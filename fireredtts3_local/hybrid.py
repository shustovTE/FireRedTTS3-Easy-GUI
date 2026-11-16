"""Strict ownership of LoRA and dense weights in the joint acoustic trainer."""
from __future__ import annotations

from contextlib import contextmanager

from .training_objective import dense_targets


@contextmanager
def exact_dense_scope(core, modules):
    """Work around PEFT's unqualified endswith matching for modules_to_save.

    Some PEFT versions match 'dit' to 'spk_proj_dit'. Temporarily remove only
    unintended top-level suffix matches while PEFT constructs/loads wrappers.
    Restore the same frozen objects before any forward, including on errors.
    No global monkeypatch and no renaming of saved model tensors.
    """
    hidden = {name: module for name, module in core.named_children()
              if name not in modules and any(name.endswith(target) for target in modules)}
    try:
        for name, module in hidden.items():
            module.requires_grad_(False)
            delattr(core, name)
        yield
    finally:
        for name, module in hidden.items():
            setattr(core, name, module)


def create_adapter(core, config):
    from peft import get_peft_model
    with exact_dense_scope(core, config.modules_to_save or []):
        return get_peft_model(core, config)


def load_adapter(core, path, *, is_trainable=False):
    from peft import PeftConfig, PeftModel
    config = PeftConfig.from_pretrained(str(path))
    with exact_dense_scope(core, config.modules_to_save or []):
        return PeftModel.from_pretrained(core, str(path), is_trainable=is_trainable)


def check_trainable(core, profile):
    from peft.tuners.lora.layer import LoraLayer
    from peft.utils.other import ModulesToSaveWrapper
    base = core.get_base_model()
    expected = set()
    prefixes = ("backbone_llm.", "dit.", "dit_head") if profile == "backbone_dit" else ("backbone_llm.",)
    for name, layer in base.named_modules():
        if isinstance(layer, LoraLayer):
            if not name.startswith(prefixes):
                raise RuntimeError(f"LoRA outside declared profile: {name}")
            for table in (layer.lora_A, layer.lora_B):
                if set(table) != {"default"}:
                    raise RuntimeError("Training requires one default adapter")
                expected.update(id(p) for p in table['default'].parameters())
    if not expected:
        raise RuntimeError("Expected trainable backbone LoRA")
    for name in dense_targets(profile):
        module = getattr(base, name)
        if not isinstance(module, ModulesToSaveWrapper) or set(module.modules_to_save) != {'default'}:
            raise RuntimeError(f"Dense module {name} must preserve its frozen original")
        if any(p.requires_grad for p in module.original_module.parameters()):
            raise RuntimeError(f"Teacher weights became trainable: {name}")
        expected.update(id(p) for p in module.modules_to_save['default'].parameters())
    named = [(n, p) for n, p in core.named_parameters() if p.requires_grad]
    if {id(p) for _, p in named} != expected:
        raise RuntimeError("Trainable parameters differ from the declared LoRA/dense profile")
    return named
