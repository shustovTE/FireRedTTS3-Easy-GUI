# Обучение на расширенной YO-выборке после Dialogs

Первый большой этап делает **один полный проход** по новым YO targets и
продолжает адаптер `russian-dialogs-04/best-validation`. Результат сохраняется
отдельно в `training/outputs/russian-yo-half-05`.

Для отчёта с 45 294 train clips / 178,5321 часа и 5033 validation clips / 18,7173
часа это 6471 обновление. В полном обновлении 7 YO + 1 Dialogs; последнее содержит
4 YO + 1 Dialogs. Все YO targets участвуют ровно один раз. Референсы выбираются
отдельно среди других записей того же локального диктора. Replay содержит только
прежние **обучающие** Dialogs; за эпоху будет 6471 выборка с повторениями,
с балансировкой дикторов. Это не 6471 разных диалогов.

Настройки первого этапа:

- LoRA rank 16 / alpha 32 / dropout 0,05, профиль backbone;
- learning rate 2e-6, новый optimizer/schedule для этого этапа;
- все целые реплики и все акустические patches для обучения;
- max_prompt_seconds / max_target_seconds = null;
- BF16, SDPA, gradient checkpointing, бюджет 2048 tokens на microbatch;
- предрасчёт признаков, выгрузка замороженных энкодеров с GPU;
- base-model preservation 0,1 на YO и Dialogs. Его teacher — исходная базовая
  модель без адаптера, а не отдельная замороженная копия dialogs-04;
- валидация до обучения, каждые 500 обновлений и в конце;
- 256 фиксированных speaker-balanced пар из YO validation и все оставшиеся
  Dialogs validation targets с фиксированными референсами/шумом;
- best-validation по среднему двух validation losses с равными весами корпусов.

256 пар — выборка с возвращением из YO holdout, а не все 5033 записи и не
гарантированно 256 разных голосов. Весь holdout сохраняется. Для валидации
используются 8 patches на пример. Сравнивать абсолютный loss с прежними этапами
на других данных некорректно; сравниваем шаг 0 и последующие шаги этого этапа.

Для одного запланированного полного прохода patience=0 отключает early stopping.
Сохраняются best-validation, финальный checkpoint и состояние для продолжения.
Остановить процесс с сохранением можно через training_job stop. Улучшение loss
не гарантирует сохранения интонации или сходства голоса: после этапа нужны
сравнение с dialogs-04 и исходной моделью на прежнем benchmark и прослушивание.

## 1. Обновить код и подготовить проверенный рецепт

```bash
cd /opt/firered-tuning
git pull --ff-only origin feat/russian-tts-adaptation

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
.venv/bin/python -u -m fireredtts3_local.yo_training \
  --expansion training/datasets/yo-half \
  --output training/plans/yo-half-training
```

Подготовка работает на CPU. Она проверяет завершение извлечения, соответствие
train/validation отчёту, WAV-хеши, отдельные референсы, разделение дикторов и
совместимость исходного адаптера. Ничего заново скачивать или извлекать не нужно.

Если текст Dialogs validation совпадает с текстом YO train или Dialogs train
(с учётом нормализации и без различий пунктуации), он исключается **из новой
копии Dialogs validation**. После этого проверяются пары и сохранение всех актёров
в проверке. Если это невозможно, подготовка завершается ошибкой. Количество
исключений записывается в отчёт; оригинальные manifests и WAV не меняются.

Выходные файлы:

- `training/plans/yo-half-training/report.json` — фактические размеры, длительности,
  fingerprints, исключения из Dialogs validation и рассчитанные шаги;
- `recipe.json` — рецепт для запуска;
- `dialogs-validation.jsonl` — проверенная копия отложенных диалогов.

Используйте подготовленный `recipe.json`; файл в `recipes/` является шаблоном.
Каталог плана должен быть новым. После успешной подготовки повторять её не нужно.

## 2. Предрасчёт признаков и пилот на 30 шагов в фоне

```bash
.venv/bin/python -m fireredtts3_local.training_job start \
  training/plans/yo-half-training/recipe.json \
  --job-dir training/jobs/yo-half-pilot \
  --precompute --stop-after 30

.venv/bin/python -m fireredtts3_local.training_job status \
  training/jobs/yo-half-pilot

tail -f training/jobs/yo-half-pilot/job.log
```

SSH можно закрыть. `Ctrl+C` у tail завершает только просмотр. Предрасчёт включает
YO train/validation, replay и Dialogs validation; совпадающие готовые признаки
переиспользуются. Время предрасчёта не входит в оценку длительности шагов обучения.

После `status: completed`, `exit_code: 0` проверьте производительность:

```bash
.venv/bin/python - <<'PY'
import json
import statistics
from pathlib import Path
root = Path('training/outputs/russian-yo-half-05')
rows = [json.loads(s) for s in (root/'performance.jsonl').read_text().splitlines() if s.strip()]
warm = [r for r in rows if r['step'] > 5] or rows
seconds = statistics.median(r['step_seconds'] for r in warm)
total = json.loads(Path('training/plans/yo-half-training/report.json').read_text())['total_steps']
print('Завершено шагов:', rows[-1]['step'], '/', total)
print(f'Медианное время шага: {seconds:.3f} сек')
print(f"Пиковая выделенная память PyTorch: {max(r['peak_allocated_gb'] for r in rows):.2f} GB")
print(f"Остаток без проверок и сохранений: {max(0, total-rows[-1]['step'])*seconds/60:.1f} мин")
PY
```

Память PyTorch относится только к этому процессу, не ко всей GPU. Скорость пилота
приблизительная: состав пар и длины следующих batches будут отличаться.

## 3. Продолжить тот же этап после успешного пилота

Снимок рецепта сохраняет тот же график learning rate и данные. Не меняйте epochs,
batch size, ratio или learning rate между пилотом и продолжением.

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from fireredtts3_local.training_job import read_status
job = Path('training/jobs/yo-half-pilot').resolve()
status = read_status(job)
assert not status['active'] and status['status'] == 'completed' and status['exit_code'] == 0, status
config = json.loads((job/'recipe.json').read_text())
checkpoint = Path(config['output_dir']).resolve()
assert (checkpoint/'trainer_state.pt').is_file(), 'Нет сохранённого состояния пилота'
config['resume_checkpoint'] = str(checkpoint)
path = Path('training/recipes/yo-half-resume.json')
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
PY

.venv/bin/python -m fireredtts3_local.training_job start \
  training/recipes/yo-half-resume.json \
  --job-dir training/jobs/yo-half-full

tail -f training/jobs/yo-half-full/job.log
```

Это продолжит с шага 30, сохранив порядок оставшихся примеров и состояние
optimizer. Повторный предрасчёт не нужен. Для остановки с сохранением:

```bash
.venv/bin/python -m fireredtts3_local.training_job stop training/jobs/yo-half-full
```

## 4. Результаты этапа

```bash
cat training/outputs/russian-yo-half-05/coverage.jsonl
cat training/outputs/russian-yo-half-05/selection_metrics.jsonl
tail -n 15 training/jobs/yo-half-full/job.log
```

Ожидается одна строка coverage: 45 294 targets, 178,5321 часа, 6471 replay draws.
Числа могут отличаться для другого исходного отчёта: они вычисляются из manifests.
`best-validation` может относиться к промежуточному шагу или к исходному шагу 0,
если обучение не улучшило выбранную метрику. Финальный адаптер находится в корне
output, checkpoint последнего шага — в `checkpoint-step-006471` для этой выборки.
Прежде чем выбирать следующий проход, оцениваем готовый адаптер на тех же
невиданных голосах, коротких вопросах и ответах.
