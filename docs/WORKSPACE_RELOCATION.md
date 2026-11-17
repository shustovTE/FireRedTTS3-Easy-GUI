# Работа из /opt/firered-tuning без симлинка

Все новые команды выполняются из `/opt/firered-tuning`. Репозиторий, `training`,
`models`, `.runtime` и виртуальные окружения уже перенесены вместе.
Старый каталог для VoxCPM workflow больше не нужен.

## Обновление и установка

```bash
cd /opt/firered-tuning
git pull --ff-only origin feat/russian-tts-adaptation
.venv/bin/python -m fireredtts3_local.voxcpm_training setup
```

Повтор `setup` обновляет editable install официального VoxCPM на новый путь.
Уже скачанные модели и пакеты используются повторно. После установки вызывайте
новое окружение явно, не через старый `activate` или сохранённые console scripts:

```bash
.venv-voxcpm2/bin/python -m fireredtts3_local.voxcpm_training check-runtime
```

Проверка читает реальные тестовые WAV через SoundFile/librosa. TorchCodec
с системными библиотеками FFmpeg для этого workflow не требуется.

## Сохранённые данные и рецепты

В прежних manifests и отчётах могут оставаться абсолютные пути
`/home/firered-tuning/...`. VoxCPM-команды сопоставляют этот известный префикс
с фактическим текущим корнем репозитория при чтении файлов. Это не симлинк:
старый путь может вообще отсутствовать.

Исходные manifests, отчёты и аудио не переписываются. Их контрольные хэши
продолжают проверяться. Новая подготовка записывает актуальные пути.
Для уже подготовленных VoxCPM-наборов при необходимости создаются небольшие
рабочие JSONL в `.runtime/voxcpm-manifests/`, чтобы официальный loader также
получал новые абсолютные пути. Тексты, порядок, reference-пары и отбор сохраняются;
хэши рабочих файлов привязываются к рецепту запуска. Повторять VAD не требуется.
Старые benchmarks и указание base в LoRA checkpoint также читаются после переноса.

Проверить уже готовый объединённый набор:

```bash
if [ -f training/datasets/voxcpm2-ru/report.json ]; then
  .venv-voxcpm2/bin/python -m fireredtts3_local.voxcpm_data audit \
    training/datasets/voxcpm2-ru
fi
```

Если вы ранее создали предложенный симлинк, он больше не используется этим
workflow. Его можно удалить отдельно; настоящие каталоги с данными удалять не нужно.
Установщик ссылки удалён из репозитория.

Далее продолжайте [инструкцию VoxCPM2](VOXCPM2_RUSSIAN_LORA.md) с нужного этапа.
Исторические FireRed resume/audit не переводятся на новую схему: эта поддержка
относится к VoxCPM и использованию существующих Dialogs/Spontaneous в нём.
