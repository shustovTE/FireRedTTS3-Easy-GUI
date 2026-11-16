# Dialogs без обрезки: новая LoRA от исходной base

Следующий отдельный этап поверх готового `dialogs-08`: [подтверждённая спонтанная
речь Common Voice до 30 секунд](COMMON_VOICE_SPONTANEOUS.md).

Этот этап использует только исходные Dialogs, без YO в обучении или валидации
и без адаптеров предыдущих этапов. `models/FireRedTTS3` должен оставаться
оригинальной base; fingerprint проверяется при загрузке.

Сохраняется прежний отбор: `neutral`, полные записи до 10 секунд, максимум
30 минут **обучения на каждого из трёх актёров**. По количеству файлов треть
отбирается с `?` в любом месте, треть — с точкой в конце и без `?`/`!`, треть —
из всех оставшихся подходящих записей. Поэтому фактическая доля вопросов может
оказаться больше одной трети. Повторный выбор файла не допускается.
Все тексты пунктуационно закончены, без многоточия.

Дополнительно отложено до 3 минут 20 секунд на каждого актёра для валидации:
другие исходные сессии и последовательности слов. Это проверка знакомых
голосов на новых записях, **не проверка незнакомых дикторов**. Используются
все отложенные targets, по одному фиксированному референсу на target.

Аудио заново импортируется из оригинального `source_dir/wavs` по тому же
сохранённому плану. ASR/VAD-обрезка, обрезка до последнего слова +200 мс,
лимитер и нормализация громкости не применяются. Остаются прежние проверки
аудио, целочисленные тройки и пересчёт целого WAV в 24 кГц PCM16.
Если фактическая длительность превышает бюджет, исключается целая тройка,
а не конец аудио. Исключения перечисляются в отчёте.

## 1. Подготовить и проверить независимый набор Dialogs

На Gauss:

```bash
cd /home/firered-tuning
git pull --ff-only origin feat/russian-tts-adaptation

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -u -m fireredtts3_local.dialogs_only

cat training/datasets/dialogs-base-08/report.json
```

Старый план находится по SHA256 из
`training/datasets/dialogs-neutral/import_report.json`, поиск выполняется в
`training/plans/dialogs*/plan.json`. Это сохраняет прежний отбор.
Если план лежит в другом месте, передайте его через `--plan ПУТЬ/plan.json`.
Сеть и повторная загрузка Parquet не нужны. Нужны оригинальные WAV из
`source_dir` плана, а не `dialogs-tail-reviewed/audio`.

При прежнем отборе ожидаются около 969 train / 123 validation, 1,50 / 0,17 часа.
Точные цифры будут в новом отчёте. Старые каталоги не перезаписываются.

Рецепт готовится в `training/datasets/dialogs-base-08/recipe.json`:
LoRA rank 16, alpha 32, dropout 0.05, backbone, LR `3e-6`, batch 8,
до 5 полных эпох, preservation 0.1 против оригинальной base на обучающих
парах Dialogs. Проверка и checkpoint после каждой эпохи, patience 2.
При 969 targets — 122 обновления на эпоху, максимум 610. Replay отсутствует.
Кэш признаков отдельный: `.runtime/training-cache-dialogs-base-08`.

## 2. Удалить прежние данные YO

Выполняйте после успешной подготовки выше, до запуска нового предрасчёта.
Сначала посмотреть перечень и объём:

```bash
.venv/bin/python -m fireredtts3_local.cleanup_yo
```

Затем удалить перечисленные данные:

```bash
.venv/bin/python -m fireredtts3_local.cleanup_yo --apply
df -h /home/firered-tuning
```

Удаляются известные каталоги raw YO, импортированные/обрезанные YO, старые
смешанные manifests, планы расширения YO, два YO-support manifest у старого
Dialogs и общий `.runtime/training-cache`. Также удаляется **только**
`datasets--NCSpeech--YO-CPT-ru` из стандартных HF Hub cache, если он есть.
Отчёт с точными путями сохраняется в `training/cleanup/yo-*.json`.

Оригинальные Dialogs, подготовленный новый набор, base, старые адаптеры,
экспорты, логи и сгенерированные тестовые WAV сохраняются. Исторические
отчёты/снимки рецептов могут по-прежнему содержать пути к удалённым данным;
старые YO-прогоны после очистки нельзя продолжить. Контейнер SGLang не меняется.

Команда отказывается удалять данные при активных заданиях обучения,
пересечении с новым набором или ссылках на внешние каталоги. Она проверяет
новый Dialogs-набор до и после удаления. Произвольные копии вне известных
путей и общий кэш блоков Xet автоматически не удаляются: он не разделён по
датасетам. Не используйте глобальное удаление всего Hugging Face cache.

## 3. Предрасчёт и пилот в фоне

```bash
.venv/bin/python -m fireredtts3_local.training_job start \
  training/datasets/dialogs-base-08/recipe.json \
  --job-dir training/jobs/dialogs-base-08-pilot \
  --precompute --stop-after 30

.venv/bin/python -m fireredtts3_local.training_job status \
  training/jobs/dialogs-base-08-pilot

tail -n 20 training/jobs/dialogs-base-08-pilot/job.log
tail -n 1 training/outputs/russian-dialogs-base-08/performance.jsonl
```

SSH можно закрыть: задание отделено от терминала. Дождитесь
`status: completed`, `exit_code: 0`, `active: false` у пилота.

## 4. Продолжить тот же запуск после пилота

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from fireredtts3_local.training_job import read_status

job = Path('training/jobs/dialogs-base-08-pilot').resolve()
state = read_status(job)
assert state['status'] == 'completed' and state['exit_code'] == 0 and not state['active'], state
config = json.loads((job / 'recipe.json').read_text())
checkpoint = Path(config['output_dir'])
assert (checkpoint / 'trainer_state.pt').is_file(), 'Нет checkpoint пилота'
config['resume_checkpoint'] = str(checkpoint)
path = Path('training/recipes/dialogs-base-08-resume.json')
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(path)
PY

.venv/bin/python -m fireredtts3_local.training_job start \
  training/recipes/dialogs-base-08-resume.json \
  --job-dir training/jobs/dialogs-base-08-full
```

## 5. Метрики

```bash
.venv/bin/python -m fireredtts3_local.training_job status \
  training/jobs/dialogs-base-08-full

tail -n 3 training/outputs/russian-dialogs-base-08/validation_metrics.jsonl
cat training/outputs/russian-dialogs-base-08/coverage.jsonl
tail -n 10 training/jobs/dialogs-base-08-full/job.log
```

Здесь одна валидация; `adaptation_validation_metrics.jsonl` и
`selection_metrics.jsonl` не создаются. `best-validation` выбирается по её loss.
Перед публикацией сравните свободный синтез с base на сохранённом голосе бота:
короткие вопросы/ответы и оба длинных диагностических текста. Хороший loss
трёх актёров сам по себе не доказывает сохранения клонирования других людей.
