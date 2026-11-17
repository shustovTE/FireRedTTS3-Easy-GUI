# Новый LoRA от исходной базы после проверки окончаний

Этот запуск не продолжает `dialogs-04`/`yo-half-05`. Он создаёт новый адаптер
на исходной `models/FireRedTTS3`, с новым optimizer и scheduler. Рецепт проверяет
fingerprint базовых весов из первого предрасчёта; подмена на опубликованную
merged-модель завершится ошибкой. Это дообучение предобученной базы через LoRA,
а не обучение всей модели со случайной инициализации.

## Какие данные используются

- YO: существующие `yo-half/train.jsonl` и `validation.jsonl`, то есть 45 294
  обучающих и 5 033 проверочных записей до обрезки. Состав, тексты, локальные
  дикторы, отдельные референсы и историческое разделение не пересоздаются.
- YO Parquet: только 190 локальных файлов из `training/plans/yo-half/plan.json`.
  Читаются столбцы метаданных, не вложенное аудио. Файлы сверяются с хешами плана.
  Новых загрузок нет. Старые WAV и исходные Parquet не удаляются.
- Для каждой YO записи конец равен
  `min(фактическая длительность, конец последнего слова + 0.200 с)`.
  `[pause...]` не считается словом. Начало, внутренние паузы и пунктуация
  сохраняются. Граница округляется вверх до семпла; тишина не дописывается.
  Обрезаются и train, и validation, поэтому и target, и reference используют
  одинаковую подготовку. При расхождении текста, ID, времени или отсутствии
  alignment подготовка прекращается, а не включает старый хвост незаметно.
- Старые 10-секундные сегменты и `yo-complete` в этот набор не добавляются:
  их исходные Parquet могли быть удалены. Для них нельзя угадывать конец слова
  по текущим WAV. Все используемые здесь YO записи имеют локальный alignment.
- Dialogs: исходная выбранная neutral-выборка до 10 с, с теми же актёрами,
  долями по количеству question/statement/random и отдельной session validation.
  Обрезка может уменьшить минуты; добор записей и смена категорий не выполняются.

## Как проверяются Dialogs

Используется уже установленный Faster-Whisper и локально закешированный
`large-v3`, затем Silero VAD из Faster-Whisper. Ревизия не скачивает модель.
Если кэш отсутствует, передайте `--asr-model /path/to/ctranslate2-model`.

Whisper запускается с word timestamps, Russian, temperature=0, beam=5,
без VAD-фильтра перед распознаванием, без подсказки исходного текста и без
переноса контекста соседних записей. Для автоматической обрезки нужны:

1. Все слова исходного текста совпали с началом распознанного текста после
   нормализации регистра, ударений, ё/е и пунктуации. Внутренний вопросительный
   знак или точка не становятся концом всей реплики.
2. Метки слов допустимы и последовательны; средняя уверенность совпавших слов
   и уверенность последнего слова не ниже 0,8. Это оценка ASR, не гарантия.
3. За последним словом +200 мс VAD обнаружил минимум 32 мс речи (один кадр
   детектора). Используются threshold=0,5, speech_pad=0, минимальные интервалы
   речи/тишины 100 мс. При выполнении этих условий WAV обрезается на границе
   слова +200 мс, а не на начале следующего островка речи.

VAD может принимать шум/дыхание за речь, а метка Whisper может ошибаться на
протяжном последнем звуке. Поэтому **прослушайте предложенные обрезки**.
Этот алгоритм — консервативный автоматический отбор, не акустическое доказательство
границы текста. Дополнительный звук внутри первых 200 мс сохраняется.

`trim` — условия обрезки выполнены; `keep` — подтверждён текст, речи за границей
не обнаружено (длинная тишина сама по себе не обрезается); `review` — граница
не установлена надёжно. В последнем случае исходное аудио остаётся в наборе,
а запись включается в `listening_audit.json`. `complete=true` означает завершение
алгоритма, не отсутствие сомнительных записей. Все `trim` и `review` требуют
прослушивания перед обучением; исходные файлы сохранены для сравнения.

## 1. Подготовка в фоне

```bash
cd /opt/firered-tuning
git pull --ff-only origin feat/russian-tts-adaptation
mkdir -p training/logs

nohup env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -u -m fireredtts3_local.yo_retrain \
  --review-dialogs \
  --tail-padding-seconds 0.2 \
  >> training/logs/trimmed-06-prepare.log 2>&1 < /dev/null &

tail -f training/logs/trimmed-06-prepare.log
```

Это сначала ревизия Dialogs, затем обрезка YO и подготовка рецепта. CPU/int8
для Whisper выбран как уже использованный на этом хосте вариант; GPU не
требуется. Если CUDA CTranslate2 на хосте проверена, можно использовать
`--asr-device cuda --asr-compute-type float16` **с первого запуска**.
Настройки входят в контракт возобновления. Смена настроек требует нового output.

Закрытие SSH не останавливает подготовку. Повтор той же команды продолжает
законченные ASR-анализы, части метаданных YO и проверенные WAV. Файловые блокировки
не позволяют двум процессам писать в один набор. Должны сохраниться примерно
ещё столько места под WAV, сколько занимает исходная отобранная выборка YO.

Проверка после завершения:

```bash
cat training/datasets/dialogs-tail-reviewed/audit_report.json
cat training/datasets/yo-trimmed-06/preparation_state.json
cat training/datasets/yo-trimmed-06/report.json
```

В `preparation_state.json` нужны `status=completed`, `complete=true`.
`report.json` показывает минуты до/после, категории Dialogs, число срезов и шагов.
Для прослушивания:

- `training/datasets/dialogs-tail-reviewed/listening_audit.json`: все спорные
  записи и обрезки, с путями оригинала и результата, текстом и причиной.
- `training/datasets/yo-trimmed-06/tail_audit.json`: десять самых больших срезов
  и двадцать случайных записей. Время из forced alignment тоже может ошибаться.

Не используйте recipe.json из незавершённой подготовки. Повторный запуск
подготовки после старта обучения не нужен: для пилота/продолжения предназначен
сохранённый рецепт, а для другого эксперимента — новые директории.

## 2. Пилот нового обучения

Подготовленный рецепт: `training/datasets/yo-trimmed-06/recipe.json`.
Параметры: rank16/alpha32, backbone LoRA, LR=5e-6, одна эпоха YO,
batch8 (7 YO + 1 Dialogs), preservation=0,1 относительно исходной базы,
BF16/SDPA, gradient checkpointing и предрасчёт признаков. Все YO targets проходят
один раз; Dialogs выбираются speaker-balanced с возвращением, около 6471
добавлений на эпоху при прежнем количестве YO. Это не гарантия однократного
покрытия каждого Dialogs clip. Для primary данных сохранён полный coverage.

Для текущих 45 294 YO targets план составит 6471 optimizer updates. Длительности
после обрезки и скорость будут известны только после подготовки/пилота.
Одна эпоха — первый проверяемый этап, а не обещание окончательного качества.

```bash
.venv/bin/python -m fireredtts3_local.training_job start \
  training/datasets/yo-trimmed-06/recipe.json \
  --job-dir training/jobs/trimmed-06-pilot \
  --precompute --stop-after 30

.venv/bin/python -m fireredtts3_local.training_job status training/jobs/trimmed-06-pilot
tail -f training/jobs/trimmed-06-pilot/job.log
```

Обрезанные WAV получают новые SHA256, поэтому старые признаки для них не
используются. Неизменившиеся записи могут безопасно переиспользовать признаки
той же базовой модели. Старый кэш удалять не требуется.

## 3. Продолжить тот же запуск после пилота

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from fireredtts3_local.training_job import read_status
job = Path('training/jobs/trimmed-06-pilot').resolve()
status = read_status(job)
assert not status['active'] and status['status'] == 'completed' and status['exit_code'] == 0, status
config = json.loads((job/'recipe.json').read_text())
checkpoint = Path(config['output_dir']).resolve()
assert (checkpoint/'trainer_state.pt').is_file()
assert not config.get('initial_adapter')
config['resume_checkpoint'] = str(checkpoint)
path = Path('training/recipes/trimmed-06-resume.json')
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
PY

.venv/bin/python -m fireredtts3_local.training_job start \
  training/recipes/trimmed-06-resume.json \
  --job-dir training/jobs/trimmed-06-full
```

## Проверка результата

Формула training loss сохранена: сравниваем эффект новой подготовки, не меняя
одновременно веса stop-классов. В validation добавлены отдельные метрики:

- `stop_terminal_probability`, `stop_terminal_bce`: последний кадр каждой реплики;
- `stop_terminal_recall_05`/`03`: доля терминальных состояний с pstop ≥0,5/0,3;
- `stop_early_any_05`/`03`: доля примеров с хотя бы одним преждевременным stop
  после минимальных шести patches, но до терминального состояния.

Это **teacher-forced** оценки с истинной аудиоисторией, не частота ошибок
свободной генерации. Нужны оба вида проверки: независимые голоса/WER и прослушивание
вопросов, коротких ответов, длинных реплик с внутренними паузами при нескольких seed.
Проверьте 0,5 и 0,3 заново; рабочий порог старой модели нельзя автоматически
приписывать новому адаптеру. Ничего не сливается и не публикуется автоматически.

## Проверка трёх проблемных вопросов после шага 6471

```bash
bash scripts/evaluate_endings.sh
```

Скрипт читает сохранённый голос `Dovbish_Anastasiia` из контейнера
`sglang-fire-tts-firered-tts3-1`, копирует тот же WAV и текст референса, затем
генерирует три вопроса при seed 1234/1235/1236 старым `yo-half-05/best-validation`
и новым `trimmed-base-06/checkpoint-step-006471`. Порог native генератора — 0,5,
без разбиения текста; параметры сохраняются в settings.json. Внешний API сервера
для синтеза не используется, поэтому обновлять работающую модель пока не нужно.

Результаты: `training/eval/endings-06/{old-05,new-06}/q1_seed1234.wav` и т.д.
Всего 18 WAV. Слушайте хвост, последнее слово целиком и продолжение после
внутреннего вопросительного знака. Сравнение выполняется в native backend;
после него отдельно проверяются клонирование на прежнем benchmark и SGLang
после локального слияния. Совпадение seed не обещает одинаковый звук между
native и SGLang. Автоматические WER/similarity здесь не рассчитываются.

Для другого контейнера/голоса или повторного запуска укажите новую папку:

```bash
bash scripts/evaluate_endings.sh \
  sglang-fire-tts-firered-tts3-1 Dovbish_Anastasiia training/eval/endings-06-retry
```

Скрипт требует непустой ref_text сохранённого голоса и новую выходную папку.
Он не перезапускает контейнер, не меняет его настройки и не публикует модель.
