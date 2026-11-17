# VoxCPM2 LoRA: Dialogs + Spontaneous + Common Voice Scripted

Готовится **новая LoRA исходной VoxCPM2**, без переноса адаптеров FireRed.
Используется официальный trainer, закреплённый по commit. Его обучающая цель
не переписана. Работа на GPU запускается на вашем хосте; команды ниже рассчитаны
на `/opt/firered-tuning` и RTX PRO 6000 Blackwell 96 GB.

## Что входит в обучение

| Корпус | Источник | Отбор |
|---|---|---|
| Dialogs | `training/datasets/dialogs-base-08` | Прежние исходные полные записи: neutral, до 10 с, категории/лимиты и split сохранены |
| Spontaneous | `training/datasets/cv-spontaneous-09` | Прежние подтверждённые записи до 30 с, сохранены разметка, категории и split |
| Common Voice Scripted 27.0 Russian | Новый проверенный архив Mozilla | `validated.tsv` + совпадающий текст в `validated_sentences.tsv`, исключены `reported.tsv`; законченная пунктуация и другие проверки текста; без дополнительного лимита длины |

Для Scripted — максимум **30 минут учебных записей на client_id**, отдельно
примерно 200 секунд валидации. Целые записи не режутся ради бюджета. Для
независимых reference/target нужно хотя бы две разные подходящие реплики в каждом
split; валидатор может превысить ориентир 200 секунд ради двух целых длинных
записей, что отражается в отчёте. Здесь нет обязательного баланса третей: он
сохраняется как исходный отбор Dialogs/Spontaneous, а не навязывается читке.

Проверенная 17 ноября 2026 версия Scripted:

- ID: `cmu5x45pn00dao107j4o9w2yv`;
- архив: `common-voice-scripted-speech-27-0-russia-72f7812a.tar.gz`;
- размер: **7 093 766 588 bytes**;
- SHA256: `3f269caf4757a7837048153067ca84e29c53fd5dee9394a55e6ecc92afb6055a`;
- корень: `cv-corpus-27.0-2026-09-11/ru`;
- все confirmed audio: 178 296 записей, 3392 client_id, около 254,4 часа;
- фактически извлечены и проверены после текстовых фильтров и квот:
  71 780 train / 10 577 validation, около 104,5002 / 16,2963 часа, 2407 client_id;
- квоты перепроверены по длительности, прочитанной импортёром из аудиофайлов:
  ещё 63 записи исключены по сравнению с планом на основе TSV-длительности.
  У MP3 это оценка libsndfile; точное число декодированных отсчётов используется
  на следующем этапе VAD и в финальных manifests VoxCPM.

Последние числа **до VAD и сверки текстов с двумя вашими корпусами**; конечные
числа берите из локального отчёта. Scripted — чтение, и при одном проходе по
всем записям оно будет преобладать по объёму. Этот рецепт не делает искусственных
повторов маленьких корпусов. Улучшение спонтанной просодии нужно проверить,
а не выводить из общего числа часов или падения loss.

Текстовый фильтр проверяет конечные `. ! ?`, исключает многоточия, явные обрывы,
небалансированные кавычки/скобки и некоторые неоднозначные окончания. Он
не дописывает слова или пунктуацию. Это консервативная формальная проверка,
**не гарантия грамматической завершённости каждого русского предложения**.

## VAD для всех трёх корпусов

Каждая выбранная запись train/validation проходит **Silero VAD 6.2.3**; все
референсы выбираются из этих же проверенных записей. Фиксируются версия пакета,
SHA256 bundled JIT-модели и параметры. ASR не используется.

- Если до первого обнаруженного голоса больше 200 мс, удаляется лишнее.
- Если после последнего голоса больше 200 мс, удаляется лишнее.
- Если край короче 200 мс, тишина не добавляется.
- Внутренние паузы сохраняются. Нового ограничения продолжительности нет.
- Нет нормализации, компрессии или gain. Выход — FLOAT WAV, непрерывный отрезок
  исходных декодированных отсчётов с исходной частотой. Копия 16 kHz нужна
  только для VAD; официальный AudioVAE позже сам получит 16 kHz.
- Исходники не перезаписываются. Отсутствие речи или некорректное аудио дают
  явное исключение в отчёте. Если потерян любой из трёх корпусов, подготовка
  завершается ошибкой.

200 мс отсчитываются **от границ Silero**, а не от вручную размеченного конца
слова: VAD может ошибаться на тихих звуках. Список всех срезов сохраняется в
`vad-audit.jsonl`; его можно проверить и воспроизвести.

MP3-длительность берётся из фактически декодированных отсчётов: число frames
в заголовке может быть приблизительным. На реальных Scripted MP3 проверены
границы VAD, точное совпадение сохранённого отрезка и неизменность исходников.
Отдельно проверены установка, официальный токенизатор, загрузчик и пересчёт
FLOAT WAV в 16 kHz. Обучение и генерация на GPU в облачной проверке не запускались.

## 1. Обновить репозиторий

```bash
cd /opt/firered-tuning
git pull --ff-only origin feat/russian-tts-adaptation
```

Если каталог перенесён из `/home/firered-tuning`, один раз восстановите доступ
по старым абсолютным путям, не меняя manifests и их контрольные хэши:

```bash
python3 scripts/relocate_workspace.py --old-root /home/firered-tuning
```

Подробности: [перенос рабочего каталога](WORKSPACE_RELOCATION.md). Готовые
VAD-аудио и скачанные модели повторно создавать не нужно.

```bash
test -s training/datasets/dialogs-base-08/train.jsonl
test -s training/datasets/dialogs-base-08/validation.jsonl
test -s training/datasets/cv-spontaneous-09/train.jsonl
test -s training/datasets/cv-spontaneous-09/validation.jsonl
```

Нужны именно исходные подготовленные корпуса, а не версии `*-tail-reviewed`.
Новые manifests не зависят от весов FireRed или его feature cache.

## 2. Отдельное окружение и исходная модель

```bash
cd /opt/firered-tuning
.venv/bin/python -m fireredtts3_local.voxcpm_training setup
```

Команда использует установленный `uv`, создаёт `.venv-voxcpm2`, устанавливает
PyTorch 2.8/cu128, официальный VoxCPM и Silero 6.2.3, скачивает исходные веса.
Прежняя `.venv` FireRed не изменяется. Для этого workflow закреплён
`datasets==3.6.0`: официальный loader читает WAV через SoundFile/librosa,
а native VoxCPM2 использует librosa для reference. Setup проверяет реальное
чтение target/reference FLOAT WAV и ресэмплинг в 16 kHz без загрузки весов.

TorchCodec остаётся установлен как зависимость upstream, но не используется
этими путями и не является обязательной проверкой. Ошибка старого setup
`Could not load libtorchcodec` с отсутствующими `libavutil.so.56–59` исправлена:
обновите репозиторий и повторите `setup`. Устанавливать старый FFmpeg или менять
PyTorch для этого workflow не требуется. Самостоятельная проверка:

```bash
.venv-voxcpm2/bin/python -m fireredtts3_local.voxcpm_training check-runtime
```

Версии:

- VoxCPM source: `300e4260b509066151d623dc9c0aa922cf82232b`, `.runtime/VoxCPM2`;
- `openbmb/VoxCPM2`: `7936080b41163874f38610f46f438e2c0757f9fd`, `models/VoxCPM2`;
- список установленных зависимостей: `.venv-voxcpm2/voxcpm-installed.txt`.

После установки используйте **`.venv-voxcpm2/bin/python`** для новых команд.
HF берёт токен из своего обычного локального хранилища/окружения, если он задан.

## 3. Скачать Scripted

```bash
.venv-voxcpm2/bin/python -m fireredtts3_local.common_voice_scripted download
```

Ключ Mozilla вводится скрыто при запросе либо берётся из переменной
`MOZILLA_DATA_COLLECTIVE_API_KEY`. Ключ и подписанная ссылка не записываются
в конфиги и логи. Не вставляйте ключ в командную строку или файл репозитория.

Архив сохранится в:

```text
training/raw/common-voice-scripted-27-ru/common-voice-scripted-speech-27-0-russia-72f7812a.tar.gz
```

При обрыве повторите ту же команду: загрузчик проверяет/продолжает свой `.part`,
а готовый архив принимает только при совпадении полного SHA256 и размера.
Архив и прежние данные автоматически не удаляются. Помимо 7,1 GB архива нужны
место для выбранных MP3, FLOAT WAV, модели и checkpoint; следите за `df -h`.

## 4. Отбор, VAD и рецепт — в фоне, без активного SSH

```bash
mkdir -p training/logs
nohup bash scripts/prepare_voxcpm2_ru.sh \
  > training/logs/voxcpm2-prepare.log 2>&1 < /dev/null &
```

Скрипт последовательно:

1. Выбирает Scripted, резервируя validation отдельно и учитывая тексты прежних
   train/validation, чтобы новый корпус не пересёк их границы.
2. Проверяет Silero все выбранные записи трёх корпусов и создаёт новые WAV.
3. Подбирает разные target/reference одного диктора внутри одного split.
4. Проверяет объединение на пересечение нормализованных текстов, исходных
   записей и хэшей аудио, включая аудио после обрезки. Сравниваются также сами
   PCM-отсчёты: разные заголовки WAV не скрывают аудиодубликат.
5. Собирает привязанный к данным рецепт и проверяет длины **всех** train/validation
   target/reference официальным токенизатором.

Просмотр:

```bash
cat training/logs/voxcpm2-prepare.status
tail -n 30 -f training/logs/voxcpm2-prepare.log
```

`Ctrl+C` закрывает только просмотр. Продолжайте после `completed` в status и
`[prepare] COMPLETE` в логе. При `failed` ошибка находится в конце лога.
Повтор скрипта проверяет уже готовые этапы; существующие результаты с другими
данными/настройками не перезаписываются. Незавершённый VAD-этап при аварийном
прерывании потребуется выполнить заново.

Краткий отчёт:

```bash
.venv-voxcpm2/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('training/datasets/voxcpm2-ru/report.json')
r = json.loads(p.read_text())
for name, corpus in r['corpora'].items():
    for split in ('train', 'validation'):
        x = corpus[split]['retained']
        print(f"{name:12} {split:10}: {x['clips']} записей, {x['hours']:.2f} ч, {x['speakers']} ID дикторов")
print('Исключения:', r['rejections'])
print('VAD:', r['vad']['version'], r['vad']['model_sha256'])
PY

cat training/recipes/voxcpm2-ru.yaml
```

Дополнительные отчёты:

- `training/datasets/cv-scripted-27/report.json` — отбор нового корпуса;
- `training/datasets/voxcpm2-ru/vad-audit.jsonl` — все успешные VAD-срезы;
- `training/datasets/voxcpm2-ru/rejections.jsonl` — явные причины исключений;
- `training/datasets/voxcpm2-ru/*-provenance.jsonl` — источник, текст, диктор,
  reference и хэши каждой финальной пары;
- `training/recipes/voxcpm2-ru.audit.json` — число обновлений и проверка контекста.

Нет лимита Scripted в 10/20/30 с. Остаётся технический предел контекста VoxCPM:
если целая пара не помещается, сборка **остановится с указанием записи**,
а не обрежет или молча выкинет её через фильтр официального loader.

## 5. Пилот 30 шагов

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
.venv-voxcpm2/bin/python -m fireredtts3_local.voxcpm_training start \
  training/recipes/voxcpm2-ru.yaml \
  --job-dir training/jobs/voxcpm2-ru-pilot \
  --pilot-steps 30
```

```bash
.venv-voxcpm2/bin/python -m fireredtts3_local.voxcpm_training status \
  training/jobs/voxcpm2-ru-pilot
tail -n 30 -f training/jobs/voxcpm2-ru-pilot/job.log
```

Пилот отдельный, output `training/outputs/voxcpm2-ru-lora-pilot-30`. Он сохраняет
scheduler полного плана, но заканчивает после 30 обновлений. Это проверка запуска,
памяти и времени; не сравнение окончательного качества и не точный resume начала
полного обучения. Основной запуск снова начинает от исходной VoxCPM2.

Перед полным запуском сравните хотя бы проблемные длинные реплики Base и пилота
по инструкции ниже. Штатный успешный exit не является оценкой звука.

## 6. Основной запуск

После завершения пилота и проверки генерации:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
.venv-voxcpm2/bin/python -m fireredtts3_local.voxcpm_training start \
  training/recipes/voxcpm2-ru.yaml \
  --job-dir training/jobs/voxcpm2-ru-full
```

Обучение продолжится при закрытии SSH. Прогресс и статус:

```bash
.venv-voxcpm2/bin/python -m fireredtts3_local.voxcpm_training status \
  training/jobs/voxcpm2-ru-full

tail -n 30 -f training/jobs/voxcpm2-ru-full/job.log
```

Остановить с запросом сохранения штатному trainer:

```bash
.venv-voxcpm2/bin/python -m fireredtts3_local.voxcpm_training stop \
  training/jobs/voxcpm2-ru-full
```

Не останавливайте процесс через `kill -9`. Этот launcher намеренно отказывается
начинать обучение поверх непустого output: неявный официальный auto-resume не
используется. Для прерванного основного запуска сначала проверьте checkpoint;
эта первая версия обвязки не обещает точного продолжения по RNG/dataloader.

## Рецепт и ограничения штатной валидации

Первый рецепт: один проход по размеру объединения, batch 1 × accumulation 8,
LoRA r32/alpha32/dropout0, обе LM + DiT, без LoRA дополнительных проекций,
AudioVAE frozen. LR1e-4 взят из официального LoRA template; это исходная настройка
для проверки, не доказанный оптимум русского. Warmup до 100 обновлений, cosine,
clip norm1. Число шагов вычисляется из фактических записей; округление accumulation
может захватить несколько примеров следующего прохода. Нет обещания, что каждая
запись встретится строго один раз и ни одна не повторится в последнем update.

`val_manifest` задан явно. Но pinned official trainer считает validation loss
только по **первым 10 batch**, а его TensorBoard sample helper не передаёт
референс клонирования. Весь отложенный corpus остаётся на диске; порядок
validation перемежает корпуса и дикторов, чтобы первые batch не оказались
только Dialogs. Это ограниченная проверка, не оценка всей валидационной выборки.
Аудит данных/VAD/контекста, напротив, охватывает все записи.

Checkpoint сохраняется каждые 100 обновлений (для коротких runs чаще) и в конце:

```text
training/outputs/voxcpm2-ru-lora/step_0000100/
training/outputs/voxcpm2-ru-lora/latest/
```

`latest` означает последний сохранённый, а не автоматически лучший по звуку.

## 7. WAV Base и LoRA: целые реплики и тот же голос

Используется существующий benchmark `training/eval/dialogs-base-08/benchmark.jsonl`.
В нём должны быть поля `id`, `text`, `reference_audio`, `reference_text`, `seed`.
Референсы и их точные транскрипты остаются одинаковыми для обоих запусков.

Base (можно сделать до обучения):

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
.venv-voxcpm2/bin/python -m fireredtts3_local.voxcpm_eval \
  --benchmark training/eval/dialogs-base-08/benchmark.jsonl \
  --model models/VoxCPM2 \
  --output training/eval/voxcpm2-ru/base
```

Пилот:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
.venv-voxcpm2/bin/python -m fireredtts3_local.voxcpm_eval \
  --benchmark training/eval/dialogs-base-08/benchmark.jsonl \
  --model models/VoxCPM2 \
  --adapter training/outputs/voxcpm2-ru-lora-pilot-30/latest \
  --output training/eval/voxcpm2-ru/pilot-30
```

После основного обучения:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
.venv-voxcpm2/bin/python -m fireredtts3_local.voxcpm_eval \
  --benchmark training/eval/dialogs-base-08/benchmark.jsonl \
  --model models/VoxCPM2 \
  --adapter training/outputs/voxcpm2-ru-lora/latest \
  --output training/eval/voxcpm2-ru/adapted
```

Получатся `case-00000.wav`, `case-00000.float.wav`, остальные cases и `report.json`.
FLOAT сохраняет сырой уровень; PCM-копия использует при необходимости один
постоянный gain, записанный в отчёте. Ни разбивки на предложения, ни повторной
генерации плохого результата, ни ASR здесь нет. Используется combined clone
mode с исходным reference и точным prompt transcript, CFG2, 10 flow steps,
не более 1000 patches (~160 с); достижение лимита явно отмечается как ошибка.

Проверять нужно полный текст, пропуски середины, хвосты, вопросительную интонацию,
клонирование и нарастание тембра/громкости. Этот benchmark — прежний небольшой
набор разработки, а не доказательство качества всех незнакомых голосов.
`report.json` не выдаёт автоматический perceptual gate.

## Источники

- [VoxCPM source, pinned](https://github.com/OpenBMB/VoxCPM/tree/300e4260b509066151d623dc9c0aa922cf82232b)
- [Официальный LoRA template](https://github.com/OpenBMB/VoxCPM/blob/300e4260b509066151d623dc9c0aa922cf82232b/conf/voxcpm_v2/voxcpm_finetune_lora.yaml)
- [Scripted dataset](https://mozilladatacollective.com/datasets/cmu5x45pn00dao107j4o9w2yv)
- [Silero VAD 6.2.3](https://pypi.org/project/silero-vad/6.2.3/)
