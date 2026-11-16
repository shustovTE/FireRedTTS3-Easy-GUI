# Один прогон от base: Dialogs + Common Voice Spontaneous

Этот режим `joint-10` начинает с **исходной FireRedTTS3 Base** и обучается на
объединении уже подготовленных `dialogs-base-08` и `cv-spontaneous-09`.
Адаптеры `08` и `09` не загружаются. Обе выборки участвуют в одном обучении;
каждый target используется ровно один раз за эпоху, с другим референсом своего
диктора. Replay и дополнительное повторение маленького корпуса отсутствуют.

Это экспериментальный режим по итогам [аудита метода](TRAINING_METHOD_AUDIT.md).
CPU-тесты проверяют математику, градиенты, замороженную base, возобновление и
слияние. Скорость, память и устранение слышимых артефактов требуют проверки на GPU.

## Что меняется в обучении

| Компонент | Режим |
|---|---|
| Backbone | Свежая LoRA, rank 16 / alpha 32, dropout 0, LR `1e-6` |
| Акустический DiT и `dit_head` | Полное обучение, LR `1e-6` |
| `stop_head` | Полное обучение, LR `1e-5`; его loss не передаёт градиент в backbone |
| PatchEncoder, CAM++, проекции голоса, RedAE | Заморожены |

Обучается около **313 млн параметров**. Полное обучение акустического DiT не
означает обучение waveform-декодера RedAE. Последний оставлен исходным: для его
обучения потребовался бы отдельный waveform objective и другой эксперимент.

PEFT сохраняет обучаемые полные модули и отдельные неизменные оригиналы для
preservation loss. При выборе модулей учтено сопоставление PEFT по окончанию
имени: `dit` не должен случайно включать `spk_proj_dit`. Загружайте гибридный
адаптер через этот репозиторий; для других runtime экспортируйте полные веса.
Слияние проверяет также точное сохранение всех обученных bias, convolution и
normalization tensors, а не только матриц LoRA.

Точность следует нативному графу: BF16 autocast внутри backbone, FP32 для
PatchEncoder, проекций, DiT и stop head. RedAE использует свой исходный режим
encode. Новый кэш `redae-campp-patches-native-v3` отделён от прежнего BF16-кэша
PatchEncoder. Поэтому необходим новый предрасчёт.

Stop loss: половина веса приходится на последний STOP, половина — на среднюю
ошибку предыдущих CONTINUE. Вклад STOP больше не уменьшается пропорционально
длине записи. Это меняет калибровку головы; прежний порог `0.3` автоматически
не переносится. Контрольная генерация использует `0.5`. Значения `loss`/`stop`
нельзя напрямую сравнивать с прогонами `08`/`09`.

Preservation `0.1` сравнивает student с исходной base на тех же обучающих парах.
Это ограничитель изменений, а не гарантия сохранения голоса или качества длинной
свободной генерации. Teacher forcing остаётся частью flow objective.

## Данные и объём

Новые загрузки не нужны. Подготовка проверяет отчёты, хэши WAV и разделение,
создаёт новые manifests со ссылками на существующие полные записи. Нового
ограничения громкости или обрезки окончаний нет. Сохраняются уже выполненные
преобразования исходных этапов, включая фиксированное уменьшение уровня отдельных
CV-файлов при импорте, если оно было нужно для записи PCM16.

| Корпус | Обучение при прежнем отборе | Отдельная валидация |
|---|---:|---:|
| Dialogs, neutral, до 10 секунд | 969 записей, 1,50 часа | 123 записи |
| Common Voice, подтверждённые транскрипты, до 30 секунд | 168 записей, 0,68 часа | 24 записи |
| Вместе | **1137 записей, около 2,18 часа** | Два отдельных отчёта |

Сохраняются категории отбора по количеству записей, лимиты на диктора и
дополнительная валидация поверх учебного бюджета. Обучение не использует
отложенные записи ни как targets, ни как референсы. Это валидация знакомых
дикторов на других записях, а не широкая проверка незнакомых голосов.

Batch 8, accumulation 1: **143 обновления на эпоху, до 5 эпох / 715 шагов**.
Последний batch неполный. Если размеры локальных выборок отличаются, правильные
числа появятся в `report.json`. Проверка после каждой эпохи; остановка после
двух проверок без допустимого улучшения.

Гибридный адаптер в FP32 занимает ориентировочно **1,25 ГБ**, полный checkpoint
с Adam — несколько ГБ. Предусмотрены ранние адаптеры каждые 50 шагов и полные
checkpoint после эпох. Для полного прогона оставьте примерно **50 ГБ свободно**,
плюс запас для нового кэша и контрольных WAV. Старые оценки 70 МБ и скорости
backbone-only LoRA для этого режима не подходят.

## 1. Обновить код и подготовить объединение

Все команды выполняются на Gauss:

```bash
cd /home/firered-tuning
git pull --ff-only origin feat/russian-tts-adaptation

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -m fireredtts3_local.joint_training prepare

.venv/bin/python -m fireredtts3_local.joint_training audit
```

Нужны исходная base в `models/FireRedTTS3`, существующие подготовленные каталоги
`training/datasets/dialogs-base-08`, `training/datasets/cv-spontaneous-09` и
`training/eval/dialogs-base-08/benchmark.jsonl` с доступными reference WAV.
Benchmark копируется с абсолютными путями и привязывается к хэшам аудио.
Проверяются пересечения его текстов/референса с обучением. Прежние адаптеры
не нужны. При нестандартных путях есть `--dialogs`, `--spontaneous`, `--recipe`.

Новый рецепт: `training/datasets/russian-joint-10/recipe.json`.
Новый output: `training/outputs/russian-joint-base-10`.
Подготовка откажется перезаписывать существующий набор или заполненный output.
Если она уже успешно выполнена, используйте только `audit`.

## 2. Предрасчёт и пилот 30 шагов в фоне

Запускайте, когда предыдущий training job завершён. Не запускайте второй пилот
или полный прогон одновременно. Worker работает после отключения SSH.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -m fireredtts3_local.training_job start \
  training/datasets/russian-joint-10/recipe.json \
  --job-dir training/jobs/joint-10-pilot \
  --precompute --stop-after 30
```

Первый запуск делает предрасчёт, валидацию base и контрольную генерацию до
первого шага. Это может занять заметно больше времени, чем сами 30 обновлений.

```bash
.venv/bin/python -m fireredtts3_local.training_job status training/jobs/joint-10-pilot
tail -n 20 training/jobs/joint-10-pilot/job.log
tail -n 1 training/outputs/russian-joint-base-10/performance.jsonl
cat training/outputs/russian-joint-base-10/generation_metrics.jsonl
```

До первых шагов `performance.jsonl` ещё нет. На пилоте контрольная генерация
относится к **step 0**, а не к step 30; это отмечено в metadata checkpoint.
Пилот проверяет исполнение, память и скорость. Сравнивать качество обучения
можно после контрольной генерации очередной эпохи.

## 3. Продолжить тот же прогон

После статуса `completed` у пилота:

```bash
.venv/bin/python -m fireredtts3_local.joint_training resume \
  --job training/jobs/joint-10-pilot \
  --output training/recipes/joint-10-resume.json

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -m fireredtts3_local.training_job start \
  training/recipes/joint-10-resume.json \
  --job-dir training/jobs/joint-10-full
```

Это продолжает step 30 с теми же optimizer/scheduler/RNG и порядком targets,
а не начинает новую стадию. Предрасчёт повторять не нужно. При повторном вызове
создания resume-файла уже существующий файл не перезаписывается.

```bash
tail -n 3 training/outputs/russian-joint-base-10/validation_metrics.jsonl
tail -n 3 training/outputs/russian-joint-base-10/adaptation_validation_metrics.jsonl
tail -n 3 training/outputs/russian-joint-base-10/generation_metrics.jsonl
tail -n 5 training/jobs/joint-10-full/job.log
```

Первый validation-файл — Dialogs, второй — Spontaneous. В `selection_metrics.jsonl`
используется равный вес **двух средних validation loss**, независимо от размера
корпусов. `optimizer_metrics.jsonl` показывает LR и нормы градиента backbone,
акустических модулей и stop head до общего clipping. `coverage.jsonl` показывает
число targets и часы на каждой законченной эпохе.

Для корректной остановки после текущего обновления:

```bash
.venv/bin/python -m fireredtts3_local.training_job stop training/jobs/joint-10-full
```

## Контроль свободной генерации и выбор checkpoint

При step 0 и каждой validation генерируются все случаи из прежнего benchmark
с теми же reference/text/seed: `do_split=False`, CFG 2, 10 flow steps, STOP 0.5.
Сохраняются:

- `generation/step-.../case-....float.wav` — исходные значения, включая превышения ±1;
- `case-....wav` — PCM16-копия для прослушивания, только с общим понижением уровня
  при необходимости; коэффициент `pcm_gain` записан в отчёт;
- `report.json` — причина завершения, stop scores, уровень акустических latents,
  посекундные RMS/peak, доля превышений full scale и разница конца/начала.

Кандидат не заменит `best-validation`, если достигает лимита генерации,
превышает ±1 более чем в 0,1% сэмплов либо его рост RMS между первыми и последними
10 секундами больше аналогичного роста base на 3 dB. Это инженерные пороги:
они не распознают каждое искажение и не заменяют проверку произнесённых слов,
тембра и просодии. PCM-копии могут звучать тише; сравнивайте также float-метрики.

`best-validation` первоначально содержит **step 0 как fallback**. Если все
обученные кандидаты провалили проверку или не улучшили loss, там останется base.
Смотрите поле `step` и `generation_guard` в `firered_lora_metadata.json`.
Корневой adapter — последний сохранённый шаг, он не обязательно лучший.
Ранние `adapter-step-*` не содержат optimizer и предназначены для сравнения;
для resume используйте корень output или `checkpoint-step-*`.

После обучения используйте прежние `adapt evaluate`/`adapt compare` на том же
benchmark и обязательно проверьте длинные контрольные WAV. Средний WER и
similarity на одном голосе уже пропускали слышимый срыв. Экспорт через
[merge/publish CLI](MERGE_AND_PUBLISH.md) поддерживает новый профиль, но запуск
обучения сам ничего не публикует в HF и не меняет работающий SGLang.
