# Адаптация FireRedTTS3-Base для русского с сохранением клонирования

Этот тренер реализует экспериментальную адаптацию по публичному графу инференса
и статье [FireRedTTS3, §2.1–2.2](https://arxiv.org/pdf/2608.17492).
Официальный код обучения не опубликован. Параметры рецепта — отправная точка
для эксперимента, а не подтверждённый оптимум. Улучшение произношения и сохранение
голоса необходимо измерять на GPU и прослушиванием.

## Что изменено

- Аудио и транскрипция никогда не обрезаются независимо друг от друга.
- Каждая пара использует разные непересекающиеся записи одного диктора и языка.
- Дикторы выбираются сбалансированно. Global ID используется для балансировки и
  предотвращения утечки; пары формируются по явно заданному локальному speaker_id.
- Validation отделяется по связанным дикторам, global ID, исходным записям и точным
  хешам файлов. Это не заменяет проверку ошибочных/пропущенных ID и near-duplicates.
- Stop loss: N нулевых меток перед N патчами, одна положительная ПОСЛЕ последнего.
  Используется средний BCE без искусственного переобучения на последнем патче.
- Flow loss: равномерная выборка нескольких патчей без замещения и t из [0,1).
  Полный каузальный backbone-контекст сохраняется. Это наш вариант реконструкции
  objective, поскольку точное распределение t/patch sampling авторы не раскрыли.
- Для CFG совместно зануляются условия backbone и диктора с вероятностью 0.1;
  чистая акустическая история сохраняется. Вероятность взята из статьи, совместная
  маска выбрана по unconditional-ветке официального инференса. LoRA dropout иной.
- LoRA применяется к полным именам разрешённых модулей, не к `all-linear`.
- Есть replay и удержание исходной модели через отключение LoRA на frozen base.
- Предрасчёт латентов RedAE, CAM++ и frozen PatchEncoder по хешам аудио и base.
- Пакетные backbone/DiT с масками разной длины; loss усредняется по примерам.
- Ограниченная фоновая загрузка CPU-признаков, pinned memory и перенос на GPU.
- Validation имеет фиксированные пары/шум для сравнения чекпоинтов и не меняет RNG
  обучения. Сохраняются sampling RNG, Python/Torch/CUDA RNG, optimizer и scheduler.
- Частичный gradient accumulation нормируется по фактическому числу примеров.
- Сохранение `best-validation`, early stopping, TensorBoard и JSONL-метрики.
  Лучший validation loss **не означает** лучшее клонирование или просодию.

## Установка и требования

Рабочая директория во всех примерах — корень репозитория. Используйте существующее
окружение Python 3.12.10. Для импорта Parquet добавлен необязательный extra:

```bash
uv sync --frozen --extra data --no-install-project
```

На Linux запускайте `.venv/bin/python`, на Windows — `.venv\Scripts\python.exe`.
Ниже `python` означает этот интерпретатор. Флаг `--no-install-project` позволяет
работать из checkout без перегенерации tracked egg-info.

Импорт, audit, split, unit/integration-тесты работают на CPU. Настоящее обучение
требует NVIDIA CUDA с BF16; SDPA используется по умолчанию, FlashAttention опционален.
Рецепт не обещает конкретное потребление VRAM: размер модели, длина текста/аудио,
число патчей, replay, preservation и профиль влияют на него. Сначала запускайте
короткую проверку на целевом GPU. При нехватке памяти уменьшайте длины при подготовке
данных, `batch_size`, `patches_per_example` или используйте `backbone`; не обрезайте транскрипции
отдельно от аудио. Кэш может занимать много места; пустой `cache_dir` отключает его.

## Импорт YO-CPT-ru

Скачайте выбранные Parquet shards из [YO-CPT-ru](https://huggingface.co/datasets/NCSpeech/YO-CPT-ru)
с фиксированным commit SHA. Можно использовать `huggingface_hub.hf_hub_download`
с `repo_type="dataset"`, `filename="data/ru000-000.parquet"`, `revision=SHA`.
Импорт читает Parquet пакетами, не загружает весь многотысячечасовой корпус в RAM.

```bash
python -m fireredtts3_local.adapt prepare-yo \
  /data/yo/ru000-000.parquet /data/yo/ru000-001.parquet \
  --output training/datasets/yo \
  --source-revision YOUR_DATASET_COMMIT_SHA \
  --min-mos 4.0 --max-seconds 10 --max-hours 50 --max-speaker-minutes 30
```

Выходная директория должна быть новой/пустой. `max-hours` — бюджет, а не гарантия
набора такого объёма: строгая фильтрация может оставить значительно меньше.
Порядок shard-файлов влияет на ограниченную выборку. Отбирайте разные источники;
первые 50 часов не являются случайной или репрезентативной выборкой корпуса.
Порог DistillMOS — инженерная настройка и несопоставим напрямую с NISQA.

Импорт проверяет язык, MOS, длительность, моно, конечность samples и сильный clipping.
RUAccent `+` перед гласными и combining acute убираются; арифметический `+` сохраняется.
Пунктуация сохраняется только при совпадении слов `text_denorm`, `text` и alignment
(игнорируются регистр, ударения и е/ё). Изменённые числа/слова отклоняются.
Отрезки выделяются по пунктуации либо измеренным паузам, строго по word alignment.
Если подходящей границы нет, запись отклоняется. Короткий остаток также приводит к
отклонению записи. После ограничения выборки исключаются дикторы без двух разных
подготовленных записей. `import_report.json` содержит причины отказов (обычно по
исходным utterance; `segments_without_distinct_reference` считается по сегментам).

Автоматические оценки не гарантируют правильные ударения, идентичность дикторов,
естественность пунктуации или отсутствие артефактов denoising. Прослушайте выборку
вместе с текстами, особенно вопросы, окончания фраз и повторяющихся ведущих.
Импорт не использует описания лиц/персон для Base.

## Manifest и собственные данные

Каждая строка JSONL:

```json
{"utt_id":"clip-001","audio":"audio/001.wav","text":"Здравствуйте! Когда вам удобно?","language":"Russian","speaker_id":"corpus:local-speaker-12","global_spk_id":"corpus:person-5","source_id":"corpus:recording-8","start":0.0,"end":4.2}
```

Обязательны `audio`, `text`, `speaker_id`, `language`. Пути относительно manifest
или абсолютные. Код `ru` преобразуется в `Russian`. Идентификаторы разных корпусов
обязательно снабжайте разными префиксами. `source_id` обозначает всю исходную запись,
а не каждый нарезанный WAV. Для отрезков одной записи нужны точные `start`/`end`.
В YO `source_id` группирует видео для split, а `segment_source_id` указывает исходный
utterance, внутри которого отсчитываются времена word alignment. Не смешивайте
локальные часы разных utterance при проверке пересечения аудио.
`global_spk_id` необязателен и учитывается только если известен. Непроверенный global
ID не используется для разрешения пар между локальными дикторами.

В GUI обычная подготовка читает `.txt` плюс `.json` с `speaker_id` и необязательными
source/global/start/end полями. Один speaker_id для всей папки разрешён только при
явном заполнении соответствующего поля. Наличие одного голоса в каждом файле не
означает, что во всех файлах один человек. Старые manifest без speaker_id требуют
миграции; идентичность автоматически не угадывается.

Для Balalaika нужно сначала восстановить надёжные speaker_id. `is_single_speaker`
не подходит в качестве идентификатора. Для русского общего назначения сохраняйте
разнообразие дикторов и стилей, а не только суммарные часы.

## Отложенные данные и audit

```bash
python -m fireredtts3_local.adapt split training/datasets/yo/manifest.jsonl \
  --output training/datasets/yo-split --fraction 0.1 --seed 1234
```

Разделение строится по связанным компонентам speaker/source/global/hash.
Крупные компоненты могут дать долю, отличающуюся от 10%; это предпочтительнее утечки.
В обоих split нужны пары одного диктора, включая достаточно короткие референсы.
Одиночные записи и непригодные пары вызывают явную ошибку. Данные следует отобрать
или дополнить, а не молча использовать чужие голоса.

Если validation_manifest не задан, trainer делает это разделение автоматически.
`validation_scope="recording"` допускается для явно однодикторного эксперимента;
такая проверка **не измеряет перенос на незнакомые голоса**. Отдельный benchmark
незнакомых дикторов нужен в любом случае. Глобальные ID неполны: дополнительно
проверьте повторяющихся людей между видео/корпусами и дубликаты после перекодирования.

```bash
python -m fireredtts3_local.adapt options > options.json
# Укажите validation_manifest в options.json, если manifest уже содержит только train.
python -m fireredtts3_local.adapt audit training/datasets/yo/manifest.jsonl
```

Audit не скачивает модель и не требует GPU. Он читает аудиометаданные и хеширует файлы:
для большого корпуса это занимает время. Неподходящие длины отклоняются, не обрезаются.

## Предрасчёт и пакетное обучение на RTX PRO 6000 Blackwell 96 GB

Для этой карты добавлен отдельный стартовый рецепт
[recipes/russian_blackwell_96gb.json](../recipes/russian_blackwell_96gb.json).
Он использует `batch_size=8`, `gradient_accumulation=1`, BF16, SDPA, без gradient
checkpointing. Эффективный batch остаётся **8 пар на optimizer step**, как у
прежнего рецепта 1×8. Это исходная конфигурация для замера, а не измеренный
предел VRAM или обещание ускорения. Старый `russian_backbone.json` сохранён
для небольших GPU и ленивого заполнения кэша.

После импорта и split проверьте пути train/validation/replay в рецепте.
Исходные веса `FireRedTeam/FireRedTTS3` должны уже находиться в
`models/FireRedTTS3` (их скачивает обычная загрузка модели в GUI).
Предрасчёт работает на GPU и загружает только RedAE, CAM++ и PatchEncoder,
не создавая Qwen backbone и DiT:

```bash
python -m fireredtts3_local.adapt precompute recipes/russian_blackwell_96gb.json
python -m fireredtts3_local.adapt train recipes/russian_blackwell_96gb.json --stop-after 30
```

`precompute` обрабатывает все клипы train, validation и replay, исключая повторные
хеши аудио. Повторный запуск проверяет и переиспользует готовые файлы. Прерывание
не требует пересчёта уже сохранённых клипов. Для восстановления повреждённых
записей используйте ту же команду с `--repair`. Файлы записываются атомарно.
Ключ кэша включает версию расчёта, полный fingerprint base/tokenizer, точность,
версию PyTorch и attention backend; CPU- и CUDA-признаки не смешиваются.
При изменении этих параметров нужен новый предрасчёт.

RedAE и CAM++ вычисляются **отдельно для каждого клипа**: официальный RedAE.encode
обрабатывает только первый элемент входа, а padding разных аудиозаписей меняет
контекст encoder и статистику CAM++. Замороженный PatchEncoder работает в eval,
включая его dropout. Его готовые выходы также сохранены. В режиме
`require_precomputed_features=true` обучение проверяет наличие всех файлов до
первого шага и запрещает скрытый запуск аудиоэнкодеров; содержимое проверяется
при загрузке. RedAE/CAM++ выгружаются в CPU RAM, освобождая VRAM. Для необязательного
Eval Audio они временно возвращаются на GPU.

Кэш хранит float32-признаки без дополнительной квантизации. Для стандартных
размерностей Base это примерно **0.21 GB на час аудио**, около 10–11 GB на 50 часов,
плюс накладные расходы файлов. Оставьте место также для исходных shards, нарезанных
WAV, весов и checkpoints. Кэш лучше держать на локальном NVMe, не сетевом диске.
WAV не удаляйте: audit при каждом запуске проверяет их метаданные и хеши.
Весь YO-CPT-ru скачивать не требуется — достаточно импортированной выборки.

Один пакетный student forward обрабатывает несколько пар разной длины; teacher
обрабатывает только примеры, где включено удержание base. Padding находится после
полной последовательности и исключён из stop/flow loss. История акустики не
смешивается с текстом или соседними примерами. Flow и stop усредняются сначала
внутри примера, затем между примерами: длинная запись не получает больший вес.
`bucket_by_length` сортирует уже выбранные пары только внутри текущего optimizer
step. Баланс дикторов и вероятность replay сохраняются, примеры не отбрасываются.

Первые 30 шагов нужны для проверки GPU, памяти и скорости. `--stop-after` завершает
заданный бюджет обновлений этого запуска и сохраняет состояние с исходным
`training_steps=1000` и расписанием LR. Для продолжения добавьте в тот же рецепт
`"resume_checkpoint": "training/outputs/russian-blackwell-01"` и запустите `train`
без `--stop-after`. Изменять batch, accumulation или остальные настройки при resume
нельзя; для сравнения конфигураций создавайте новый output/run.

В `performance.jsonl` записываются время шага с CUDA synchronization, пары/сек,
объём целевого аудио, время ожидания данных/постановки копирования и пиковая
allocated VRAM. CUDA allocator reserve и память драйвера в allocated VRAM не входят;
смотрите также `nvidia-smi`. После разогрева оцените время 1000 шагов по медиане:

```bash
python - <<'PYTIME'
import json, statistics
from pathlib import Path
rows = [json.loads(line) for line in Path(
    'training/outputs/russian-blackwell-01/performance.jsonl').read_text().splitlines()]
seconds = statistics.median(row['step_seconds'] for row in rows[5:])
print(f'{seconds:.2f} s/step; 1000 updates: {1000*seconds/3600:.2f} h')
PYTIME
```

Эта оценка не включает предрасчёт, validation, генерацию Eval Audio и сохранение.
Их время прибавляется отдельно. При OOM начните новый run с `batch_size=4` и
`gradient_accumulation=2` (те же 8 пар), уменьшите также `validation_batch_size`;
при необходимости включите `gradient_checkpointing`. При наличии запаса VRAM
сравнивайте скорости на одинаковом наборе и бюджете пар. Увеличение batch до 16
при accumulation=1 удваивает примеры на шаг и меняет эксперимент.

## Второй этап: каждый обучающий клип целиком

[recipes/russian_coverage_96gb.json](../recipes/russian_coverage_96gb.json) продолжает
адаптацию от `training/outputs/russian-blackwell-01/best-validation` в новом output
`training/outputs/russian-coverage-02`. Используется прежний split: validation
остаётся вне обучения. Готовый кэш полностью переиспользуется.

Это новый этап с загрузкой весов LoRA через `initial_adapter`, новым optimizer и
новым расписанием LR. `resume_checkpoint` продолжает прерванный этап с его optimizer,
RNG и scheduler; менять бюджет завершённого этапа через resume по-прежнему нельзя.
Перед загрузкой проверяются fingerprint base, профиль, rank/alpha/dropout и точные
имена адаптируемых модулей. Хеш исходного адаптера записывается в рецепт нового
этапа; сохраняйте исходную папку адаптера неизменной, в том числе для resume.

Рецепт второго этапа:

- `sampling_strategy: "coverage"`, `length_mode: "Epochs"`: каждый обучающий
  фрагмент становится целевым ровно один раз за эпоху; порядок меняется по эпохам.
  Референс случайно выбирается из других непересекающихся записей того же диктора.
- `patches_per_example: "all"`: flow loss включает все целевые акустические патчи,
  с независимыми шумом и временем t для каждого; это увеличивает память и вычисления.
- `validation_patches_per_example: 8`: прежние validation-пары и шум остаются
  сопоставимыми с первым пилотом. Менять это число между сравниваемыми запусками
  без новой исходной оценки нельзя. `null` по умолчанию наследует обучающий бюджет.
- LR `5e-6`, до пяти эпох, validation и checkpoint после каждой эпохи;
  `validate_every: "epoch"`, `patience: 2`. Если две эпохи подряд не улучшат validation,
  обучение закончится раньше. Выбирайте `best-validation`, затем проверяйте синтез.
- `backbone`, удержание исходной base с весом 0.1; исходная base остаётся teacher,
  даже когда student начинает с уже обученной LoRA.

Пять эпох и LR — стартовые настройки эксперимента, не установленный оптимум.
`coverage` даёт каждой записи один проход: вклад дикторов определяется числом их
записей и уже выполненным ограничением корпуса. Равномерный выбор дикторов остаётся
в режиме `speaker_balanced`. Пока `coverage` требует пустой `replay_manifest`;
регуляризация против оригинальной base при этом работает.

Для 9483 обучающих файлов, batch 8 и accumulation 1 одна эпоха содержит 1186
обновлений (последний пакет — 3 файла), а пять эпох — максимум 5930 обновлений.
После полной эпохи появляется строка `[coverage]` и запись в `coverage.jsonl` с
числом целей и часами целевого аудио. Остановка/возобновление посреди эпохи и
фоновая загрузка сохраняют тот же порядок и не пропускают оставшиеся файлы.

```bash
python -m fireredtts3_local.adapt train recipes/russian_coverage_96gb.json --stop-after 30
```

Это новый режим нагрузки, поэтому сначала измерьте VRAM и время по
`training/outputs/russian-coverage-02/performance.jsonl`. Время шага старого режима
с восемью патчами не является прогнозом для `"all"`.
Для продолжения скопируйте рецепт в `training/recipes/coverage-resume.json`, добавьте
`"resume_checkpoint": "training/outputs/russian-coverage-02"` и запустите без
`--stop-after`. Сохраните `initial_adapter` в этом рецепте для проверки происхождения.
При нехватке памяти начните новый output с batch 4 и accumulation 2, при
необходимости включите gradient checkpointing. Бюджет полных проходов сохранится.

## Обучение

Исходный рецепт: [recipes/russian_backbone.json](../recipes/russian_backbone.json).

```bash
python -m fireredtts3_local.adapt train recipes/russian_backbone.json
```

`backbone` изменяет только линейные слои Qwen. `backbone_dit` добавляет DiT и
`dit_head` с LR, умноженным на `dit_lr_scale` (по умолчанию 0.25). RedAE, CAM++,
PatchEncoder, embeddings, проекции диктора и stop head заморожены. Stop loss всё
равно обучает состояния backbone, поступающие в frozen head.

Первый пилот — проверка метода, а не обещание качества. Сравните backbone и
backbone_dit при одном split и сопоставимом бюджете. Только после подтверждённого
улучшения расширяйте корпус, например с 20–50 до 200–500 часов. Эти числа не взяты
из официального fine-tuning recipe.

Дополнительные опции:

| Опция | Смысл |
|---|---|
| `replay_manifest` | Дополнительные настоящие пары для удержания разнообразия; не могут пересекаться с validation |
| `replay_ratio` | Вероятность выбрать replay для каждой пары; 0.2 — исходная гипотеза |
| `preservation_weight` | MSE flow-предсказаний и вероятностей остановки против base с отключённой LoRA |
| `validation_pairs` | Число фиксированных, воспроизводимо выбранных пар для оценки loss |
| `patience` | Число validation без улучшения; 0 отключает early stopping |
| `patches_per_example` | Число выбранных целевых патчей или `"all"` для всех патчей |
| `validation_patches_per_example` | Отдельный бюджет validation; `null` наследует обучающий |
| `sampling_strategy` | `speaker_balanced` или полные эпохи `coverage` |
| `validate_every` | Период в шагах или `"epoch"` в режиме Epochs |
| `max_prompt_seconds`, `max_target_seconds` | Пределы целых подготовленных клипов; не команды обрезания |
| `batch_size`, `validation_batch_size` | Число пар в одном пакетном проходе; default 1 для совместимости |
| `require_precomputed_features` | Читать только готовый кэш; не вычислять аудиопризнаки во время обучения |
| `offload_frozen_encoders` | Убрать RedAE/CAM++ с GPU; требует готового кэша |
| `loader_workers`, `prefetch_batches` | Потоки CPU-загрузки и ограничение очереди; работают в режиме готового кэша |
| `bucket_by_length` | Группировка по длине только внутри текущего optimizer step |
| `gradient_checkpointing` | Non-reentrant checkpointing backbone для уменьшения памяти |

Если replay задан, preservation применяется к replay. Иначе он явно применяется
к основному корпусу. Значение 0 отключает teacher. Teacher использует те же шум,
t, историю и маску CFG; его LoRA отключена, dropout выключен, градиентов нет.
Это дополнительная регуляризация, не часть заявленного авторами рецепта. Слишком
сильный teacher может подавлять полезную адаптацию. Только русский replay не
подтверждает сохранение других языков или межъязыкового клонирования.

Для возобновления добавьте в исходный JSON `resume_checkpoint` с путём к checkpoint
или корню завершённого/остановленного run. Остальные настройки, total_steps, входные
файлы, split и base должны совпадать. Checkpoints старого scalar-тренера (schema 2) и all-linear несовместимы с resume
нового пакетного тренера (schema 3). Существующие adapters можно использовать
для инференса. Stop завершает текущий optimizer step целиком и затем сохраняет
состояние. Последний неполный batch/шаг эпохи нормируется по фактическому числу пар.
Предварительно загруженные будущие пары не продвигают сохраняемый sampler RNG. Аварийное прерывание можно
возобновить с последнего полностью записанного checkpoint. Не редактируйте аудио
в процессе обучения.

Выход содержит recipe, точные train/validation/replay manifest, список trainable
параметров, хеш base/tokenizer, метрики и RNG/optimizer state. `best-validation`
может остаться на step 0, если адаптация не улучшила loss. Это корректный результат,
а не повод объявлять последний checkpoint лучшим. Непустой output без явного
resume не перезаписывается.

В GUI рецепт находится в LoRA Training → Russian adaptation / validation recipe.
Он сохраняется вместе с проектом. Можно вставить путь к импортированному manifest.
В GUI вставляйте в это поле только объект `options` из JSON; остальные поля
рецепта задаются отдельными контролами. Для рецепта Blackwell установите Gradient
Accumulation = 1, иначе 8×8 даст 64 пары на шаг. Предрасчёт запускается из CLI.
Одиночное Eval Audio предназначено для быстрого прослушивания и не заменяет benchmark.

## Проверка качества клонирования

Подготовьте JSONL с одинаковыми текстами для нескольких незнакомых дикторов:

```json
{"id":"speaker-a-question-1","speaker_id":"heldout:a","reference_audio":"refs/a.wav","reference_text":"Точный текст референса.","text":"Подскажите, пожалуйста, когда вам удобно встретиться?","language":"Russian","seed":1234}
```

Не включайте этих людей/записи в обучение; benchmark — отдельный от validation loss
набор. Включите короткие ответы, вопросы, редкие слова/ударения, длинные фразы и
несколько референсов на человека. Для межъязыкового клонирования добавьте такие
примеры явно. Числа и сокращения приведите к произносимой форме одинаково для обеих
моделей. Ударения оценивайте слухом: ASR WER их часто не видит.

```bash
python -m fireredtts3_local.adapt evaluate benchmark.jsonl --output training/eval/base \
  --speaker-model microsoft/wavlm-base-plus-sv --asr-model large-v3
python -m fireredtts3_local.adapt evaluate benchmark.jsonl --output training/eval/adapted \
  --adapter training/outputs/russian-backbone-01/best-validation \
  --speaker-model microsoft/wavlm-base-plus-sv --asr-model large-v3
python -m fireredtts3_local.adapt compare training/eval/base/report.json \
  training/eval/adapted/report.json --max-sim-drop 0.02 --max-wer-increase 0.01
```

Скореры опциональны для генерации аудио, но оба обязательны для автоматического
сравнения. Они работают на CPU; их загрузка требует отдельного дискового места.
Для воспроизводимости используйте локальные зафиксированные snapshots скореров.
WavLM-base-plus-sv независим от CAM++, но это не тот же fine-tuned WavLM-Large,
которым авторы измеряли таблицы статьи: абсолютные значения несопоставимы.

Gate проверяет средний WER и худшее среднее падение similarity по диктору, чтобы
улучшение других голосов не скрыло потерю одного. Пороги 0.02/0.01 — настраиваемые
инженерные допуски, не научно установленные перцептивные границы. Ни gate, ни loss
не измеряют всю просодию. Прослушайте результаты вслепую и сравните окончания,
ударения, вопросы, различимость голосов и стабильность тембра на длинных фразах.

## Экспорт полного checkpoint для SGLang Omni

```bash
python -m fireredtts3_local.adapt export \
  --base models/FireRedTTS3 \
  --adapter training/outputs/russian-backbone-01/best-validation \
  --sglang-config /data/ckermit/config.json \
  --output training/exports/russian-backbone-01
```

Передайте config из совместимой зафиксированной ревизии CKermit/FireRedTTS3-Base.
Экспорт на CPU проверяет исходные артефакты по fingerprint, размеры backbone/core/
RedAE, делает `merge_and_unload(safe_merge=True)` всего core, проверяет сохранённые
ключи и shapes и копирует tokenizer, RedAE, CAM++. В provenance явно записано, что
загрузка SGLang ещё не проверена. Не переносите только backbone: при профиле
backbone_dit обучены и акустические компоненты. Проверьте исходную и адаптированную
модели в зафиксированной версии интеграции SGLang, включая сходство синтеза с PyTorch.

## Проверки разработчика

```bash
python -m unittest discover -s tests -v
```

Тесты используют настоящие Qwen/PatchEncoder/DiT уменьшенного размера, PEFT,
optimizer/checkpoints, синтетическое аудио и Parquet. Проверяют градиенты, causal
teacher-forcing против incremental generation, stop-target, CFG, teacher, split,
replay leakage, cache invalidation и идентичность resumed/uninterrupted весов.
Пакетный граф сверяется по loss и градиентам с независимым scalar-графом на
разных длинах, смешанном preservation и короткой акустической истории; проверяются
кэш против online-признаков, gradient checkpointing, предзагрузка и неполные batch.
Они не заменяют GPU-прогон полной модели, оценку качества или проверку SGLang.
