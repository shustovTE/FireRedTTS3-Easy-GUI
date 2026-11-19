# LoRA и Full по рецептам VoxCPM2 для FireRedTTS3-Base

Для повтора выбранной версии колл-центра используйте [рецепт backbone шага 500](FIRERED_BACKBONE_500_RU.md). В нём зафиксированы точный прогон, состав адаптера при синтезе и результаты остальных экспериментов.

Два независимых эксперимента стартуют с оригинального FireRedTTS3-Base. Источник — проверенные Dialogs и Spontaneous из `training/datasets/voxcpm2-dialogs-spontaneous-full-1000` на Gauss. Программа сверяет отчёт, исходники, разбиение, WAV и тексты.

На каждого диктора приходится максимум 30 минут уникального учебного аудио. Длительность измеряется по сэмплам готового WAV после VAD и обработки шума в Spontaneous. Записи сортируются от длинных к коротким, при равенстве — по `utt_id`. Отбор заканчивается перед первой записью, с которой сумма превысила бы 1 800 секунд. Поэтому у диктора может получиться, например, 29:56, даже если среди исключённых есть трёхсекундная реплика. Все записи используются целиком. Референсы для клонирования берутся из оставшихся записей того же диктора. Цель без отдельного подходящего референса исключается и учитывается в отчёте.

Отложенная валидация идёт сверх лимита, с исходными парами. После отбора программа заново распределяет 16 000 учебных показов с прежним правилом равномерных повторов и минимум 30% реплик со знаком вопроса среди показов. LoRA и Full получают одни и те же пары в одном порядке.

| Параметр | LoRA | Full |
| --- | --- | --- |
| Обучаемые веса | Q/K/V/O attention в Qwen и FireRed DiT, rank 32, alpha 32, dropout 0 | Весь FireRed Base core, включая PatchEncoder, Qwen, DiT, проекции и STOP |
| LR | 0.0001 | 0.00001 |
| Общие настройки | 1 000 обновлений, микробатч 2, накопление 8 | Те же |
| Оптимизатор | AdamW, weight decay 0.01, клиппинг нормы 1, warmup 100, cosine до нуля | Те же |
| Обучающая цель | Flow + STOP, вес каждого 1, все действительные позиции микробатча | Та же |

RedAE и CAM++ заморожены. Времена для flow matching берутся из logit-normal распределения VoxCPM2 и обращаются для направления ODE FireRed. FireRed использует собственный бинарный STOP, свою историю, токенизатор и клонирование с текстом референса. Обучение идёт по истории записи. В Full PatchEncoder вычисляется с градиентом из латентов RedAE. Сохранённые эмбеддинги PatchEncoder используются только для LoRA. При валидации учитываются все отложенные пары. В штатном тренере VoxCPM2 число проверочных батчей ограничено.

Это экспериментальное соответствие рецептам на другой архитектуре. Первичные источники — [LoRA](https://github.com/OpenBMB/VoxCPM/blob/300e4260b509066151d623dc9c0aa922cf82232b/conf/voxcpm_v2/voxcpm_finetune_lora.yaml), [Full](https://github.com/OpenBMB/VoxCPM/blob/300e4260b509066151d623dc9c0aa922cf82232b/conf/voxcpm_v2/voxcpm_finetune_all.yaml), [flow matching VoxCPM2](https://github.com/OpenBMB/VoxCPM/blob/300e4260b509066151d623dc9c0aa922cf82232b/src/voxcpm/modules/locdit/unified_cfm.py), [FireRedTTS3, §2.2](https://arxiv.org/pdf/2608.17492).

## 1. Проверить рабочую директорию

Команды выполняются на Gauss в окружении FireRed. Для Full потребуется место под полный генератор и два состояния Adam для каждого сохранённого шага. Пилот, шаг 500 и шаг 1000 сохраняются отдельно.

```bash
cd /opt/firered-tuning
df -h .
ls training/datasets/voxcpm2-dialogs-spontaneous-full-1000/report.json
.venv/bin/python -m fireredtts3_local.vox_recipe_transfer selection
mkdir -p training/logs
```

Команда `selection` выполняется на CPU и печатает число оставшихся записей, их суммарную длительность и путь к отчёту. Посмотреть длительности и число исключений по каждому диктору можно так.

```bash
jq '.speakers' training/outputs/firered-vox-recipe-12-cap30m/speaker_selection.json
```

## 2. Запустить два пилота последовательно

Каждый пилот выполняет первые 30 из 1 000 обновлений с исходным расписанием LR. Первый запуск подготовит кэш FireRed. Фоновая оболочка переживает закрытие SSH и запускает Full после успешного завершения LoRA.

```bash
nohup bash scripts/firered_vox_pair.sh 30 > training/logs/firered-vox-cap30m-pilot.log 2>&1 < /dev/null &
```

## 3. Послушать пилот

```bash
tail -n 30 -f training/logs/firered-vox-cap30m-pilot.log
```

Последняя строка успешного сравнения указывает на `comparison-0030-…/report.json`. Рядом лежат папки `base/`, `lora/`, `full/` с исходными float WAV. Имена файлов соответствуют порядку в `cases.json`. По умолчанию используются один отложенный референс Dialogs, вопрос, полные тексты «по городам» и «Тимофей», seed 1234 и 1235. Для своего голоса передайте команде `compare --benchmark путь/к/benchmark.jsonl` с полями `id`, `text`, `reference_audio`, `reference_text`, `seed` и `language`. Путь к WAV может быть относительным к JSONL. Для повторных сравнений используйте тот же файл. ASR и проверка диктора при первом запуске могут скачать large-v3 и microsoft/wavlm-base-plus-sv. Параметры `--asr` и `--speaker-model` принимают и локальные каталоги.

В отчёте есть распознанный текст, предполагаемые пропущенные фрагменты, сходство голоса целиком и в последней трети, причина остановки, клиппинг и изменение RMS. Флаги отмечают три подряд пропущенных слова, рост WER больше 0.05, снижение сходства больше 0.02 и дополнительный рост уровня больше 3 дБ относительно базы. У попадания в предел длительности и доли клиппинга выше 0.1% отдельные флаги. Это рабочие пороги. Прослушайте окончания и сверяйте выделенные пропуски со звучанием. База также может получить флаг.

## 4. Продолжить оба режима

После проверки пилотов можно продолжить с сохранёнными весами, оптимизатором, расписанием и состоянием генераторов случайных чисел. Оба запуска используют одинаковые следующие пары.

```bash
nohup bash scripts/firered_vox_pair.sh 500 > training/logs/firered-vox-cap30m-500.log 2>&1 < /dev/null &
```

После прослушивания шага 500 запускается заключительная половина. Скрипт использует сохранённый чекпойнт каждого режима и позволяет повторить сравнение готовой пары. Необязательный второй аргумент — путь к своему benchmark.jsonl, одинаковому для всех шагов.

```bash
nohup bash scripts/firered_vox_pair.sh 1000 > training/logs/firered-vox-cap30m-full.log 2>&1 < /dev/null &
```

## 5. Сравнить результаты

```bash
tail -n 30 training/logs/firered-vox-cap30m-full.log
find training/outputs/firered-vox-recipe-12-cap30m -name report.json
```

Чекпойнты находятся в `lora/step-0500`, `full/step-0500`, `lora/step-1000`, `full/step-1000`. Внутри LoRA лежат файлы PEFT, внутри Full — каталог `fireredtts3_base` с весами и конфигурацией генератора для исходного комплекта FireRed. Эксперимент сохраняет кандидатов отдельно от исходной модели. Выбор версии делается по полным записям и отчётам. Для независимого эксперимента можно указать новое значение `--output` для всех команд.

## Повтор пилота с выравниванием громкости Dialogs и Spontaneous

Эксперимент применяет две операции ко всем готовым WAV учебной и отложенной частей. Сначала программа оценивает RMS в 50-миллисекундных кадрах и сглаживает уровень активных кадров в окне 3 с. Длительные громкие участки ослабляются относительно нижнего квартиля огибающей этой реплики. Зона нечувствительности составляет 1 дБ, максимальное ослабление 6 дБ, скорость изменения до 2 дБ/с. Уровень тихих пауз исключён из расчёта огибающей.

Затем постоянное усиление приводит каждую реплику к **−23 LUFS integrated** по измерителю R128 в FFmpeg. Если при этом пики выйдут за предел, применяется лимитер с четырёхкратной передискретизацией и компенсацией задержки. Проверка читает готовый PCM WAV и требует **−23 ± 0,2 LU и true peak ≤ −1 dBTP**. Для записей короче измерительного окна 400 мс только вход измерителя дополняется тишиной. Частота дискретизации, число сэмплов, тексты, разбиение, пары и порядок показов сохраняются. Для полностью беззвучной записи программа укажет путь и остановит подготовку.

Одна команда запускает CPU-подготовку, пересчёт признаков, два свежих пилота от Base и обе серии контрольных генераций. Для прямого сравнения укажите отчёты прошлого шага 30, рядом с каждым нужен его исходный `benchmark.jsonl`. Это даёт те же референсы, тексты, seeds и параметры инференса. Выходные WAV контрольных генераций сохраняются с исходной громкостью.

```bash
cd /opt/firered-tuning
git pull --ff-only origin feat/russian-tts-adaptation
ffmpeg -version | head -n 1
mkdir -p training/logs
old=training/outputs/firered-vox-recipe-12-cap30m
nohup bash scripts/firered_vox_pair.sh --levels 30 \
  "$old/comparison-0030-2hjv4g5z/report.json" \
  "$old/comparison-0030-9e51l0dc/report.json" \
  > training/logs/firered-vox-levels-30.log 2>&1 < /dev/null &
tail -n 30 -f training/logs/firered-vox-levels-30.log
```

Копии всего корпуса и отчёт по каждому WAV находятся в `training/datasets/firered-vox-cap30m-levels`. В `summary` показаны фактический диапазон LUFS, максимальный true peak, число записей, которым понадобился лимитер, и изменение уровня между первой и последней третью активных кадров. Для 624 уникальных целей обучения отдельно формируется `speaker_selection.json` после 30-минутной квоты. Веса, признаки и контрольные WAV лежат в `training/outputs/firered-vox-recipe-12-cap30m-levels`. Повтор команды сверяет подготовленные WAV и использует готовые чекпойнты.

```bash
jq '.summary' training/datasets/firered-vox-cap30m-levels/level_report.json
run=training/outputs/firered-vox-recipe-12-cap30m-levels
jq '.level_summary' "$run/speaker_selection.json"
jq -r '.results.full[] | [.id, .delta_vs_unlevelled.end_minus_start_rms_db, .delta_vs_unlevelled.wer, .delta_vs_unlevelled.final_voice_similarity] | @tsv' \
  "$run"/comparison-0030-*/report.json
```

Колонки — ID, изменение дрейфа громкости в дБ, изменение WER и изменение сходства голоса в конце. Дельты считаются относительно прошлого прогона того же режима и seed. Для положительного исходного дрейфа уменьшение в сторону нуля означает более ровный уровень. Сопоставляйте это с полнотой текста и звучанием концовки. Все три версии, включая Base, генерируются повторно. Дельты Base служат контролем воспроизводимости.

## Продолжить выбранную LoRA с шага 30 до 500

Параметр `--resume-levelled-lora` продолжает сохранённую LoRA на тех же нормализованных WAV. Из последнего чекпойнта восстанавливаются адаптер, AdamW, расписание LR и состояние генераторов случайных чисел. При старте с `lora/step-0030` выполняются ещё 470 обновлений. Обе серии проверок генерируют Base и LoRA на тех же голосе, текстах и seed, что и в указанных отчётах пилота.

```bash
cd /opt/firered-tuning
git pull --ff-only origin feat/russian-tts-adaptation
run=training/outputs/firered-vox-recipe-12-cap30m-levels
nohup bash scripts/firered_vox_pair.sh --resume-levelled-lora 500 \
  "$run/comparison-0030-yrslvj2s/report.json" \
  "$run/comparison-0030-ql21g14n/report.json" \
  > training/logs/firered-vox-levels-lora-500.log 2>&1 < /dev/null &
tail -n 30 -f training/logs/firered-vox-levels-lora-500.log
```

Результат обучения — `lora/step-0500` внутри `$run`. Каждая серия сравнения печатает путь к своему `comparison-0500-…/report.json`. WAV адаптера лежат в соседней папке `lora/`. Поле `compared_to` указывает соответствующий отчёт шага 30. В `delta_vs_previous_step` значения равны разнице «шаг 500 минус шаг 30» для данного режима и запроса. Флаги в `flags` используют исходную Base при текущем сравнении.

## Промежуточные адаптеры и WAV между шагами 30 и 500

Чтобы сопоставить полноту текста и тембр конца, запустите отдельный прогон LoRA от нормализованного `lora/step-0030`. Он выполняет обновления 31–400 с прежним расписанием на 1 000 шагов, состоянием AdamW, RNG и тем же порядком данных. Готовые признаки читаются из основного прогона. На каждом из шагов 100, 200, 300 и 400 сохраняется возобновляемый адаптер, затем прогон генерирует полные контрольные реплики и сравнивает их с пилотом. Следующие обновления начинаются после обеих серий прослушивания. Результаты пишутся в отдельный подкаталог `lora-0030-to-0400`.

```bash
cd /opt/firered-tuning
git pull --ff-only origin feat/russian-tts-adaptation
run=training/outputs/firered-vox-recipe-12-cap30m-levels
mkdir -p training/logs
nohup .venv/bin/python -u -m fireredtts3_local.vox_recipe_sweep \
  "$run/comparison-0030-yrslvj2s/report.json" \
  "$run/comparison-0030-ql21g14n/report.json" \
  > training/logs/firered-lora-100-400.log 2>&1 < /dev/null &
tail -n 30 -f training/logs/firered-lora-100-400.log
```

Адаптер первого среза лежит в `$run/lora-0030-to-0400/lora/step-0100/`, его WAV — в `$run/lora-0030-to-0400/audio/step-0100/series-1/lora/` и `series-2/lora/`. Для следующих срезов подставьте `step-0200`, `step-0300` и `step-0400`. Номера серий соответствуют порядку переданных отчётов. Рядом с каждой папкой `lora/` лежит `report.json` с текстами, WER, сходством голоса в последней трети, изменением громкости и дельтами к шагу 30. Сопоставляйте один и тот же ID на разных шагах. Повтор команды после прерывания продолжает с завершённого среза и формирует оставшиеся серии.

### Остановить прогон и проверить отдельно шаг 50

Если уже слышно искажение на шаге 100, закройте просмотр `tail -f` через `Ctrl+C` и остановите фоновый процесс на Gauss по открытому файлу блокировки прогона. Дождитесь сообщения об остановке.

```bash
cd /opt/firered-tuning
run=training/outputs/firered-vox-recipe-12-cap30m-levels
lock="$run/lora-0030-to-0400/.run.lock"
fuser -k -TERM "$lock"
flock -w 60 "$lock" true && echo 'Прогон 100–400 остановлен'
```

Для отдельного сравнения задайте абсолютную границу `--until 50`. Обучение выполнит 20 обновлений от сохранённого шага 30 с тем же расписанием LR. Восстановите те же две серии запросов в прежнем порядке.

```bash
cd /opt/firered-tuning
git pull --ff-only origin feat/russian-tts-adaptation
run=training/outputs/firered-vox-recipe-12-cap30m-levels
mkdir -p training/logs
nohup .venv/bin/python -u -m fireredtts3_local.vox_recipe_sweep --until 50 \
  "$run/comparison-0030-yrslvj2s/report.json" \
  "$run/comparison-0030-ql21g14n/report.json" \
  >> training/logs/firered-lora-0050.log 2>&1 < /dev/null &
tail -n 30 -f training/logs/firered-lora-0050.log
```

Адаптер появится в `$run/lora-0030-to-0050/lora/step-0050/`. Для сравнения проблемного `cities` откройте одноимённые WAV в `$run/comparison-0030-ql21g14n/lora/`, `$run/lora-0030-to-0050/audio/step-0050/series-2/lora/` и сохранённой серии шага 100.

### Сравнение на шаге 50 с обучением на искажённой акустической истории

`--until 50 --recovery` последовательно выполняет обычный прогон 30→50 и экспериментальный 30→50. Готовый обычный прогон из предыдущего раздела используется повторно. Запускайте после остановки старого прогона. Обе LoRA стартуют из одного пилота с его AdamW и расписанием LR, получают одинаковые целевые записи, референсы, порядок данных и основной loss. Для каждого варианта генерируются полные реплики по двум исходным сериям.

```bash
cd /opt/firered-tuning
git pull --ff-only origin feat/russian-tts-adaptation
run=training/outputs/firered-vox-recipe-12-cap30m-levels
mkdir -p training/logs
nohup .venv/bin/python -u -m fireredtts3_local.vox_recipe_sweep --until 50 --recovery \
  "$run/comparison-0030-yrslvj2s/report.json" \
  "$run/comparison-0030-ql21g14n/report.json" \
  >> training/logs/firered-lora-recovery-0050.log 2>&1 < /dev/null &
tail -n 30 -f training/logs/firered-lora-recovery-0050.log
```

В эксперименте к обычной задаче добавлена одна точка предсказания на подходящую запись. Случайный срез выбирается во второй половине реплики. В предшествующей части WAV на протяжении максимум восьми последних патчей плавно поднимается уровень на 3–6 дБ. Примерно в половине случаев подмешивается короткая задержанная копия сигнала (2–5 мс) для имитации металлической окраски. Замороженный RedAE кодирует этот префикс с тем же выравниванием по отсчётам, которое использовал чистый кэш. Модель получает исходный текст, исходный голосовой референс и изменённую историю. Эталон для следующего патча берётся из чистой записи на том же временном индексе. Общая функция потерь включает основной flow, STOP и дополнительный recovery flow с коэффициентом 0,15. В логе `[recovery]` показывает число примеров и обе составляющие до применения коэффициента. Это эксперимент с синтетическим искажением, его перенос на дефект при свободной генерации проверяется по WAV.

Обычные результаты лежат в `$run/lora-0030-to-0050`, экспериментальные — в `$run/lora-0030-to-0050-recovery`. У обоих одинаковая структура. В `lora/step-0050/` хранится адаптер, в `audio/step-0050/series-2/lora/` лежат четыре варианта `cities` на seed 1236–1239. В отчёте эксперимента `delta_vs_clean_control` даёт разницу относительно обычной LoRA того же шага и seed. Встроенные флаги по-прежнему используют Base.

```bash
jq -r '.results.lora[] | [.id, .audio, .asr.wer, .delta_vs_clean_control.wer, .delta_vs_clean_control.end_minus_start_rms_db, .delta_vs_clean_control.final_voice_similarity] | @tsv' \
  "$run/lora-0030-to-0050-recovery/audio/step-0050/series-2/report.json"
```

Колонки содержат ID, WAV, WER эксперимента и три разницы (WER, дрейф уровня в дБ, сходство голоса в последней трети). Проверьте одни и те же WAV на полноту перечислений и сохранение тембра до финального вопроса. Сокращение реплики из-за пропуска слов тоже может уменьшить измеренный дрейф.

## Новая recovery-LoRA от исходной Base до шага 100

Режим `--recovery-from-base` создаёт новую LoRA на исходной FireRedTTS3-Base с новым AdamW и включает задачу восстановления акустической истории с первого шага. Он использует ту же нормализованную выборку Dialogs и Spontaneous, лимит 30 минут на диктора, кэш чистых признаков и первые 100 обновлений прежнего расписания LR (100 шагов разогрева). На шагах 50 и 100 сохраняются веса и состояние обучения, затем синтезируются полные реплики с соответствующим адаптером.

Два отчёта прошлого прогона задают серии запросов и содержат готовые результаты Base. Программа сверяет отпечаток модели, тексты, seed, референсы и настройки, затем использует метрики из `results.base` для сравнения. Сами WAV исходной Base уже лежат возле этих отчётов. В новом каталоге будут только WAV адаптера. Наличие старого чекпойнта LoRA для этого режима не требуется.

```bash
cd /opt/firered-tuning
git pull --ff-only origin feat/russian-tts-adaptation
run=training/outputs/firered-vox-recipe-12-cap30m-levels
mkdir -p training/logs
nohup .venv/bin/python -u -m fireredtts3_local.vox_recipe_sweep \
  --recovery-from-base \
  "$run/comparison-0030-yrslvj2s/report.json" \
  "$run/comparison-0030-ql21g14n/report.json" \
  >> training/logs/firered-recovery-base-0100.log 2>&1 < /dev/null &
tail -n 30 -f training/logs/firered-recovery-base-0100.log
```

```bash
out=training/outputs/firered-vox-recipe-12-cap30m-levels/lora-base-recovery-0100
ls "$out"/lora/step-{0050,0100}/adapter_model.safetensors
ls "$out"/audio/step-{0050,0100}/series-2/lora/
```

Каждая завершённая серия сразу доступна для прослушивания. В `series-1` шесть запросов, в `series-2` четыре варианта длинного `cities`; сравнивайте одноимённые WAV внутри одной серии. Рядом с каждой папкой `lora/` лежит `report.json`. Поле `delta_vs_base` даёт разницу с исходной Base для того же запроса, а `compared_to` указывает на исходный отчёт. Повтор команды с теми же отчётами после прерывания использует последний завершённый срез и достраивает его прослушивание перед дальнейшим обучением.

## Эксперимент с более сильной металлической окраской истории

Ключ `--stronger-comb` увеличивает коэффициент задержанной копии сигнала с 0,35 до 0,50 в тех же случайно выбранных окнах. Это углубляет провалы гребенчатого фильтра. По-прежнему примерно половина окон получает эту окраску. Подъём уровня 3–6 дБ, задержка 2–5 мс, восьмипатчевый переход и вес дополнительного loss 0,15 одинаковы в обоих опытах. Искажается только входная история при обучении. Новая LoRA стартует от исходной Base, использует прежний порядок записей и заканчивает на шаге 100. На шагах 50 и 100 она сохраняет адаптер и синтезирует обе серии целых реплик. Метрики Base берутся из имеющихся отчётов.

```bash
cd /opt/firered-tuning
git pull --ff-only origin feat/russian-tts-adaptation
run=training/outputs/firered-vox-recipe-12-cap30m-levels
mkdir -p training/logs
nohup .venv/bin/python -u -m fireredtts3_local.vox_recipe_sweep \
  --recovery-from-base --stronger-comb \
  "$run/comparison-0030-yrslvj2s/report.json" \
  "$run/comparison-0030-ql21g14n/report.json" \
  >> training/logs/firered-recovery-comb050.log 2>&1 < /dev/null &
tail -n 30 -f training/logs/firered-recovery-comb050.log
```

Новый каталог — `$run/lora-base-recovery-comb050-0100`. Для первого прослушивания сравните его шаг 50 с шагом 50 из `$run/lora-base-recovery-0100`. После каждой серии файл `paired-recovery.json` содержит пути к парам WAV и разницы метрик «усиленная окраска минус исходная» для одинаковых текста, референса, seed и шага. Если исходный отчёт отсутствует или относится к другому эксперименту, причина записывается в поле `reason`, обучение продолжается.

```bash
run=training/outputs/firered-vox-recipe-12-cap30m-levels
out="$run/lora-base-recovery-comb050-0100"
jq -r '.cases[] | [.id, .standard_wav, .stronger_wav] | @tsv' \
  "$out/audio/step-0050/series-2/paired-recovery.json"
```

В `series-2` сравните четыре варианта `cities`. Слушайте, произнесены ли перечисления целиком и сохраняется ли тембр до последнего вопроса. Поля `delta_vs_standard.wer`, `end_minus_start_rms_db` и `final_voice_similarity` в парном отчёте помогают найти случаи для прослушивания.

## Восстановление до четырёх патчей от исходной Base

Режим `--multistep-from-base` создаёт свежую LoRA и AdamW на исходной FireRedTTS3 Base. На обновлениях **100 и 500** сохраняются адаптер, состояние обучения и аудио. Срез 100 служит точкой продолжения до 500. Применяется прежнее расписание на 1000 обновлений с разогревом 100. За 500 обновлений модель получает 8000 предъявлений целевых записей из той же выборки Dialogs и подтверждённой Spontaneous.

Вспомогательное задание начинается на первом обновлении. Для каждого подходящего клипа выбирается срез во второй половине записи. Последние восемь патчей префикса получают тот же подъём на 3–6 дБ и ту же вероятность гребенчатого искажения с коэффициентом 0,50, что и в эксперименте с усиленной окраской. Первый эталон для flow берётся непосредственно после среза. Для второго, третьего и четвёртого эталонов модель получает в историю свои предыдущие патчи, сгенерированные текущими весами через 10 шагов интегрирования с CFG 2. Эталоном в каждой позиции служит соответствующий чистый патч записи. Градиент распространяется через задачу предсказания в каждой позиции. При построении входной истории сэмплы предыдущих позиций отделены от графа.

Четыре последовательных патча охватывают до 640 мс. Конец записи или срабатывание STOP ограничивают доступный горизонт. Ошибки усредняются по доступным позициям каждого клипа, поэтому вес вспомогательной задачи остаётся 0,15. Основное обучение flow и STOP по целой чистой записи также выполняется. По полю `targets_by_hop` в логе видно, сколько примеров на текущем обновлении получили эталон на каждой из четырёх позиций. На поздних позициях предполагается близость темпа модели к записи. Результат проверяется свободной генерацией.

### Запуск на Gauss

```bash
cd /opt/firered-tuning
git pull --ff-only origin feat/russian-tts-adaptation
run=training/outputs/firered-vox-recipe-12-cap30m-levels
mkdir -p training/logs
nohup .venv/bin/python -u -m fireredtts3_local.vox_recipe_sweep \
  --multistep-from-base \
  "$run/comparison-0030-yrslvj2s/report.json" \
  "$run/comparison-0030-ql21g14n/report.json" \
  >> training/logs/firered-recovery-4patch-0500.log 2>&1 < /dev/null &
tail -n 25 -f training/logs/firered-recovery-4patch-0500.log
```

Повтор той же команды использует завершённые срезы и пары аудио. При прерывании между сохранениями будут повторно выполнены обновления после последнего сохранения. На шаге 100 эксперимент сначала завершает прослушивания, затем продолжает обучение.

### Как сравнить восстановление

В новом тесте используются три ранее не входивших в этот бенчмарк продолжения про выезд инженера, доставку и передачу пожеланий. У каждого два seed. Для всех весов и текстов используется одна реальная начальная реплика из валидации Dialogs. Для клонирования голоса подаётся отдельная чистая запись того же диктора. В одном варианте начало чистое, во втором его последние примерно 1,28 с обработаны гребенчатым фильтром (задержка 3 мс, коэффициент 0,50) с нарастающим усилением до +6 дБ. Одинаковый seed задаёт шум для парных продолжений. Обе версии начала попадают в акустическую историю backbone и DiT. Весь дальнейший текст модель произносит самостоятельно до STOP или нативного лимита. Каждый WAV содержит начало и продолжение. Base проходит этот новый тест один раз.

```bash
cd /opt/firered-tuning
out=training/outputs/firered-vox-recipe-12-cap30m-levels/lora-base-recovery-4patch-0500
ls "$out"/lora/step-{0100,0500}/adapter_model.safetensors
ls "$out"/prefix-test/{base,step-0100,step-0500}/visit-seed1234/{clean,metal}.wav
jq '.pairs[] | {case,seed,prefix_seconds,wer_delta,first_words:.matched_words[:4]}' \
  "$out/prefix-test/step-0500/report.json"
```

Послушайте одну пару `clean.wav` и `metal.wav` начиная с отметки `prefix_seconds` и сравните те же слова у Base, 100 и 500. Сам подмешанный звук можно услышать в `prefix-test/prefix-metal.wav`. В полях `variants` отчёт содержит распознанный текст продолжения, WER, пропущенные слова, причину остановки и посекундный RMS. `matched_words` показывает номера и время совпавших слов от начала продолжения, разницу RMS между вариантами и расстояние между формами их логарифмических спектров по восьми частотным полосам. Интонация, скорость речи и точность границ ASR влияют на эти показатели. Момент исчезновения металлического призвука отмечайте на слух. Дополнительно в `audio/step-0100` и `audio/step-0500` сохраняются прежние серии обычной генерации целиком, включая `cities`. Они показывают, возникает ли дефект сам по себе.

## Проверка роста уровня после каждой попытки обновления LoRA

Для четырёхпатчевого прогона можно включить `--rise-limit-percent 20`. После каждого пробного шага AdamW модель в режиме инференса генерирует целиком `cities_seed1237`. Референс, текст, seed и настройки совпадают с сохранённым отчётом Base. На исходном WAV берутся два неперекрывающихся окна, каждое длиной в треть записи и максимум 10 секунд. Если отношение RMS последнего окна к RMS первого более чем в 1,2 раза превышает то же отношение у Base, программа откатывает LoRA и состояние оптимизатора. Например, при росте у Base на 2,0 дБ предельный рост составляет примерно 3,58 дБ по формуле `2 + 20 × log10(1,2)`.

Следующая попытка использует следующий батч. Расписание LR продвигается только при принятом обновлении. Поэтому 100 и 500 обозначают номера попыток. Число принятых и отклонённых обновлений сохраняется в `checkpoint.json` и в журнале. При возобновлении программа продолжает с последнего сохранённого среза и переписывает решения для повторно обработанных батчей.

```bash
cd /opt/firered-tuning
git pull --ff-only origin feat/russian-tts-adaptation
run=training/outputs/firered-vox-recipe-12-cap30m-levels
mkdir -p training/logs
nohup .venv/bin/python -u -m fireredtts3_local.vox_recipe_sweep \
  --multistep-from-base --rise-limit-percent 20 \
  "$run/comparison-0030-yrslvj2s/report.json" \
  "$run/comparison-0030-ql21g14n/report.json" \
  >> training/logs/firered-4patch-rise20.log 2>&1 < /dev/null &
tail -n 25 -f training/logs/firered-4patch-rise20.log
```

```bash
out=training/outputs/firered-vox-recipe-12-cap30m-levels/lora-base-recovery-4patch-0500-rise20pct
jq '.[-1] | {attempt,accepted,accepted_total,rejected_total,check_seconds,cases}' "$out/rise_decisions.json"
```

Для 500 попыток с одним контрольным примером добавится 500 синтезов. Оценить затраты на Gauss можно по полю `check_seconds` в первых записях. Чтобы проверять несколько вариантов на каждой попытке, передайте по одному `--rise-case ID` на каждый, включая `--rise-case cities_seed1237`, если хотите его сохранить. Любое превышение порога отклоняет обновление. Для другого набора случаев используется отдельный каталог, путь выводится в логе.

Успех на реплике, участвовавшей в отборе весов, ожидаем. Сравните оставшиеся seed и тексты с обычным прогоном на тех же срезах. Тихий хвост вследствие пропуска слов тоже может пройти проверку RMS. Полноту текста оценивайте по WER и прослушиванию, тембр — по парным WAV `clean.wav` и `metal.wav`.

### Прослушивание новых реплик на чекпойнте 500 с порогом +20%

Команда запуска из `/opt/firered-tuning`

```bash
.venv/bin/python -u -m fireredtts3_local.vox_listen
```

Скрипт читает восемь коротких и четыре длинных текста из `examples/diagnostics/listening-ru-500.json` и синтезирует каждый целиком с seed 1234 и 1235. Голосовой образец и его транскрипт берутся из контрольного примера `cities_seed1237` в сохранённом чекпойнте.

WAV появляются в `training/eval/listening-rise20-step0500/{short,long}/`. В `texts.txt` лежат исходные реплики, в `report.json` для каждого файла записаны длительность, причина остановки и разница RMS между концом и началом. Целевая длительность длинных ответов составляет 40–50 секунд. Сохраняются все результаты синтеза с исходной амплитудой, в том числе более короткие и длинные. Для отдельного повторного прогона можно указать, например, `--seeds 1236 --output training/eval/listening-rise20-seed1236`.

### Узкие резонансы в конце длинных реплик

В восьми WAV одного голоса из прогона `rise20pct` к концу усилились полосы около 2,22, 2,96 и 5,19 кГц. Вырос и контраст с соседними частотами, в том числе вследствие спада в области 3,4–4,6 кГц. Частоты подобраны по этим восьми файлам. Проверьте обобщение на других текстах и голосах.

Спектрограммы, таблицу и три варианта последних восьми секунд каждого файла можно собрать на CPU. Команда использует отдельное окружение uv.

```bash
cd /opt/firered-tuning
uv run --no-project --with numpy --with scipy --with matplotlib python \
  scripts/inspect_tail_resonances.py \
  training/eval/listening-rise20-step0500/long \
  training/eval/spectrum-rise20-step0500
```

В `AB/` сравните `original.wav`, `reduced.wav` и `rebalanced.wav` с одинаковым префиксом имени. У всех трёх одинаковый RMS и общий запас до перегруза. В `reduced` ослаблены три резонанса, в `rebalanced` дополнительно приподнята область 4 кГц. Исходные FLOAT WAV остаются в папке прослушивания.

Для спектрального обучения используется оригинальная Base и та же чистая выборка. В половине окон на конец реальной акустической истории накладываются три плавно усиливающихся узких пика (+6…+15 дБ), широкая впадина (−8…−16 дБ) и рост уровня (+3…+6 дБ). Частоты и ширина фильтров слегка меняются. Длина нарастания 24–100 патчей, ограничена доступным префиксом. В остальных окнах воспроизводится только рост уровня. Цели — до четырёх следующих настоящих патчей; при обучении на последующих позициях сеть получает собственные предсказания в историю. Основная задача на чистой речи остаётся в функции потерь. При обычном синтезе фильтры не применяются.

Первый эксперимент сохраняет вес вспомогательной ошибки 0,15 и прежний порог RMS. Срезы после попыток 100 и 500 попадают в отдельный каталог.

```bash
cd /opt/firered-tuning
git pull --ff-only origin feat/russian-tts-adaptation
run=training/outputs/firered-vox-recipe-12-cap30m-levels
mkdir -p training/logs
nohup .venv/bin/python -u -m fireredtts3_local.vox_recipe_sweep \
  --multistep-from-base --resonant-history --rise-limit-percent 20 \
  "$run/comparison-0030-yrslvj2s/report.json" \
  "$run/comparison-0030-ql21g14n/report.json" \
  >> training/logs/firered-resonant-w015.log 2>&1 < /dev/null &
tail -n 20 -f training/logs/firered-resonant-w015.log
```

В папке `prefix-test/` исходная Base и оба среза получают один и тот же реальный префикс в чистом и резонансно искажённом варианте. Сравнивайте продолжение после метки `prefix_seconds`.

После сохранения нужного среза сравните новые WAV со старыми при тех же тексте, референсе и seed. Для прослушивания среза 100 укажите `--step 100` и отдельное имя в `--output`.

```bash
cd /opt/firered-tuning
run=training/outputs/firered-vox-recipe-12-cap30m-levels
.venv/bin/python -u -m fireredtts3_local.vox_listen \
  --run "$run/lora-base-recovery-4patch-0500-rise20pct-resonant-w015" \
  --step 500 --output training/eval/listening-resonant-w015-0500
uv run --no-project --with numpy --with scipy --with matplotlib python \
  scripts/inspect_tail_resonances.py training/eval/listening-resonant-w015-0500/long \
  training/eval/spectrum-resonant-w015-0500
```

Для отдельного сравнения веса ошибки можно добавить `--stronger-recovery-loss` к команде обучения и использовать новый лог. Вес удвоится до 0,30, каталог будет оканчиваться на `-resonant-w030`. Сверяйте число принятых обновлений, контраст полос, дословность последней фразы и сходство голоса. На длинном независимом тексте проверяйте всю запись одним вызовом. На коротких репликах слушайте, сохранились ли звонкость и разборчивость согласных.

### Сравнение backbone и DiT у резонансного среза 100 и предыдущего среза 500

На Gauss скрипт остановки проверяет наличие сохранённого `step-0100`, отправляет TERM по файлу блокировки резонансного прогона `w015` и ждёт освобождения блокировки. Команда генерации работает в фоне.

```bash
cd /opt/firered-tuning
git pull --ff-only origin feat/russian-tts-adaptation
bash scripts/stop_resonant_w015.sh
```

После сообщения `Training lock is free` запустите сравнение.

```bash
cd /opt/firered-tuning
mkdir -p training/logs
nohup .venv/bin/python -u -m fireredtts3_local.vox_attention_ablation \
  >> training/logs/attention-resonant100-previous500.log 2>&1 < /dev/null &
tail -n 20 -f training/logs/attention-resonant100-previous500.log
```

Результаты появятся в `training/eval/attention-resonant100-previous500/`. Там семь папок с WAV: `base/`, `resonant-0100/{backbone,dit,both}/` и `previous-0500/{backbone,dit,both}/`. В каждой те же четыре длинные реплики из `listening-ru-500.json` с seed 1234 и 1235. Общая Base генерируется один раз. Скрипт проверяет исходные веса, выборку, профили обучения, WAV и транскрипт референса из `cities_seed1237`. В частичных вариантах включаются только указанные проекции внимания LoRA. Аудио сохраняется целиком в исходном FLOAT WAV.

В `report.json` сохраняются параметры, число принятых обновлений и результаты каждой записи. После синтеза всех 56 WAV появляется `summary.csv`. В нём есть длительность, причина остановки, разность RMS между концом и началом, контраст трёх полос в первой и последней третях и отдельно в интервале 15–19 с. Пустое значение означает, что интервал слишком короток или в нём нет кадров выше порога активности. Затем Whisper large-v3 на CPU добавляет WER и возможные пропуски. Параметр `--audio-only` позволяет отложить распознавание. Повтор команды проверяет SHA-256 готовых WAV и продолжает с недостающих результатов.

Для прослушивания сравните один и тот же файл, например `02-invoice-seed1234.wav`, во всех семи папках. Проигрывайте при одном уровне усиления. Убедитесь, что длительность и произнесённый текст сопоставимы, прежде чем интерпретировать разницу спектра.

Для сравнения двух срезов DiT на прежних текстах про города и Тимофея выполните команду из корня репозитория.

```bash
.venv/bin/python -u -m fireredtts3_local.vox_attention_ablation --dit-cities-timofey --audio-only
```

В `training/eval/dit-cities-timofey-resonant100-previous500/` появятся папки `resonant-0100/dit/` и `previous-0500/dit/`. Каждая содержит четыре WAV для двух текстов и seed 1234 и 1235.

Для повтора конкретной серии передайте её папку `lora`. Скрипт берёт тексты, референсы и seed из её `benchmark.jsonl`, сверяет их с отчётом и загружает веса того же прогона и шага.

```bash
run=training/outputs/firered-vox-recipe-12-cap30m-levels/lora-base-recovery-4patch-0500
.venv/bin/python -u -m fireredtts3_local.vox_attention_ablation \
  --dit-series "$run/audio/step-0500/series-2/lora" --audio-only
```

Для прослушивания по парам откройте одноимённые файлы `case-000.wav`, `case-001.wav` и следующие в соседних папках `series-2/lora/` и `series-2/dit/`.

Чтобы сгенерировать ещё два варианта для той же серии, укажите компоненты LoRA.

```bash
.venv/bin/python -u -m fireredtts3_local.vox_attention_ablation \
  --series "$run/audio/step-0500/series-2/lora" --variants backbone both --audio-only
```

В `series-2/backbone/` появятся записи с LoRA только в backbone. В `series-2/both/` обе части адаптера включены, как и в исходных записях `series-2/lora/`. В каждой папке номера WAV соответствуют тем же примерам в `series-2/dit/`.

Чтобы услышать backbone шага 500 на новых репликах, передайте программе папку этого варианта.

```bash
cd /opt/firered-tuning
run=training/outputs/firered-vox-recipe-12-cap30m-levels/lora-base-recovery-4patch-0500
.venv/bin/python -u -m fireredtts3_local.vox_attention_ablation \
  --listen-from "$run/audio/step-0500/series-2/backbone" --audio-only
```

Папки `backbone/listening/short/` и `backbone/listening/long/` содержат соответственно 16 и 8 WAV. Двенадцать фраз из `examples/diagnostics/listening-ru-calls.json` генерируются целиком с seed 1236 и 1237. Голос и транскрипт референса берутся из первого примера серии. Тексты, seed и пути к файлам записаны в `listening/report.json`, фактические длительности — в `listening/summary.csv`.

### Публикация выбранного backbone шага 500

На Gauss войдите в Hugging Face под `shustov-te` с правом записи в приватный `shustov-te/FireRedTTS3-Base-Russian`. Если вход уже выполнен, запустите команду из репозитория.

```bash
cd /opt/firered-tuning
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -u -m fireredtts3_local.publish_backbone
```

Команда сверяет контрольные суммы адаптера с отчётом `lora-base-recovery-4patch-0500/audio/step-0500/series-2/backbone/report.json`, сливает на CPU только LoRA внимания backbone и проверяет тензоры полного пакета для SGLang Omni. Публикация создаёт коммит в `main` существующего приватного репозитория. Повтор после обрыва загрузки использует уже собранный пакет. Новый SHA и ссылка на проверенную ревизию сохраняются в `training/exports/FireRedTTS3-Base-Russian-4patch-0500-backbone.published.json`.
