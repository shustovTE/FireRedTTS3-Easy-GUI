# Rollout-11: Dialogs + Spontaneous с собственной акустической историей

**Статус эксперимента:** исправление не подтверждено. Step 240 пропускает
смысловые блоки, несмотря на улучшение тембра; на step 286 пользователь вновь
сообщил об артефактах. Продолжение этого режима не является рекомендуемым
следующим шагом. См. [пересмотр стратегии](TRAINING_STRATEGY_REVIEW_2026_11.md).
Команды ниже сохранены для воспроизводимости и диагностики.

Один новый прогон от **исходной Base**, без адаптеров 08/09/10. Используются
существующие `dialogs-base-08` и `cv-spontaneous-09`, их исходные полные WAV,
подтверждённые транскрипты, прежние категории, лимиты на дикторов и отложенные
записи. Dialogs остаётся до 10 секунд, Spontaneous — до 30 секунд. Новых загрузок,
обрезки окончаний и нормализации нет. При прежнем отборе: 1137 targets, 2,18 часа;
отдельная валидация Dialogs — 123, Spontaneous — 24 записи.

Это эксперимент **truncated scheduled sampling**, а не официальный trainer
FireRedTTS3 и не доказанное исправление металлического тембра. Диагностика
поддерживает гипотезу накопления ошибок в собственной истории. Короткий горизонт
уменьшает риск рассинхронизации с записью, но не доказывает совпадение фонем.
Не увеличивайте горизонт до целого высказывания: модель меняет темп, и одинаковый
номер патча перестаёт означать одинаковое место текста.

## Что обучается

| Компонент | Режим |
|---|---|
| Backbone | Новая LoRA rank 16, alpha 32, dropout 0, LR `3e-6` |
| DiT, `dit_head`, `stop_head` | Веса заморожены |
| PatchEncoder, CAM++, проекции голоса, RedAE | Заморожены |

Замороженные веса stop head не означают неизменный STOP: его вход меняется
вместе с backbone. Обычный clean-history stop loss сохраняется (`mean`, вес
`0.05`), на сгенерированной истории STOP не обучается. Precision — native:
BF16 autocast внутри backbone, исходная точность остальных модулей.

Каждый target участвует в обычном clean-history flow/stop loss. Preservation
`0.1` сравнивает с исходной Base **только на реальной истории**. Дополнительно:

1. Выбирается случайная позиция внутри записи; перед ней не менее 6 реальных
   патчей, после обучаемого продолжения остаётся не менее 6 патчей.
2. Текущая модель генерирует несколько патчей: native cosine-time Euler,
   10 ODE steps, CFG 2. Они подаются и в backbone, и в акустическую историю DiT.
3. После этой истории модель учится предсказывать **следующий реальный патч**.
   Самостоятельно предсказанные патчи не становятся эталонами. Их embeddings
   пересчитываются; кэш исходной записи здесь не подставляется.
4. Сэмплирование истории идёт без графа градиентов. Затем conditioning
   пересчитывается с градиентом в LoRA. Через всю цепочку ODE backward не идёт.

Вклад дополнительного flow loss — `0.25`, с усреднением по всему batch.
Участки с STOP ≥0.5 до/после короткого rollout исключаются из дополнительного
loss; обычное обучение на этих записях сохраняется. Слишком короткие записи
тоже остаются в обычном обучении. Оба случая видны в статистике, они не должны
незаметно превращать эксперимент в обычный teacher forcing.

| Обновления optimizer | Собственная история |
|---|---|
| 1–20 | Только обычное обучение |
| 21–120 | 1 патч, примерно 0,16 секунды |
| 121–220 | 2 патча, примерно 0,32 секунды |
| От 221 | 4 патча, примерно 0,64 секунды |

Короткие участки используются для обучения. **Контрольная генерация всегда
целой репликой (`do_split=False`)**, включая прежние длинные проблемные тексты.

## 1. Обновить и подготовить

На Gauss, после завершения других обучающих jobs:

```bash
cd /opt/firered-tuning
git pull --ff-only origin feat/russian-tts-adaptation

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -m fireredtts3_local.rollout_training prepare

.venv/bin/python -m fireredtts3_local.rollout_training audit
cat training/datasets/russian-rollout-11/report.json
```

Нужны исходная Base в `models/FireRedTTS3`, два подготовленных каталога выше
и `training/eval/dialogs-base-08/benchmark.jsonl` с доступными reference WAV.
Проверяются хэши исходных данных, разделение и пересечения текстов/референсов.
Старые адаптеры не читаются. Каталоги предыдущих прогонов не перезаписываются.
Если подготовка уже прошла, повторяйте `audit`, а не `prepare`.

Новый рецепт: `training/datasets/russian-rollout-11/recipe.json`.
Output: `training/outputs/russian-rollout-base-11`.
Batch 8, accumulation 1, до 5 эпох: **715 шагов** при прежних 1137 targets.
Числа вычисляются из фактического manifest. Каждая эпоха покрывает все targets
один раз, без дополнительного повторения маленького корпуса.

## 2. Предрасчёт и пилот в фоне

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -m fireredtts3_local.training_job start \
  training/datasets/russian-rollout-11/recipe.json \
  --job-dir training/jobs/rollout-11-pilot \
  --precompute --stop-after 240
```

Здесь пилот **240 шагов**, чтобы реально проверить все горизонты, включая 4
патча, и получить контрольную генерацию после первой эпохи. Пилот из 30 шагов
проверил бы только горизонт 1 и дал бы неполную оценку скорости/памяти.
Пилот — начало того же обучения, не отдельный расходуемый прогон. Ранняя
остановка по регрессиям может завершить обучение раньше запрошенного шага.

Предрасчёт повторно использует совместимый native-v3 кэш в
`.runtime/training-cache-joint-10`; отсутствующие признаки рассчитываются.
Старт включает валидацию и генерацию Base до первого обновления. Собственная
история считается заново на GPU при обучении, её нельзя заранее закэшировать.

```bash
.venv/bin/python -m fireredtts3_local.training_job status training/jobs/rollout-11-pilot
tail -n 30 -f training/jobs/rollout-11-pilot/job.log
```

`Ctrl+C` закрывает просмотр; обучение продолжится, в том числе без SSH.

```bash
tail -n 3 training/outputs/russian-rollout-base-11/rollout_metrics.jsonl
tail -n 3 training/outputs/russian-rollout-base-11/performance.jsonl
cat training/outputs/russian-rollout-base-11/generation_metrics.jsonl
```

`rollout_accepted_fraction` — доля batch, получившая дополнительный loss;
`rollout_attempted_fraction` — доля достаточно длинных записей;
`rollout_stop_rejected_fraction` — доля batch, отклонённая из-за STOP.
`rollout_flow` усредняется по всему batch, включая нулевой вклад пропущенных;
`rollout_contribution` — уже взвешенная добавка к loss. `teacher_forced_loss`
показывает обычный loss отдельно. Если после warmup accepted постоянно равен
нулю, собственная история не участвует в обучении — сначала разберите причину.

Оценку времени и VRAM берите по шагам **221–240**, когда горизонт максимален.
Здесь нет замеров на настоящей GPU; скорость прежней backbone LoRA не переносится
на новый режим. Sampling пакетный, истории detached; чистая и дополнительная
ветви выполняют backward последовательно, чтобы не держать оба графа в памяти.

## 3. Продолжить пилот

Если контрольный WAV срывается в бессмысленную речь, сначала выполните проверку
ниже. `Completed successfully` означает штатное завершение процесса, а не
прохождение проверки речи. Успешный `accepted` относится к учебным участкам.

После `completed`, при отсутствии ошибок и регрессий в контрольных WAV:

```bash
.venv/bin/python -m fireredtts3_local.rollout_training resume \
  --job training/jobs/rollout-11-pilot \
  --output training/recipes/rollout-11-resume.json

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -m fireredtts3_local.training_job start \
  training/recipes/rollout-11-resume.json \
  --job-dir training/jobs/rollout-11-full
```

Optimizer, scheduler, RNG и текущий горизонт продолжаются из checkpoint.
Параметры метода и хэши реализации зафиксированы: подмена objective при resume
завершится ошибкой. Предрасчёт повторять не нужно.

```bash
tail -n 3 training/outputs/russian-rollout-base-11/validation_metrics.jsonl
tail -n 3 training/outputs/russian-rollout-base-11/adaptation_validation_metrics.jsonl
tail -n 3 training/outputs/russian-rollout-base-11/rollout_metrics.jsonl
tail -n 3 training/outputs/russian-rollout-base-11/generation_metrics.jsonl
tail -n 10 training/jobs/rollout-11-full/job.log
```

Для корректной остановки:

```bash
.venv/bin/python -m fireredtts3_local.training_job stop training/jobs/rollout-11-full
```

## Результат и ограничения контроля

`validation_metrics.jsonl` относится к Dialogs, `adaptation_validation_metrics.jsonl`
— к Spontaneous. Это прежняя teacher-forced валидация, без rollout-добавки.
Она сопоставима между шагами этого прогона, но сама не измеряет металлический тембр.

При step 0 и после эпох генерируются все случаи из benchmark с CFG 2, STOP 0.5,
10 ODE steps. WAV и отчёты лежат в `generation/step-.../`. Отдельно сохраняются
исходные float WAV и PCM16-копии с записанным коэффициентом уменьшения уровня.
Файл `generation_metrics.jsonl` указывает точный путь к каждому отчёту.

Для этого режима разница уровня считается по **неперекрывающимся первой и
последней трети**, максимум по 10 секунд. Так короткая реплика больше не получает
автоматически нулевой drift из-за сравнения одних и тех же сэмплов.
Регрессии по лимиту генерации, клиппингу и росту RMS блокируют замену
`best-validation`. Остановка — после трёх проверок без допустимого улучшения.

Эти пороги не распознают все искажения, пропущенные слова или слишком раннее
завершение. Итог проверяется прослушиванием целых длинных реплик, ASR и сходством
голоса на тех же текстах/reference/seed. Хороший WER сам по себе недостаточен.
Валидация исходных корпусов проверяет другие записи знакомых дикторов;
для вывода о клонировании нужны дополнительные незнакомые голоса.

`best-validation/firered_lora_metadata.json` содержит выбранный `step` и
`generation_guard`. **Step 0 — fallback Base**, а не успешно обученный адаптер.
Корень output содержит последний checkpoint, `adapter-step-*` — промежуточные
адаптеры каждые 50 шагов. Полный checkpoint с optimizer сохраняется после эпох
и при остановке. Слияние поддерживается прежними средствами репозитория.
Этот запуск не публикует веса и не меняет SGLang.

## Если контрольная речь испортилась: проверить текущий checkpoint без обучения

На пилоте из 240 шагов штатная контрольная генерация была на **143**, когда
rollout-горизонт равнялся **2**. Четыре патча используются с обновления 221;
сохранённый на 240 адаптер ещё не проходил автоматическую генерацию. Это не
обещание улучшения на 240, а причина проверить именно сохранённые веса, прежде
чем продолжать обучение. Повышать горизонт/вес loss по этому признаку не нужно.

Например, для проблемного `case-00006.wav`:

```bash
cd /opt/firered-tuning
git pull --ff-only origin feat/russian-tts-adaptation

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -m fireredtts3_local.rollout_training check \
  --audio case-00006.wav
```

Команда работает в фоне. Получатся **три целые реплики**: Base, step 143 и step
240, с теми же текстом, голосом, seed, CFG 2 и STOP 0.5. ID случая определяется
по имени WAV в исходном отчёте, а не угадывается. При другой длине пилота есть
`--steps 143 250`. Корневой checkpoint используется только при точном совпадении
его шага, recipe и fingerprint. `best-validation` не подставляется вместо него.
Исходные веса, optimizer и обучающие журналы не перезаписываются. Обучающий
цикл и его хэши этим обновлением не меняются.

```bash
.venv/bin/python -m fireredtts3_local.training_job status training/jobs/rollout-11-check
tail -n 20 -f training/jobs/rollout-11-check/job.log
```

После завершения:

```bash
cat training/eval/rollout-11-check/summary.md
ls -lh training/eval/rollout-11-check/diagnostics-*.zip
```

WAV: `training/eval/rollout-11-check/{base,step-000143,step-000240}/cfg-2/case-00.wav`.
Здесь `case-00` — первый выбранный случай, то есть прежний `case-00006`.
Для разбора пригодятся `diagnostics-report.zip` и `diagnostics-audio-01.zip`.
В `previous_monitor` проверяется воспроизведение прежних Base/143. Если оно
существенно расходится, сначала разбирается воспроизводимость.
64 секунды обычно означают исчерпание 400 патчей; отчёт покажет точную причину
`max_patches`/`stop`. Автоматические флаги не заменяют проверку содержания речи;
эта короткая проверка не запускает ASR и не является полным benchmark.
