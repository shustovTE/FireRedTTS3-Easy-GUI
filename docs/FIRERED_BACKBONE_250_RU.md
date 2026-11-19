# FireRedTTS3 — 250 обновлений и полная запись контрольных звонков

В основе лежит [выбранный рецепт backbone-500](FIRERED_BACKBONE_500_RU.md). Обучение начинается со свежей LoRA на оригинальной FireRedTTS3 Base. Совместно обновляются адаптеры внимания backbone и DiT. Дополнительная задача предсказывает до четырёх чистых патчей после искажённой истории с весом 0,15. Каждое обновление применяется. Для итогового прослушивания активна только LoRA backbone.

Сохраняются тот же отбор и порядок Dialogs и Spontaneous, seed 1234, микробатч 2, накопление 8, пиковый LR `1e-4`, разогрев 100 и cosine по исходному расписанию на 1 000. Прогон заканчивается после первых **250 обновлений, или 4 000 предъявлений**. Адаптер и состояние оптимизатора сохраняются на шагах 100 и 250 в отдельном каталоге `lora-base-recovery-4patch-0250`. На обоих шагах штатная проверка записывает обе контрольные серии с совместным адаптером; в `prefix-test/` сохраняется отдельный тест восстановления с чистым и искажённым префиксом. Полный комплект с активным backbone записывается на шаге 250.

## Запуск на Gauss

Используется подготовленный нормализованный датасет `training/datasets/firered-vox-cap30m-levels` с исходными манифестами и обе прежние серии пилота в `training/outputs/firered-vox-recipe-12-cap30m-levels`. Для первой серии берётся `comparison-0030-yrslvj2s`, для второй — `comparison-0030-ql21g14n`. Нужны их отчёты, манифесты и голосовые референсы. Для оценок используются те же модели ASR и сходства голосов, что и в предыдущем эксперименте.

```bash
cd /opt/firered-tuning
git pull --ff-only origin feat/russian-tts-adaptation
mkdir -p training/logs
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nohup bash scripts/firered_backbone_250.sh \
  >> training/logs/firered-backbone-250.log 2>&1 < /dev/null &
```

```bash
tail -n 30 -f training/logs/firered-backbone-250.log
```

Повтор команды запуска после прерывания подхватывает последний готовый срез. При записи итогового комплекта рендерер сверяет SHA-256 WAV и достраивает отсутствующие или повреждённые файлы. Выйти из просмотра лога можно клавишами Ctrl+C.

## Что слушать

Формат всех итоговых записей — FLOAT WAV с исходным уровнем генерации. Каждая реплика произносится целиком одним запросом, с CFG 2, 10 шагами интегрирования и порогом STOP 0,5. Референс и его текст повторяют соответствующую прежнюю серию. Длительность длинных реплик определяется фактической генерацией.

| Каталог в `backbone-complete/` | Реплики и seed | WAV |
|---|---|---:|
| `controls/` | «Города» × 6 (1234–1239), «Тимофей» × 2 (1234–1235), контрольный вопрос × 2 (1234–1235) | 10 |
| `listening-ru-500/{short,long}/` | 8 коротких и 4 длинных текста, включая ремонт пола и уточнение платежа. Seed 1234 и 1235 | 24 |
| `listening-ru-calls/{short,long}/` | 8 коротких и 4 длинных звонка, включая встречу, обмен, показания счётчика и передачу ключей. Seed 1236 и 1237 | 24 |

```bash
cd /opt/firered-tuning
run=training/outputs/firered-vox-recipe-12-cap30m-levels/lora-base-recovery-4patch-0250
pack="$run/audio/step-0250/backbone-complete"
ls "$run/lora/step-0250/adapter_model.safetensors"
find "$pack" -name '*.wav' -type f | wc -l
jq '{audio_complete, asr_complete, variants: .plan.variants, wavs: (.results | length)}' "$pack/report.json"
```

Признак готовности аудио — `audio_complete: true` и 58 WAV. После этого Whisper расшифровывает записи. В `report.json` сохраняются тексты, seed, референсы, настройки генерации, хеши чекпойнта и WAV. В `summary.csv` — путь к каждому WAV, длительность, причина остановки, изменение RMS от начала к концу, спектральные показатели, WER и найденные ASR пропуски. Для прослушивания на другой машине после завершения можно упаковать этот каталог целиком.

```bash
tar -C "$pack" -czf "$run/backbone-0250-audio.tar.gz" .
```

## Обучение только Q/K/V/O backbone от Base, 250 шагов

Этот отдельный запуск обучает LoRA в 112 проекциях внимания backbone (28 блоков × Q/K/V/O). DiT работает с исходными весами и передаёт градиент в backbone. Параметры адаптера — rank 32, alpha 32 и dropout 0. Программа берёт те же нормализованные Dialogs и Spontaneous, сверяет отбор с двумя исходными сериями и использует прежнюю задачу восстановления четырёх патчей с весом 0,15.

```bash
cd /opt/firered-tuning
git pull --ff-only origin feat/russian-tts-adaptation
mkdir -p training/logs
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nohup .venv/bin/python -u \
  -m fireredtts3_local.train_backbone_attention_250 \
  >> training/logs/firered-backbone-only-250.log 2>&1 < /dev/null &
```

```bash
tail -n 30 -f /opt/firered-tuning/training/logs/firered-backbone-only-250.log
```

На шаге 100 сохраняется промежуточный чекпойнт для продолжения. На шаге 250 сохраняется итоговый адаптер, затем записываются все 58 WAV из таблицы выше с теми же голосами и seed. После записи Whisper рассчитывает WER. При повторном запуске скрипт продолжает с последнего сохранённого чекпойнта этого эксперимента и сверяет хеши уже записанных WAV.

```bash
cd /opt/firered-tuning
run=training/outputs/firered-vox-recipe-12-cap30m-levels/lora-backbone-only-recovery-4patch-0250
pack="$run/audio/step-0250/backbone-complete"
ls "$run/lora/step-0250/adapter_model.safetensors"
jq '{audio_complete, asr_complete, wavs: (.results | length)}' "$pack/report.json"
```

### Продолжить backbone-only с 250 до 500

Ключ `--continue-to-500` берёт `lora/step-0250` из `lora-backbone-only-recovery-4patch-0250`. С шага 251 восстанавливаются веса LoRA, состояние AdamW, расписание LR и генераторы случайных чисел. Обучение использует следующие 4 000 предъявлений из того же порядка примеров. На шаге 500 сохраняется чекпойнт, генерируются те же 58 WAV и запускается Whisper.

```bash
cd /opt/firered-tuning
git pull --ff-only origin feat/russian-tts-adaptation
mkdir -p training/logs
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nohup .venv/bin/python -u \
  -m fireredtts3_local.train_backbone_attention_250 --continue-to-500 \
  >> training/logs/firered-backbone-only-500.log 2>&1 < /dev/null &
```

```bash
tail -n 30 -f /opt/firered-tuning/training/logs/firered-backbone-only-500.log
```

После загрузки чекпойнта в логе появится `updates 251..500`. Новый каталог содержит адаптер, отчёт и аудио шага 500. Относительные имена WAV совпадают с комплектом шага 250.

```bash
cd /opt/firered-tuning
run=training/outputs/firered-vox-recipe-12-cap30m-levels/lora-backbone-only-recovery-4patch-0500
pack="$run/audio/step-0500/backbone-complete"
ls "$run/lora/step-0500/adapter_model.safetensors"
jq '{audio_complete, asr_complete, wavs: (.results | length)}' "$pack/report.json"
```
