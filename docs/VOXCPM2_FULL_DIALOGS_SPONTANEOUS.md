# VoxCPM2 Full на русских Dialogs и Spontaneous

Один запуск с исходной `openbmb/VoxCPM2` ревизии `7936080b41163874f38610f46f438e2c0757f9fd`. Официальный Full-тренер обновляет веса модели с замороженным AudioVAE. На одной GPU он делает 1 000 обновлений, микробатч 2, накопление 8, пиковый LR `1e-5`. Обучение использует 16 000 предъявлений из нового манифеста.

## Как подготовлены записи

| Корпус | Условия отбора | Обработка каждого WAV |
|---|---|---|
| Dialogs | Все доступные нейтральные записи исходной ревизии. Проверяются аудио, клиппинг и соответствие длительности метаданным. Квот по дикторам нет. | Silero VAD 6.2.3 обрезает только края и оставляет до 200 мс вокруг обнаруженной речи. Сохраняется частота исходника. |
| Common Voice Spontaneous 5.0 Russian | Исходная запись с непустой транскрипцией, минимум одним голосом подтверждения и статусом dev/test в релизе. Исключаются жалобы, проблемные метки качества и неоднозначные текстовые аннотации. | После того же VAD очищенная копия записывается в моно WAV float32, 16 кГц. Используется CPU-версия DPDFNet8 из `dpdfnet==0.6.0`, параметр `attn_limit_db=15`. |

У обеих коллекций текст проверяется на завершение точкой, вопросительным или восклицательным знаком с учётом кавычек и скобок. Финальное многоточие исключается. Это формальная проверка пунктуации. В обучении и цель, и референс целиком длятся от 3 до 30 секунд после VAD. Валидация допускает также клипы короче 3 секунд. Референс содержит другой текст того же диктора из того же корпуса и раздела. Сессии Dialogs, совпадающие записи и одинаковые последовательности слов разделены между train и validation. Дикторы могут встречаться в обоих разделах.

DPDFNet работает на целой реплике с новым состоянием для каждой записи. При 15 дБ штатный алгоритм подмешивает часть исходного спектра и тем самым ограничивает подавление. Ни исходный MP3, ни промежуточный WAV после VAD не перезаписываются. В обучении, валидации и клонировании внутри рецепта пути для Spontaneous указывают на очищенную копию. Сигнал в ней имеет полосу до 8 кГц. Версии, ревизия ONNX и её SHA-256 закреплены. [DPDFNet 0.6.0](https://github.com/ceva-ip/DPDFNet/tree/v0.6.0) · [Веса DPDFNet8](https://huggingface.co/Ceva-IP/DPDFNet/tree/dd6818d00f50c836fed43a6243ebe49116de5964/onnx)

Каждая прошедшая отбор учебная запись используется минимум один раз. Не менее 4 800 из 16 000 предъявлений содержат `?` в любом месте текста. При естественной доле вопросов от 30% число предъявлений отдельных записей отличается максимум на одно. Если для выполнения квоты вопросы надо показывать чаще, то внутри каждой из двух групп число предъявлений тоже отличается максимум на одно. Дополнительные предъявления распределяются по длительности. Секунды округляются к ближайшей целой, половины вверх. Целевая доля пустого интервала переходит к ближайшим заполненным интервалам по обе стороны, на краю диапазона — к одному. Фактические доли зависят от ограничения на число повторов. Валидация содержит уникальные цели.

## 1. Обновить код на Gauss

```bash
cd /opt/firered-tuning
git switch feat/russian-tts-adaptation
git pull --ff-only origin feat/russian-tts-adaptation
```

Используются уже установленные `.venv-voxcpm2` и `models/VoxCPM2`. Проверьте архив Spontaneous командой ниже. При отсутствии архива загрузчик запросит ключ Mozilla в терминале.

```bash
.venv-voxcpm2/bin/python -m fireredtts3_local.common_voice_spontaneous download
```

## 2. Подготовить данные в фоне

```bash
cd /opt/firered-tuning
mkdir -p training/logs
nohup bash scripts/prepare_voxcpm2_dialogs_spontaneous.sh \
  > training/logs/voxcpm2-dialogs-spontaneous-prepare.log 2>&1 < /dev/null &
tail -n 20 -f training/logs/voxcpm2-dialogs-spontaneous-prepare.log
```

Скрипт устанавливает DPDFNet в окружение VoxCPM2, использует проверенные WAV Dialogs на диске и докачивает недостающие. При первом запуске потребуются веса DPDFNet с Hugging Face (около 15 МБ). Признак готовности — строка `[prepare] COMPLETE`. С помощью `Ctrl+C` можно выйти из просмотра лога.

## 3. Проверить состав и послушать пары

```bash
cd /opt/firered-tuning
ds=training/datasets/voxcpm2-dialogs-spontaneous-full-1000
jq '{denoised_clips, balance: (.balance | {target_draws, unique_targets, corpus_draws, question_draws, question_fraction, draw_histogram, empty_seconds})}' "$ds/report.json"
jq '.balance.buckets' "$ds/report.json"
head -n 1 "$ds/corpora/spontaneous/validation.jsonl" | jq '{text, after_vad: .trim.destination, after_dpdfnet: .audio}'
```

`draw_histogram` показывает число уникальных записей для каждого количества предъявлений. Для прослушивания одной и той же реплики используйте два пути из последней команды.

## 4. Запустить Full в фоне

```bash
cd /opt/firered-tuning
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv-voxcpm2/bin/python -m fireredtts3_local.voxcpm_training start \
  training/recipes/voxcpm2-dialogs-spontaneous-full-1000.yaml \
  --job-dir training/jobs/voxcpm2-dialogs-spontaneous-full-1000
```

## 5. Посмотреть ход обучения

```bash
cd /opt/firered-tuning
tail -n 30 -f training/jobs/voxcpm2-dialogs-spontaneous-full-1000/job.log
```

Итоговые веса и логи TensorBoard находятся в `training/outputs/voxcpm2-dialogs-spontaneous-full-1000/`. Для автоматической валидации официальный тренер использует до десяти батчей за проверку.
