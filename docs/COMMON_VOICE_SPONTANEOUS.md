# Спонтанная русская речь поверх Dialogs-08

Этап `russian-spontaneous-09` продолжает LoRA
`training/outputs/russian-dialogs-base-08/best-validation` на оригинальной
`models/FireRedTTS3`. Начальный адаптер копируется в подготовленный набор;
новые optimizer/scheduler и выходной каталог отделены от Dialogs-08.
YO не используется. Этот эксперимент направлен на разговорную манеру;
устранение металлического тембра или улучшение длинных монологов не гарантируется.

## Источник и отбор

[Common Voice Spontaneous Speech 5.0 — Russian](https://mozilladatacollective.com/datasets/cmu5mg3pr00simh07epeylc55),
`sps-corpus-5.0-2026-09-11-ru`, архив 93 031 233 байта, SHA256
`aa0910f5fef7f24afccf5bca075995f0b7a970d96da72626c80507d629d05a6f`.
Другая версия архива отклоняется. Лицензия CC0; условия источника запрещают
установление личности участников и перезалив самого датасета. Аудио, тексты
и API-ключ не добавляются в Git или публичный пакет весов.

- Только непустой `transcription`, `votes >= 1`, официальный `dev`/`test`.
  У всех подтверждённых записей этой версии один одобрительный голос.
  `prompt` — показанный участнику вопрос, не транскрипция; он не используется.
  ASR для дополнения/исправления текста не запускается.
- Целые записи 2–30 секунд, полное пунктуационное окончание `. ! ?`, без
  многоточия в конце. Внутренние многоточия, повторения, «эм», «ну», растяжки,
  самопоправки и разговорная лексика сохраняются. Нет лимита символов.
- `[pause]`, `[dramatic pause]`, `[silent]`, `[silence]` внутри реплики переходят
  в `…`, у границ текста убирается только служебная метка. Аудио паузы остаётся.
  `[disfluency|эм]` восстанавливает указанный человеком `эм`. Преобразования и
  исходный текст хранятся в manifest. Неизвестные слова, `[unclear]`, `[laugh]`,
  `[noise]`, `[disfluency]` без расшифровки и паузы внутри слова приводят к
  исключению всей записи: для этих меток не изобретаются слова/управляющие токены.
- Остаются проверки языка, длительности, mono/finite, практически пустого
  аудио, дубликатов и подозрительно большой доли отсчётов около полной шкалы
  (`abs >= .999` более 0.1% отсчётов). Это инженерный фильтр, не доказательство
  клиппинга исходной записи. Отклоняются сообщения о проблемных audio_id.
- Пересчёт целой записи в 24 kHz PCM16. Если пик после декодирования/ресэмплинга
  превышает 1, ко всему файлу применяется один коэффициент `0.98 / peak`, чтобы
  не внести клиппинг при записи PCM16. Внутренняя динамика сохраняется. Нет
  лимитера, компрессора, нормализации к LUFS, VAD или обрезки окончаний.
- Метки `emotion` у этого корпуса нет: нейтральность не предполагается и не
  определяется автоматически.

Для каждого оставшегося `client_id` отдельно: треть записей с `?` в любом месте
транскрипции, треть с точкой и без `?`/`!`, треть случайная из оставшихся, без
повторного выбора. Фактических вопросов может быть больше трети. Тройки
распределяются на train и дополнительный holdout: ориентир 10% троек, минимум
одна тройка проверки и одна обучения на диктора. Бюджеты независимы: максимум
30 минут train и до 3 минут 20 секунд validation на диктора. Если записей мало,
бюджеты не заполняются повторами. Неподходящие дикторы явно перечисляются в отчёте.

Проверочные записи и последовательности слов не встречаются в обучении.
Референс — другая запись того же `client_id` внутри той же части, также <=30s.
Совпадающие слова с Dialogs исключаются. У Common Voice нет надёжного session ID;
это **проверка знакомых голосов на других записях**, не session/speaker holdout.
Пустой официальный train и исходные dev/test переразбиваются для нашего
эксперимента. Его результаты нельзя выдавать за независимый официальный ASR test.

Проверка настоящего архива до исключения пересечений с локальным Dialogs дала
328 кандидатов, 168 train (40.85 мин) и 24 validation (5.70 мин), 6 дикторов.
В обеих частях трети точные. Семь других дикторов не смогли дать обе части с
третями. Цифры на хосте определяет `report.json`, а не эти ориентиры.

## 1. Обновить код и скачать архив

На Gauss, без активного задания обучения:

```bash
cd /home/firered-tuning
git pull --ff-only origin feat/russian-tts-adaptation

.venv/bin/python -m fireredtts3_local.common_voice_spontaneous download
```

Введите Mozilla Data Collective API key в скрытое поле. Это не токен Hugging Face.
Ключ не сохраняется и не передаётся серверу хранения с подписанным URL.
Также поддерживается `MOZILLA_DATA_COLLECTIVE_API_KEY` из окружения. Не вписывайте
ключ в рецепт/команды/репозиторий. При сбое повторите download: готовый архив
переиспользуется только после сверки SHA256; незавершённый файл не устанавливается.

Загрузчик отправляет явный User-Agent: стандартный `Python-urllib` может быть
отклонён Cloudflare с HTTP403 / error1010 до проверки ключа. При сбое выводятся
HTTP-статус и безопасная классификация причины (авторизация, фильтр запросов,
лимит запросов, DNS/TLS/тайм-аут), без содержимого ответа, ключа или подписанного
URL. HTTP403 сам по себе не доказывает, что ключ неверный.

Если архив уже есть, укажите его через `prepare --archive /путь/архив.tar.gz`.
Повторно скачивать его не нужно. Для отдельного анализа без модели:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -m fireredtts3_local.common_voice_spontaneous inspect
```

## 2. Подготовить этап

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -u -m fireredtts3_local.common_voice_spontaneous prepare

cat training/datasets/cv-spontaneous-09/report.json

.venv/bin/python -m fireredtts3_local.common_voice_spontaneous audit
```

Нужны подготовленный `training/datasets/dialogs-base-08` и его best-validation.
Связь адаптера с train/validation Dialogs-08 проверяется по fingerprint; адаптер
из старой цепочки YO не принимается. Существующие выходные каталоги не
перезаписываются. При ошибке не переходите к обучению и не удаляйте предыдущий run.

Подготовка создаёт `training/datasets/cv-spontaneous-09/recipe.json`, WAV, manifests,
`report.json`, `filter_audit.json` и фиксированную копию стартового адаптера.
Исходный архив, Dialogs и его веса сохраняются.

## 3. Предрасчёт и обучение в фоне

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -m fireredtts3_local.training_job start \
    training/datasets/cv-spontaneous-09/recipe.json \
    --job-dir training/jobs/spontaneous-09-full \
    --precompute
```

SSH можно закрыть. Отдельный кэш: `.runtime/training-cache-spontaneous-09`.
Полный шаг содержит 4 Common Voice + 4 исходных Dialogs. Каждый выбранный
Common Voice target проходит ровно один раз за эпоху; Dialogs выбирается как
replay с балансировкой дикторов. Replay не означает полный проход по Dialogs.
LoRA backbone rank16/alpha32/dropout0.05, LR3e-6, до 5 эпох, preservation0.1
против исходной base. При 168 targets: 42 шага/эпоху, максимум 210 шагов.

Каждую эпоху оцениваются **все** CV-validation и **все** исходные Dialogs-validation,
все flow-патчи, фиксированные пары/шум. Best выбирается по среднему loss двух
корпусов; patience2. Нельзя напрямую сравнивать эти loss с логами другого рецепта;
сравнивайте с шагом 0 именно этого запуска.

## 4. Наблюдение

```bash
.venv/bin/python -m fireredtts3_local.training_job status \
  training/jobs/spontaneous-09-full

tail -n 3 training/outputs/russian-spontaneous-09/validation_metrics.jsonl
tail -n 3 training/outputs/russian-spontaneous-09/adaptation_validation_metrics.jsonl
tail -n 3 training/outputs/russian-spontaneous-09/selection_metrics.jsonl
cat training/outputs/russian-spontaneous-09/coverage.jsonl
tail -n 15 training/jobs/spontaneous-09-full/job.log
```

`validation_metrics` — Common Voice; `adaptation_validation_metrics` — сохранённый
Dialogs holdout. В `selection_metrics` старое имя `unseen_voice_loss` обозначает
первую проверку Common Voice, **не** незнакомых дикторов; `adaptation_loss` — Dialogs.
Успех задания: completed, exit_code0, activefalse. Лучший адаптер:
`training/outputs/russian-spontaneous-09/best-validation`.

Для первого запуска можно вместо полного задания использовать другой `--job-dir`
и `--stop-after 30`, затем продолжить сохранённый checkpoint:

```bash
.venv/bin/python -m fireredtts3_local.training_job start \
  training/datasets/cv-spontaneous-09/recipe.json \
  --job-dir training/jobs/spontaneous-09-pilot --precompute --stop-after 30
```

Не запускайте пилот после полного старта в тот же output. После успешного пилота:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from fireredtts3_local.training_job import read_status
job = Path('training/jobs/spontaneous-09-pilot').resolve()
state = read_status(job)
assert state['status'] == 'completed' and state['exit_code'] == 0 and not state['active'], state
config = json.loads((job/'recipe.json').read_text())
checkpoint = Path(config['output_dir'])
assert (checkpoint/'trainer_state.pt').is_file()
config['resume_checkpoint'] = str(checkpoint)
path = Path('training/recipes/spontaneous-09-resume.json')
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
PY

.venv/bin/python -m fireredtts3_local.training_job start \
  training/recipes/spontaneous-09-resume.json \
  --job-dir training/jobs/spontaneous-09-full
```

Сравнивайте генерацию с Dialogs-08 на одном голосе/тексте/seed: обычные вопросы,
реплики с явно записанными междометиями/паузами и прежние длинные тексты. Для
голоса бота пригоден сохранённый benchmark `training/eval/dialogs-base-08/benchmark.jsonl`.
Естественность пауз и самопоправок оценивается отдельно от WER: нежелательные
добавленные звуки, потерянные слова и металлический тембр остаются регрессиями.
