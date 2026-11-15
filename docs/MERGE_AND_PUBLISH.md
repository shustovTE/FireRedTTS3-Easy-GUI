# Слияние LoRA с пакетом CKermit и публикация для SGLang Omni

Для текущего русского адаптера используем
`training/outputs/russian-yo-half-05/best-validation`. В нём уже накоплены изменения
предыдущих этапов обучения. Экспорт создаёт полную модель; отдельная LoRA при
обслуживании запросов больше не нужна.

Проверенная версия пакета:

- `CKermit/FireRedTTS3-Base` — commit `53f8b59c3252f15d539a9205fbcab3e334a36b9f`;
- его core, RedAE и CAM++ побайтно совпадают с файлами исходной
  `FireRedTeam/FireRedTTS3`, используемой при обучении;
- оригинальный загрузчик добавляет языковые/управляющие токены при запуске.
  У CKermit они уже сохранены. Проверены совпадение полного backend токенизатора,
  словаря из 151924 токенов и кодирования контрольных текстов;
- runtime-интеграция находится в [SGLang Omni PR #3005](https://github.com/sgl-project/sglang-omni/pull/3005),
  проверенный исходный код — commit `718b51d52a2cffe2218f9c6a1ae076756eb8a505`.

Поэтому экспорт переиспользует **проверенные по SHA256** большие файлы из локальной
обучающей базы, а с Hub скачивает конфигурацию, токенизатор и карточку CKermit.
Это слияние с теми же исходными весами, без повторной загрузки примерно 12 ГБ.
Если локальные веса отличаются, экспорт останавливается.

## 1. Обновить код и войти с токеном для записи

Команды выполняются на GPU-машине, где находятся модели и обученный адаптер:

```bash
cd /home/firered-tuning
git pull --ff-only origin feat/russian-tts-adaptation

.venv/bin/hf auth login --force
.venv/bin/hf auth whoami
```

Токен создаётся на <https://huggingface.co/settings/tokens>. Для публикации нужны
права **Write**, либо fine-grained права, позволяющие создать и записать нужный
репозиторий моделей. Токен Read, использованный для скачивания датасета, этих прав
не даёт. `--force` нужен для замены уже активного Read-токена. В меню входа выберите
**Paste an access token**, введите токен
в скрытом поле; git credentials для этой задачи не нужны (`n`).

Логин берётся из `whoami` этого токена, а не из GitHub-логина или имени Linux.
Программа использует `token=True`; значение токена не передаётся в аргументах,
не записывается в модель, manifest или отчёт публикации.
Если токен задан переменной окружения `HF_TOKEN`, она имеет приоритет над кешем;
у этого токена также должны быть права записи.

## 2. Создать проверенный пакет модели

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
.venv/bin/python -u -m fireredtts3_local.adapt export \
  --base models/FireRedTTS3 \
  --adapter training/outputs/russian-yo-half-05/best-validation \
  --output training/exports/FireRedTTS3-Base-Russian \
  --evaluation-baseline training/eval/pilot-01/base/report.json \
  --evaluation-adapted training/eval/pilot-01/yo-half-05/report.json
```

Слияние работает на **CPU в FP32** и использует оперативную память. Ориентируйтесь
на порядка 20 ГБ доступной RAM и 15 ГБ свободного места для пакета. Это оценка
ресурсов, а не замер на конкретном сервере. Параметры исходной модели и адаптера
не изменяются. Итоговая модель сохраняет число параметров и приблизительный
размер оригинального пакета: около 12,3 ГБ, включая RedAE/CAM++.

Проверяются:

1. Fingerprint всех оригинальных артефактов и токенизатора из метаданных LoRA.
2. SHA256 больших файлов относительно зафиксированной версии CKermit.
3. Размерности и настройки core, backbone, RedAE, вариант Base и токенизатор.
4. Точные LoRA targets с учётом сокращённых PEFT суффиксов; неподдерживаемые
   конфигурации DoRA, произвольные saved modules, bias и т. п. отклоняются.
5. Выходы каждого LoRA-слоя через настоящий PEFT до merge и того же слоя после
   merge на фиксированных входах (`rtol=1e-4`, `atol=1e-5`).
6. Неизменность тензоров вне объявленных LoRA weights, отсутствие NaN/Inf,
   полнота и содержимое каждого сохранённого тензора.
7. Совпадение скопированных RedAE/CAM++ и файлов токенизатора.

Пакет собирается во временном каталоге и становится доступен по итоговому пути
после успешного завершения. Готовый каталог не перезаписывается. При обычной
ошибке сборки временные файлы убираются; после аварийного отключения процесса
может остаться скрытый временный каталог рядом с export.

В пакете будут веса полного core (возможны несколько shards и index), `config.json`,
токенизатор CKermit, `redae/`, `campp/`, лицензия Apache-2.0, исходная карточка,
новая карточка модели, `export_provenance.json` и `release_manifest.json` с хешами.
Сводка оценки публикуется без текстов, имён дикторов и референсного аудио.
Параметры `--evaluation-*` необязательны для других адаптеров, но передаются парой.

Проверки слияния не означают идентичность синтеза в SGLang: runtime, точность
вычислений и настройки генерации отличаются. В provenance остаётся честное
`sglang_runtime_validated: false` до отдельной проверки сервера.

## 3. Опубликовать под своим логином HF

```bash
.venv/bin/python -u -m fireredtts3_local.adapt publish \
  training/exports/FireRedTTS3-Base-Russian \
  --repo-name FireRedTTS3-Base-Russian \
  --public
```

Это создаст публичный `<логин из токена>/FireRedTTS3-Base-Russian`. `--public`
явно задаёт публичность; без него новый репозиторий приватный. Команда:

- повторно сверяет локальные файлы по manifest и отклоняет лишние файлы;
- загружает только явный список файлов готового пакета, без `.cache`, аудио,
  manifests обучения, adapter/trainer state и секретов;
- не перезаписывает другой существующий репозиторий с отличающимся содержимым;
- привязывает commit к исходной ревизии назначения и проверяет удалённые хеши;
- печатает URL и commit SHA опубликованной модели.

Отчёт записывается **рядом с пакетом**, чтобы не менять запечатанный список файлов:

```bash
cat training/exports/FireRedTTS3-Base-Russian.published.json
```

При сетевом сбое повторите ту же команду publish с тем же пакетом. Уже загруженное
содержимое может переиспользоваться Hub; если commit успел завершиться, повторный
запуск проверит совпадение и вернёт существующую ревизию. Другой релиз публикуйте
под новым именем: этот первоначальный workflow не затирает предыдущую модель.

Для фоновой загрузки вместо команды выше:

```bash
mkdir -p training/logs
nohup env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -u -m fireredtts3_local.adapt publish \
  training/exports/FireRedTTS3-Base-Russian \
  --repo-name FireRedTTS3-Base-Russian --public \
  >> training/logs/publish-russian.log 2>&1 < /dev/null &

tail -f training/logs/publish-russian.log
```

SSH можно закрыть, Ctrl+C у tail завершает только просмотр. При ошибке прав доступа
проверьте локальный аккаунт и права Write через `hf auth login`; токен не нужно
отправлять в чат или добавлять в Git.

## 4. Использовать в SGLang Omni

Используйте отдельное окружение SGLang Omni с интеграцией FireRedTTS3.
В проверенной на момент подготовки ветке main этой интеграции ещё нет. Для нового
checkout зафиксируйте ревизию PR; установку CUDA/UCX/FlashAttention выполните по
[официальной инструкции этой ревизии](https://github.com/sgl-project/sglang-omni/blob/718b51d52a2cffe2218f9c6a1ae076756eb8a505/docs/get_started/installation.md).
Обучающее `.venv` для этой установки не используется.

```bash
git clone https://github.com/sgl-project/sglang-omni.git /home/sglang-omni-firered
cd /home/sglang-omni-firered
git fetch origin refs/pull/3005/head
git checkout 718b51d52a2cffe2218f9c6a1ae076756eb8a505

uv venv .venv -p 3.12
uv pip install --python .venv/bin/python --prerelease=allow -e '.[firered-tts3]'
```

Если совместимая версия уже установлена, используйте её. Для проверки наличия
конфигурации в окружении SGLang:

```bash
.venv/bin/python -c 'from sglang_omni.models.firered_tts3.hf_config import FireRedTTS3Config; print("FireRedTTS3 integration available")'
```

На том же сервере можно запустить уже собранный локальный пакет. Следующую
команду выполняйте **из checkout SGLang Omni**, с его окружением:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/sgl-omni serve \
  --model-path /home/firered-tuning/training/exports/FireRedTTS3-Base-Russian \
  --config examples/configs/firered_tts3_base.yaml \
  --allowed-local-media-path /home/firered-tuning/training \
  --port 8000
```

Путь с референсами должен соответствовать фактическому размещению ваших WAV.
Дождитесь HTTP 200 от `curl --fail http://127.0.0.1:8000/health`. Проверка health
подтверждает готовность сервера, но качество оценивается генерацией и прослушиванием.
Параметры API и потокового PCM сохранены в карточке пакета и
[исходной инструкции CKermit](https://huggingface.co/CKermit/FireRedTTS3-Base/blob/53f8b59c3252f15d539a9205fbcab3e334a36b9f/README.md).

Контрольный запрос из первого примера прежнего benchmark можно подготовить в
окружении **обучающего репозитория**, пока сервер работает в другом терминале:

```bash
cd /home/firered-tuning
.venv/bin/python - <<'PY'
import json
from pathlib import Path
benchmark = Path('training/eval/pilot-01/benchmark.jsonl').resolve()
row = next(json.loads(s) for s in benchmark.read_text().splitlines() if s.strip())
ref = Path(row['reference_audio'])
ref = ref.resolve() if ref.is_absolute() else (benchmark.parent/ref).resolve()
assert ref.is_file(), ref
assert len(row['text']) <= 400, 'Для smoke test выберите текст не длиннее 400 символов'
payload = dict(input=row['text'], language=row.get('language', 'Russian'),
               references=[dict(audio_path=str(ref), text=row['reference_text'])],
               seed=row.get('seed', 1234), response_format='wav')
output = Path('training/eval/sglang-smoke')
output.mkdir(parents=True, exist_ok=True)
(output/'request.json').write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
print('Референс должен быть разрешён через --allowed-local-media-path:', ref)
PY

curl --fail-with-body http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  --data-binary @training/eval/sglang-smoke/request.json \
  --output training/eval/sglang-smoke/output.wav
```

Прослушайте результат рядом с соответствующим WAV из `pilot-01/yo-half-05`.
Совпадение seed не гарантирует побайтово одинаковое аудио между runtime.

На другой машине скачайте **весь** свой репозиторий (подставьте repo_id и commit из
отчёта публикации):

```bash
hf download YOUR_LOGIN/FireRedTTS3-Base-Russian \
  --revision COMMIT_FROM_PUBLISH_REPORT \
  --local-dir ./models/FireRedTTS3-Base-Russian
```

Передайте этот каталог в `--model-path`. Для русского задайте `language: Russian`,
один reference WAV с точной расшифровкой и целевой текст. В проверенной версии API
лимит `input` — 400 символов; звук — 24 кГц mono. Проверяйте короткие вопросы,
ответы и длинные фразы на тех же референсах, что использовались при оценке адаптера.

## Старый локальный экспорт

`adapt export --sglang-config /path/config.json ...` сохранён для совместимости.
Он также проверяет слияние core, но не привязывает пакет к Hub-версии CKermit и
не формирует запечатанный релиз для команды publish. Для текущей задачи используйте
команды выше без `--sglang-config`.
