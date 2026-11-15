# Новый этап от base: YO + Dialogs, готовые WAV до 20 секунд

Используются только уже подготовленные файлы из `training/datasets/yo-trimmed-06`:
YO после `text_alignment` +200 мс и Dialogs после выполненной ASR/VAD-ревизии.
Реплики с действием `review` в Dialogs остаются в ранее принятом виде.
Повторного скачивания, распаковки, обрезки или копирования WAV нет.

Фильтр — **фактическая длительность WAV ≤20.0 с** после прежней обработки окончаний.
20 секунд ровно разрешены; даже один лишний семпл исключает запись целиком.
Текст не укорачивается, ограничения количества символов нет. Правило применяется
к YO/Dialogs, train/validation, target/reference. Старое разделение дикторов
и сессий сохраняется: записи между частями не перемещаются. Если после фильтра
у реплики нет другого допустимого референса своего диктора, она исключается
с отдельной причиной в отчёте. Прежние фильтры качества и лимит на диктора
наследуются от исходного набора; добор данных не выполняется.

Это свежая **LoRA от оригинальной pretrained base**, без адаптеров 05/06.
Обучение всей модели с нуля здесь не используется. Fingerprint оригинальных
весов проверяется перед предрасчётом и обучением. Остаются rank16/alpha32,
backbone, LR=5e-6, одна эпоха YO, 7 YO + 1 Dialogs в полном шаге,
preservation=0.1 относительно базы и обе проверки качества.
Фиксированные 6471 шаг здесь не задаются: число шагов рассчитывается после отбора.

## Подготовить и запустить пилот

```bash
cd /home/firered-tuning
git pull --ff-only origin feat/russian-tts-adaptation

.venv/bin/python -u -m fireredtts3_local.duration_retrain
cat training/datasets/trimmed-20s-07/report.json

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -m fireredtts3_local.training_job start \
  training/datasets/trimmed-20s-07/recipe.json \
  --job-dir training/jobs/trimmed-20s-07-pilot \
  --precompute --stop-after 30

.venv/bin/python -m fireredtts3_local.training_job status \
  training/jobs/trimmed-20s-07-pilot
tail -n 20 training/jobs/trimmed-20s-07-pilot/job.log
```

Проверка исходных WAV читает их для SHA256 и может занять несколько минут.
Готовая подготовка появляется атомарно. При ошибке без готового output можно
повторить команду. Если output уже есть, используйте его `recipe.json`.
Старый `yo-trimmed-06` и его аудио должны остаться на диске: новые manifests
ссылаются на них. `--precompute` переиспользует признаки тех же WAV при совпадении
fingerprint базы, версии кэша и режима вычислений. Рабочие признаки не удаляются.

Пилот работает независимо от SSH. Его новый output:
`training/outputs/russian-trimmed-20s-base-07`.
В `report.json` есть часы/дикторы до и после, причины исключения и `total_steps`.

## Продолжить после успешных 30 шагов

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from fireredtts3_local.training_job import read_status
job = Path('training/jobs/trimmed-20s-07-pilot').resolve()
status = read_status(job)
assert not status['active'] and status['status'] == 'completed' and status['exit_code'] == 0, status
config = json.loads((job / 'recipe.json').read_text())
assert not config.get('initial_adapter')
output = Path(config['output_dir']).resolve()
assert (output / 'trainer_state.pt').is_file()
config['resume_checkpoint'] = str(output)
path = Path('training/recipes/trimmed-20s-07-resume.json')
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
PY

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -m fireredtts3_local.training_job start \
  training/recipes/trimmed-20s-07-resume.json \
  --job-dir training/jobs/trimmed-20s-07-full

.venv/bin/python -m fireredtts3_local.training_job status training/jobs/trimmed-20s-07-full
tail -f training/jobs/trimmed-20s-07-full/job.log
```

После обучения сравните фиксированный benchmark с базой и 06, а также вопросы
и проблемную длинную реплику. Новый отбор — эксперимент, не доказанное лечение
металлического звучания. Сводный loss разных выборок напрямую несопоставим.

## Слияние и новый коммит в том же HF-репозитории

После получения нового адаптера и проверки звука можно собрать отдельный пакет:

```bash
.venv/bin/python -u -m fireredtts3_local.adapt export \
  --base models/FireRedTTS3 \
  --adapter training/outputs/russian-trimmed-20s-base-07/best-validation \
  --output training/exports/FireRedTTS3-Base-Russian-20s-07

.venv/bin/python -u -m fireredtts3_local.adapt publish \
  training/exports/FireRedTTS3-Base-Russian-20s-07 \
  --repo-id shustov-te/FireRedTTS3-Base-Russian \
  --public --update-existing
```

Это будущие команды для **готовых** весов, а не часть подготовки/пилота.
Для включения результатов benchmark в карточку передайте export оба параметра
`--evaluation-baseline` и `--evaluation-adapted` с отчётами именно этих весов.
Прежняя HF-версия остаётся доступна по commit SHA. Для SGLang следует закрепить
новый SHA из `FireRedTTS3-Base-Russian-20s-07.published.json`, иначе контейнер
с прежним `revision` продолжит использовать старые веса.
Подробности: [обновление HF](MERGE_AND_PUBLISH.md).
