# Короткие нейтральные вопросы и ответы: Dialogs после YO

Источник: [langswap/dialogs-ru-emotional-conversations](https://huggingface.co/datasets/langswap/dialogs-ru-emotional-conversations),
revision `a74ff44d8d27ca3d21098ea3ed4d31a32923fbdd`. Используем только авторский
`train.csv`; оригинальные `val.csv` и `test.csv` в новый этап не включаем.
Лицензия — [OpenRAIL](https://huggingface.co/datasets/langswap/dialogs-ru-emotional-conversations/blob/main/LICENSE.md).
Метаданные и лицензионный файл скачиваются вместе и фиксируются SHA256 в плане.

## Правило отбора

- Только `emotion=neutral`, положительная длительность **не более 10 секунд**.
- Целая исходная реплика с окончанием `.`, `!`, `?`, включая знак перед кавычкой.
  Многоточия и несколько точек в окончании исключаются; текст не дописывается.
- Для **каждого актёра и каждого split** треть записей отбирается среди реплик с
  `?` в любом месте; треть — среди реплик без `?` и `!`, с точкой в конце;
  последняя треть выбирается из
  всех оставшихся подходящих записей. Файл повторно не выбирается.
- Это трети **по числу записей**, а не по секундам. Случайная треть тоже может
  содержать вопросы и точки: конечная доля вопросов может быть больше трети.
- Внутри актёра/split убираются повторы нормализованного текста. Пунктуационные
  контрасты «Готово.» / «Готово?» при этом различаются. Выбор перемежает исходные
  сессии и интервалы длины 0–2, 2–4, 4–6, 6–8, 8–10 с; seed фиксирован.
- Отбор идёт тройками question/statement/random до бюджета: звук не обрезается,
  поэтому до точных 30 минут может не хватить нескольких секунд.

## Дополнительная валидация

Обучающий бюджет — **30 минут на каждого актёра**. Сверх него резервируется
до **3 минут 20 секунд** проверочных записей каждого актёра. Сначала целые сессии
записи (`masha_dima_part*`, `masha_sveta_part*`) выделяются под проверку, затем
выбираются реплики. Остальные записи проверочных сессий в train не попадают.
Между Dialogs train/validation исключаются также одинаковые последовательности
слов, даже если различаются регистр, ё/е или пунктуация.

Это проверка новых текстов/сессий **знакомых актёров**, а не новых голосов:
всех трёх актёров мы намеренно используем в обучении. Дополнительно сохраняем
отдельную speaker-disjoint проверку YO; она не смешивается с Dialogs validation.

Во всём `metadata.csv` **6240 нейтральных реплик / 10,73 часа**. Из них до 10 с —
5219 / 6,68 часа; после проверки законченного пунктуационного окончания —
**5166 / 6,60 часа**. Это ещё до исключения авторских val/test и повторов текста.

Подсчёт по CSV с seed 1234, до проверки реальных WAV:

| Актёр | Подходит в официальном train | Выбрано в train | Минут train | Дополнительная validation |
|---|---:|---:|---:|---:|
| Дима | 678 / 75,24 мин | 279 = 93 + 93 + 93 | 29,91 | 36 / 3,22 мин |
| Маша | 2390 / 183,11 мин | 339 = 113 + 113 + 113 | 29,98 | 39 / 3,32 мин |
| Света | 2036 / 132,96 мин | 360 = 120 + 120 + 120 | 30,00 | 48 / 3,30 мин |

Всего **978 обучающих записей / 89,89 минуты**, плюс **123 проверочные / 9,84 минуты**.
В этих 978 записях фактически 443 с вопросительным знаком: 326 из question-слота
и ещё 117 из случайного. Ни одной записи дважды в плане нет.
Проверочные сессии при этом seed: `masha_dima_part2`, `masha_dima_part9`,
`masha_sveta_part10`. Счётчики окончательного импорта могут уменьшиться после
проверки файлов.

## Подготовка

Все команды — из корня репозитория. Для обновления кода дождитесь окончания
текущего GPU-запуска. Новые каталоги должны быть пустыми/отсутствовать.

```bash
cd /home/firered-tuning
git pull --ff-only origin feat/russian-tts-adaptation

.venv/bin/python -m fireredtts3_local.adapt plan-dialogs \
  --source training/raw/dialogs \
  --output training/plans/dialogs-neutral \
  --train-minutes 30 --validation-minutes 3.3333333333333335 \
  --max-seconds 10 --seed 1234 --download
```

`plan-dialogs` скачивает только CSV и лицензию, без всего аудиокорпуса.
`training/plans/dialogs-neutral/report.json` содержит часы, трети, длины и число
сессий отдельно по актёрам. `plan.json` сохраняет точный выбор и хеши исходников.

```bash
.venv/bin/python -u -m fireredtts3_local.adapt prepare-dialogs \
  training/plans/dialogs-neutral/plan.json \
  --output training/datasets/dialogs-neutral \
  --replay-manifest training/datasets/yo-split/train.jsonl \
  --validation-manifest training/datasets/yo-split/validation.jsonl \
  --download
```

Скачиваются только выбранные WAV, до 8 файлов одновременно. Аудио проверяется
на моно, конечность значений, длительность, сильный clipping и дубликаты, затем
целиком пересчитывается в 24 кГц. Пары используют разные файлы того же актёра.
Если WAV не проходит проверку, удаляется вся его отборочная тройка, чтобы
отношение 1:1:1 не изменилось; это явно отражается в `import_report.json`.
Автоматического заполнения освободившихся секунд другими WAV нет. Если после
проверок исчез актёр или остались записи без отдельного референса, импорт
останавливается до обучения.

У Dialogs нет исходного MOS и word alignment, как у YO: они не выдумываются и
не подменяются проверкой пунктуации. Нужно прослушать часть аудио с текстом;
в метаданных встречаются повторы и опечатки. Расставленные в `accent_text`
ударения в тренер не подаются: используем поле `text`.

Из существующих YO train/validation отдельно выбираются законченные реплики
до 10 секунд и исключаются записи без другого подходящего референса. Исходное
разделение дикторов сохраняется. На диске остаются ссылки на прежние YO WAV,
копирования аудио нет. Получаются:

- `train.jsonl` — выбранные Dialogs для обучения;
- `validation.jsonl` — дополнительные Dialogs из других сессий;
- `replay.jsonl` — короткие законченные YO для удержания разнообразия;
- `unseen-validation.jsonl` — короткие законченные YO с отложенными дикторами.

Метка `neutral` относится к Dialogs: YO не имеет такой разметки эмоций.

## Новый этап и фоновый запуск

[Рецепт](../recipes/russian_dialogs_96gb.json) начинает **новый этап** с
`training/outputs/russian-complete-03/best-validation`, с новым optimizer/scheduler,
LR `3e-6`, до 5 эпох. Исходный адаптер должен уже перестать изменяться.

`coverage_replay` использует по **4 целевые пары Dialogs + 4 пары YO** в полном
обновлении. Каждая обучающая запись Dialogs участвует ровно один раз за эпоху;
YO выбирается с балансировкой дикторов, не обязательно с полным покрытием YO.
Последний неполный шаг сохраняет соотношение 1:1. На обеих частях действует
preservation против frozen base. Для рецепта требуется целое число replay-пар
в эффективном batch. Все flow-патчи сохраняются; DiT считает их частями.

При 978 обучающих Dialogs получается **245 шагов/эпоху, максимум 1225 шагов**.
После аудиопроверки количество может измениться. `performance.jsonl.examples`
включает и Dialogs, и replay, а `coverage.jsonl.targets` — только Dialogs;
`replay_targets` показывает отдельный счётчик повторений YO.

```bash
.venv/bin/python -m fireredtts3_local.training_job start \
  recipes/russian_dialogs_96gb.json \
  --job-dir training/jobs/dialogs-pilot --precompute --stop-after 30

.venv/bin/python -m fireredtts3_local.training_job status training/jobs/dialogs-pilot
tail -f training/jobs/dialogs-pilot/job.log
```

SSH можно закрыть; `Ctrl+C` у `tail` завершает просмотр. Предрасчёт учитывает
обе проверки и replay; старые совпадающие признаки YO переиспользуются.
После успешного пилота создайте продолжение из его снимка рецепта:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
config = json.loads(Path('training/jobs/dialogs-pilot/recipe.json').read_text())
checkpoint = Path(config['output_dir'])
assert (checkpoint/'trainer_state.pt').is_file(), 'Не найден checkpoint пилота'
config['resume_checkpoint'] = str(checkpoint)
path = Path('training/recipes/dialogs-resume.json')
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
PY

.venv/bin/python -m fireredtts3_local.training_job start \
  training/recipes/dialogs-resume.json --job-dir training/jobs/dialogs-full
```

`validation_metrics.jsonl` содержит YO-проверку на фиксированных 32 парах;
`adaptation_validation_metrics.jsonl` — **все** дополнительные Dialogs targets
с фиксированными референсами/шумом. `selection_metrics.jsonl` показывает их
среднее с равным весом двух корпусов: по нему выбираются `best-validation` и
early stopping. Низкий loss не доказывает улучшения интонации или клонирования.

Сравнивайте исходный, `russian-complete-03` и новый адаптер на одинаковых
невиданных голосах. Помимо прежнего benchmark нужны короткие контрасты:
«Готово.» / «Готово?», «Хорошо.» / «Хорошо?», «Вы уже отправили файл?» /
«Когда вы отправите файл?», «Да, уже отправил.», «Нет, пока не отправил.».
WER не измеряет вопросительную интонацию: сравнение на слух остаётся необходимым.
