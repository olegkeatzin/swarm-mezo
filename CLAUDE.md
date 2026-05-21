# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Swarm-MeZO — проект для программы «Мультиагентные технологии и роевой интеллект»

Центр тяжести проекта — теоретический документ
[`теория/swarm-mezo.md`](теория/swarm-mezo.md) и санитарная симуляция в
`теория/swarm_mezo/`. PyTorch-пайплайн в `src/` переносит ту же математику на
RoBERTa-base + SST-2.

## Команды

Пакетный менеджер: **uv** (`pyproject.toml`, не `requirements.txt`).

```bash
uv sync                                       # установить зависимости в .venv
uv run pytest                                 # все тесты PyTorch-пайплайна
uv run pytest теория/swarm_mezo/tests/        # тесты numpy-симуляции (E1–E3)
uv run python теория/swarm_mezo/run.py        # прогон E1/E2/E3 (~1 мин)
uv run python scripts/run_fedavg.py           # Day 2: FedAvg-MeZO → outputs/day2_fedavg.json
uv run python scripts/run_consensus.py        # Day 3: топологии на non-IID → outputs/day3_consensus.json
uv run python scripts/run_reputation.py       # Day 4: репутационная W → outputs/day5_reputation.json
uv run python scripts/run_trimmed.py          # Day 5: trimmed-mean W → outputs/day5_trimmed.json
uv run python scripts/run_k10.py              # K=10: частый консенсус, β×trim → outputs/day6_k10.json
uv run python scripts/verify_prompt.py        # проверка prompt-based подхода
```

Результаты сохраняются в `outputs/` (в .gitignore). Скрипты идемпотентны — уже выполненные конфигурации пропускаются.

## Состояние репозитория

Реализованы:

- `src/mezo.py`, `src/data.py`, `src/train.py`, `src/prompt.py`,
  `src/federated.py`, `src/consensus.py`, `src/reputation.py`
- санитарные тесты (`tests/test_perturbation.py`, `test_toy.py`,
  `test_gradient_sign.py`), `tests/test_federated_vmap.py` (8 тестов на
  vmap-helpers), `tests/test_consensus.py` (30 тестов на doubly-stochastic /
  спектр / contraction rate), `tests/test_reputation.py` (12 тестов на
  репутационную W, включая контрольные ветки conformity и trimmed-mean)
- `scripts/run_fedavg.py`, `scripts/run_consensus.py`, `scripts/run_reputation.py`,
  `scripts/run_trimmed.py`, `scripts/verify_prompt.py`, `scripts/smoke_test_vmap.py`,
  `scripts/smoke_test_reputation.py`, `scripts/pilot_throughput.py`
- `notebooks/01_sanity_visual.ipynb`, `02_day1_baselines.ipynb`,
  `03_day2_fedavg.ipynb`, `04_day3_consensus.ipynb`,
  `05_day4_reputation.ipynb`, `06_reputation_iid.ipynb`,
  `07_conformity_control.ipynb`, `08_trimmed_control.ipynb`
  (визуализаторы из `outputs/*.json`),
  `colab_run_reputation.ipynb`, `colab_run_reputation_iid.ipynb`,
  `colab_run_conformity.ipynb`, `colab_run_trimmed.ipynb`,
  `colab_run_k10.ipynb` (прогоны в Colab)
- теоретический документ `теория/swarm-mezo.md` + numpy-симуляция в
  `теория/swarm_mezo/` (E1, E2, E3 — все три гипотезы подтверждены)

## Workflow: scripts для прогонов, notebooks для визуализации

- **Тяжёлые прогоны** живут в `scripts/*.py`. Они инкрементально пишут результаты в `outputs/*.json`.
- **Визуализация** живёт в `notebooks/*.ipynb`, читает JSON из `outputs/`, рисует — никаких training loops в ноутбуках.
- **Импортируемая логика** — в `src/`, и скрипты, и ноутбуки её импортируют.

## Контекст

Одиночная работа на программе **Сириус, 18–22 мая 2026** под руководством проф. О.Н. Граничина (СПбГУ). За 5 дней очного этапа нужно представить решение по проекту **«Federated MeZO»** из трека 3 («Мультиагентная стохастическая безградиентная настройка нейросетевых моделей»).

Программа: https://siriusuniversity.ru/admission/educational-modules-and-activities/scientific-center-for-information-technologies-and-artificial-intelligence/multiagentnye-tekhnologii-i-roevoy-intellekt/

**Постановка от автора проекта (Е. Крохалёв):** распространить идеи distributed accelerated SPSA на задачу fine-tuning'а языковых моделей. Построить **Decentralized Federated MeZO**, где:
- каждый агент обучает локальную копию LLM;
- обновления вычисляются через MeZO (zeroth-order, без backprop);
- согласование происходит peer-to-peer;
- используется consensus mixing;
- добавляется Nesterov-like acceleration.

## Теоретическая база (что нужно понимать)

Полная и канонизированная версия — в [`теория/swarm-mezo.md`](теория/swarm-mezo.md). Краткий конспект:

**MeZO (Malladi et al., NeurIPS 2023):**
- Memory-efficient zeroth-order оптимизатор для fine-tuning LLM.
- Адаптация SPSA (Spall 1992) для нейросетей.
- Ключевой трюк: in-place perturbation через переиспользование seed'а — память = памяти инференса.
- Один шаг: `θ ← θ − η · projected_grad · z`, где `projected_grad = (L(θ+εz) − L(θ−εz)) / (2ε)` — скаляр, `z` — нормальный шум.
- Цена: высокая дисперсия SPSA-оценки → медленная сходимость.
- Статья: https://arxiv.org/abs/2305.17333 · Репо: https://github.com/princeton-nlp/MeZO

**Distributed SPSA / consensus** (школа Граничина):
- N агентов независимо считают локальные SPSA-оценки.
- Периодически усредняют веса через **consensus matrix W** (NxN).
- Топология графа влияет на скорость через **спектральный gap** матрицы W.
- Consensus mixing = механизм variance reduction (усреднение N независимых шумных оценок).
- Distributed ZO через consensus (Mhanna & Assaad): https://arxiv.org/abs/2210.05618

**Закон `1/N` и риск корреляции возмущений (E1, E2):**
- При независимых `z_i` MSE усреднённой SPSA-оценки падает как `1/N` (E1 даёт log-log slope −0.996 при теории −1).
- Общий банк сидов размера `K` даёт плато `~Var(N=1)/K` — это конкретное требование к FedKSeed.

**Репутационная модуляция W (§4 теории, E3):**
- `W_ij = r_j / Σ_k r_k` — row-stochastic, все строки одинаковые → за один шаг все агенты приходят в общий взвешенный центроид `(r^⊤ θ) / Σr`.
- Закон эволюции: `r_i ← r_i / (γ_r + β · |L_i − L_min|)`.
- Спектр по `β`: `β=0` → симметричный FedAvg (модель Де Гроота, ровно `W = (1/N)·J`); на синтетике E3 окно `β ∈ [0.05, 0.5]` давало ускорение ≈30%, но на RoBERTa+SST-2 (Day 4, §4.6–4.7 теории) это окно **не воспроизвелось** — ни на non-IID, ни на IID; рабочий режим — `β=0`. Большие `β` дают каскад.
- Сходимость опирается на вектор Перрона row-stochastic матрицы (простое собственное значение 1, остальные внутри единичного круга при наличии остовного дерева).
- Две ветки закона репутации: `mode="loss"` — `penalty=|L_i−L_min|` (заземление на качество, §4.2); `mode="conformity"` — `penalty=|L_i−L̄|` (дословное правило лекции, контроль). Conformity строго хуже loss во всех точках и каскадит сильнее — см. §4.8 теории.

**Federated learning (для контекста):**
- FedAvg (McMahan 2016) — каждый клиент делает local SGD-шаги, затем усреднение.
- Federated MeZO ≡ FedAvg, но локальный шаг — MeZO вместо SGD.
- Главное преимущество: **каждый агент укладывается в память инференса**, потому что не нужен backprop.

## Цель проекта в одну фразу

Реализовать федеративный MeZO с репутационной модуляцией матрицы консенсуса: N
агентов независимо файнтюнят копию языковой модели через SPSA-оценки градиента,
периодически согласуют веса через `W`, где вес агента зависит от его лосса,
и эмпирически найти рабочее окно гиперпараметра отбора `β`.

## Технический стек

- **Модель:** RoBERTa-base (125M параметров).
  - https://huggingface.co/FacebookAI/roberta-base
- **Датасеты:** SST-2 (и при наличии времени RTE, CoLA из GLUE).
  - https://huggingface.co/datasets/nyu-mll/glue
- **Фреймворк:** PyTorch + HuggingFace Transformers + datasets.
- **Симуляция агентов:** N копий весов модели стэкаются по leading dim'у через `torch.func.stack_module_state`, и `vmap(functional_call)` гонит все N forward'ов одним батчевым GPU-вызовом — реальная параллельность, не Python-threading. См. `src/federated.py`.
- **Подход к задаче классификации:** **prompt-based MLM** (`src/prompt.py`) — шаблон `"{sentence} It was <mask>."`, предсказываем ` terrible` / ` great` через `RobertaForMaskedLM`. Это следует оригинальному MeZO paper и не требует fine-tuning головы. НЕ используем `AutoModelForSequenceClassification`.
- **Логирование экспериментов:** результаты в JSON (`outputs/`), инкрементально.
- **Железо:** одна GPU 16+ GB достаточно для RoBERTa-base. На RTX 4060 Ti N=8 с BATCH=16, MAX_LEN=128 берёт ~12 GB.

## Грабли с vmap + HuggingFace на Windows

Production-пайплайн `src/federated.py` пробивается тремя обязательными мерами:

1. **`from datasets import ...` ДО `import torch`** во всех скриптах. Иначе pyarrow и torch на Windows конфликтуют DLL'ками и процесс падает segfault'ом без Python traceback (exit 5/139). Идиома стоит во всех `scripts/*.py`.
2. **`AutoModelForMaskedLM.from_pretrained(..., attn_implementation="eager", dtype=torch.bfloat16)`** при загрузке модели. `attn_implementation="eager"` — для совместимости с vmap (ИЛИ полагаться на монки-патч из п.3; сама по себе мера не помогает, но лежит в скриптах для документирования). `dtype=torch.bfloat16` — для скорости: forward на 4060 Ti ускоряется ~1.7–2× с тензорными ядрами Ada. Чтобы `(L+ − L−)/(2ε)` не терял ε's worth of precision на ε=1e-3, в `_make_prompt_loss` и `_eval_one_agent` логиты приведены к `.float()` перед `cross_entropy` — CE считается в fp32 даже когда модель в bf16.
3. **Монки-патч `transformers.masking_utils._ignore_bidirectional_mask_sdpa`** — функция содержит `padding_mask.all()` в Python `if`, что vmap отвергает как data-dependent control flow. Патч (`_patch_transformers_for_vmap()`) глушит fast-path всегда False; полная маска строится дальше тензорно. Вызывается на импорте `src/federated.py`, ничего больше делать не надо.

Также в HF MLM-моделях входной embedding tied с lm_head. `stack_module_state` дедуплицирует по `data_ptr` → tied-параметр выпадает из stacked params, и MeZO его не возмущает. `_untie_weights_inplace` развязывает их перед стэкингом.

## Ключевые гиперпараметры (стартовые значения из MeZO paper)

- `lr` = 1e-6 (на 2-3 порядка меньше Adam!)
- `eps` = 1e-3 (масштаб возмущения)
- `batch_size` = 16
- `model.eval()` обязательно — dropout убьёт SPSA-оценку
- `torch.no_grad()` обязательно — иначе пропадает экономия памяти

Для federated части:
- `N` (число агентов) = 2, 4, 8
- `local_steps` (шагов между consensus-раундами) = 1, 10, 50, 200, 1000 — главный гиперпараметр, отвечающий за explore/exploit trade-off
- `β` (репутационная модуляция, см. `src/reputation.py`) — на синтетике E3 окно `[0.05, 0.5]`, но на RoBERTa+SST-2 рабочего окна нет (Day 4): `β=0` (FedAvg) — рекомендуемый режим, `β>0` нейтрален или вредит

## Структура репозитория

```
swarm-mezo/
├── CLAUDE.md
├── pyproject.toml              # зависимости (uv), pytest config
├── теория/
│   ├── swarm-mezo.md           # ⭐ теоретическая база
│   ├── swarm-mezo-spec.md      # ТЗ на санитарную симуляцию
│   └── swarm_mezo/             # NumPy E1/E2/E3 + results/
├── src/
│   ├── mezo.py                 # MeZOOptimizer: per-instance torch.Generator
│   ├── federated.py            # train_fedavg_mezo: N агентов через vmap
│   ├── consensus.py            # матрицы W: ring, star, full + apply_consensus
│   ├── reputation.py           # W (§4): mode=loss / conformity + trim_k (robust-agg)
│   ├── data.py                 # SST2Loaders
│   ├── prompt.py               # prompt-based MLM
│   └── train.py                # train_mezo, train_adamw, evaluate; TrainHistory
├── scripts/
│   ├── run_fedavg.py           # Day 2: N-sweep и K-sweep
│   ├── run_consensus.py        # Day 3: ring/star/full на non-IID
│   ├── run_reputation.py       # Day 4: β-sweep, ветки loss + conformity, IID через SHARDING=iid
│   ├── run_trimmed.py          # Day 5: trimmed-mean W, IID, сетка β × trim_k∈{2,4}
│   ├── run_k10.py              # K=10: частый консенсус (vs K=100), IID, β × trim_k∈{0,2}
│   ├── smoke_test_vmap.py, smoke_test_reputation.py
│   ├── verify_prompt.py, pilot_throughput.py
├── tests/
│   ├── conftest.py             # добавляет корень в sys.path
│   ├── test_perturbation.py    # Test A: обратимость возмущения
│   ├── test_toy.py             # Test B: сходимость MeZO на линейной регрессии
│   ├── test_gradient_sign.py   # Test C: <SPSA-оценка, истинный градиент> > 0
│   ├── test_federated_vmap.py  # vmap-helpers
│   ├── test_consensus.py       # топологии W, спектральный gap
│   └── test_reputation.py      # репутационная W: row-stochastic, β=0=FedAvg, β→∞ winner-take-all, conformity- и trimmed-контроль
├── notebooks/
│   ├── 01_sanity_visual.ipynb
│   ├── 02_day1_baselines.ipynb
│   ├── 03_day2_fedavg.ipynb
│   ├── 04_day3_consensus.ipynb
│   ├── 05_day4_reputation.ipynb
│   ├── 06_reputation_iid.ipynb       # Day 4 IID-контроль
│   ├── 07_conformity_control.ipynb   # ветка loss vs conformity
│   ├── 08_trimmed_control.ipynb      # Day 5 trimmed-mean: сетка β × trim_k
│   ├── colab_run_reputation.ipynb    # прогон Day 4 в Google Colab
│   ├── colab_run_reputation_iid.ipynb # прогон Day 4 IID-контроля в Colab
│   ├── colab_run_conformity.ipynb    # прогон conformity-ветки в Colab
│   ├── colab_run_trimmed.ipynb       # прогон trimmed-mean ветки в Colab
│   └── colab_run_k10.ipynb           # прогон K=10 (частый консенсус) в Colab
└── outputs/                    # JSON с результатами (gitignore)
```

## План работы на 5 дней

### День 1 (понедельник, 18 мая): Baseline на одном агенте
- Воспроизвести single-agent MeZO на RoBERTa-base + SST-2.
- Прогнать standard fine-tuning через AdamW — верхний бейзлайн.

### День 2 (вторник, 19 мая): FedAvg-MeZO
- N агентов, у каждого свой MeZO, периодически усредняем веса.
- Эксперимент: разные N и local_steps.

### День 3 (среда, 20 мая): Consensus mixing с топологиями на non-IID
- Заменить FedAvg на consensus mixing через матрицу W.
- Топологии: ring, star, full. Данные шардятся non-IID через Dirichlet(α=0.5).
- Headline-плот: log ‖θ − θ̄‖ vs round для каждой топологии с теоретической линией log|λ₂| — спектральный gap эмпирически.

### День 4 (четверг, 21 мая): Репутационная W — выполнен, результат отрицательный
- Закон `r_i ← r_i/(γ_r + β·|L_i − L_min|)`, `W_ij = r_j/Σr` (см. `src/reputation.py`).
- Sweep `β ∈ {0, 0.1, 0.5, 1, 10}` на non-IID Dirichlet split'е (как Day 3) и
  IID-контроль (`SHARDING=iid`).
- **Результат:** окно `β` из E3 на RoBERTa+SST-2 не воспроизвелось — ни на
  non-IID (каскад, −8 п.п.), ни на IID (ничья с FedAvg). Рекомендуемый режим
  `β=0`. Полный разбор — §4.6–4.7 теории.
- Контрольная ветка `mode="conformity"` (дословное правило лекции
  `penalty=|L_i−L̄|`): строго хуже loss-ветки, каскадит сильнее — §4.8 теории.

### День 5 (пятница, 22 мая): Презентация
- 10 слайдов, 10 минут.
- Структура: мотивация → MeZO → закон 1/N (E1) → риск корреляции (E2) → топологии (Day 3) → репутационная W и спектр β (E3 + Day 4) → выводы.

## 8 граблей в реализации MeZO (главная техническая опасность)

1. **Один и тот же seed для всех трёх перебираний параметров** (forward+, forward−, update). Если разный — метод математически развалится.
2. **Dropout должен быть отключён** через `model.eval()`. Иначе L(θ+εz) и L(θ−εz) посчитаются на разных масках dropout'а, SPSA-оценка станет мусором.
3. **`@torch.no_grad()` обязательно** — без него autograd копит активации, вся экономия памяти теряется.
4. **`loss.item()` берёт скаляр** — иначе могут остаться ссылки на графы.
5. **Learning rate 1e-6, не 1e-4.** Adam-like LR взорвёт модель за пару шагов.
6. **Параметры с `requires_grad=False` пропускать** — иначе возмущается то, что не обновляется.
7. **Один батч на оба forward pass'а** внутри одного шага. Иначе разность будет содержать разницу между батчами.
8. **Per-instance `torch.Generator` вместо глобального RNG.** В `MeZOOptimizer._rng` и в `src/federated.py` (`rng = torch.Generator(device=device)`) — это и thread-safe для будущих расширений, и удобно: re-seed одним числом полностью воспроизводит всю последовательность z. При расширении не трогать глобальный RNG внутри `_perturb`/`_perturb_stacked`.

## Ключевые детали реализации (src/mezo.py)

1. **Per-instance `torch.Generator`** вместо `torch.manual_seed` — не трогает глобальный RNG. Генератор лениво создаётся на устройстве первого параметра.
2. **`torch.randint`** для выбора seed шага. На vmap-пути в `src/federated.py` глобальный torch RNG нам не мешает, и однородный API лучше mixing'а с `random` модулем.

```python
self._rng = torch.Generator(device=device)   # per-instance, не global

def _perturb(self, scaling, seed):
    rng = self._get_rng()
    rng.manual_seed(seed)
    for p in self.params:
        z = torch.empty_like(p).normal_(generator=rng)
        p.data.add_(z, alpha=scaling)

def step(self, loss_fn):
    seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    ...
```

В `src/federated.py` тот же приём разворачивается на стэк параметров: один `torch.empty(N, *shape).normal_(generator=rng)` даёт независимые z по агентам (разные куски RNG-stream), а пересеивание `rng` тем же per-step seed'ом точно их воспроизводит.

Референсная реализация single-agent (для сверки): https://github.com/princeton-nlp/MeZO/blob/main/large_models/trainer.py

## Санитарные тесты (запускать перед любым большим прогоном)

Все три теста реализованы в `tests/`. Запуск: `uv run pytest`.

- **Test A** (`test_perturbation.py`) — обратимость возмущения: `_perturb(+ε, seed)` + `_perturb(−ε, seed)` = identity. Также проверяет, что шаг воспроизводим при фиксированном seed.
- **Test B** (`test_toy.py`) — MeZO за 3000 шагов снижает loss линейной регрессии минимум вдвое.
- **Test C** (`test_gradient_sign.py`) — по 200 seed'ам `<projected_grad · z, истинный_градиент>` > 0 в среднем.

Дополнительно: `test_consensus.py` (30 тестов на матрицы W), `test_federated_vmap.py` (8 тестов на vmap-helpers), `test_reputation.py` (12 тестов на репутационную W, включая контрольные ветки conformity и trimmed-mean).

## Concept map (что с чем связано)

- **MeZO** = **SPSA** + in-place trick (Princeton 2023)
- **SPSA** = детище школы Спалла и Граничина (~30 лет работы)
- **Federated MeZO** = FedAvg, но local step = MeZO instead of SGD
- **Consensus matrix W** управляет «как именно агенты общаются»:
  - Симметричная `W = (1/N)·J` → FedAvg (модель Де Гроота)
  - Кольцо / звезда → меньший спектральный gap, медленнее consensus
  - Репутационная `W_ij = r_j/Σr` → row-stochastic, точка консенсуса смещена к лучшим
- **Спектральный gap** управляет скоростью сходимости consensus-части
- **`β`** в репутационной W задаёт спектр «кооперация ↔ отбор»; окно `[0.05, 0.5]` есть на синтетике E3, но на RoBERTa+SST-2 не воспроизводится — рабочий режим `β=0` (Day 4)
- **`local_steps`** управляет балансом коммуникация vs локальный прогресс
- **Consensus mixing = variance reduction** для SPSA-оценок (закон `1/N`, E1)

## Что говорить на защите (заготовки)

**Мотивация:** «Backprop требует памяти в 5–10 раз больше инференса из-за активаций. MeZO решает это через zeroth-order оценку градиента, но цена — высокая дисперсия. Я предлагаю уменьшить дисперсию через N агентов с consensus mixing.»

**Закон 1/N:** «E1 на квадратике даёт log-log slope −0.996 при теоретическом −1 — закон `1/N` для усреднённой SPSA-оценки экспериментально подтверждён.»

**Связь с курсом:** «Consensus mixing — это распределённая стохастическая аппроксимация в духе работ О.Н. Граничина. Спектральный gap матрицы W управляет скоростью согласования; в Day 3 я показываю эту связь эмпирически на LLM (rate ‖θ−θ̄‖ совпадает с |λ₂| в пределах 2.5% для star-топологии).»

**Про роевой интеллект:** «Репутационная модуляция `W_ij = r_j/Σr` с гиперпараметром `β` — это непрерывный спектр между чистой кооперацией (`β=0` = модель Де Гроота, FedAvg) и жёстким отбором (`β→∞` = `gbest` из PSO). E3 на гладком синтетическом ландшафте (QuadraticWithWells M=10) показывает рабочее окно `β ∈ [0.05, 0.5]` с ускорением раннего descent ≈30%. Но Day 4 — честный отрицательный результат: на RoBERTa+SST-2 это окно не воспроизводится. На non-IID отбор по probe-лоссу путает фитнес с репрезентативностью шарда и даёт каскад (−8 п.п.); IID-контроль каскад убирает, подтверждая диагноз, но окна нет и там. Контрольная ветка `mode="conformity"` (дословное правило лекции, штраф за отклонение от консенсуса вместо качества) — строго хуже и каскадит сильнее, что показывает: заземление репутации на лосс — это суть метода, а не косметика. Рекомендуемый режим — `β=0` (FedAvg-MeZO).»

**Про коммуникацию:** «В MeZO каждый шаг полностью описывается двумя числами — projected_grad и seed. Это означает потенциал коммуникации в несколько байт на раунд, что на порядки меньше FedAvg. E2 количественно показывает: при общем банке сидов размера `K` плато MSE ровно ×`K` — это конкретное требование к реализации FedKSeed.»

## Все ссылки одним списком

**Основное:**
- MeZO paper: https://arxiv.org/abs/2305.17333
- MeZO репо: https://github.com/princeton-nlp/MeZO
- MeZO блог: https://princeton-nlp.github.io/mezo/

**Distributed / consensus:**
- Distributed ZO через consensus: https://arxiv.org/abs/2210.05618
- Adaptation-diffusion consensus: https://arxiv.org/abs/1410.6956
- ε-консенсус под шумом и задержками (Amelina et al. 2015): https://www.sciencedirect.com/science/article/pii/S0005109814005044
- FedAvg: https://arxiv.org/abs/1602.05629

**Теоретический фундамент:**
- Spall overview SPSA: https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF
- Variance-reduced ZO для LM: https://arxiv.org/abs/2404.08080
- Nesterov & Spokoiny (accelerated gradient-free): https://arxiv.org/abs/1502.03811

**Связанные методы:**
- FedKSeed (federated MeZO с seed-обменом): https://arxiv.org/abs/2312.06353
- Dirichlet partition для non-IID (Hsu et al. 2019): https://arxiv.org/abs/1909.06335

**Инструменты:**
- RoBERTa-base: https://huggingface.co/FacebookAI/roberta-base
- GLUE: https://huggingface.co/datasets/nyu-mll/glue
- PyTorch: https://pytorch.org/
- HuggingFace Transformers: https://huggingface.co/docs/transformers

**Программа Сириуса:**
- https://siriusuniversity.ru/admission/educational-modules-and-activities/scientific-center-for-information-technologies-and-artificial-intelligence/multiagentnye-tekhnologii-i-roevoy-intellekt/
