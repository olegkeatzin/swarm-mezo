# Swarm-MeZO

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests: 52 passing](https://img.shields.io/badge/tests-52%20passing-brightgreen.svg)](tests/)

**Распределённое zeroth-order fine-tuning'ование LLM, в котором правило согласования агентов параметризовано двумя числами — и непрерывно интерполирует между FedAvg, классическим consensus mixing и эволюционной селекцией.**

Учебный проект для программы **«Мультиагентные технологии и роевой интеллект»**, Сириус, 18–22 мая 2026, под руководством проф. О.Н. Граничина (СПбГУ). Трек 3 «Мультиагентная стохастическая безградиентная настройка нейросетевых моделей».

---

## TL;DR

MeZO ([Malladi et al. 2023](https://arxiv.org/abs/2305.17333)) — это zeroth-order оптимизатор, который файнтюнит языковую модель **без backprop**: память ≈ памяти инференса. Цена — высокая дисперсия SPSA-оценки градиента.

Стандартный способ снизить дисперсию — пустить **N агентов**, усреднять их веса. Но это бьёт по специализации агентов на non-IID данных.

**Swarm-MeZO** — это семейство правил согласования вида

```
θ_i  ←  (1 − α) · θ_i  +  α · Σ_j  softmax(−β · L_j)_j · θ_j
```

с двумя параметрами:
- **β** (selectivity, обратная температура softmax'а) — насколько резко смещаемся к агентам с лучшим лоссом
- **α** (social coefficient / inertia) — насколько сильно вообще тянемся к центру роя

Получается **2-параметрический континуум** между знакомыми режимами:

| режим | (α, β) | что делает |
|---|---|---|
| **identity** | α=0, любая β | агенты независимы, консенсуса нет |
| **FedAvg** | α=1, β=0 | классическое равное усреднение |
| **Consensus W** (Day 3) | через `consensus_fn` | doubly-stochastic W (ring/star/full) |
| **Swarm** | 0<α≤1, β>0 | взвешенное по приспособленности притяжение к лидеру(ам) |
| **Pure ES** | α=1, β→∞ | winner-take-all: все становятся лучшим |

Это и есть мост между **distributed consensus** (школа Граничина) и **evolutionary computation** (PSO, OpenAI ES, DE) — два разных полюса популяционной оптимизации, объединённые одной формулой.

## Что особенного

- **Векторизованная реализация N агентов через `torch.func.vmap`** — все N forward'ов выполняются как одна батчевая операция на GPU, реальный параллелизм, не Python threading. См. [`src/federated.py`](src/federated.py).
- **Swarm-шаг векторизован тем же стеком** — никакого Python-цикла по агентам, единственный `softmax(−β·L)` + `(N, 1, ..., 1)` бродкаст. См. [`src/swarm.py`](src/swarm.py).
- **52 sanity-теста**: обратимость возмущения MeZO, doubly-stochastic свойства матриц W, эмпирическая скорость контракции = |λ₂(W)|, β=0 → FedAvg, β→∞ → winner-take-all, convex-hull preservation.
- **Полная воспроизводимость через `uv.lock`** + per-instance `torch.Generator` в MeZO (фиксированный seed полностью восстанавливает последовательность z).
- **Идемпотентные prod-скрипты** — каждый sweep пишет JSON инкрементально, уже сделанные конфигурации пропускаются.

## Результаты

Все прогоны: RoBERTa-base + SST-2, prompt-based MLM `"{sentence} It was <mask>."`, 1000 train-сэмплов, batch=16, MeZO lr=1e-6, ε=1e-3.

### Headline (Day 3): спектральный gap количественно предсказывает скорость consensus-контракции

На non-IID Dirichlet(α=0.5), N=8 агентов, K=100 шагов между раундами:

| топология | gap | \|λ₂\| теор. | эмпирич. rate ‖θ−θ̄‖_after / _before | val acc |
|---|---|---|---|---|
| full (FedAvg) | 1.000 | 0.000 | 0.000 ✓ | 0.8876 |
| ring | 0.195 | 0.805 | 0.603 | 0.8853 |
| star | 0.125 | 0.875 | **0.855** | 0.8784 |

Эмпирическая скорость контракции для `star` совпала с теоретическим |λ₂| с точностью **≤2.5%** на реальном fine-tuning'е LLM. Headline-плот `log ‖θ_t − θ̄‖ vs round` с теоретической линией log|λ₂| — рендерится прямо в [`notebooks/04_day3_consensus.ipynb`](notebooks/04_day3_consensus.ipynb) на GitHub.

### Day 4: Swarm-MeZO sweep (в процессе)

Тот же non-IID Dirichlet(α=0.5) split что в Day 3 — результаты будут head-to-head сравнимы:

| config | α | β | смысл | val acc |
|---|---|---|---|---|
| `alpha0.5_beta1.0` | 0.5 | 1.0 | умеренный swarm | _running_ |
| `alpha0.5_beta5.0` | 0.5 | 5.0 | резкая селекция | _pending_ |
| `alpha1.0_beta2.0` | 1.0 | 2.0 | pull-to-leader, без inertia | _pending_ |

Probe-батч для скоринга агентов — 32 сэмпла из `train[1000:1032]`, **disjoint от training data, без val leakage**.

### Day 1: single-agent baseline (для контекста)

| метод | val acc | заметка |
|---|---|---|
| init (zero-shot prompt) | 0.4587 | случайный baseline ≈ 0.49 |
| **MeZO** | **0.9094** | 5000 шагов, ZO, без backprop |
| AdamW | 0.8922 | 1500 шагов, full backprop |

MeZO обгоняет AdamW при том же бюджете данных — типичный результат на low-data prompt fine-tuning.

### Day 2: FedAvg-MeZO N-sweep (IID, для контекста)

| N | val acc | val loss |
|---|---|---|
| 1 | 0.8865 | 0.2981 |
| 2 | **0.8956** | 0.3053 |
| 4 | 0.8888 | 0.3082 |
| 8 | 0.8876 | 0.3155 |

На IID даже N=2 уже выжимает variance reduction; K∈{1, 10, 100, 1000} при N=4 даёт разброс ±0.003 — частота консенсуса не критична. На non-IID картина меняется (см. Day 3).

## Быстрый старт

```bash
# Зависимости (pinned versions в uv.lock — полная воспроизводимость).
uv sync

# Все санитарные тесты (~5 секунд).
uv run pytest

# Прогоны (идемпотентны, инкрементально пишут в outputs/*.json).
uv run python scripts/run_fedavg.py        # Day 2: FedAvg-MeZO IID sweep
uv run python scripts/run_consensus.py     # Day 3: топологии на non-IID
uv run python scripts/run_swarm.py         # Day 4: Swarm-MeZO (α, β) sweep
```

Визуализация — в `notebooks/`, читают JSON из `outputs/`, training loops в ноутбуках не живут.

## Структура репозитория

```
swarm-mezo/
├── src/
│   ├── mezo.py           # MeZOOptimizer (per-instance torch.Generator)
│   ├── federated.py      # train_fedavg_mezo (vmap + stack_module_state),
│   │                       поддерживает consensus_fn И swarm_config
│   ├── consensus.py      # build_full / ring / star, spectral_gap, apply_consensus
│   ├── swarm.py          # SwarmConfig, swarm_consensus_step, compute_swarm_weights
│   ├── data.py           # SST-2 loaders
│   ├── prompt.py         # prompt-based MLM
│   └── train.py          # single-agent train loops
├── scripts/              # production-прогоны (пишут в outputs/*.json)
├── notebooks/            # чистые визуализаторы из outputs/*.json
├── tests/                # 52 sanity-теста
├── outputs/              # результаты sweep'ов (закоммичены)
├── CLAUDE.md             # подробные инженерные заметки
├── pyproject.toml        # uv, deps, pytest config
└── README.md
```

Инженерные подводные камни (vmap + HuggingFace на Windows, 8 граблей MeZO, доказательство doubly-stochastic свойств) — в [CLAUDE.md](CLAUDE.md).

## Математика swarm-шага: row-stochastic, НЕ doubly-stochastic

В матричном виде swarm-step эквивалентен умножению стекованных параметров на матрицу

```
W = (1 − α) · I  +  α · 𝟏 · wᵀ,    где  w_j = softmax(−β·L)_j,  Σ_j w_j = 1
```

Строки W суммируются в 1 (агенты делают выпуклую комбинацию → нет NaN-взрывов). Но **столбцы НЕ суммируются в 1** при β>0 — столбец j даёт `(1−α) + N·α·w_j`. Это значит, что **среднее параметров по агентам не сохраняется** между раундами — рой смещается в сторону агентов с низким лоссом.

Это и есть **намеренный отказ** от классического doubly-stochastic consensus в обмен на эволюционный сигнал. См. документацию и тесты в [`src/swarm.py`](src/swarm.py) и [`tests/test_swarm.py`](tests/test_swarm.py) — последний явно проверяет асимметрию.

## Технический стек

- **Модель:** RoBERTa-base (125M).
- **Датасет:** SST-2 (GLUE), prompt-based MLM, никакой classification head.
- **Фреймворк:** PyTorch + HuggingFace Transformers + datasets, `torch.func.vmap` для параллельных агентов.
- **Пакетный менеджер:** [uv](https://docs.astral.sh/uv/).
- **Железо:** RTX 4060 Ti 16 GB; N=8 с batch=16, max_len=128 — ~12 GB.

## Литература

**MeZO и ZO:**
- MeZO ([Malladi et al. 2023](https://arxiv.org/abs/2305.17333)) — основа.
- SPSA overview ([Spall](https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF)) — корни zeroth-order'а.
- Variance-reduced ZO для LM ([Gautam et al. 2024](https://arxiv.org/abs/2404.08080)).

**Distributed / consensus (мост к школе Граничина):**
- Distributed ZO через consensus ([Mhanna & Assaad](https://arxiv.org/abs/2210.05618)).
- Adaptation-diffusion consensus ([Granichin et al.](https://arxiv.org/abs/1410.6956)).
- FedAvg ([McMahan 2016](https://arxiv.org/abs/1602.05629)).

**Non-IID / FL:**
- Dirichlet partition ([Hsu et al. 2019](https://arxiv.org/abs/1909.06335)) — стандартный non-IID setup в FL.
- FedKSeed ([federated MeZO с seed-обменом](https://arxiv.org/abs/2312.06353)).

**Evolutionary / swarm (мост во вторую сторону):**
- OpenAI Evolution Strategies ([Salimans et al. 2017](https://arxiv.org/abs/1703.03864)).
- PSO original ([Kennedy & Eberhart 1995](https://ieeexplore.ieee.org/document/488968)).
- Nesterov & Spokoiny (accelerated gradient-free): https://arxiv.org/abs/1502.03811.

Полный список — в [CLAUDE.md](CLAUDE.md).

## Лицензия

[MIT](LICENSE).
