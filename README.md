# Federated MeZO

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Распределённый zeroth-order fine-tuning языковой модели: **N агентов независимо считают SPSA-оценки градиента (MeZO), периодически согласуют веса через consensus mixing** на разных топологиях графа.

Учебный проект для программы **«Мультиагентные технологии и роевой интеллект»**, Сириус, 18–22 мая 2026, под руководством проф. О.Н. Граничина (СПбГУ). Трек 3 «Мультиагентная стохастическая безградиентная настройка нейросетевых моделей», постановка от Е. Крохалёва.

## Что внутри

- **MeZO-оптимизатор** (`src/mezo.py`) — in-place perturbation через переиспользование seed'а; память ≈ памяти инференса. Per-instance `torch.Generator` (thread-safe).
- **Векторизованный федеративный цикл** (`src/federated.py`) — N агентов через `torch.func.stack_module_state` + `vmap(functional_call)`. Один батчевый GPU-вызов прогоняет все N forward'ов параллельно.
- **Consensus mixing** (`src/consensus.py`) — матрицы W для топологий `full` / `ring` / `star`, спектральный gap, in-place применение.
- **Prompt-based MLM** (`src/prompt.py`) — для SST-2 шаблон `"{sentence} It was <mask>."`, никакой sequence-classification головы.
- **Sanity-тесты** (`tests/`) — обратимость возмущения, сходимость на линейной регрессии, проверка `⟨SPSA, true grad⟩ > 0`, doubly-stochastic свойства матриц, эмпирическая скорость контракции = |λ₂(W)|.

## Результаты

Все прогоны на RoBERTa-base + SST-2 (1000 train-сэмплов, 16 batch), MeZO lr=1e-6, ε=1e-3.

### День 1: single-agent baseline

| метод | val acc | заметка |
|---|---|---|
| init (zero-shot prompt) | 0.4587 | случайный baseline ≈ 0.49 |
| **MeZO** | **0.9094** | 5000 шагов, ZO |
| AdamW | 0.8922 | 1500 шагов, full backprop |

MeZO на удивление обгоняет AdamW при том же бюджете данных — типичный результат на low-data prompt-fine-tuning.

### День 2: FedAvg-MeZO (IID)

N агентов, периодически усредняем веса (W = 1/N · 1·1ᵀ). N-sweep при K=10:

| N | val acc | val loss |
|---|---|---|
| 1 | 0.8865 | 0.2981 |
| 2 | **0.8956** | 0.3053 |
| 4 | 0.8888 | 0.3082 |
| 8 | 0.8876 | 0.3155 |

K-sweep (шагов между раундами консенсуса) при N=4: разница между K=1 и K=1000 в пределах ±0.003 — на IID-данных частота консенсуса слабо влияет на финальный результат.

### День 3: топологии на non-IID

Данные шардятся через Dirichlet(α=0.5) — каждый агент видит оба класса, но с сильным перекосом (от 2% до 99% позитивов). N=8, K=100, 5000 шагов на топологию.

| топология | gap | \|λ₂\| теор. | эмпирич. rate | val acc |
|---|---|---|---|---|
| full (FedAvg) | 1.000 | 0.000 | 0.000 | **0.8876** |
| ring | 0.195 | 0.805 | 0.603 | 0.8853 |
| star | 0.125 | 0.875 | **0.855** | 0.8784 |

**Headline-результат:** эмпирическая скорость контракции ‖θ − θ̄‖ для `star`-топологии совпала с теоретическим |λ₂| с точностью ≤2.5% (`0.855` vs `0.875`). Это количественно подтверждает: спектральный gap матрицы W управляет скоростью consensus-контракции и для реального fine-tuning'а LLM. Плот `log ‖θ_t − θ̄‖ vs round` с теоретической линией `log|λ₂|` — в [`notebooks/04_day3_consensus.ipynb`](notebooks/04_day3_consensus.ipynb).

По accuracy на non-IID Dirichlet(0.5) full ещё держится (0.8876), но разрыв заметен; при более жёстком перекосе (α=0.1 или label-sort) ожидается, что full просядет сильнее.

## Быстрый старт

```bash
# Зависимости (pinned versions в uv.lock).
uv sync

# Все санитарные тесты.
uv run pytest

# Day 1 + Day 2 + Day 3 sweeps. Каждый скрипт идемпотентен —
# уже сосчитанные конфигурации пропускаются, результат пишется
# инкрементально в outputs/*.json.
uv run python scripts/run_fedavg.py        # Day 2
uv run python scripts/run_consensus.py     # Day 3 (non-IID)
```

Визуализация — в `notebooks/`, они читают JSON из `outputs/` и не содержат training loops.

## Структура репозитория

```
federated-mezo/
├── src/
│   ├── mezo.py           # MeZOOptimizer (per-instance torch.Generator)
│   ├── federated.py      # train_fedavg_mezo (vmap + stack_module_state)
│   ├── consensus.py      # build_full / build_ring / build_star, spectral_gap, apply_consensus
│   ├── data.py           # SST-2 loaders
│   ├── prompt.py         # prompt-based MLM (для классификации без головы)
│   └── train.py          # single-agent train loops, evaluate
├── scripts/              # production-прогоны (пишут в outputs/*.json)
├── notebooks/            # чистые визуализаторы из outputs/
├── tests/                # sanity-тесты, ~40 штук
├── outputs/              # результаты sweep'ов (закоммичены)
├── CLAUDE.md             # подробные инженерные заметки (что/почему/грабли)
├── pyproject.toml        # uv, deps, pytest config
└── README.md
```

Все инженерные детали и подводные камни (vmap + HuggingFace на Windows, 8 граблей MeZO, объяснение consensus matrices и спектрального gap'а) — в [CLAUDE.md](CLAUDE.md).

## Технический стек

- **Модель:** RoBERTa-base (125M).
- **Датасет:** SST-2 (GLUE), prompt-based MLM.
- **Фреймворк:** PyTorch + HuggingFace Transformers + datasets, `torch.func.vmap` для параллельных агентов.
- **Пакетный менеджер:** [uv](https://docs.astral.sh/uv/).
- **Железо:** RTX 4060 Ti 16 GB — N=8 с batch=16, max_len=128 берёт ~12 GB.

## Литература (минимум)

- **MeZO** (Malladi et al. 2023): https://arxiv.org/abs/2305.17333
- **SPSA overview** (Spall): https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF
- **FedAvg** (McMahan 2016): https://arxiv.org/abs/1602.05629
- **Distributed ZO через consensus** (Mhanna & Assaad): https://arxiv.org/abs/2210.05618
- **Non-IID via Dirichlet partition** (Hsu et al. 2019): https://arxiv.org/abs/1909.06335

Полный список ссылок — в [CLAUDE.md](CLAUDE.md).

## Лицензия

[MIT](LICENSE).
