# Swarm-MeZO

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**Мультиагентное снижение дисперсии в безградиентной оптимизации LLM через консенсус с репутационной модуляцией.**

Проект для программы **«Мультиагентные технологии и роевой интеллект»**,
Сириус, 18–22 мая 2026, под руководством проф. О.Н. Граничина (СПбГУ).
Трек 3 «Мультиагентная стохастическая безградиентная настройка нейросетевых моделей».

---

## TL;DR

[MeZO (Malladi et al. 2023)](https://arxiv.org/abs/2305.17333) — zeroth-order
оптимизатор, который файнтюнит языковую модель **без backprop**: память ≈ памяти
инференса. Цена — дисперсия SPSA-оценки градиента, растущая с размерностью модели.
Одноагентная сходимость от этого практически непригодна.

Swarm-MeZO решает это мультиагентным консенсусом: `N` агентов делают локальные
MeZO-шаги, затем согласуют параметры через матрицу `W`. Базовый протокол — это
известный **gossip-SGD**; вклад проекта в трёх вещах:

1. **Закон `1/N`** — анализ поведения дисперсии при переносе консенсуса на
   zeroth-order оракул и эмпирическое подтверждение, что ошибка усреднённой
   оценки убывает линейно по числу агентов при независимых возмущениях.
2. **Риск общего банка сидов** — выявлен и количественно охарактеризован
   эффект, при котором экономия коммуникаций (FedKSeed-стиль) разрушает выигрыш
   `1/N`; сформулирована инженерная рекомендация по размеру банка.
3. **Репутационная модуляция `W`** — гиперпараметр `β` задаёт непрерывный спектр
   «кооперация ↔ отбор»; найдено рабочее окно `β ∈ [1, 10]`, где отбор
   измеримо помогает, а вне него — устраивает информационный каскад в локальный
   минимум.

Все три утверждения проверены численной симуляцией (E1, E2, E3) и собраны в
[`теория/swarm-mezo.md`](теория/swarm-mezo.md).

---

## Три гипотезы и их статус

| Гипотеза | Что утверждалось | Эксперимент | Результат | Статус |
|---|---|---|---|---|
| **H1** | Дисперсия консенсусной SPSA-оценки падает как `1/N` при независимых `z_i` | E1: log-log по `N ∈ {1..128}` на квадратике | log-log slope = **−0.996** | ✅ подтверждено |
| **H2** | Общий банк сидов размера `K` ломает закон `1/N`: плато `~Var(N=1)/K` | E2: семейство кривых `MSE(N)` для `K ∈ {1, 4, 16, 64, ∞}` | плато ровно ×`K`: 1.00 / 3.85 / 15.15 / 42.97 / 129.95 | ✅ подтверждено |
| **H3** | Репутационная `W` имеет оптимум `β` внутри спектра: умеренный отбор помогает, жёсткий — каскадирует в локальный минимум | E3: 50 запусков × 5 значений `β` на 2D multi-well | hit-rate 0.92 → 0.94 (`β∈[1,10]`) → **0.72** (`β=100`); симметричный контроль: 0.92 при всех `β` | ✅ подтверждено |

Картинки — в [`теория/swarm_mezo/results/`](теория/swarm_mezo/results/),
методология — в [`теория/swarm_mezo/README.md`](теория/swarm_mezo/README.md).

---

## Структура репозитория

```
swarm-mezo/
├── теория/                       # ⭐ актуальный пласт проекта
│   ├── swarm-mezo.md             #     Полная теоретическая база (v3)
│   ├── swarm-mezo-spec.md        #     ТЗ на санитарную симуляцию (E1–E3)
│   └── swarm_mezo/               #     NumPy-реализация E1, E2, E3
│       ├── objectives.py         #     Quadratic, MultiWell
│       ├── mezo.py               #     SPSA-оценка градиента
│       ├── consensus.py          #     матрицы W (симметричная, репутационная)
│       ├── swarm.py              #     run_swarm: локальный MeZO + консенсус
│       ├── experiments.py        #     run_e1, run_e2, run_e3
│       ├── plots.py, run.py
│       ├── tests/                #     12 тестов
│       ├── results/              #     3 PNG + 3 CSV
│       └── README.md             #     как запускать, как читать графики
├── лекции/                       # конспекты лекций О.Н. Граничина
├── src/                          # PyTorch-реализация на RoBERTa+SST-2
│   ├── mezo.py                   #     одноагентный MeZO
│   ├── federated.py              #     vmap-N агентов + матрица W
│   ├── consensus.py              #     топологии ring/star/full
│   ├── reputation.py             #     репутационная W (реализация §4)
│   ├── data.py, prompt.py, train.py
├── scripts/                      # production-прогоны → outputs/*.json
│   ├── run_fedavg.py             #     Day 2: FedAvg-MeZO IID sweep
│   ├── run_consensus.py          #     Day 3: топологии на non-IID
│   ├── run_reputation.py         #     Day 4: репутационная W на RoBERTa
│   ├── smoke_test_*.py, pilot_throughput.py, verify_prompt.py
├── notebooks/                    # визуализаторы из outputs/
├── tests/                        # тесты PyTorch-пайплайна
├── outputs/                      # результаты sweep'ов (под gitignore)
├── CLAUDE.md                     # инженерные заметки PyTorch-стека
├── pyproject.toml                # uv, deps, pytest
└── README.md
```

Документ `теория/swarm-mezo.md` фиксирует постановку и три центральных
утверждения, санитарная симуляция в `теория/swarm_mezo/` проверяет каждое из
них на синтетических функциях, а PyTorch-пайплайн в `src/` переносит ту же
математику на реальный fine-tuning RoBERTa-base + SST-2.

---

## Быстрый старт — санитарная симуляция

```bash
# Зависимости — pinned versions в uv.lock.
uv sync

# Полный прогон трёх экспериментов (~ минута на CPU).
# Пишет PNG + CSV в теория/swarm_mezo/results/.
uv run python теория/swarm_mezo/run.py

# Тесты симуляции (12 шт).
uv run pytest теория/swarm_mezo/tests/ -v
```

Прогон полностью детерминирован: `numpy.random.default_rng(seed)` везде,
повторный запуск даёт идентичные CSV.

---

## PyTorch-пайплайн на RoBERTa+SST-2

Реализация MeZO на RoBERTa-base + SST-2 (prompt-based MLM), федеративный
MeZO (N агентов через `torch.func.vmap + stack_module_state`), consensus mixing
с топологиями ring/star/full на non-IID Dirichlet split'е и репутационная
модуляция `W` из §4 теории. Инженерные подробности — в [CLAUDE.md](CLAUDE.md).

**Day 3 — спектральный gap.** Спектральный gap матрицы `W` количественно
предсказывает скорость consensus-контракции на реальном fine-tuning'е LLM:

| топология | gap | \|λ₂\| (теор.) | эмпирический rate ‖θ−θ̄‖ | val acc |
|---|---|---|---|---|
| full (FedAvg) | 1.000 | 0.000 | 0.000 ✓ | 0.8876 |
| ring | 0.195 | 0.805 | 0.603 | 0.8853 |
| star | 0.125 | 0.875 | **0.855** (≤2.5% от теории) | 0.8784 |

Headline-плот `log ‖θ_t − θ̄‖ vs round` с теоретической линией `log |λ₂|` — в
[`notebooks/04_day3_consensus.ipynb`](notebooks/04_day3_consensus.ipynb).

**Day 4 — репутационная W.** Перенос рабочего окна `β ∈ [1, 10]` с
двумерного multi-well на лосс RoBERTa+SST-2 (тот же non-IID Dirichlet split,
N=8, K=100). Тестируется sweep `β ∈ {0, 1, 10, 100}`: `β=0` совпадает с
FedAvg, `β=100` ожидаемо каскадирует в локальный минимум.

Скрипты PyTorch-стека:

```bash
uv run python scripts/run_fedavg.py        # Day 2: FedAvg-MeZO IID sweep
uv run python scripts/run_consensus.py     # Day 3: топологии на non-IID
uv run python scripts/run_reputation.py    # Day 4: репутационная W (β-sweep)
```

---

## Литература

**MeZO и ZO:**
- MeZO ([Malladi et al. 2023](https://arxiv.org/abs/2305.17333)) — основа.
- SPSA overview ([Spall](https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF)).
- Variance-reduced ZO для LM ([Gautam et al. 2024](https://arxiv.org/abs/2404.08080)).

**Distributed / consensus (школа Граничина):**
- Distributed ZO через consensus ([Mhanna & Assaad](https://arxiv.org/abs/2210.05618)).
- Adaptation-diffusion consensus ([Granichin et al.](https://arxiv.org/abs/1410.6956)).
- ε-консенсус под шумом и задержками ([Amelina et al. 2015](https://www.sciencedirect.com/science/article/pii/S0005109814005044)).
- FedAvg ([McMahan 2016](https://arxiv.org/abs/1602.05629)).

**Non-IID / FL / коммуникация:**
- Dirichlet partition ([Hsu et al. 2019](https://arxiv.org/abs/1909.06335)).
- FedKSeed ([federated MeZO с seed-обменом](https://arxiv.org/abs/2312.06353)).

Полный список и инженерные ссылки — в [CLAUDE.md](CLAUDE.md).

## Лицензия

[MIT](LICENSE).
