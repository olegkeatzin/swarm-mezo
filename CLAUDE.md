# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Federated MeZO — проект для программы «Мультиагентные технологии и роевой интеллект»

## Команды

Пакетный менеджер: **uv** (`pyproject.toml`, не `requirements.txt`).

```bash
uv sync                                       # установить зависимости в .venv
uv run pytest                                 # все тесты
uv run pytest tests/test_perturbation.py -v   # один файл
uv run python scripts/run_fedavg.py           # FedAvg-эксперименты → outputs/day2_fedavg.json
uv run python scripts/verify_prompt.py        # проверка prompt-based подхода
```

Результаты сохраняются в `outputs/` (в .gitignore). Скрипты идемпотентны — уже выполненные конфигурации пропускаются.

## Состояние репозитория

На 2026-05-16 Дни 1–2 в коде завершены, идёт production-прогон Day-3 sweep'а (non-IID Dirichlet α=0.5, 3 топологии × 5000 шагов × N=8 ≈ 7.5ч). Реализованы:
- `src/mezo.py`, `src/data.py`, `src/train.py`, `src/prompt.py`, `src/federated.py`, `src/consensus.py`
- все три санитарных теста + `tests/test_federated_vmap.py` (8 тестов на vmap-helpers) + `tests/test_consensus.py` (30 тестов на doubly-stochastic / спектр / contraction rate)
- `scripts/run_fedavg.py`, `scripts/run_consensus.py`, `scripts/verify_prompt.py`, `scripts/smoke_test_vmap.py`, `scripts/pilot_throughput.py`
- `notebooks/01_sanity_visual.ipynb`, `notebooks/02_day1_baselines.ipynb`, `notebooks/03_day2_fedavg.ipynb`, `notebooks/04_day3_consensus.ipynb` (последние два — чистые визуализаторы из `outputs/*.json`)

**Что ещё не реализовано:** `scripts/run_evolution.py` (взвешенный consensus по validation score) — День 4.

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

**MeZO (Malladi et al., NeurIPS 2023):**
- Memory-efficient zeroth-order optimizer для fine-tuning LLM.
- Адаптация SPSA (Spall 1992) для нейросетей.
- Ключевой трюк: in-place perturbation через переиспользование seed'а — память = памяти инференса.
- Один шаг: `θ ← θ − η · projected_grad · z`, где `projected_grad = (L(θ+εz) − L(θ−εz)) / (2ε)` — скаляр, `z` — нормальный шум.
- Цена: высокая дисперсия SPSA-оценки → медленная сходимость.
- Статья: https://arxiv.org/abs/2305.17333
- Репо: https://github.com/princeton-nlp/MeZO
- Блог: https://princeton-nlp.github.io/mezo/

**Distributed SPSA / consensus** (школа Граничина):
- N агентов независимо считают локальные SPSA-оценки.
- Периодически усредняют веса через **consensus matrix W** (doubly stochastic, NxN).
- Топология графа влияет на скорость через **спектральный gap** матрицы W.
- Consensus mixing = механизм variance reduction (усреднение N независимых шумных оценок).
- Distributed ZO через consensus (Mhanna & Assaad): https://arxiv.org/abs/2210.05618
- Adaptation-diffusion consensus: https://arxiv.org/abs/1410.6956

**Federated learning (для контекста):**
- FedAvg (McMahan 2016) — каждый клиент делает local SGD-шаги, затем усреднение.
- Federated MeZO ≡ FedAvg, но локальный шаг — MeZO вместо SGD.
- Главное преимущество: **каждый агент укладывается в память инференса**, потому что не нужен backprop.
- FedAvg: https://arxiv.org/abs/1602.05629

**Variance reduction для ZO:**
- Известно, что MeZO шумит и требует variance reduction.
- Consensus mixing можно интерпретировать как форму variance reduction.
- Variance-reduced ZO для LM (Gautam et al.): https://arxiv.org/abs/2404.08080

**Nesterov acceleration для ZO:**
- Существует теоретическая база для accelerated gradient-free методов.
- Nesterov & Spokoiny: https://arxiv.org/abs/1502.03811

## Цель проекта в одну фразу

Реализовать федеративный MeZO: N агентов независимо файнтюнят копию языковой модели через SPSA-оценки градиента, периодически согласуют веса через consensus mixing, и сравнить со всеми разумными бейзлайнами.

## Технический стек

- **Модель:** RoBERTa-base (125M параметров). Не больше — иначе не уложимся в 5 дней.
  - https://huggingface.co/FacebookAI/roberta-base
- **Датасеты:** SST-2, RTE, возможно CoLA (все из GLUE).
  - https://huggingface.co/datasets/nyu-mll/glue
- **Фреймворк:** PyTorch + HuggingFace Transformers + datasets.
  - https://pytorch.org/
  - https://huggingface.co/docs/transformers
- **Симуляция агентов:** N копий весов модели стэкаются по leading dim'у через `torch.func.stack_module_state`, и `vmap(functional_call)` гонит все N forward'ов одним батчевым GPU-вызовом — реальная параллельность, не Python-threading. См. `src/federated.py`.
- **Подход к задаче классификации:** **prompt-based MLM** (`src/prompt.py`) — шаблон `"{sentence} It was <mask>."`, предсказываем ` terrible` / ` great` через `RobertaForMaskedLM`. Это следует оригинальному MeZO paper и не требует fine-tuning головы. НЕ используем `AutoModelForSequenceClassification`.
- **Логирование экспериментов:** результаты в JSON (`outputs/`), инкрементально. wandb — опционально.
- **Железо:** одна GPU 16+ GB достаточно для RoBERTa-base. На RTX 4060 Ti N=8 с BATCH=16, MAX_LEN=128 берёт ~12 GB.

## Грабли с vmap + HuggingFace на Windows (валидировано на 2026-05-15)

Production-пайплайн `src/federated.py` пробивается тремя обязательными мерами:

1. **`from datasets import ...` ДО `import torch`** во всех скриптах. Иначе pyarrow и torch на Windows конфликтуют DLL'ками и процесс падает segfault'ом без Python traceback (exit 5/139). Идиома стоит во всех `scripts/*.py`.
2. **`AutoModelForMaskedLM.from_pretrained(..., attn_implementation="eager")`** при загрузке модели, ИЛИ полагаться на монки-патч (см. п.3). Сама по себе эта мера не помогает (eager-mask внутри тоже зовёт sdpa_mask), но она лежит в скриптах для документирования.
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

## Структура репозитория

```
federated-mezo/
├── CLAUDE.md
├── pyproject.toml              # зависимости (uv), pytest config
├── src/
│   ├── mezo.py                 # MeZOOptimizer: per-instance torch.Generator (thread-safe)
│   ├── federated.py            # FedAvgMeZO: N агентов + ThreadPoolExecutor + fedavg_consensus
│   ├── consensus.py            # [НУЖНО] матрицы W: ring, star, full — День 3
│   ├── data.py                 # SST2Loaders: get_sst2_loaders, move_batch
│   ├── prompt.py               # prompt-based MLM: build_prompt_dataset, prompt_loss_fn, prompt_evaluate
│   └── train.py                # train_mezo, train_adamw, evaluate; TrainHistory dataclass
├── scripts/
│   ├── run_fedavg.py           # N-sweep и K-sweep; результаты → outputs/day2_fedavg.json
│   ├── verify_prompt.py        # проверка prompt-based подхода
│   ├── run_consensus.py        # [НУЖНО] День 3
│   └── run_evolution.py        # [НУЖНО] День 4
├── tests/
│   ├── conftest.py             # добавляет корень в sys.path
│   ├── test_perturbation.py    # Test A: обратимость возмущения
│   ├── test_toy.py             # Test B: сходимость MeZO на линейной регрессии
│   └── test_gradient_sign.py   # Test C: <SPSA-оценка, истинный градиент> > 0
├── notebooks/
│   ├── 01_sanity_visual.ipynb  # визуальные санитарные проверки
│   ├── 02_day1_baselines.ipynb # MeZO vs AdamW на SST-2
│   └── 03_day2_fedavg.ipynb    # FedAvg-MeZO эксперименты
└── outputs/                    # JSON с результатами (gitignore)
```

## План работы на 5 дней

### День 1 (понедельник, 18 мая): Baseline на одном агенте
- Воспроизвести single-agent MeZO на RoBERTa-base + SST-2.
- Прогнать standard fine-tuning через AdamW — верхний бейзлайн.
- Результат: две кривые на SST-2.

### День 2 (вторник, 19 мая): FedAvg-MeZO
- N агентов, у каждого свой MeZO, периодически усредняем веса.
- Эксперимент: разные N и local_steps.
- Результат: график «FedAvg-MeZO с N=4 vs single-MeZO».

### День 3 (среда, 20 мая): Consensus mixing с разными топологиями на non-IID
- Заменить FedAvg на consensus mixing через матрицу W.
- Топологии: ring, star, full.
- **Данные шардятся non-IID через Dirichlet(α=0.5)** (Hsu et al. 2019) — на IID все топологии вырождаются в FedAvg и эксперимент бессмыслен. На non-IID full-граф (gap=1) усредняет агрессивно и убивает per-agent специализацию, а ring/star (меньший gap) её сохраняют — топология действительно начинает иметь значение.
- Эксперимент: одна картинка — три кривые + бейзлайны.
- Headline-плот: log ‖θ − θ̄‖ vs round для каждой топологии с теоретической линией log|λ₂| — спектральный gap эмпирически.

### День 4 (четверг, 21 мая): Nesterov + эволюционный гибрид
**Ветка А — momentum/Nesterov:** локальный momentum в MeZO-шаге.
**Ветка Б (если время) — эволюционный гибрид:** взвешенный consensus mixing по качеству агентов на валидации:
```
W_ij ∝ exp(β · score_j)
```
β=0 → FedAvg, β→∞ → отбор как в ES. Промежуточное β — смесь кооперации и эволюционного отбора. Это попадает в название программы («роевой интеллект») и даёт научный нерв.

### День 5 (пятница, 22 мая): Презентация
- 10 слайдов, 10 минут.
- Структура: мотивация → MeZO → Federated MeZO → эксперименты (4 шт) → выводы.

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

Актуальная реализация отличается от минимального примера двумя моментами:

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

## Concept map (что с чем связано)

- **MeZO** = **SPSA** + in-place trick (Princeton 2023)
- **SPSA** = детище школы Спалла и Граничина (~30 лет работы)
  - Spall overview: https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF
- **Federated MeZO** = FedAvg, но local step = MeZO instead of SGD
- **Consensus matrix W** управляет «как именно агенты общаются»:
  - Полная (W_ij = 1/N) → FedAvg
  - Кольцо → медленное согласование, лучше exploration
  - Звезда → централизованная топология
- **Спектральный gap** W управляет скоростью сходимости consensus-части
- **Local_steps** управляет балансом коммуникация vs локальный прогресс
- **Consensus mixing = variance reduction** для SPSA-оценок
- **Эволюционный гибрид** = взвешивание W по качеству агентов → мост к ES/PSO/DE → попадание в тему «роевой интеллект»
  - OpenAI ES: https://arxiv.org/abs/1703.03864
  - PSO original: https://ieeexplore.ieee.org/document/488968

## Что говорить на защите (заготовки)

**Мотивация:** «Backprop требует памяти в 5–10 раз больше инференса из-за активаций. MeZO решает это через zeroth-order оценку градиента, но цена — высокая дисперсия. Я предлагаю уменьшить дисперсию через N агентов с consensus mixing, получая первую известную мне distributed accelerated MeZO для LLM fine-tuning.»

**Связь с курсом:** «Consensus mixing — это распределённая стохастическая аппроксимация в духе работ О.Н. Граничина. Спектральный gap матрицы W управляет скоростью согласования; в моём эксперименте я показываю эту связь эмпирически на LLM.»

**Про роевой интеллект:** «Federated MeZO как кооперативное усреднение — один полюс популяционной оптимизации. Эволюционные методы (PSO, ES, DE) — другой полюс. Я построил гибрид через взвешенное по качеству consensus mixing, превратив W в непрерывный спектр между кооперацией и отбором.»

**Про коммуникацию:** «В MeZO каждый шаг полностью описывается двумя числами — projected_grad и seed. Это означает потенциал коммуникации в несколько байт на раунд, что на порядки меньше FedAvg.»

## Все ссылки одним списком

**Основное:**
- MeZO paper: https://arxiv.org/abs/2305.17333
- MeZO репо: https://github.com/princeton-nlp/MeZO
- MeZO блог: https://princeton-nlp.github.io/mezo/

**Distributed / consensus:**
- Distributed ZO через consensus: https://arxiv.org/abs/2210.05618
- Adaptation-diffusion consensus: https://arxiv.org/abs/1410.6956
- FedAvg: https://arxiv.org/abs/1602.05629

**Теоретический фундамент:**
- Spall overview SPSA: https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF
- Variance-reduced ZO для LM: https://arxiv.org/abs/2404.08080
- Nesterov & Spokoiny (accelerated gradient-free): https://arxiv.org/abs/1502.03811

**Эволюционные методы (для гибрида):**
- OpenAI Evolution Strategies: https://arxiv.org/abs/1703.03864
- PSO original (Kennedy & Eberhart 1995): https://ieeexplore.ieee.org/document/488968

**Связанные методы:**
- FedKSeed (federated MeZO с seed-обменом): https://arxiv.org/abs/2312.06353
- TRM (для контекста темы «консенсус сквозь время»): https://arxiv.org/abs/2510.04871

**Инструменты:**
- RoBERTa-base: https://huggingface.co/FacebookAI/roberta-base
- GLUE: https://huggingface.co/datasets/nyu-mll/glue
- PyTorch: https://pytorch.org/
- HuggingFace Transformers: https://huggingface.co/docs/transformers
- wandb: https://docs.wandb.ai/

**Программа Сириуса:**
- https://siriusuniversity.ru/admission/educational-modules-and-activities/scientific-center-for-information-technologies-and-artificial-intelligence/multiagentnye-tekhnologii-i-roevoy-intellekt/