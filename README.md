# pinn_accel

Минимальная версия проекта для быстрых экспериментов с PINN, весами loss-компонент и компактным линейным RL-агентом.

Что изменено относительно старой структуры:

- `train.py` только читает конфиг/CLI и запускает эксперимент.
- Уравнения лежат отдельно в `src/pinn_accel/equations`.
- Награды агента лежат отдельно в `src/pinn_accel/rewards.py`.
- Агент лежит отдельно в `src/pinn_accel/agents`.
- Сохранение истории, графиков и checkpoint-ов унифицировано через `ArtifactStore`.
- В тренировочном цикле нет обязательной повторной оценки loss после каждого `optimizer.step()`, поэтому шаг обучения дешевле.
- Для ускорения доступны фиксированные sampling pools и `torch.compile` через конфиг.

## Быстрый запуск

Из директории `pinn_accel`:

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

Для Burgers-конфига положите HDF5-файл рядом с `train.py` или задайте свой путь в `equation_params.data_path`:

```json
"equation_params": {
  "nu": 0.001,
  "data_path": "1D_Burgers_Sols_Nu0.001.hdf5",
  "sample_id": 0,
  "target_time": 0.0
}
```

```bash
python train.py --config configs/burgers_tiny_loss_weight.json
```

Быстрый smoke-run:

```bash
python train.py --equation heat --controllers fixed,tiny_loss_weight,softadapt,relobralo,gradnorm --steps 20 --no-plots
```

Артефакты сохраняются в `artifacts/<timestamp>/<equation>/<controller>/`:

- `history.json`
- `batch_info.json`
- `checkpoint.pt`
- `plots/*.pdf`

Для HDF5 Burgers training всегда full-batch: размеры `pde`, `ic`, `bc` берутся из размеров координат в файле и сохраняются в `batch_info.json`.
Если доступно истинное решение, в `history.json` также пишется `relative_l2`, а графики сохраняются как `relative_l2.pdf` и `comparison_relative_l2.pdf`.
Сравнение решений по срезам времени из `solution_slice_times` сохраняется в `comparison_solution_slices.pdf`.

## Контроллеры весов

Встроенные контроллеры:

- `fixed`
- `tiny_loss_weight`
- `softadapt`
- `relobralo`
- `gradnorm`

Если перечислить несколько контроллеров в `controllers`, для каждого сохраняются свои history/checkpoint/plots, а общие графики сравнения лежат в `comparison/plots`.

## Режимы оптимизации PINN

`training.optimizer_mode` поддерживает:

- `adam`
- `adam_lbfgs`
- `lbfgs`

Для `adam_lbfgs` можно явно задать `adam_steps` и `lbfgs_steps`. Если их не задать, `steps` делится примерно 80/20 между Adam и L-BFGS.
По умолчанию `freeze_weights_during_lbfgs=true`: в фазе L-BFGS adaptive-веса не пересчитываются, а L-BFGS оптимизирует только параметры PINN с последними весами из Adam.

## Добавить новое уравнение

1. Создать файл в `src/pinn_accel/equations`, например `klein_gordon.py`.
2. Вернуть из него `EquationSpec`: домен, residual и constraints.
3. Зарегистрировать builder в `src/pinn_accel/equations/__init__.py`.

Минимальная форма residual:

```python
def residual(model, xt):
    xt = xt.clone().detach().requires_grad_(True)
    u = model(xt)
    grads = gradients(u, xt)
    u_x = grads[:, 0:1]
    u_t = grads[:, 1:2]
    return u_t + u_x
```

## Изменить функцию награды

Добавить класс в `src/pinn_accel/rewards.py` и зарегистрировать его в `REWARD_REGISTRY`.
После этого указать имя в конфиге:

```json
{
  "controller_params": {
    "tiny_loss_weight": {
      "reward": "relative_improvement"
    }
  }
}
```

Доступные baseline reward для агента включают `baseline_gap`,
`normalized_baseline_gap` и `normalized_baseline_gap_delta`.
Также есть reward по ошибке решения: `relative_l2_improvement`,
`relative_l2_baseline_gap` и `relative_l2_baseline_gap_delta`.

## Изменить агента

Текущий агент — `TinyLossWeightAgent` в `src/pinn_accel/agents/tiny.py`. Его состояние:

- `log_losses`
- `dlog_losses`
- `log_lambdas`
- `log(loss / initial_loss)`
- `progress`

Политика — по умолчанию один линейный слой `LinearRLPolicy` с нормальным шумом действия и без bias (`policy_bias=false`). Если задать `policy_hidden_dim`, включится скрытый слой `Linear -> Tanh -> Linear`.

Агент должен наследоваться от `BaseWeightAgent` и реализовать:

- `select_action(state)`
- `update(state, action, reward, next_state, done)`

Затем его нужно добавить в `make_agent()` в `src/pinn_accel/agents/__init__.py`.
