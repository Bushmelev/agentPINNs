# pinn_accel

Минимальная версия проекта для быстрых экспериментов с PINN, весами loss-компонент и RL-агентами.

Что изменено относительно старой структуры:

- `train.py` только читает конфиг/CLI и запускает эксперимент.
- Уравнения лежат отдельно в `src/pinn_accel/equations`.
- Награды агента лежат отдельно в `src/pinn_accel/rewards.py`.
- Агенты лежат отдельно в `src/pinn_accel/agents`.
- Сохранение истории, графиков и checkpoint-ов унифицировано через `ArtifactStore`.
- В тренировочном цикле нет обязательной повторной оценки loss после каждого `optimizer.step()`, поэтому шаг обучения дешевле.
- Для ускорения доступны фиксированные sampling pools и `torch.compile` через конфиг.

## Быстрый запуск

Из директории `pinn_accel`:

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

```bash
python train.py --config configs/burgers_actor_critic.json
```

Быстрый smoke-run:

```bash
python train.py --equation heat --controllers fixed --steps 20 --no-plots
```

Артефакты сохраняются в `artifacts/<timestamp>/<equation>/<controller>/`:

- `history.json`
- `checkpoint.pt`
- `plots/*.png`

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
    "actor_critic": {
      "reward": "relative_improvement"
    }
  }
}
```

## Изменить агента

Агент должен наследоваться от `BaseWeightAgent` в `src/pinn_accel/agents/base.py` и реализовать:

- `select_action(state)`
- `update(state, action, reward, next_state, done)`

Затем его нужно добавить в `make_agent()` в `src/pinn_accel/agents/__init__.py`.
