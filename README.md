# optuna-memorydump

Dump Optuna in-memory storage to RDB storage. This tool ensures idempotence and thread-safe.

```python
import optuna
from optuna_memorydump import Callback


def objective(trial):
    x1 = trial.suggest_uniform("x1", -100, 100)
    x2 = trial.suggest_uniform("x2", -100, 100)
    return 100 * (x2 - x1 ** 2) ** 2 + (x1 - 1) ** 2


if __name__ == "__main__":
    study = optuna.create_study(
        study_name="dumped",
        sampler=optuna.samplers.CmaEsSampler())
    dump_storage = optuna.storages.RDBStorage('sqlite:///db-dump.sqlite3')
    study.optimize(
        objective, timeout=30, n_jobs=8, gc_after_trial=False,
        callbacks=[Callback(interval=100, storage=dump_storage)]
    )
    print("Best value: {} (params: {})\n".format(
        study.best_value, study.best_params))
```

```console
$ python examples/rosenbrock.py
[I 2020-04-12 18:33:39,200] Finished trial#0 with value: 227720050.8621735 with parameters: {'x1': -39.47837834000014, 'x2': 49.50818721667139}. Best is trial#0 with value: 227720050.8621735.
[I 2020-04-12 18:33:39,204] Finished trial#2 with value: 17419243.42417588 with parameters: {'x1': 20.127939426831727, 'x2': -12.225342646901677}. Best is trial#2 with value: 17419243.42417588.
[I 2020-04-12 18:33:39,205] Finished trial#1 with value: 593435738.9867325 with parameters: {'x1': -50.020580636993536, 'x2': 66.01017926994089}. Best is trial#2 with value: 17419243.42417588.
[I 2020-04-12 18:33:39,209] Finished trial#3 with value: 192279821.98413566 with parameters: {'x1': 37.17603190771658, 'x2': -4.587931683783628}. Best is trial#2 with value: 17419243.42417588.
[I 2020-04-12 18:33:39,211] Finished trial#6 with value: 33941733.71122515 with parameters: {'x1': -24.987962707794185, 'x2': 41.80873062855637}. Best is trial#2 with value: 17419243.42417588.
:
[I 2020-04-12 18:33:39,924] Finished trial#100 with value: 714804.0405193055 with parameters: {'x1': 8.90255814998557, 'x2': -5.286849533643432}. Best is trial#98 with value: 13900.353268681309.
[I 2020-04-12 18:33:39,931] memorydump is triggered at trial 100 (thread=123145615126528).
```
