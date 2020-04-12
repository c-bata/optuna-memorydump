# optuna-memorydump

Dump Optuna in-memory storage to RDB storage. This tool ensures idempotence and thread-safe.

## Installation

```console
$ pip install git+https://github.com/c-bata/optuna-memorydump.git
```

## Usage

```python
import optuna
from optuna_memorydump import MemoryDumpCallback


def objective(trial):
    x1 = trial.suggest_uniform("x1", -100, 100)
    x2 = trial.suggest_uniform("x2", -100, 100)
    return 100 * (x2 - x1 ** 2) ** 2 + (x1 - 1) ** 2


if __name__ == "__main__":
    study = optuna.create_study(
        study_name="dumped", sampler=optuna.samplers.CmaEsSampler(),
    )
    study.optimize(
        objective, timeout=30, n_jobs=4, gc_after_trial=False,
        callbacks=[MemoryDumpCallback("sqlite:///db.sqlite3", interval=100)],
    )
    print("Best value: {} (params: {})\n".format(study.best_value, study.best_params))
```

```console
$ python examples/rosenbrock.py
[I 2020-04-12 18:33:39,200] Finished trial#0 with value: ...
[I 2020-04-12 18:33:39,204] Finished trial#2 with value: ...
[I 2020-04-12 18:33:39,205] Finished trial#1 with value: ...
:
[I 2020-04-12 18:33:39,924] Finished trial#100 with value: ...
[I 2020-04-12 18:33:39,931] memorydump is triggered at trial 100 (thread=123145615126528).
[I 2020-04-12 23:40:20,013] memorydump of trial 100 is finished in 1.938s (thread=123145458638848).
:
[I 2020-04-12 18:33:39,924] Finished trial#200 with value: ...
[I 2020-04-12 23:40:21,505] memorydump of trial 200 is finished in 0.105s (thread=123145458638848).
[I 2020-04-12 23:40:24,061] memorydump is triggered at trial 300 (thread=123145425059840).
```
