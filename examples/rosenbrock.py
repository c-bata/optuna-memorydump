import logging
import optuna
from optuna_memorydump import MemoryDumpCallback


def objective(trial):
    x1 = trial.suggest_uniform("x1", -100, 100)
    x2 = trial.suggest_uniform("x2", -100, 100)
    return 100 * (x2 - x1 ** 2) ** 2 + (x1 - 1) ** 2


if __name__ == "__main__":
    logging.getLogger("optuna.study").setLevel(logging.ERROR)
    study = optuna.create_study(
        study_name="dumped", sampler=optuna.samplers.CmaEsSampler(),
    )
    study.optimize(
        objective,
        n_trials=600 + 1,
        n_jobs=3,
        gc_after_trial=False,
        callbacks=[MemoryDumpCallback("sqlite:///db.sqlite3", interval=100)],
    )
    print("Best value: {} (params: {})\n".format(study.best_value, study.best_params))
