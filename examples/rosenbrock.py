import optuna
from optuna_memorydump import dump

study = optuna.create_study(
    study_name="dumped",
    sampler=optuna.samplers.CmaEsSampler())
dump_storage = optuna.storages.RDBStorage('sqlite:///db-dump.sqlite3')


def objective(trial):
    x1 = trial.suggest_uniform("x1", -100, 100)
    x2 = trial.suggest_uniform("x2", -100, 100)

    if (trial.number + 1) % 100 == 0:
        print("check point:", trial.number)
        dump(study, dump_storage)

    return 100 * (x2 - x1 ** 2) ** 2 + (x1 - 1) ** 2


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    study = optuna.create_study(
        study_name="dumped",
        sampler=optuna.samplers.CmaEsSampler())
    study.optimize(
        objective, timeout=30,
        n_jobs=8, gc_after_trial=False)
    print("Best value: {} (params: {})\n".format(
        study.best_value, study.best_params))
