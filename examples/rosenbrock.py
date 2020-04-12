import optuna

from optuna_memorydump import dump


def objective(trial):
    x1 = trial.suggest_uniform("x1", -100, 100)
    x2 = trial.suggest_uniform("x2", -100, 100)
    return 100 * (x2 - x1 ** 2) ** 2 + (x1 - 1) ** 2


if __name__ == "__main__":
    study = optuna.create_study(study_name="dumped", sampler=optuna.samplers.CmaEsSampler())
    study.optimize(objective, n_trials=10)
    print("Best value: {} (params: {})\n".format(study.best_value, study.best_params))

    dump_storage = optuna.storages.RDBStorage('sqlite:///db-dump.sqlite3')
    dump(study, dump_storage)

