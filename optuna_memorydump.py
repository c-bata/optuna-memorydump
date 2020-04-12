import logging
import time
from typing import Dict, Optional
import optuna
import threading


_logger = logging.getLogger("optuna.memorydump")
_lock = threading.Lock()


def _sync_study(from_study: optuna.Study, to_study: optuna.Study) -> None:
    if from_study.system_attrs != to_study.system_attrs:
        for k in from_study.system_attrs:
            if (
                k in to_study.system_attrs
                and from_study.system_attrs[k] == to_study.system_attrs[k]
            ):
                continue
            from_study.set_system_attr(k, from_study.system_attrs[k])
    if from_study.user_attrs != to_study.user_attrs:
        for k in from_study.user_attrs:
            if (
                k in to_study.user_attrs
                and from_study.user_attrs[k] == to_study.user_attrs[k]
            ):
                continue
            from_study.set_user_attr(k, from_study.user_attrs[k])


def _append_trial(
    storage: optuna.storages.BaseStorage,
    study_id: int,
    base_trial: optuna.structs.FrozenTrial,
) -> None:
    trial_id = storage.create_new_trial(study_id, template_trial=base_trial)

    number = storage.get_trial_number_from_id(trial_id)
    assert number == base_trial.number, "integrity error"


def _sync_trial(
    storage: optuna.storages.BaseStorage,
    from_trial: optuna.structs.FrozenTrial,
    to_trial: optuna.structs.FrozenTrial,
) -> None:
    if from_trial.system_attrs != to_trial.system_attrs:
        for k in from_trial.system_attrs:
            if (
                k in to_trial.system_attrs
                and from_trial.system_attrs[k] == to_trial.system_attrs[k]
            ):
                continue
            storage.set_trial_system_attr(
                to_trial._trial_id, k, from_trial.system_attrs[k]
            )

    if from_trial.user_attrs != to_trial.user_attrs:
        for k in from_trial.user_attrs:
            if (
                k in to_trial.user_attrs
                and from_trial.user_attrs[k] == to_trial.user_attrs[k]
            ):
                continue
            storage.set_trial_user_attr(to_trial._trial_id, k, from_trial.user_attrs[k])

    if from_trial.intermediate_values != to_trial.intermediate_values:
        for step in from_trial.intermediate_values:
            if step in to_trial.intermediate_values:
                continue
            assert storage.set_trial_intermediate_value(
                to_trial._trial_id, step, from_trial.intermediate_values[step]
            )

    if (
        from_trial.distributions != to_trial.distributions
        or from_trial.params != to_trial.params
    ):
        names = set(from_trial.distributions.keys())
        names.intersection_update(set(from_trial.distributions.keys()))
        for name in names:
            from_distribution = from_trial.distributions[name]
            from_external = from_trial.params[name]
            if (
                (name in to_trial.distributions)
                and (name in to_trial.params)
                and (from_distribution == to_trial.distributions[name])
                and (from_external == to_trial.params[name])
            ):
                continue
            from_internal = from_distribution.to_internal_repr(from_external)
            assert storage.set_trial_param(
                to_trial._trial_id, name, from_internal, from_distribution
            )

    if from_trial.value != to_trial.value:
        storage.set_trial_value(to_trial._trial_id, from_trial.value)

    if from_trial.state != to_trial.state:
        storage.set_trial_state(to_trial._trial_id, from_trial.state)


def _dump(
    study: optuna.Study, storage: optuna.storages.BaseStorage, study_id: int,
) -> None:
    dumped_trial_map: Dict[int, optuna.structs.FrozenTrial] = {
        t.number: t for t in storage.get_all_trials(study_id)
    }

    for trial in study.trials:
        dt = dumped_trial_map.get(trial.number, None)
        if dt is None:
            _append_trial(storage, study_id, trial)
            continue

        if dt.state.is_finished():
            continue

        _sync_trial(storage, trial, dt)


class Callback:
    def __init__(
        self,
        interval: int,
        storage: optuna.storages.BaseStorage,
        sync_study_attr_always: bool = False,
    ) -> None:
        self._interval = interval
        self._storage = storage
        self._sync_study_always = sync_study_attr_always

        self._first_call = True
        self._to_study: Optional[optuna.Study] = None

    def __call__(self, study: optuna.Study, trial: optuna.structs.FrozenTrial) -> None:
        if trial.number == 0 or trial.number % self._interval != 0:
            return

        if not _lock.acquire(blocking=False):
            _logger.info(
                f"memorydump is skipped at trial {trial.number}"
                f" (thread={threading.get_ident()})."
            )
            return

        _logger.info(
            f"memorydump is triggered at trial {trial.number}"
            f" (thread={threading.get_ident()})."
        )

        if self._first_call:
            to_study = optuna.create_study(
                storage=self._storage, study_name=study.study_name, load_if_exists=True
            )
            to_study._storage.set_study_direction(to_study.study_id, study.direction)
            self._to_study = to_study
            self._first_call = False

        assert self._to_study is not None

        if self._sync_study_always or self._first_call:
            _sync_study(from_study=study, to_study=self._to_study)

        start = time.time()
        _dump(study, self._storage, self._to_study.study_id)
        elapsed = time.time() - start

        if self._first_call:
            self._first_call = False

        _lock.release()
        _logger.info(
            f"memorydump of trial {trial.number} is finished"
            f" in {elapsed:.3f}s (thread={threading.get_ident()})."
        )
