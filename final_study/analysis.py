"""Eight-schedule specification using the unchanged, reviewed statistical math.

Load an isolated copy so adapting its specification validator cannot weaken the
historical 32-schedule validator used elsewhere in the repository.
"""
import importlib.util

import precollection_analysis as historical

MODELS, METHODS, CONDITIONS = historical.MODELS, historical.METHODS, historical.CONDITIONS
ROSTER, IDENTITY = historical.ROSTER, historical.IDENTITY


def load_spec():
    spec = historical.load_spec()
    spec.update(version='final-eight-schedule-analysis-v1',
                status='ADOPTED_EIGHT_EQUAL_SCHEDULES',
                schedule_seeds=list(range(44000, 44008)),
                precision_scenario_half_width=None)
    return spec


def validate_spec(spec):
    if spec != load_spec():
        raise ValueError('Eight-schedule analysis changed; version and review it.')


_module_spec = importlib.util.spec_from_file_location('_final_schedule_math', historical.__file__)
_math = importlib.util.module_from_spec(_module_spec)
_module_spec.loader.exec_module(_math)
_math.validate_spec = validate_spec


def design(spec):
    return _math.design(spec)


def primary_contrasts(rows, spec):
    return _math.primary_contrasts(rows, spec)
