import pandas as pd

from src.calendar_fr import school_holiday


def flag(day):
    return school_holiday([pd.Timestamp(day)])[0]


def test_vacances_et_hors_vacances():
    assert flag("2008-12-25") == 1.0   # Noel 2008
    assert flag("2009-08-15") == 1.0   # ete
    assert flag("2009-10-15") == 0.0   # periode scolaire


def test_bornes():
    assert flag("2009-02-14") == 1.0   # premier jour (samedi)
    assert flag("2009-03-02") == 0.0   # jour de reprise : exclu
    assert flag("2009-03-01") == 1.0   # veille de la reprise


def test_longueur_et_type():
    out = school_holiday(pd.date_range("2008-01-01", periods=10))
    assert len(out) == 10 and set(out) <= {0.0, 1.0}
