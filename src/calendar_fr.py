"""Vacances scolaires, zone C (Sceaux, academie de Versailles).

Source : calendriers officiels de l'Education nationale (BO, archives du calendrier scolaire).
Chaque periode va du premier jour de vacances (samedi) au jour de reprise des cours EXCLU.
"""
import pandas as pd

# (premier jour de vacances, jour de reprise des cours)
ZONE_C = [
    ("2006-10-25", "2006-11-06"), ("2006-12-23", "2007-01-08"), ("2007-02-17", "2007-03-05"),
    ("2007-04-07", "2007-04-23"), ("2007-07-04", "2007-09-03"),
    ("2007-10-27", "2007-11-08"), ("2007-12-22", "2008-01-07"), ("2008-02-23", "2008-03-10"),
    ("2008-04-19", "2008-05-05"), ("2008-07-03", "2008-09-02"),
    ("2008-10-25", "2008-11-06"), ("2008-12-20", "2009-01-05"), ("2009-02-14", "2009-03-02"),
    ("2009-04-11", "2009-04-27"), ("2009-07-02", "2009-09-01"),
    ("2009-10-24", "2009-11-05"), ("2009-12-19", "2010-01-04"), ("2010-02-20", "2010-03-08"),
    ("2010-04-17", "2010-05-03"), ("2010-07-02", "2010-09-02"),
    ("2010-10-23", "2010-11-04"), ("2010-12-18", "2011-01-03"),
]


def school_holiday(index):
    """1.0 si le jour est en vacances scolaires (zone C), 0.0 sinon."""
    days = pd.DatetimeIndex(index).normalize()
    flag = pd.Series(0.0, index=days)
    for start, end in ZONE_C:
        flag[(days >= start) & (days < end)] = 1.0
    return flag.to_numpy()
