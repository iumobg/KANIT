"""The two sets of graded patterns of a model.

``candidates``: nominated by the final model and by stability selection;
their grades are the primary results. ``association``: the patterns
that pass the pyseer layer but are not candidates, at most ``max_patterns`` with the
smallest p values. Steps 09, 10, 12 and 14b compute every evidence layer and the grades of
either set (``--set``), the association set in its own directory with its own
multiple-testing families, as the candidates have theirs. The association set enters no
hypothesis test and no primary result; the knowledge base stores it with the route
``association`` (the candidates with ``model``).
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path

import pandas as pd

CANDIDATES, ASSOCIATION = "candidates", "association"
SETS = (CANDIDATES, ASSOCIATION)
ROUTE = {CANDIDATES: "model", ASSOCIATION: "association"}
LABEL = {CANDIDATES: "candidate", ASSOCIATION: "association"}     # "<label> patterns"
ASSOCIATION_COLUMNS = ("pattern_id", "rank", "lrt_pvalue")


class _Paths(Mapping[str, Path]):
    """The patterns file and the directories of one set, each resolved when asked for."""

    IN_ASSOCIATION_DIR = {"patterns": "association.csv", "card_layer_dir": "card_layer",
                          "layers_dir": "layers", "grades_dir": "grades"}

    def __init__(self, name: str, path: Callable[[str], Path]):
        if name not in SETS:
            raise ValueError(f"unknown pattern set {name!r}; one of {SETS}")
        self.name, self.path = name, path

    def __getitem__(self, key: str) -> Path:
        if key not in self.IN_ASSOCIATION_DIR:
            raise KeyError(key)
        if self.name == CANDIDATES:
            return self.path("candidates_file" if key == "patterns" else key)
        return self.path("association_dir") / self.IN_ASSOCIATION_DIR[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.IN_ASSOCIATION_DIR)

    def __len__(self) -> int:
        return len(self.IN_ASSOCIATION_DIR)


def paths(name: str, path: Callable[[str], Path]) -> Mapping[str, Path]:
    """``patterns`` (the set's file), ``card_layer_dir``, ``layers_dir`` and ``grades_dir`` of
    one set; ``path(key)`` resolves a paths_organism key of the model. The candidates use
    the keys themselves (candidates_file for the patterns), the association set the
    directories under association_dir."""
    return _Paths(name, path)


def read_patterns(path: Path) -> list[int]:
    """The pattern ids of a set's file, sorted."""
    return sorted(set(pd.read_csv(path)["pattern_id"].astype(int)))


def association(tested: pd.DataFrame, candidates: Iterable[int], max_patterns: int) -> pd.DataFrame:
    """The association set from pyseer's tested patterns (``pattern_id``, ``lrt-pvalue``,
    ``significant``): the significant ones that are not candidates, the ``max_patterns``
    with the smallest p values (ties by pattern id), ranked from 1."""
    cand = {int(c) for c in candidates}
    pid = tested["pattern_id"].astype(int)
    significant = tested["significant"].astype(str).str.lower().isin({"true", "1", "1.0"})
    d = tested.loc[significant & ~pid.isin(cand), ["pattern_id", "lrt-pvalue"]].copy()
    d["pattern_id"] = d["pattern_id"].astype(int)
    d = d.sort_values(["lrt-pvalue", "pattern_id"], kind="mergesort").head(max_patterns)
    return pd.DataFrame({"pattern_id": d["pattern_id"].to_numpy(),
                         "rank": range(1, len(d) + 1),
                         "lrt_pvalue": d["lrt-pvalue"].to_numpy()},
                        columns=list(ASSOCIATION_COLUMNS))
