#!/usr/bin/env python3
"""The patterns nominated by association (lib/pattern_sets.py): the
choice of the set, its paths and the layers' tables when it is empty. The steps run on it
in tests/test_knowledge_base.py."""
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib import card_layer, cpss, mda, pattern_sets  # noqa: E402

pytestmark = pytest.mark.unit


def _tested():
    return pd.DataFrame({"pattern_id": [1, 2, 3, 4, 5, 6, 7],
                         "lrt-pvalue": [1e-9, 1e-8, 1e-8, 1e-3, 2e-9, 5e-9, 0.5],
                         "significant": [True, True, True, False, True, True, False]})


def test_the_set_is_the_significant_patterns_that_are_not_candidates():
    s = pattern_sets.association(_tested(), candidates=[1, 6], max_patterns=10)
    assert s["pattern_id"].tolist() == [5, 2, 3]           # by p, ties by pattern id
    assert s["rank"].tolist() == [1, 2, 3] and list(s.columns) == list(pattern_sets.ASSOCIATION_COLUMNS)
    assert pattern_sets.association(_tested(), [1, 6], max_patterns=2)["pattern_id"].tolist() == [5, 2]


def test_significance_read_back_from_a_csv_counts():
    t = _tested().assign(significant=lambda d: d["significant"].map({True: "True", False: "False"}))
    assert pattern_sets.association(t, [], 3)["pattern_id"].tolist() == [1, 5, 6]


def test_an_empty_set_keeps_its_columns():
    s = pattern_sets.association(_tested(), candidates=range(10), max_patterns=50)
    assert s.empty and list(s.columns) == list(pattern_sets.ASSOCIATION_COLUMNS)
    assert pattern_sets.association(_tested(), [], 0).empty


def test_the_set_has_its_own_directory():
    def path(key):
        return Path("/r") / key

    c = pattern_sets.paths(pattern_sets.CANDIDATES, path)
    a = pattern_sets.paths(pattern_sets.ASSOCIATION, path)
    assert dict(c) == {"patterns": Path("/r/candidates_file"),
                       "card_layer_dir": Path("/r/card_layer_dir"),
                       "layers_dir": Path("/r/layers_dir"), "grades_dir": Path("/r/grades_dir")}
    assert dict(a) == {"patterns": Path("/r/association_dir/association.csv"),
                       "card_layer_dir": Path("/r/association_dir/card_layer"),
                       "layers_dir": Path("/r/association_dir/layers"),
                       "grades_dir": Path("/r/association_dir/grades")}
    with pytest.raises(ValueError, match="unknown pattern set"):
        pattern_sets.paths("other", path)


def test_a_step_resolves_only_the_paths_it_uses():
    def path(key):
        if key != "layers_dir":
            raise KeyError(key)
        return Path("/r/layers")

    assert pattern_sets.paths(pattern_sets.CANDIDATES, path)["layers_dir"] == Path("/r/layers")


def test_cpss_layer_outside_the_prefilter():
    table = pd.DataFrame({"pattern_id": [1, 2], "chi2": [9.0, 4.0], "pi": [0.7, 0.2]})
    layer = cpss.layer([2, 1, 3], table, 0.6)
    assert layer["pattern_id"].tolist() == [2, 1, 3]
    assert layer["passes"].tolist() == [False, True, False]
    assert layer["in_prefilter"].tolist() == [True, True, False] and layer["pi"].iloc[2] == 0.0
    assert cpss.layer([], table, 0.6).empty


def test_empty_layers_have_the_columns_of_full_ones():
    main, clusters = mda.empty_layer()
    assert list(main.columns) == list(mda.MAIN_COLUMNS) and main.empty
    assert list(clusters.columns) == list(mda.CLUSTER_COLUMNS) and clusters.empty
    assert card_layer.UNITIG_COLUMNS[:5] == ("pattern_id", "unitig_index", "sequence", "length",
                                             "located_genomes")
    assert all(f"{m}_n_b" in card_layer.UNITIG_COLUMNS for m in card_layer.MODES)
    assert card_layer.PATTERN_COLUMNS[:2] == ("pattern_id", "n_members")
