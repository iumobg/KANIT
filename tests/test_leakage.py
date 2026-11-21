#!/usr/bin/env python3
"""No lineage on both sides of any split: every outer fold of every
repeat of the lineage-aware arm, and every inner split made inside each outer training
set. The lineage-blind arm, which does not group, does mix lineages."""
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib import folds  # noqa: E402
from lib.lineage import no_group_leakage  # noqa: E402

train = pytest.importorskip("lib.train")
pytestmark = pytest.mark.unit


def _data(seed=0):
    rng = np.random.default_rng(seed)
    sizes = rng.geometric(0.08, size=150)                      # skewed lineage sizes
    groups = np.repeat(np.arange(sizes.size), sizes)
    rate = rng.uniform(0.05, 0.9, size=sizes.size)[groups]     # resistance runs in lineages
    y = (rng.random(groups.size) < rate).astype(int)
    return y, groups


def test_no_lineage_crosses_an_outer_or_inner_split():
    y, groups = _data()
    splits = folds.assign_all(y, groups, n_repeats=3, n_folds=5, min_minority=5,
                              max_attempts=50)
    aware = [s for s in splits if s.arm == folds.LINEAGE_AWARE]
    assert aware and all(s.evaluable for s in aware)
    n_inner = 0
    for s in aware:
        for k in range(5):
            tr, te = s.fold_of != k, s.fold_of == k
            assert no_group_leakage(tr, te, groups), (s.repeat, k)
            idx = np.flatnonzero(tr)
            itr, iva = train.inner_split(y[idx], groups[idx], folds.LINEAGE_AWARE, seed=k)
            m_tr = np.zeros(idx.size, bool)
            m_va = np.zeros(idx.size, bool)
            m_tr[itr], m_va[iva] = True, True
            assert no_group_leakage(m_tr, m_va, groups[idx]), (s.repeat, k)
            assert not set(groups[idx][iva]) & set(groups[te])      # validation never sees test
            n_inner += 1
    assert n_inner == 15
    blind = [s for s in splits if s.arm == folds.LINEAGE_BLIND and s.evaluable]
    assert any(not no_group_leakage(s.fold_of != k, s.fold_of == k, groups)
               for s in blind for k in range(5))                    # the test can see leakage
