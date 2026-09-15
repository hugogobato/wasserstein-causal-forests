"""Contract checks for the symmetric confirmatory rerun."""

import json

import numpy as np

from research.checks.wcf_confirmatory_make_colab_notebooks import (
    DEFAULT_SHARDS,
    FOREST_SETUP_PINNED,
    PAPER_COMMIT,
    PAPER_REPOSITORY,
    PAPER_SETUP_PATH,
    PAPER_SETUP_SHA256,
    PAPER_STUDY_PATH,
    PAPER_STUDY_SHA256,
    allocate,
    causal_drf_paper_pin,
    download,
    group_cells,
)
from research.run_wcf_confirmatory import (
    ORDINARY_METHODS,
    PAPER_DGPS_ALL,
    SEEDS,
    ZERO_METHODS,
    build_manifest,
)
from wasserstein_causal_forests.cwdb.dr_calibration import FunctionalAIPW
from wasserstein_causal_forests.g3.confirmatory_evaluation import (
    moderator_bins_for_dgp,
    moderator_column,
)


def test_manifest_is_complete_distinct_and_fully_checksummed():
    manifest = build_manifest()
    assert manifest["n_cells"] == 14 * 50 * 3 == 2100
    assert manifest["replication_seeds"] == list(SEEDS)
    assert len({cell["cell_key"] for cell in manifest["cells"]}) == 2100
    assert manifest["grid_id"] == "wcf_confirmatory_r50_v1"
    assert manifest["nuisance_folds"] == 3
    for label, source in PAPER_DGPS_ALL.items():
        methods = {
            cell["method"] for cell in manifest["cells"] if cell["dgp"] == source
        }
        expected = ZERO_METHODS if label.startswith("Z") else ORDINARY_METHODS
        assert methods == set(expected)


def test_active_income_moderators_use_paper_coordinates():
    assert moderator_column("IC1") == 3
    assert moderator_column("IC2") == 2
    assert moderator_column("IC3") == 3
    assert moderator_column("IC0") == 0
    X = np.zeros((4, 6))
    X[:, 3] = [-0.75, -0.25, 0.25, 0.75]
    assert np.array_equal(moderator_bins_for_dgp("IC1", X), np.arange(4))


def test_functional_aipw_retains_unit_scores():
    n = 40
    X = np.zeros((n, 2))
    treatment = np.tile([0, 1], n // 2)
    observed = {"h": np.linspace(-1, 1, n)}
    oof = {"h": {0: np.zeros(n), 1: np.ones(n)}}
    model = FunctionalAIPW(n_bins=4).fit(
        observed=observed,
        oof_arm_means=oof,
        X=X,
        treatment=treatment,
        random_state=7,
        oracle_propensity=np.full(n, 0.5),
    )
    assert model.scores_["h"].shape == (n,)
    assert np.isclose(model.marginal_["h"], model.scores_["h"].mean())


def test_default_shards_preserve_complete_paired_replications():
    manifest = build_manifest()
    shards, loads = allocate(manifest["cells"], DEFAULT_SHARDS)
    assert DEFAULT_SHARDS == 26
    assert len(shards) == DEFAULT_SHARDS
    assert sum(map(len, shards)) == 2100
    assert sum(len(group_cells(shard)) for shard in shards) == 700
    assert max(loads) / 3600 < 8.0
    assert min(loads) / 3600 > 5.0
    for shard in shards:
        for group in group_cells(shard):
            expected = ZERO_METHODS if group[0]["dgp"].startswith("ZI") else ORDINARY_METHODS
            assert {cell["method"] for cell in group} == set(expected)


def test_causal_drf_paper_pin_is_content_addressed():
    text = causal_drf_paper_pin()
    compile(text, "<causal_drf_paper_pin>", "exec")
    for fragment in (
        PAPER_REPOSITORY,
        PAPER_COMMIT,
        PAPER_SETUP_PATH,
        PAPER_SETUP_SHA256,
        PAPER_STUDY_PATH,
        PAPER_STUDY_SHA256,
        "CAUSAL_CLEAN_DRF_COMMIT",
    ):
        assert fragment in text
    assert "0a1a508444176b5b1553f13e832be93a374b0af2" in text


def test_notebook_download_and_r_setup_are_pinned():
    source = download(3, "0123456789abcdef")
    assert "wcf_confirmatory_shard_03_0123456789ab.zip" in source
    assert "files.download(output_file)" in source
    assert "(Not on Colab / download skipped):" in source
    assert 'install_version("drf", version="1.3.1"' in FOREST_SETUP_PINNED
    assert "0a1a508444176b5b1553f13e832be93a374b0af2" in FOREST_SETUP_PINNED
