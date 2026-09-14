"""In-memory checks for the sample-size sensitivity manifests."""

from research.run_wcf_sample_size_sensitivity import (
    DEFAULT_N_VALUES,
    DEFAULT_SEEDS,
    ORDINARY_METHODS,
    PAPER_DGPS_ALL,
    ZERO_METHODS,
    build_manifest,
)


def test_all_roster_has_expected_size_and_family_methods():
    manifest = build_manifest()
    assert manifest["n_cells"] == 5 * len(DEFAULT_N_VALUES) * len(DEFAULT_SEEDS) * 3

    all_manifest = build_manifest(roster="all")
    assert all_manifest["n_cells"] == 14 * len(DEFAULT_N_VALUES) * len(DEFAULT_SEEDS) * 3
    assert all_manifest["paper_dgp_map"] == PAPER_DGPS_ALL

    for label, source in PAPER_DGPS_ALL.items():
        methods = {
            cell["method"]
            for cell in all_manifest["cells"]
            if cell["dgp"] == source
        }
        expected = set(ZERO_METHODS if label.startswith("Z") else ORDINARY_METHODS)
        assert methods == expected
        if label.startswith("Z"):
            assert "cwdb_dr" not in methods


def test_all_roster_registry_contains_two_part_adapter():
    manifest = build_manifest(roster="all")
    assert manifest["method_registry"]["cwdb_zipt"]["adapter"] == "zipt"
    assert manifest["method_roster_by_family"]["Z"] == list(ZERO_METHODS)
