# Superseded first execution of the G3 manifest

Complete and internally consistent: 4110/4110 cells, 97650 rows, zero failures,
merge audit `PASS`, and the independent gate recomputation agreed exactly.

It is superseded because of a defect in the adapter layer, not in the data. In
`methods.py::_output_from_laws` the law-producing methods were handed only the
*training* functionals, so C-WDB, W-DRF-T, and Causal-DRF reported
`not_applicable` for `grid_skewness` and `grid_upper_tail_mean`. Those are
exactly the functionals excluded from every training manifest to test the D7
transfer claim, so that claim measured nothing: rule 3's win was carried by
`grid_mean` and `grid_sd`, which every method was trained on.

The remaining rules were unaffected, and for the record this run returned
`NOT-GO`, failing rule 1 (D2 false-effect ratio 2.69 against a 1.25 cap) and
rule 4 (PTA-S beat C-WDB on the rule 3 target).

Kept so the defect and the correction are both auditable. Do not merge these
rows with the current run: they were produced by a different code revision.
