# KVCA Sweep Notes

## Current Best Parameter Sweep Result

Parameter sweep on the `A_sc` placement currently shows:

- Best run: `yolov8_kvca_a_sc_group_weight_p2sr4_p3sr4_n`
- Base placement: `yolov8_kvca_placement_a_sc.yaml`
- KVCA placement: P2 and P3 gates before `Detect`
- Compression mode: `group_weight`
- Spatial reduction: P2 `sr=4`, P3 `sr=4`

This suggests learned weighted compression inside each `sr x sr` group preserves the small-object Varroa signal better than fixed `avg`, `avg_dwk`, or `dw_stride` compression for this placement.

## Follow-Up

Use `group_weight` as the default KVCA compression mode for the next round.

Recommended next checks:

- Sweep `group_weight` around reduction ratios:
  - P2 `sr=2`, P3 `sr=4`
  - P2 `sr=4`, P3 `sr=4`
  - P2 `sr=8`, P3 `sr=4`
  - P2 `sr=2`, P3 `sr=2` if GPU memory is acceptable
- Re-test stronger placements with `group_weight`, especially `B_sc` and `C_sc`, because the best compression mode may change the placement ranking.
