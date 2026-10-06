from train_scripts.train_copy_paste import _mosaic_modes, _remote_prefix
from copy_paste_protocol import variant_overrides


def test_mosaic_interaction_is_available_for_all_datasets():
    expected = [(True, 1.0, 10)]
    for dataset in ("levir", "tinyperson", "visdrone"):
        assert _mosaic_modes(dataset, True, 1.0, 10) == expected


def test_default_visdrone_matrix_remains_matched_pair():
    assert _mosaic_modes("visdrone", False, 1.0, 10) == [(False, 0.0, 0), (True, 1.0, 10)]
    assert _mosaic_modes("levir", False, 1.0, 10) == [(False, 0.0, 0)]


def test_remote_prefix_is_mosaic_qualified_for_every_dataset(monkeypatch):
    monkeypatch.setenv("COPY_PASTE_MOSAIC", "1")
    assert _remote_prefix("levir", "stcp", 42).endswith("/stcp/mosaic/seed_42")
    assert _remote_prefix("tinyperson", "stcp", 42).endswith("/stcp/mosaic/seed_42")
    monkeypatch.setenv("COPY_PASTE_MOSAIC", "0")
    assert _remote_prefix("levir", "stcp", 42).endswith("/stcp/no_mosaic/seed_42")


def test_all_image_canvas_variants_preserve_r2_r4_geometry_and_scope():
    r2 = variant_overrides("all_canvas_r2")
    r4 = variant_overrides("all_canvas_r4")
    assert r2["copy_paste_mode"] == "all_canvas"
    assert r2["negative_cp_target_policy"] == "deficit"
    assert r2["negative_cp_donor_policy"] == "matched"
    assert r2["negative_cp_degradation"] == "none"
    assert r4["copy_paste_mode"] == "all_canvas"
    assert r4["negative_cp_target_policy"] == "deficit"
    assert r4["negative_cp_donor_policy"] == "larger"
    assert r4["negative_cp_degradation"] == "weak_blur"
