"""烧成试片研究：研究版本、试片登记、定稿/复制、Scheffé 拟合、配比搜索与结果冻结。

线性试验布局（比例步长 0.25）：
  A(1,0) B(0.75,0.25) C(0.5,0.5) D(0.25,0.75) E(0,1)
试片测量按已知函数生成（精确值，便于校验系数还原）：
  L*  = 50·x1 + 60·x2 + 10·x1·x2   （真二次曲面）
  光泽 = 80·x1 + 70·x2              （真线性）
  厚度 = 1.0·x1 + 1.2·x2
  a*  = 5·x1 + 2·x2，b* = -3·x1 + 4·x2
缺陷：针孔仅 E 格发生（等级 1）；缩釉仅 D 格第 2 片（等级 2）；流釉全无。
"""
from __future__ import annotations

import pytest


def _id_map(client):
    return {m["name"]: m["id"] for m in client.get("/materials").json()}


def _freeze(client, items):
    resp = client.post("/versions", json={"items": items})
    assert resp.status_code == 201, resp.text
    return resp.json()["version"]["id"]


def _sources(client):
    ids = _id_map(client)
    a = _freeze(client, [
        {"material_id": ids["钾长石"], "amount": 60.0},
        {"material_id": ids["方解石"], "amount": 20.0},
        {"material_id": ids["石英"], "amount": 20.0},
    ])
    b = _freeze(client, [
        {"material_id": ids["钾长石"], "amount": 40.0},
        {"material_id": ids["方解石"], "amount": 10.0},
        {"material_id": ids["白云石"], "amount": 10.0},
        {"material_id": ids["石英"], "amount": 40.0},
    ])
    c = _freeze(client, [
        {"material_id": ids["钠长石"], "amount": 50.0},
        {"material_id": ids["高岭土"], "amount": 20.0},
        {"material_id": ids["石英"], "amount": 30.0},
    ])
    return ids, a, b, c


CELLS = [
    ("A", 1.0, 0.0),
    ("B", 0.75, 0.25),
    ("C", 0.5, 0.5),
    ("D", 0.25, 0.75),
    ("E", 0.0, 1.0),
]


def _make_experiment(client, cells=None, ratio_step=0.25, **overrides):
    ids, a, b, c = _sources(client)
    cells = cells if cells is not None else CELLS
    payload = {
        "sources": [a, b],
        "mode": "linear",
        "cells": [
            {"position": pos, "weights": [x1, x2]} for pos, x1, x2 in cells
        ],
        "ratio_step": ratio_step,
        "dry_mass_per_tile_g": 100.0,
        "scale_division_g": 0.5,
        "minimum_weighed_g": 1.0,
    }
    payload.update(overrides)
    resp = client.post("/blend-experiments", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()["experiment"]["id"], (a, b, c)


def _make_ternary_experiment(client, cells, ratio_step=0.25):
    ids, a, b, c = _sources(client)
    payload = {
        "sources": [a, b, c],
        "mode": "ternary",
        "cells": [
            {"position": pos, "weights": list(w)} for pos, w in cells
        ],
        "ratio_step": ratio_step,
        "dry_mass_per_tile_g": 100.0,
        "scale_division_g": 0.5,
        "minimum_weighed_g": 1.0,
    }
    resp = client.post("/blend-experiments", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()["experiment"]["id"]


def _make_study(client, experiment_id, **overrides):
    payload = {
        "experiment_id": experiment_id,
        "kiln_run": "2026-09-10-#2",
        "body": "白瓷坯",
        "firing_curve": [
            {"time_min": 0.0, "temp_c": 25.0},
            {"time_min": 120.0, "temp_c": 600.0},
            {"time_min": 360.0, "temp_c": 1280.0},
        ],
        "atmosphere": "氧化",
        "kiln_position": "第3层中部",
        "fired_on": "2026-09-10",
    }
    payload.update(overrides)
    resp = client.post("/firing-studies", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()


def _tile(position, rep, x1, x2, **overrides):
    tile = {
        "position": position,
        "replicate_no": rep,
        "l_star": round(50 * x1 + 60 * x2 + 10 * x1 * x2, 6),
        "a_star": round(5 * x1 + 2 * x2, 6),
        "b_star": round(-3 * x1 + 4 * x2, 6),
        "gloss60": round(80 * x1 + 70 * x2, 6),
        "thickness_mm": round(1.0 * x1 + 1.2 * x2, 6),
        "pinhole": 1 if x2 == 1.0 else 0,
        "crawling": 0,
        "running": 0,
    }
    tile.update(overrides)
    return tile


def _seed_tiles(client, study_id, cells=CELLS, reps=2):
    for pos, x1, x2 in cells:
        for rep in range(1, reps + 1):
            overrides = {}
            if pos == "D" and rep == 2:
                overrides["crawling"] = 2
            resp = client.post(
                f"/firing-studies/{study_id}/tiles",
                json=_tile(pos, rep, x1, x2, **overrides),
            )
            assert resp.status_code == 201, resp.text


def _ready_study(client, cells=CELLS, reps=2, finalize=True, **exp_overrides):
    experiment_id, _ = _make_experiment(client, cells=cells, **exp_overrides)
    study = _make_study(client, experiment_id)
    _seed_tiles(client, study["id"], cells=cells, reps=reps)
    if finalize:
        resp = client.post(f"/firing-studies/{study['id']}/finalize", json={})
        assert resp.status_code == 201, resp.text
    return study["id"], experiment_id


# ---------------------------------------------------------------------------
# 研究创建与校验
# ---------------------------------------------------------------------------

def test_create_study_happy(client):
    experiment_id, sources = _make_experiment(client)
    study = _make_study(client, experiment_id, note="第一窑")
    assert study["status"] == "draft"
    assert study["version_no"] == 1
    assert study["experiment_id"] == experiment_id
    assert study["sources"] == list(sources[:2])
    assert study["kiln_run"] == "2026-09-10-#2"
    assert study["body"] == "白瓷坯"
    assert study["kiln_position"] == "第3层中部"
    assert len(study["firing_curve"]) == 3
    assert study["n_tiles"] == 0
    assert study["unmeasured_cells"] == [pos for pos, _, _ in CELLS]

    again = client.get(f"/firing-studies/{study['id']}")
    assert again.status_code == 200
    listed = client.get("/firing-studies").json()
    assert any(s["id"] == study["id"] for s in listed)


def test_create_study_experiment_not_found(client):
    resp = client.post("/firing-studies", json={
        "experiment_id": "no-such-experiment",
        "kiln_run": "r1",
        "body": "白瓷坯",
        "firing_curve": [{"time_min": 0.0, "temp_c": 25.0}],
        "kiln_position": "第1层",
    })
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "blend_experiment_not_found"


def test_create_study_invalid_curve(client):
    experiment_id, _ = _make_experiment(client)
    resp = client.post("/firing-studies", json={
        "experiment_id": experiment_id,
        "kiln_run": "r1",
        "body": "白瓷坯",
        "firing_curve": [
            {"time_min": 10.0, "temp_c": 100.0},
            {"time_min": 10.0, "temp_c": 200.0},
        ],
        "kiln_position": "第1层",
    })
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_firing_curve"


def test_create_study_blank_fields(client):
    experiment_id, _ = _make_experiment(client)
    resp = client.post("/firing-studies", json={
        "experiment_id": experiment_id,
        "kiln_run": "   ",
        "body": "白瓷坯",
        "firing_curve": [{"time_min": 0.0, "temp_c": 25.0}],
        "kiln_position": "第1层",
    })
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "blank_field"


# ---------------------------------------------------------------------------
# 试片登记（草稿期）
# ---------------------------------------------------------------------------

def test_add_tiles_and_study_view(client):
    study_id, _ = _ready_study(client, finalize=False)
    view = client.get(f"/firing-studies/{study_id}").json()
    assert view["n_tiles"] == 10
    assert view["unmeasured_cells"] == []
    cells = {c["position"]: c for c in view["cells"]}
    assert cells["A"]["n_tiles"] == 2
    assert cells["A"]["l_star_mean"] == pytest.approx(50.0)
    assert cells["E"]["l_star_mean"] == pytest.approx(60.0)
    assert cells["E"]["defects"]["pinhole"]["rate"] == pytest.approx(1.0)
    assert cells["D"]["defects"]["crawling"]["n_occurred"] == 1
    assert cells["D"]["defects"]["crawling"]["rate"] == pytest.approx(0.5)


def test_add_tile_unknown_position(client):
    study_id, _ = _ready_study(client, finalize=False)
    resp = client.post(
        f"/firing-studies/{study_id}/tiles",
        json=_tile("Z9", 1, 0.5, 0.5),
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "unknown_cell_position"


def test_add_tile_duplicate_replicate(client):
    study_id, _ = _ready_study(client, finalize=False)
    # 完全相同的测量值：幂等返回旧记录
    same = client.post(
        f"/firing-studies/{study_id}/tiles", json=_tile("A", 1, 1.0, 0.0)
    )
    assert same.status_code == 201
    tiles = client.get(f"/firing-studies/{study_id}").json()["tiles"]
    assert len([t for t in tiles if t["position"] == "A" and t["replicate_no"] == 1]) == 1
    # 同编号不同测量值：冲突
    conflict = client.post(
        f"/firing-studies/{study_id}/tiles",
        json=_tile("A", 1, 1.0, 0.0, l_star=51.0),
    )
    assert conflict.status_code == 422
    assert conflict.json()["error"]["code"] == "duplicate_replicate"


def test_add_tile_invalid_unit(client):
    study_id, _ = _ready_study(client, finalize=False)
    resp = client.post(
        f"/firing-studies/{study_id}/tiles",
        json=_tile("A", 3, 1.0, 0.0, thickness_unit="cm"),
    )
    assert resp.status_code == 422


def test_add_tile_incomplete_measurement(client):
    study_id, _ = _ready_study(client, finalize=False)
    payload = _tile("A", 3, 1.0, 0.0)
    del payload["gloss60"]
    resp = client.post(f"/firing-studies/{study_id}/tiles", json=payload)
    assert resp.status_code == 422


def test_add_tile_grade_out_of_range(client):
    study_id, _ = _ready_study(client, finalize=False)
    resp = client.post(
        f"/firing-studies/{study_id}/tiles",
        json=_tile("A", 3, 1.0, 0.0, pinhole=5),
    )
    assert resp.status_code == 422


def test_delete_tile(client):
    study_id, _ = _ready_study(client, finalize=False)
    view = client.get(f"/firing-studies/{study_id}").json()
    tile_id = view["tiles"][0]["id"]
    resp = client.delete(f"/firing-studies/{study_id}/tiles/{tile_id}")
    assert resp.status_code == 200
    assert resp.json()["n_tiles"] == 9
    missing = client.delete(f"/firing-studies/{study_id}/tiles/99999")
    assert missing.status_code == 404


# ---------------------------------------------------------------------------
# 定稿、只读与复制新版
# ---------------------------------------------------------------------------

def test_finalize_idempotent_and_readonly(client):
    study_id, _ = _ready_study(client, finalize=False)
    first = client.post(f"/firing-studies/{study_id}/finalize", json={})
    assert first.status_code == 201
    assert first.json()["created"] is True
    freeze = first.json()["freeze"]
    assert freeze["final_state"]["n_tiles"] == 10
    assert freeze["snapshot"]["study"]["kiln_run"] == "2026-09-10-#2"
    assert len(freeze["snapshot"]["tiles"]) == 10

    second = client.post(f"/firing-studies/{study_id}/finalize", json={})
    assert second.json()["created"] is False
    assert second.json()["freeze"]["id"] == freeze["id"]

    got = client.get(f"/firing-freezes/{freeze['id']}")
    assert got.status_code == 200
    assert got.json()["study_id"] == study_id

    # 定稿后只读：追加/删除试片均被拒绝
    add = client.post(
        f"/firing-studies/{study_id}/tiles", json=_tile("A", 3, 1.0, 0.0)
    )
    assert add.status_code == 422
    assert add.json()["error"]["code"] == "invalid_firing_status"
    tile_id = freeze["snapshot"]["tiles"][0]["id"]
    delete = client.delete(f"/firing-studies/{study_id}/tiles/{tile_id}")
    assert delete.status_code == 422
    assert delete.json()["error"]["code"] == "invalid_firing_status"


def test_finalize_empty_study(client):
    experiment_id, _ = _make_experiment(client)
    study = _make_study(client, experiment_id)
    resp = client.post(f"/firing-studies/{study['id']}/finalize", json={})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "empty_firing_study"


def test_copy_for_supplementary_measurement(client):
    study_id, _ = _ready_study(client, finalize=True)
    copied = client.post(f"/firing-studies/{study_id}/copy", json={})
    assert copied.status_code == 201
    new_study = copied.json()
    assert new_study["status"] == "draft"
    assert new_study["version_no"] == 2
    assert new_study["parent_id"] == study_id
    assert new_study["root_id"] == study_id
    assert new_study["kiln_run"] == "2026-09-10-#2"
    assert new_study["n_tiles"] == 10

    # 补测：新版本可追加试片并定稿
    add = client.post(
        f"/firing-studies/{new_study['id']}/tiles",
        json=_tile("C", 3, 0.5, 0.5),
    )
    assert add.status_code == 201
    fin = client.post(f"/firing-studies/{new_study['id']}/finalize", json={})
    assert fin.status_code == 201
    assert fin.json()["freeze"]["final_state"]["n_tiles"] == 11

    # 再复制：版本号继续递增
    copied2 = client.post(f"/firing-studies/{new_study['id']}/copy", json={})
    assert copied2.json()["version_no"] == 3
    assert copied2.json()["root_id"] == study_id


def test_copy_requires_finalized(client):
    study_id, _ = _ready_study(client, finalize=False)
    resp = client.post(f"/firing-studies/{study_id}/copy", json={})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_firing_status"


# ---------------------------------------------------------------------------
# Scheffé 响应面拟合
# ---------------------------------------------------------------------------

def test_fit_quadratic_recovers_coefficients(client):
    study_id, _ = _ready_study(client, finalize=False)
    resp = client.post(f"/firing-studies/{study_id}/fit", json={"model_order": 2})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["terms"] == ["x1", "x2", "x1x2"]
    assert body["n_cells"] == 5
    assert body["n_tiles"] == 10
    assert body["df"] == 7

    coefs = {
        c["term"]: c["value"]
        for c in body["metrics"]["l_star"]["coefficients"]
    }
    assert coefs["x1"] == pytest.approx(50.0, abs=1e-4)
    assert coefs["x2"] == pytest.approx(60.0, abs=1e-4)
    assert coefs["x1x2"] == pytest.approx(10.0, abs=1e-4)
    # 精确数据：残差为 0，R²=1，CV 误差为 0
    assert body["metrics"]["l_star"]["r2"] == pytest.approx(1.0)
    assert body["metrics"]["l_star"]["cv"]["rmse"] == pytest.approx(0.0)
    assert body["metrics"]["l_star"]["cv"]["method"] == "leave_one_cell_out"
    # 真线性指标在二次模型下交叉项系数为 0
    gloss = {
        c["term"]: c["value"]
        for c in body["metrics"]["gloss60"]["coefficients"]
    }
    assert gloss["x1"] == pytest.approx(80.0, abs=1e-4)
    assert gloss["x2"] == pytest.approx(70.0, abs=1e-4)
    assert gloss["x1x2"] == pytest.approx(0.0, abs=1e-4)
    # 系数置信区间结构
    for c in body["metrics"]["l_star"]["coefficients"]:
        assert c["ci_low"] <= c["value"] <= c["ci_high"]

    # 缺陷发生率按重复试片统计
    pinhole = {c["position"]: c for c in body["defect_rates"]["pinhole"]}
    assert pinhole["E"]["rate"] == pytest.approx(1.0)
    assert pinhole["E"]["n_occurred"] == 2
    assert 0.0 < pinhole["E"]["wilson_low"] < 1.0
    assert pinhole["A"]["rate"] == pytest.approx(0.0)
    crawling = {c["position"]: c for c in body["defect_rates"]["crawling"]}
    assert crawling["D"]["rate"] == pytest.approx(0.5)


def test_fit_linear_model(client):
    study_id, _ = _ready_study(client, finalize=False)
    resp = client.post(f"/firing-studies/{study_id}/fit", json={"model_order": 1})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["terms"] == ["x1", "x2"]
    gloss = {
        c["term"]: c["value"]
        for c in body["metrics"]["gloss60"]["coefficients"]
    }
    assert gloss["x1"] == pytest.approx(80.0, abs=1e-4)
    assert gloss["x2"] == pytest.approx(70.0, abs=1e-4)


def test_fit_outlier_tile_flagged(client):
    study_id, _ = _ready_study(client, finalize=False)
    resp = client.post(
        f"/firing-studies/{study_id}/tiles",
        json=_tile("C", 3, 0.5, 0.5, l_star=70.0),
    )
    assert resp.status_code == 201
    fit = client.post(f"/firing-studies/{study_id}/fit", json={"model_order": 2})
    assert fit.status_code == 200, fit.text
    outliers = fit.json()["metrics"]["l_star"]["outliers"]
    flagged = {(o["position"], o["replicate_no"]) for o in outliers}
    assert ("C", 3) in flagged
    for o in outliers:
        assert abs(o["studentized_residual"]) >= 2.5


def test_fit_insufficient_cells(client):
    # 仅两端点格：二次模型 3 项需要 3 个格位
    cells = [("A", 1.0, 0.0), ("E", 0.0, 1.0)]
    study_id, _ = _ready_study(client, cells=cells, finalize=False)
    resp = client.post(f"/firing-studies/{study_id}/fit", json={"model_order": 2})
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["code"] == "insufficient_design"
    regions = err["details"]["missing_regions"]
    assert any(r["type"] == "missing_edge_interior" for r in regions)


def test_fit_insufficient_tiles(client):
    # 3 格各 1 片：项数 = 试片数，无剩余自由度
    cells = [("A", 1.0, 0.0), ("C", 0.5, 0.5), ("E", 0.0, 1.0)]
    study_id, _ = _ready_study(client, cells=cells, reps=1, finalize=False)
    resp = client.post(f"/firing-studies/{study_id}/fit", json={"model_order": 2})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "insufficient_design"


def test_fit_rank_deficient(client):
    # 全部格位同一比例：设计矩阵欠秩
    cells = [("C1", 0.5, 0.5), ("C2", 0.5, 0.5), ("C3", 0.5, 0.5)]
    study_id, _ = _ready_study(client, cells=cells, finalize=False)
    resp = client.post(f"/firing-studies/{study_id}/fit", json={"model_order": 1})
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["code"] == "rank_deficient_design"
    assert err["details"]["rank"] < err["details"]["n_terms"]


def test_fit_ternary_happy(client):
    cells = [
        ("P", (1.0, 0.0, 0.0)),
        ("Q", (0.0, 1.0, 0.0)),
        ("R", (0.0, 0.0, 1.0)),
        ("S", (0.5, 0.25, 0.25)),
    ]
    experiment_id = _make_ternary_experiment(client, cells)
    study = _make_study(client, experiment_id)
    for pos, w in cells:
        x1, x2, x3 = w
        for rep in (1, 2):
            resp = client.post(
                f"/firing-studies/{study['id']}/tiles",
                json={
                    "position": pos,
                    "replicate_no": rep,
                    "l_star": round(50 * x1 + 60 * x2 + 70 * x3, 6),
                    "a_star": round(5 * x1 + 3 * x2 + 1 * x3, 6),
                    "b_star": round(-3 * x1 + 2 * x2 + 4 * x3, 6),
                    "gloss60": round(80 * x1 + 75 * x2 + 70 * x3, 6),
                    "thickness_mm": round(1.0 * x1 + 1.1 * x2 + 1.2 * x3, 6),
                    "pinhole": 0,
                    "crawling": 0,
                    "running": 0,
                },
            )
            assert resp.status_code == 201
    fit = client.post(
        f"/firing-studies/{study['id']}/fit", json={"model_order": 1}
    )
    assert fit.status_code == 200, fit.text
    body = fit.json()
    assert body["mode"] == "ternary"
    assert body["terms"] == ["x1", "x2", "x3"]
    coefs = {
        c["term"]: c["value"]
        for c in body["metrics"]["l_star"]["coefficients"]
    }
    assert coefs["x1"] == pytest.approx(50.0, abs=1e-4)
    assert coefs["x2"] == pytest.approx(60.0, abs=1e-4)
    assert coefs["x3"] == pytest.approx(70.0, abs=1e-4)


def test_fit_ternary_collinear_rejected(client):
    cells = [
        ("P", (1.0, 0.0, 0.0)),
        ("M", (0.5, 0.5, 0.0)),
        ("Q", (0.0, 1.0, 0.0)),
    ]
    experiment_id = _make_ternary_experiment(client, cells)
    study = _make_study(client, experiment_id)
    for pos, w in cells:
        for rep in (1, 2):
            client.post(
                f"/firing-studies/{study['id']}/tiles",
                json=_tile(pos, rep, w[0], w[1]),
            )
    resp = client.post(f"/firing-studies/{study['id']}/fit", json={"model_order": 1})
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["code"] == "rank_deficient_design"
    types = {r["type"] for r in err["details"]["missing_regions"]}
    assert "unused_source" in types
    assert "collinear_cells" in types


# ---------------------------------------------------------------------------
# 配比搜索
# ---------------------------------------------------------------------------

def test_search_happy(client):
    study_id, _ = _ready_study(client, finalize=True)
    resp = client.post(f"/firing-studies/{study_id}/search", json={
        "model_order": 2,
        "limits": {"l_star": 56.0, "pinhole_rate": 0.5},
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["domain"]["n_grid_points"] == 5
    assert body["domain"]["n_extrapolated_excluded"] == 0
    candidates = body["candidates"]
    assert candidates
    # 排序键字典序：违规数 -> 不确定度 -> 中心偏差
    keys = [
        (c["n_violations"], c["uncertainty"], c["center_deviation"])
        for c in candidates
    ]
    assert keys == sorted(keys)
    best = body["best"]
    assert best["n_violations"] == 0
    assert best["predictions"]["l_star"]["value"] <= 56.0
    assert best["predictions"]["pinhole_rate"]["value"] <= 0.5
    # E 格预测 L*=60 与针孔率 1.0，必违规
    e_cand = [c for c in candidates if c["weights"] == [0.0, 1.0]]
    assert e_cand and set(e_cand[0]["violations"]) == {"l_star", "pinhole_rate"}


def test_search_extrapolation_excluded(client):
    # 仅 B/C/D 有试片：凸包为 x2 ∈ [0.25, 0.75]，端点 0/1 被剔除
    cells = [("B", 0.75, 0.25), ("C", 0.5, 0.5), ("D", 0.25, 0.75)]
    study_id, _ = _ready_study(client, cells=cells, finalize=True)
    resp = client.post(f"/firing-studies/{study_id}/search", json={
        "model_order": 2,
        "limits": {"l_star": 99.0},
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["domain"]["n_extrapolated_excluded"] == 2
    assert body["domain"]["n_in_hull"] == 3
    for c in body["candidates"]:
        assert 0.25 <= c["weights"][1] <= 0.75


def test_search_with_targets_center_deviation(client):
    study_id, _ = _ready_study(client, finalize=True)
    resp = client.post(f"/firing-studies/{study_id}/search", json={
        "model_order": 2,
        "targets": {"l_star": {"low": 50.0, "high": 54.0}},
    })
    assert resp.status_code == 200, resp.text
    best = resp.json()["best"]
    assert best is not None
    assert best["n_violations"] == 0
    assert best["center_deviation"] >= 0.0


def test_search_unknown_metric(client):
    study_id, _ = _ready_study(client, finalize=True)
    resp = client.post(f"/firing-studies/{study_id}/search", json={
        "limits": {"not_a_metric": 1.0},
    })
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "unknown_metric"


def test_search_invalid_rate_limit(client):
    study_id, _ = _ready_study(client, finalize=True)
    resp = client.post(f"/firing-studies/{study_id}/search", json={
        "limits": {"pinhole_rate": 1.5},
    })
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_rate_limit"


def test_search_step_not_compatible(client):
    study_id, _ = _ready_study(client, finalize=True)
    resp = client.post(f"/firing-studies/{study_id}/search", json={
        "ratio_step": 0.3,
    })
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "step_not_compatible"


def test_search_ratio_step_missing(client):
    study_id, _ = _ready_study(client, finalize=True, ratio_step=None)
    resp = client.post(f"/firing-studies/{study_id}/search", json={})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "ratio_step_missing"


# ---------------------------------------------------------------------------
# 结果冻结
# ---------------------------------------------------------------------------

def test_freeze_result_happy_and_idempotent(client):
    study_id, _ = _ready_study(client, finalize=True)
    payload = {
        "model_order": 2,
        "limits": {"l_star": 56.0},
        "candidate_index": 0,
        "note": "选定配比",
    }
    first = client.post(f"/firing-studies/{study_id}/result-freezes", json=payload)
    assert first.status_code == 201, first.text
    assert first.json()["created"] is True
    freeze = first.json()["freeze"]
    assert freeze["id"] == freeze["input_hash"]
    assert freeze["study_id"] == study_id
    assert freeze["fit_options"]["model_order"] == 2
    assert freeze["selected"]["n_violations"] == 0
    assert freeze["selected"]["predictions"]["l_star"]["value"] <= 56.0
    assert len(freeze["source_snapshot"]["tiles"]) == 10
    assert freeze["source_snapshot"]["study"]["kiln_run"] == "2026-09-10-#2"

    second = client.post(f"/firing-studies/{study_id}/result-freezes", json=payload)
    assert second.json()["created"] is False
    assert second.json()["freeze"]["id"] == freeze["id"]

    got = client.get(f"/firing-result-freezes/{freeze['id']}")
    assert got.status_code == 200
    assert got.json()["selected"] == freeze["selected"]
    listed = client.get(f"/firing-studies/{study_id}/result-freezes").json()
    assert len(listed) == 1


def test_freeze_result_requires_finalized(client):
    study_id, _ = _ready_study(client, finalize=False)
    resp = client.post(f"/firing-studies/{study_id}/result-freezes", json={})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_firing_status"


def test_freeze_result_candidate_out_of_range(client):
    study_id, _ = _ready_study(client, finalize=True)
    resp = client.post(f"/firing-studies/{study_id}/result-freezes", json={
        "candidate_index": 99,
    })
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "candidate_index_out_of_range"


def test_result_freeze_immutable_across_kiln_runs(client):
    experiment_id, _ = _make_experiment(client)
    # 第一窑
    study1 = _make_study(client, experiment_id, kiln_run="run-1")
    _seed_tiles(client, study1["id"])
    client.post(f"/firing-studies/{study1['id']}/finalize", json={})
    payload = {"model_order": 2, "limits": {"l_star": 56.0}}
    first = client.post(
        f"/firing-studies/{study1['id']}/result-freezes", json=payload
    )
    assert first.status_code == 201, first.text
    freeze1 = first.json()["freeze"]

    # 后续窑次：同一试验、不同试片数据、相同搜索约束
    study2 = _make_study(client, experiment_id, kiln_run="run-2")
    for pos, x1, x2 in CELLS:
        for rep in (1, 2):
            client.post(
                f"/firing-studies/{study2['id']}/tiles",
                json=_tile(pos, rep, x1, x2, l_star=round(
                    55 * x1 + 65 * x2 + 10 * x1 * x2, 6
                )),
            )
    client.post(f"/firing-studies/{study2['id']}/finalize", json={})
    second = client.post(
        f"/firing-studies/{study2['id']}/result-freezes", json=payload
    )
    assert second.status_code == 201, second.text
    freeze2 = second.json()["freeze"]
    assert freeze2["id"] != freeze1["id"]

    # 第一窑的冻结结果不被改写
    got = client.get(f"/firing-result-freezes/{freeze1['id']}").json()
    assert got["selected"] == freeze1["selected"]
    assert got["source_snapshot"]["study"]["kiln_run"] == "run-1"
