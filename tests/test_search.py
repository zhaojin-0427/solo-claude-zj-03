"""搜索 / 重配接口测试：锁定、禁用、单项上下限、排序与无解诊断。"""
from __future__ import annotations

import pytest


def _id_map(client):
    return {m["name"]: m["id"] for m in client.get("/materials").json()}


GLAZE_TARGETS = {
    "K2O":  {"low": 0.15, "high": 0.35},
    "CaO":  {"low": 0.35, "high": 0.6},
    "MgO":  {"low": 0.05, "high": 0.2},
    "ZnO":  {"low": 0.05, "high": 0.15},
    "Al2O3": {"low": 0.3, "high": 0.5},
    "SiO2": {"low": 2.5, "high": 3.5},
}


def test_search_finds_feasible_recipe(client):
    resp = client.post("/search", json={
        "batch_size": 100.0, "step": 0.5, "batch_tolerance": 0.5,
        "targets": GLAZE_TARGETS,
    })
    assert resp.status_code == 200, resp.text
    r = resp.json()
    assert r["status"] == "optimal"
    best = r["best"]
    assert best["all_targets_met"] is True
    assert best["n_violations"] == 0
    assert abs(best["batch_mass"] - 100.0) <= 0.5 + 1e-9
    # 用量均在步进网格上
    for item in best["items"]:
        n = item["amount"] / 0.5
        assert abs(n - round(n)) < 1e-6
    # 助熔归一化
    seger = best["seger"]
    flux_sum = sum(v for k, v in seger.items()
                   if k in ("K2O", "Na2O", "CaO", "MgO", "BaO", "ZnO",
                            "Li2O", "SrO", "PbO", "MnO"))
    assert flux_sum == pytest.approx(1.0, abs=1e-6)
    assert len(r["candidates"]) >= 2
    # 候选按 (越界, 加权偏差, 种数, 成本) 排序
    keys = [(c["n_violations"], c["weighted_deviation"], c["n_materials"],
             round(c["cost"], 6)) for c in r["candidates"]]
    assert keys == sorted(keys)


def test_search_respects_locked_and_forbidden(client):
    ids = _id_map(client)
    resp = client.post("/search", json={
        "batch_size": 100.0, "step": 0.5, "batch_tolerance": 0.5,
        "targets": {
            "CaO": {"low": 0.4, "high": 0.7},
            "Al2O3": {"low": 0.3, "high": 0.5},
            "SiO2": {"low": 2.5, "high": 3.5},
        },
        "forbidden": [ids["钾长石"]],
        "locked": {str(ids["石英"]): 30.0},
        "required": [ids["方解石"]],
    })
    r = resp.json()
    assert r["status"] == "optimal", r
    used = {i["material_id"]: i["amount"] for i in r["best"]["items"]}
    assert ids["钾长石"] not in used
    assert used.get(ids["石英"]) == 30.0
    assert ids["方解石"] in used


def test_search_material_limit(client):
    ids = _id_map(client)
    resp = client.post("/search", json={
        "batch_size": 100.0, "step": 0.5,
        "targets": GLAZE_TARGETS,
        "limits": {str(ids["钾长石"]): {"low": 60.0, "high": 80.0}},
    })
    r = resp.json()
    assert resp.status_code == 200
    amt = {i["material_id"]: i["amount"] for i in r["best"]["items"]}.get(ids["钾长石"], 0)
    assert 60.0 - 1e-9 <= amt <= 80.0 + 1e-9


def test_search_stock_limit(client):
    ids = _id_map(client)
    # 把氧化锌库存压到 0.5 kg 再搜：ZnO 下限 0.05 釉式对 100 kg 批量
    # 需要约 1kg 量级，应当出现越界
    client.patch(f"/materials/{ids['氧化锌']}", json={"available": 0.5})
    resp = client.post("/search", json={
        "batch_size": 100.0, "step": 0.5,
        "targets": GLAZE_TARGETS,
    })
    r = resp.json()
    assert r["status"] in ("optimal", "target_unreachable")
    zinc_used = sum(
        i["amount"] for i in r["best"]["items"] if i["material_id"] == ids["氧化锌"]
    )
    assert zinc_used <= 0.5 + 1e-9


def test_search_target_unreachable_diagnosis(client):
    resp = client.post("/search", json={
        "batch_size": 100.0, "step": 0.5,
        "targets": {"Al2O3": {"low": 5.0}},
    })
    r = resp.json()
    assert r["status"] == "target_unreachable"
    assert r["best"]["n_violations"] >= 1
    limiting = r["diagnosis"]["limiting_oxides"]
    assert any(x["oxide"] == "Al2O3" for x in limiting)
    info = next(x for x in limiting if x["oxide"] == "Al2O3")
    assert info["achievable_high"] < 5.0
    assert "Al2O3" in r["diagnosis"]["note"]


def test_search_structural_infeasible(client):
    resp = client.post("/search", json={
        "batch_size": 1_000_000.0, "step": 1.0,
        "targets": {},
    })
    r = resp.json()
    assert r["status"] == "infeasible"
    assert r["best"] is None
    assert r["candidates"] == []
    assert r["diagnosis"]["structural_issues"]


def test_search_required_forbidden_conflict(client):
    resp = client.post("/search", json={
        "batch_size": 10.0, "step": 1.0,
        "required": [1], "forbidden": [1],
    })
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "required_forbidden_conflict"


def test_search_lock_off_grid(client):
    resp = client.post("/search", json={
        "batch_size": 10.0, "step": 1.0,
        "locked": {"1": 0.15},
    })
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "lock_not_on_grid"


def test_search_lock_exceeds_batch(client):
    resp = client.post("/search", json={
        "batch_size": 10.0, "step": 1.0,
        "locked": {"1": 20.0},
    })
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "locked_exceeds_batch"


def test_search_unknown_oxide_target(client):
    resp = client.post("/search", json={
        "batch_size": 10.0, "step": 1.0,
        "targets": {"XO": {"low": 0.1}},
    })
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "unknown_oxide"


def test_search_contradictory_bounds(client):
    resp = client.post("/search", json={
        "batch_size": 10.0, "step": 1.0,
        "targets": {"SiO2": {"low": 3.0, "high": 2.0}},
    })
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "contradictory_bounds"


def test_search_nonexistent_material_reference(client):
    resp = client.post("/search", json={
        "batch_size": 10.0, "step": 1.0,
        "required": [9999],
    })
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "material_not_found"
