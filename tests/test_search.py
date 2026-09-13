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


def test_search_forbidden_and_locked_conflict(client):
    """同一原料同时 forbidden 与 locked：必须拒绝而非忽略锁定。"""
    resp = client.post("/search", json={
        "batch_size": 100.0, "step": 0.5,
        "targets": GLAZE_TARGETS,
        "forbidden": [1],
        "locked": {"1": 40.0},
    })
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "forbidden_locked_conflict"


def test_search_forbidden_with_limit_rejected(client):
    """禁用原料又给正用量上下限同样拒绝。"""
    resp = client.post("/search", json={
        "batch_size": 100.0, "step": 0.5,
        "targets": GLAZE_TARGETS,
        "forbidden": [2],
        "limits": {str(2): {"low": 5.0}},
    })
    assert resp.status_code == 422


def test_search_target_results_complete(client):
    """每个候选都要带每个 target 的当前值、区间与偏差。"""
    resp = client.post("/search", json={
        "batch_size": 100.0, "step": 0.5, "batch_tolerance": 0.5,
        "targets": {
            "K2O": {"low": 0.15, "high": 0.35, "weight": 2.0},
            "CaO": {"low": 0.35, "high": 0.6},
            "SiO2": {"low": 2.5, "high": 3.5},
        },
    })
    r = resp.json()
    tr = r["best"]["target_results"]
    assert set(tr) == {"K2O", "CaO", "SiO2"}
    for oxide, t in tr.items():
        assert "value" in t and "low" in t and "high" in t
        assert "deviation" in t and "weighted_deviation" in t
        assert "weight" in t and "in_range" in t
    assert tr["K2O"]["weight"] == 2.0
    # 达标项偏差为 0 且 in_range；未达标项偏差为正
    for oxide, t in tr.items():
        if t["in_range"]:
            assert t["deviation"] == 0
        else:
            assert t["deviation"] > 0


def test_search_weighted_deviation_optimal_with_required(client):
    """同为必用的两个原料：越界数相同时必须选加权偏差更小的方案。

    在小原料库（3 种原料）上对整数网格暴力枚举，验证优化器返回的
    (越界数, 加权偏差) 等于真实字典序最优。
    """
    import itertools
    from app.chemistry import calc_batch, deviation_summary

    created = []
    for name, ox in [
        ("A料", {"K2O": 20.0, "SiO2": 80.0}),
        ("B料", {"CaO": 50.0, "SiO2": 50.0}),
        ("C料", {"SiO2": 100.0}),
    ]:
        resp = client.post("/materials", json={
            "name": name, "oxides": ox, "loi": 0.0,
            "price": 1.0, "available": 100.0,
        })
        created.append(resp.json()["id"])
    A, B, C = created
    others = [m["id"] for m in client.get("/materials").json()
              if m["id"] not in created]

    body = {
        "batch_size": 10.0, "step": 1.0, "batch_tolerance": 0.0,
        "targets": {
            "K2O":  {"low": 0.30, "high": 0.30, "weight": 3.0},
            "CaO":  {"low": 0.40, "high": 0.40, "weight": 2.0},
            "SiO2": {"low": 3.0, "high": 3.0, "weight": 1.0},
        },
        "required": [A, B],
        "forbidden": others,
    }
    r = client.post("/search", json=body).json()
    opt_grid = tuple(
        int(round(next((i["amount"] for i in r["best"]["items"]
                        if i["material_id"] == m), 0.0)))
        for m in (A, B, C)
    )

    # 暴力枚举全部整数网格
    class _T:
        def __init__(self, d):
            self.low, self.high, self.weight = d["low"], d["high"], d["weight"]

    mats = {m["id"]: m for m in client.get("/materials").json()}
    scored = []
    for ka, kb, kc in itertools.product(range(0, 11), repeat=3):
        if ka < 1 or kb < 1 or ka + kb + kc != 10:
            continue
        tuples = []
        for mid, g in ((A, ka), (B, kb), (C, kc)):
            if g == 0:
                continue
            m = mats[mid]
            tuples.append((m["id"], m["name"], m["oxides"], m["loi"],
                           m["price"], float(g)))
        tg = {o: _T(body["targets"][o]) for o in body["targets"]}
        res = calc_batch(tuples, targets=tg)
        s = deviation_summary(res, tg)
        scored.append(((s["n_violations"], round(s["weighted_deviation"], 12)),
                       (ka, kb, kc)))
    scored.sort(key=lambda x: x[0])
    opt_key = next(k for k, g in scored if g == opt_grid)
    assert opt_key == scored[0][0]


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
