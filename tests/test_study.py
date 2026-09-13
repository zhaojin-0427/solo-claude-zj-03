"""原料批次波动研究：创建校验、重采样统计、稳健搜索与结果冻结。

测试配方（冻结来源）：钾长石 60 / 方解石 10 / 石英 30（批量 100 kg）。
化验批次设计（目标 CaO ∈ [0.46, 0.55]）：
  钾长石 K1: K2O 16.92   钾长石 K2: K2O 15.5
  方解石 C1: CaO 56.03   方解石 C2: CaO 50.0（loi 50.0）
独立抽样时仅 (K1, C2) 组合越界（CaO≈0.453），联合达标率约 0.75；
若声明联动 (K1↔C1, K2↔C2)，所有场景达标，联合达标率恰为 1.0。
"""
from __future__ import annotations

import pytest


def _id_map(client):
    return {m["name"]: m["id"] for m in client.get("/materials").json()}


def _freeze_source(client, **extra):
    ids = _id_map(client)
    payload = {"items": [
        {"material_id": ids["钾长石"], "amount": 60.0},
        {"material_id": ids["方解石"], "amount": 10.0},
        {"material_id": ids["石英"], "amount": 30.0},
    ]}
    payload.update(extra)
    resp = client.post("/versions", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()["version"], ids


def _batches(ids, feldspar_labels=("K1", "K2"), calcite_labels=("C1", "C2")):
    return [
        {"material_id": ids["钾长石"], "batch": feldspar_labels[0],
         "oxides": {"K2O": 16.92, "Al2O3": 18.32, "SiO2": 64.76},
         "loi": 0.0, "price": 2.4, "available": 500.0},
        {"material_id": ids["钾长石"], "batch": feldspar_labels[1],
         "oxides": {"K2O": 15.5, "Al2O3": 18.32, "SiO2": 66.18},
         "loi": 0.0, "price": 2.4, "available": 500.0},
        {"material_id": ids["方解石"], "batch": calcite_labels[0],
         "oxides": {"CaO": 56.03}, "loi": 43.97, "price": 0.8, "available": 800.0},
        {"material_id": ids["方解石"], "batch": calcite_labels[1],
         "oxides": {"CaO": 50.0}, "loi": 50.0, "price": 0.6, "available": 800.0},
        {"material_id": ids["石英"], "batch": "Q1",
         "oxides": {"SiO2": 100.0}, "loi": 0.0, "price": 1.0, "available": 1000.0},
    ]


def _study_payload(version_id, ids, **overrides):
    payload = {
        "version_id": version_id,
        "batches": _batches(ids),
        "targets": {"CaO": {"low": 0.46, "high": 0.55}},
        "n_resamples": 200,
        "seed": 7,
    }
    payload.update(overrides)
    return payload


def _create_study(client, **overrides):
    version, ids = _freeze_source(client)
    resp = client.post("/studies", json=_study_payload(version["id"], ids, **overrides))
    return resp, version, ids


# ---------------------------------------------------------------------------
# 创建与统计
# ---------------------------------------------------------------------------

def test_create_study_happy(client):
    resp, version, ids = _create_study(client)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["created"] is True
    study = body["study"]
    assert study["version_id"] == version["id"]
    assert study["n_resamples"] == 200
    assert study["seed"] == 7

    r = study["result"]
    assert r["n_resamples"] == 200
    assert r["seed"] == 7
    assert r["batch_mass"] == pytest.approx(100.0)
    # 仅 (K1, C2) 组合越界 → 联合达标率约 0.75
    assert 0.6 < r["joint_pass_rate"] < 0.9
    assert r["n_violation_scenarios"] > 0

    # 各氧化物分位数
    assert {"CaO", "K2O", "Al2O3", "SiO2"} <= set(r["oxide_stats"])
    cao = r["oxide_stats"]["CaO"]
    assert cao["p5"] <= cao["p50"] <= cao["p95"]
    assert cao["p5"] < cao["p95"]  # 批次波动确实传导到釉式

    # 单项目标达标率（单目标时等于联合达标率）
    t = r["target_stats"]["CaO"]
    assert t["low"] == 0.46 and t["high"] == 0.55
    assert t["pass_rate"] == pytest.approx(r["joint_pass_rate"])
    assert t["mean_deviation"] > 0

    # 常见越界组合：只有 CaO 单独越界
    combos = r["violation_combinations"]
    assert len(combos) == 1
    assert combos[0]["oxides"] == ["CaO"]
    assert combos[0]["count"] == r["n_violation_scenarios"]
    assert combos[0]["share_of_violations"] == pytest.approx(1.0)

    # 原料敏感度：石英只有一批 → 0；方解石批次决定成败 → 明显为正
    sens = {s["material_id"]: s for s in r["material_sensitivity"]}
    assert set(sens) == {ids["钾长石"], ids["方解石"], ids["石英"]}
    assert sens[ids["石英"]]["spread"] == 0.0
    assert sens[ids["方解石"]]["spread"] > 0.2
    assert sens[ids["方解石"]]["batch_pass_rates"]["C1"] == pytest.approx(1.0)
    spreads = [s["spread"] for s in r["material_sensitivity"]]
    assert spreads == sorted(spreads, reverse=True)

    # 烧后质量与成本统计（方解石 LOI/价格随批波动）
    assert r["fired_mass"]["p5"] < r["fired_mass"]["p95"]
    assert r["cost"]["p5"] <= r["cost"]["p50"] <= r["cost"]["p95"]


def test_create_study_idempotent(client):
    resp1, version, ids = _create_study(client)
    s1 = resp1.json()["study"]
    resp2 = client.post("/studies", json=_study_payload(version["id"], ids))
    body2 = resp2.json()
    assert body2["created"] is False
    assert body2["study"]["id"] == s1["id"]
    assert body2["study"]["result"] == s1["result"]

    # GET 与列表一致
    got = client.get(f"/studies/{s1['id']}")
    assert got.status_code == 200
    assert got.json()["result"] == s1["result"]
    listing = client.get("/studies").json()
    assert any(s["id"] == s1["id"] for s in listing)


def test_study_missing_batch_data(client):
    version, ids = _freeze_source(client)
    payload = _study_payload(version["id"], ids)
    payload["batches"] = [b for b in payload["batches"] if b["material_id"] != ids["石英"]]
    resp = client.post("/studies", json=payload)
    assert resp.status_code == 422
    body = resp.json()["error"]
    assert body["code"] == "missing_batch_data"
    assert body["details"]["material_ids"] == [ids["石英"]]


def test_study_batch_for_foreign_material(client):
    version, ids = _freeze_source(client)
    payload = _study_payload(version["id"], ids)
    payload["batches"].append({
        "material_id": ids["白云石"], "batch": "D1",
        "oxides": {"CaO": 30.41, "MgO": 21.86}, "loi": 47.73,
        "price": 0.9, "available": 100.0,
    })
    resp = client.post("/studies", json=payload)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "material_not_in_recipe"


def test_study_analysis_sum_error(client):
    version, ids = _freeze_source(client)
    payload = _study_payload(version["id"], ids)
    payload["batches"][2]["oxides"] = {"CaO": 30.0}
    payload["batches"][2]["loi"] = 40.0  # 合计 70，超出默认 ±2
    resp = client.post("/studies", json=payload)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "analysis_sum_error"


def test_study_unknown_oxide(client):
    version, ids = _freeze_source(client)
    payload = _study_payload(version["id"], ids)
    payload["batches"][0]["oxides"] = {"K2O": 16.92, "XxO": 83.08}
    resp = client.post("/studies", json=payload)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "unknown_oxide"


def test_study_duplicate_batch(client):
    version, ids = _freeze_source(client)
    payload = _study_payload(version["id"], ids)
    payload["batches"].append(dict(payload["batches"][0]))
    resp = client.post("/studies", json=payload)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "duplicate_batch"


def test_study_linked_pairing_incomplete(client):
    version, ids = _freeze_source(client)
    payload = _study_payload(version["id"], ids)
    payload["linked_groups"] = [[ids["钾长石"], ids["方解石"]]]  # K1/K2 vs C1/C2
    resp = client.post("/studies", json=payload)
    assert resp.status_code == 422
    body = resp.json()["error"]
    assert body["code"] == "batch_pairing_incomplete"
    assert body["details"]["missing"] or body["details"]["extra"]


def test_study_linked_group_conflict(client):
    version, ids = _freeze_source(client)
    payload = _study_payload(version["id"], ids)
    payload["linked_groups"] = [
        [ids["钾长石"], ids["方解石"]],
        [ids["方解石"], ids["石英"]],
    ]
    resp = client.post("/studies", json=payload)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "linked_group_conflict"

    payload["linked_groups"] = [[ids["钾长石"]]]
    resp = client.post("/studies", json=payload)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "linked_group_conflict"


def test_linked_sampling_joint_rate(client):
    """同批联动：K1↔C1、K2↔C2 配对后所有场景达标；独立抽样则否。"""
    version, ids = _freeze_source(client)
    linked_payload = _study_payload(
        version["id"], ids,
        batches=_batches(ids, feldspar_labels=("L1", "L2"),
                         calcite_labels=("L1", "L2")),
        linked_groups=[[ids["钾长石"], ids["方解石"]]],
    )
    resp = client.post("/studies", json=linked_payload)
    assert resp.status_code == 201, resp.text
    assert resp.json()["study"]["result"]["joint_pass_rate"] == pytest.approx(1.0)

    unlinked = _study_payload(version["id"], ids)
    resp2 = client.post("/studies", json=unlinked)
    assert resp2.json()["study"]["result"]["joint_pass_rate"] < 1.0


def test_study_version_not_found(client):
    _, ids = _freeze_source(client)
    resp = client.post("/studies", json=_study_payload("deadbeef", ids))
    assert resp.status_code == 404


def test_study_no_flux_rejected(client):
    version, ids = _freeze_source(client)
    payload = _study_payload(version["id"], ids)
    for b in payload["batches"]:
        b["oxides"] = {"SiO2": 100.0}
        b["loi"] = 0.0
    resp = client.post("/studies", json=payload)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "no_flux"


def test_study_targets_default_from_version(client):
    """研究未给 targets 时沿用来源版本搜索约束中的目标。"""
    ids = _id_map(client)
    resp = client.post("/versions", json={
        "items": [
            {"material_id": ids["钾长石"], "amount": 60.0},
            {"material_id": ids["方解石"], "amount": 10.0},
            {"material_id": ids["石英"], "amount": 30.0},
        ],
        "search_constraints": {
            "mode": "search",
            "targets": {"CaO": {"low": 0.4, "high": 0.6, "weight": 2.0}},
        },
    })
    version = resp.json()["version"]
    payload = _study_payload(version["id"], ids)
    del payload["targets"]
    resp = client.post("/studies", json=payload)
    assert resp.status_code == 201, resp.text
    study = resp.json()["study"]
    assert study["targets"]["CaO"]["low"] == 0.4
    assert study["targets"]["CaO"]["weight"] == 2.0
    assert "CaO" in study["result"]["target_stats"]


def test_draw_plans_linked_and_deterministic():
    from app.variability import draw_plans

    batches = {
        1: {"L1": None, "L2": None},
        2: {"L1": None, "L2": None},
        3: {"A": None, "B": None},
    }
    plans = draw_plans([1, 2, 3], batches, [[1, 2]], 200, 42)
    # 联动组始终同批号
    assert all(p[1] == p[2] for p in plans)
    assert {p[1] for p in plans} <= {"L1", "L2"}
    # 同种子完全可复现
    assert draw_plans([1, 2, 3], batches, [[1, 2]], 200, 42) == plans
    # 非联动原料独立抽取：2×2 组合都会出现
    assert len({(p[1], p[3]) for p in plans}) == 4
    # 不声明联动时两原料同样独立
    plans2 = draw_plans([1, 2], batches, [], 200, 42)
    assert len({(p[1], p[2]) for p in plans2}) == 4


# ---------------------------------------------------------------------------
# 稳健配方搜索
# ---------------------------------------------------------------------------

def _make_study(client, **overrides):
    resp, version, ids = _create_study(client, **overrides)
    assert resp.status_code == 201, resp.text
    return resp.json()["study"], version, ids


def test_robust_search_improves_joint_rate(client):
    study, _, ids = _make_study(client)
    resp = client.post(
        f"/studies/{study['id']}/robust-search",
        json={"max_change": 10.0, "step": 0.5},
    )
    assert resp.status_code == 200, resp.text
    r = resp.json()
    assert r["status"] == "ok"
    assert r["search_constraints"]["n_resamples"] == 200
    assert r["search_constraints"]["seed"] == 7

    best = r["best"]
    assert best["joint_pass_rate"] == pytest.approx(1.0)
    assert best["worst_quantile_deviation"] == pytest.approx(0.0)
    # 最小改动：方解石 +0.5（转移 0.5 kg，L1 改动量 1.0）
    assert best["change_amount"] == pytest.approx(1.0)
    amounts = {i["material_id"]: i["amount"] for i in best["items"]}
    assert amounts[ids["方解石"]] == pytest.approx(10.5)
    assert sum(amounts.values()) == pytest.approx(100.0)  # 批量不变

    # 来源配方作为 0 改动候选一并返回
    source = next(c for c in r["candidates"] if c["is_source"])
    assert source["change_amount"] == 0.0
    assert source["joint_pass_rate"] == pytest.approx(
        study["result"]["joint_pass_rate"]
    )
    # 候选按 (联合达标率降, 最差分位偏差, 改动量, 成本) 排序
    keys = [
        (-c["joint_pass_rate"], c["worst_quantile_deviation"],
         c["change_amount"], c["cost_mean"])
        for c in r["candidates"]
    ]
    assert keys == sorted(keys)


def test_robust_search_respects_locked(client):
    study, _, ids = _make_study(client)
    resp = client.post(
        f"/studies/{study['id']}/robust-search",
        json={"max_change": 10.0, "step": 0.5, "locked": [ids["钾长石"]]},
    )
    r = resp.json()
    best = r["best"]
    assert best["joint_pass_rate"] == pytest.approx(1.0)
    amounts = {i["material_id"]: i["amount"] for i in best["items"]}
    assert amounts[ids["钾长石"]] == pytest.approx(60.0)  # 锁定不动
    assert amounts[ids["方解石"]] == pytest.approx(10.5)
    assert amounts[ids["石英"]] == pytest.approx(29.5)


def test_robust_search_all_locked_returns_source(client):
    study, _, ids = _make_study(client)
    resp = client.post(
        f"/studies/{study['id']}/robust-search",
        json={"max_change": 10.0, "step": 0.5,
              "locked": [ids["钾长石"], ids["方解石"]]},
    )
    r = resp.json()
    assert len(r["candidates"]) == 1
    assert r["best"]["is_source"] is True
    assert r["best"]["change_amount"] == 0.0


def test_robust_search_budget_too_small(client):
    study, _, _ = _make_study(client)
    # 最小转移改动量为 2×step = 1.0 kg，预算 0.5 下无可行移动
    resp = client.post(
        f"/studies/{study['id']}/robust-search",
        json={"max_change": 0.5, "step": 0.5},
    )
    r = resp.json()
    assert len(r["candidates"]) == 1
    assert r["best"]["change_amount"] == 0.0


def test_robust_search_stock_cap(client):
    """方解石各批最小可用量 10.4 kg → 无法增至 10.5，搜索退回来源配方。"""
    version, ids = _freeze_source(client)
    payload = _study_payload(version["id"], ids)
    for b in payload["batches"]:
        if b["material_id"] == ids["方解石"]:
            b["available"] = 10.4
    study = client.post("/studies", json=payload).json()["study"]

    resp = client.post(
        f"/studies/{study['id']}/robust-search",
        json={"max_change": 10.0, "step": 0.5, "locked": [ids["钾长石"]]},
    )
    r = resp.json()
    assert r["stock_basis"][str(ids["方解石"])] == pytest.approx(10.4)
    assert len(r["candidates"]) == 1
    assert r["best"]["change_amount"] == 0.0


def test_robust_search_deterministic(client):
    study, _, _ = _make_study(client)
    body = {"max_change": 10.0, "step": 0.5}
    r1 = client.post(f"/studies/{study['id']}/robust-search", json=body).json()
    r2 = client.post(f"/studies/{study['id']}/robust-search", json=body).json()
    assert r1 == r2


def test_robust_search_study_not_found(client):
    resp = client.post(
        "/studies/deadbeef/robust-search", json={"max_change": 1.0}
    )
    assert resp.status_code == 404


def test_robust_search_locked_outsider(client):
    study, _, ids = _make_study(client)
    resp = client.post(
        f"/studies/{study['id']}/robust-search",
        json={"max_change": 5.0, "locked": [ids["白云石"]]},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "material_not_in_recipe"


# ---------------------------------------------------------------------------
# 稳健结果冻结
# ---------------------------------------------------------------------------

def _search_best(client, study_id):
    r = client.post(
        f"/studies/{study_id}/robust-search",
        json={"max_change": 10.0, "step": 0.5},
    ).json()
    return r


def test_freeze_robust_result(client):
    study, version, ids = _make_study(client)
    search = _search_best(client, study["id"])
    best = search["best"]

    resp = client.post(f"/studies/{study['id']}/freeze", json={
        "items": best["items"],
        "search_constraints": search["search_constraints"],
        "note": "稳健配方 v1",
    })
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["created"] is True
    freeze = body["freeze"]
    assert freeze["study_id"] == study["id"]
    assert freeze["note"] == "稳健配方 v1"

    # 冻结重算与搜索评估一致（同种子同场景）
    assert freeze["result"]["joint_pass_rate"] == pytest.approx(
        best["joint_pass_rate"]
    )
    assert freeze["result"]["change_amount"] == pytest.approx(
        best["change_amount"]
    )

    # 来源配方、化验数据、抽样规则与随机种子整体存档
    snap = freeze["source_snapshot"]
    assert snap["version_id"] == version["id"]
    assert snap["seed"] == 7
    assert snap["n_resamples"] == 200
    assert len(snap["batches"]) == 5
    assert snap["linked_groups"] == []
    assert snap["constants_snapshot"]["oxides"]["CaO"]["molwt"]

    # 幂等：同输入重复冻结返回同一记录
    resp2 = client.post(f"/studies/{study['id']}/freeze", json={
        "items": list(reversed(best["items"])),
        "search_constraints": search["search_constraints"],
        "note": "稳健配方 v1",
    })
    body2 = resp2.json()
    assert body2["created"] is False
    assert body2["freeze"]["id"] == freeze["id"]

    # GET 一致
    got = client.get(f"/robust-versions/{freeze['id']}")
    assert got.status_code == 200
    assert got.json()["result"] == freeze["result"]
    freezes = client.get(f"/studies/{study['id']}/freezes").json()
    assert any(f["id"] == freeze["id"] for f in freezes)


def test_freeze_source_matches_study(client):
    """以来源配方原样冻结：联合达标率应与研究结果一致。"""
    study, _, _ = _make_study(client)
    resp = client.post(f"/studies/{study['id']}/freeze", json={
        "items": study["source_items"],
    })
    assert resp.status_code == 201, resp.text
    freeze = resp.json()["freeze"]
    assert freeze["result"]["joint_pass_rate"] == pytest.approx(
        study["result"]["joint_pass_rate"]
    )
    assert freeze["result"]["change_amount"] == 0.0


def test_freeze_foreign_material(client):
    study, _, ids = _make_study(client)
    resp = client.post(f"/studies/{study['id']}/freeze", json={
        "items": [
            {"material_id": ids["钾长石"], "amount": 60.0},
            {"material_id": ids["白云石"], "amount": 40.0},
        ],
    })
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "material_not_in_recipe"


def test_freeze_empty_batch(client):
    study, _, ids = _make_study(client)
    resp = client.post(f"/studies/{study['id']}/freeze", json={
        "items": [{"material_id": ids["石英"], "amount": 0.0}],
    })
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "empty_batch"


def test_freeze_study_not_found(client):
    resp = client.post("/studies/deadbeef/freeze", json={
        "items": [{"material_id": 1, "amount": 1.0}],
    })
    assert resp.status_code == 404


def test_robust_version_not_found(client):
    assert client.get("/robust-versions/deadbeef").status_code == 404
    assert client.get("/studies/deadbeef").status_code == 404
    assert client.get("/studies/deadbeef/freezes").status_code == 404
