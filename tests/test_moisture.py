"""原料含水测定：不可变版本登记、湿料称量换算、配方接口批号选择与釉浆扣水。"""
from __future__ import annotations

import pytest

TODAY = "2026-09-14"


def _ids(client):
    return {m["name"]: m["id"] for m in client.get("/materials").json()}


def _add_measurement(client, mid, lot="L1", sample=110.0, dried=99.0,
                     valid_from="2026-01-01", valid_to="2026-12-31", **extra):
    payload = {
        "lot": lot, "sample_mass_g": sample, "dried_mass_g": dried,
        "valid_from": valid_from, "valid_to": valid_to,
    }
    payload.update(extra)
    return client.post(f"/materials/{mid}/moisture-measurements", json=payload)


def _freeze(client, items, **extra):
    payload = {"items": items}
    payload.update(extra)
    return client.post("/versions", json=payload)


# ---------------------------------------------------------------------------
# 测定登记：湿基含水率、校验与不可变
# ---------------------------------------------------------------------------

def test_measurement_wet_basis_fraction_and_listing(client):
    ids = _ids(client)
    r = _add_measurement(client, ids["钾长石"], sample=110.0, dried=99.0)
    assert r.status_code == 201, r.text
    rec = r.json()
    assert rec["moisture_fraction"] == pytest.approx(0.1)
    assert rec["moisture_pct"] == pytest.approx(10.0)
    assert rec["lot"] == "L1"
    listing = client.get(f"/materials/{ids['钾长石']}/moisture-measurements").json()
    assert [m["id"] for m in listing] == [rec["id"]]
    assert client.get("/moisture-measurements").json()[0]["id"] == rec["id"]


def test_measurement_mass_inverted_rejected(client):
    ids = _ids(client)
    r = _add_measurement(client, ids["钾长石"], sample=90.0, dried=100.0)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "moisture_mass_inverted"


def test_measurement_interval_inverted_rejected(client):
    ids = _ids(client)
    r = _add_measurement(
        client, ids["钾长石"], valid_from="2026-06-01", valid_to="2026-01-01"
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "moisture_interval_inverted"


def test_measurement_lot_missing_rejected(client):
    ids = _ids(client)
    r = client.post(
        f"/materials/{ids['钾长石']}/moisture-measurements",
        json={"lot": "   ", "sample_mass_g": 100.0, "dried_mass_g": 90.0,
              "valid_from": "2026-01-01", "valid_to": "2026-12-31"},
    )
    assert r.status_code == 422
    # 批号字段本身 min_length=1；纯空白亦不接受
    assert r.status_code == 422

    # 计算接口给空批号同样 422
    r = client.post("/compute", json={
        "items": [{"material_id": ids["钾长石"], "amount": 10.0}],
        "lots": {str(ids["钾长石"]): ""},
    })
    assert r.json()["error"]["code"] == "moisture_lot_missing"


def test_interval_overlap_same_lot_rejected_adjacent_allowed(client):
    ids = _ids(client)
    assert _add_measurement(
        client, ids["钾长石"], lot="A",
        valid_from="2026-01-01", valid_to="2026-03-31"
    ).status_code == 201
    # 重叠区间
    r = _add_measurement(
        client, ids["钾长石"], lot="A",
        valid_from="2026-03-31", valid_to="2026-06-30"
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "moisture_interval_overlap"
    # 紧邻（前一区间次日起）允许：闭区间下 4/1 起不相交
    r = _add_measurement(
        client, ids["钾长石"], lot="A",
        valid_from="2026-04-01", valid_to="2026-06-30"
    )
    assert r.status_code == 201, r.text
    # 不同批号区间重叠允许
    r = _add_measurement(
        client, ids["钾长石"], lot="B",
        valid_from="2026-02-01", valid_to="2026-05-31"
    )
    assert r.status_code == 201, r.text


def test_measurement_unknown_material_404(client):
    r = _add_measurement(client, 9999)
    assert r.status_code == 404


def test_measurement_is_immutable_and_as_of_respected(client):
    ids = _ids(client)
    _add_measurement(
        client, ids["钾长石"], sample=100.0, dried=95.0,
        valid_from="2026-01-01", valid_to="2026-03-31",
    )
    _add_measurement(
        client, ids["钾长石"], sample=100.0, dried=90.0,
        valid_from="2026-04-01", valid_to="2026-12-31",
    )
    items = [{"material_id": ids["钾长石"], "amount": 90.0}]
    feb = client.post("/compute", json={
        "items": items, "lots": {str(ids["钾长石"]): "L1"},
        "moisture_as_of": "2026-02-01",
    }).json()["moisture"]["lines"][0]
    may = client.post("/compute", json={
        "items": items, "lots": {str(ids["钾长石"]): "L1"},
        "moisture_as_of": "2026-05-01",
    }).json()["moisture"]["lines"][0]
    assert feb["moisture_fraction"] == pytest.approx(0.05)
    assert feb["wet_kg"] == pytest.approx(90 / 0.95)
    assert may["moisture_fraction"] == pytest.approx(0.10)
    assert may["wet_kg"] == pytest.approx(100.0)
    # 区间外无有效测定 -> 按干料
    out = client.post("/compute", json={
        "items": items, "lots": {str(ids["钾长石"]): "L1"},
        "moisture_as_of": "2027-01-01",
    }).json()["moisture"]["lines"][0]
    assert out["basis"] == "dry"
    assert out["reason"] == "no_effective_measurement"


# ---------------------------------------------------------------------------
# /compute：干基化学 + 湿料称量换算
# ---------------------------------------------------------------------------

def test_compute_wet_weighing_water_stock_cost(client):
    ids = _ids(client)
    kf, cc = ids["钾长石"], ids["方解石"]
    _add_measurement(client, kf, lot="K1", sample=110.0, dried=99.0)  # 10%
    mats = {m["id"]: m for m in client.get("/materials").json()}
    r = client.post("/compute", json={
        "items": [
            {"material_id": kf, "amount": 90.0},
            {"material_id": cc, "amount": 10.0},
        ],
        "lots": {str(kf): "K1"},
        "moisture_as_of": TODAY,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    # 化学分析仍按干料
    assert body["batch_mass"] == pytest.approx(100.0)
    mp = body["moisture"]
    by = {l["material_id"]: l for l in mp["lines"]}
    assert by[kf]["basis"] == "wet"
    assert by[kf]["wet_kg"] == pytest.approx(100.0)
    assert by[kf]["carried_water_kg"] == pytest.approx(10.0)
    assert by[kf]["cost"] == pytest.approx(100.0 * mats[kf]["price"])
    assert by[kf]["stock_used_kg"] == pytest.approx(100.0)
    assert by[kf]["stock_remaining_kg"] == pytest.approx(mats[kf]["available"] - 100.0)
    # 未指定批号的方解石按干料，明确标注
    assert by[cc]["basis"] == "dry"
    assert by[cc]["reason"] == "lot_not_specified"
    assert by[cc]["wet_kg"] == pytest.approx(10.0)
    assert by[cc]["carried_water_kg"] == 0.0
    assert mp["totals"]["wet_kg"] == pytest.approx(110.0)
    assert mp["totals"]["carried_water_kg"] == pytest.approx(10.0)
    assert mp["totals"]["cost"] == pytest.approx(
        100.0 * mats[kf]["price"] + 10.0 * mats[cc]["price"]
    )


def test_compute_missing_effective_measurement_marked_dry(client):
    ids = _ids(client)
    kf = ids["钾长石"]
    r = client.post("/compute", json={
        "items": [{"material_id": kf, "amount": 10.0}],
        "lots": {str(kf): "GHOST"},
        "moisture_as_of": TODAY,
    })
    mp = r.json()["moisture"]
    line = mp["lines"][0]
    assert line["basis"] == "dry"
    assert line["reason"] == "no_effective_measurement"
    assert [u["material_id"] for u in mp["unresolved"]] == [kf]
    assert mp["totals"]["wet_kg"] == pytest.approx(10.0)


def test_compute_stock_shortfall_flagged(client):
    ids = _ids(client)
    kf = ids["钾长石"]
    _add_measurement(client, kf, lot="K1", sample=110.0, dried=99.0)  # 10%
    # 库存 500 kg 湿料：干料 460 kg 已需要湿料 > 500
    r = client.post("/compute", json={
        "items": [{"material_id": kf, "amount": 460.0}],
        "lots": {str(kf): "K1"},
        "moisture_as_of": TODAY,
    })
    line = r.json()["moisture"]["lines"][0]
    assert line["within_stock"] is False
    assert r.json()["moisture"]["stock_shortfall"][0]["material_id"] == kf


def test_compute_lot_outside_items_rejected(client):
    ids = _ids(client)
    r = client.post("/compute", json={
        "items": [{"material_id": ids["钾长石"], "amount": 10.0}],
        "lots": {str(ids["石英"]): "Q1"},
    })
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "moisture_lot_not_in_items"


# ---------------------------------------------------------------------------
# /search：选批换算库存与成本，缺测定按干料
# ---------------------------------------------------------------------------

def test_search_lots_wet_plan_and_unresolved(client):
    ids = _ids(client)
    kf, cc, zn = ids["钾长石"], ids["方解石"], ids["氧化锌"]
    _add_measurement(client, kf, lot="K1", sample=110.0, dried=99.0)
    r = client.post("/search", json={
        "batch_size": 100.0, "step": 0.5, "batch_tolerance": 0.5,
        "targets": {
            "K2O": {"low": 0.15, "high": 0.35},
            "CaO": {"low": 0.35, "high": 0.6},
            "Al2O3": {"low": 0.3, "high": 0.5},
            "SiO2": {"low": 2.5, "high": 3.5},
        },
        "forbidden": [zn],
        "lots": {str(kf): "K1", str(cc): "MISSING"},
        "moisture_as_of": TODAY,
    })
    assert r.status_code == 200, r.text
    sr = r.json()
    assert sr["status"] == "optimal"
    assert sr["moisture"]["resolved_lots"] == [kf]
    assert [u["material_id"] for u in sr["moisture"]["unresolved"]] == [cc]
    best = sr["best"]
    by = {l["material_id"]: l for l in best["moisture"]["lines"]}
    assert by[kf]["basis"] == "wet"
    assert by[kf]["wet_kg"] == pytest.approx(by[kf]["dry_kg"] / 0.9)
    # 化学成本仍按干料，湿料成本单独给出且更大
    assert best["cost_wet"] > best["cost"]
    # 搜索约束回传含批号，客户端可原样冻结
    assert sr["search_constraints"]["lots"][str(kf)] == "K1"


# ---------------------------------------------------------------------------
# 冻结：湿料称量方案随版本不可变
# ---------------------------------------------------------------------------

def _three_item_freeze(client, ids, lots=None):
    items = [
        {"material_id": ids["钾长石"], "amount": 60.0},
        {"material_id": ids["方解石"], "amount": 20.0},
        {"material_id": ids["石英"], "amount": 20.0},
    ]
    return _freeze(
        client, items, lots=lots, moisture_as_of=TODAY
    ) if lots is not None else _freeze(client, items)


def test_freeze_with_lots_stores_plan_and_is_idempotent(client):
    ids = _ids(client)
    kf = ids["钾长石"]
    _add_measurement(client, kf, lot="K1", sample=110.0, dried=99.0)
    r = _three_item_freeze(client, ids, lots={str(kf): "K1"})
    assert r.status_code == 201, r.text
    v = r.json()["version"]
    plan = v["moisture_plan"]
    by = {l["material_id"]: l for l in plan["lines"]}
    assert by[kf]["basis"] == "wet"
    assert by[kf]["measurement"]["lot"] == "K1"
    # 干基化学结果不因含水改变
    assert v["result"]["batch_mass"] == pytest.approx(100.0)

    r2 = _three_item_freeze(client, ids, lots={str(kf): "K1"})
    assert r2.json()["version"]["id"] == v["id"]
    assert r2.json()["created"] is False


def test_freeze_without_lots_has_no_plan_and_stable_hash(client):
    ids = _ids(client)
    kf = ids["钾长石"]
    _add_measurement(client, kf, lot="K1", sample=110.0, dried=99.0)
    r1 = _three_item_freeze(client, ids)
    r2 = _three_item_freeze(client, ids)
    assert r1.json()["version"]["moisture_plan"] is None
    assert r1.json()["version"]["id"] == r2.json()["version"]["id"]
    # 选批版本与干料版本是两个不同版本
    r3 = _three_item_freeze(client, ids, lots={str(kf): "K1"})
    assert r3.json()["version"]["id"] != r1.json()["version"]["id"]


def test_frozen_version_unaffected_by_later_measurement(client):
    ids = _ids(client)
    kf = ids["钾长石"]
    _add_measurement(client, kf, lot="K1", sample=110.0, dried=99.0)  # 10%
    v = _three_item_freeze(client, ids, lots={str(kf): "K1"}).json()["version"]
    frozen_fraction = v["moisture_plan"]["lines"][0]["moisture_fraction"]

    # 新测定只影响后续（同日重叠会被拒，改为次年新区间，验证已冻结版本不变）
    _add_measurement(
        client, kf, lot="K1", sample=100.0, dried=80.0,
        valid_from="2027-01-01", valid_to="2027-12-31",
    )
    got = client.get(f"/versions/{v['id']}").json()
    assert got["moisture_plan"]["lines"][0]["moisture_fraction"] == frozen_fraction
    assert got["result"]["seger"] == v["result"]["seger"]


# ---------------------------------------------------------------------------
# 釉浆批次：初始加水扣除原料自带水，快照隔离
# ---------------------------------------------------------------------------

def _slurry_payload(version_id, **overrides):
    payload = {
        "version_id": version_id,
        "target_dry_mass_kg": 10.0,
        "solids_low": 0.60, "solids_high": 0.70,
        "density_low": 1.50, "density_high": 1.70,
        "powder_true_density_kg_l": 2.6,
        "water_temp_c": 20.0,
        "container_capacity_l": 20.0,
        "additive_ratio": 0.0,
    }
    payload.update(overrides)
    return payload


def test_slurry_initial_water_deducts_carried_moisture(client):
    ids = _ids(client)
    kf = ids["钾长石"]
    _add_measurement(client, kf, lot="K1", sample=110.0, dried=99.0)  # 10%
    vid = _three_item_freeze(client, ids, lots={str(kf): "K1"}).json()["version"]["id"]
    r = client.post("/slurry-batches", json=_slurry_payload(vid))
    assert r.status_code == 201, r.text
    plan = r.json()["initial_plan"]
    by = {l["material_id"]: l for l in plan["materials"]}
    # 钾长石需干料 6000 g：湿料 6666.667 g，带入 666.667 g 水
    assert by[kf]["weighing_basis"] == "wet"
    assert by[kf]["wet_mass_g"] == pytest.approx(6000.0 / 0.9)
    assert by[kf]["carried_water_g"] == pytest.approx(6000.0 * 0.1 / 0.9)
    assert plan["carried_water_g"] == pytest.approx(6000.0 * 0.1 / 0.9)
    total_water = 10000.0 / 0.65 - 10000.0
    assert plan["initial_total_water_g"] == pytest.approx(total_water)
    assert plan["initial_water_g"] == pytest.approx(
        total_water - plan["carried_water_g"]
    )
    assert plan["weighed_wet_mass_g"] == pytest.approx(
        10000.0 + plan["carried_water_g"]
    )


def test_slurry_negative_net_water_lists_contributors(client):
    ids = _ids(client)
    kf = ids["钾长石"]
    _add_measurement(client, kf, lot="WET", sample=100.0, dried=70.0)  # 30%
    vid = _three_item_freeze(client, ids, lots={str(kf): "WET"}).json()["version"]["id"]
    # 固含率 96~98%：总用水 ≈ 10000/0.97-10000 ≈ 309 g，
    # 钾长石 6000 g 干料带入 6000*0.3/0.7 ≈ 2571 g -> 净加水为负
    r = client.post("/slurry-batches", json=_slurry_payload(
        vid, solids_low=0.96, solids_high=0.98, density_high=2.5
    ))
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "moisture_water_exceeds_plan"
    assert err["details"]["excess_water_g"] == pytest.approx(
        err["details"]["carried_water_g"] - err["details"]["initial_total_water_g"],
        abs=1e-6,
    )
    contributors = err["details"]["contributors"]
    assert contributors[0]["material_id"] == kf
    assert "钾长石" in contributors[0]["name"]


def test_existing_slurry_batch_isolated_from_new_measurements(client):
    ids = _ids(client)
    kf = ids["钾长石"]
    _add_measurement(client, kf, lot="K1", sample=110.0, dried=99.0)
    vid = _three_item_freeze(client, ids, lots={str(kf): "K1"}).json()["version"]["id"]
    bid = client.post(
        "/slurry-batches", json=_slurry_payload(vid)
    ).json()["id"]
    net_before = client.get(f"/slurry-batches/{bid}").json()[
        "initial_plan"
    ]["initial_water_g"]

    # 新批号测定（不影响已冻结版本与已建批次）
    _add_measurement(
        client, kf, lot="K2", sample=100.0, dried=80.0,
        valid_from="2026-01-01", valid_to="2026-12-31",
    )
    net_after = client.get(f"/slurry-batches/{bid}").json()[
        "initial_plan"
    ]["initial_water_g"]
    assert net_after == net_before
