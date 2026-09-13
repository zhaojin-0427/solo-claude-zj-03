"""不可变配方版本测试：冻结、哈希幂等、快照隔离与重算一致。"""
from __future__ import annotations


def _id_map(client):
    return {m["name"]: m["id"] for m in client.get("/materials").json()}


def _freeze(client, payload):
    return client.post("/versions", json=payload)


def test_freeze_creates_version(client):
    ids = _id_map(client)
    payload = {"items": [
        {"material_id": ids["钾长石"], "amount": 40.0},
        {"material_id": ids["方解石"], "amount": 20.0},
        {"material_id": ids["石英"], "amount": 40.0},
    ], "note": "第一次冻结"}
    resp = _freeze(client, payload)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["created"] is True
    v = body["version"]
    assert len(v["id"]) == 32
    assert v["note"] == "第一次冻结"
    assert v["constants_version"]

    # GET 一致
    got = client.get(f"/versions/{v['id']}").json()
    assert got["result"]["seger"] == v["result"]["seger"]
    assert got["material_snapshot"] == v["material_snapshot"]


def test_freeze_idempotent_same_hash(client):
    ids = _id_map(client)
    payload = {"items": [
        {"material_id": ids["石英"], "amount": 50.0},
        {"material_id": ids["钾长石"], "amount": 50.0},
    ]}
    r1 = _freeze(client, payload).json()
    # 调换顺序、同料同量 → 同一输入哈希
    payload2 = {"items": list(reversed(payload["items"]))}
    r2 = _freeze(client, payload2).json()
    assert r1["version"]["id"] == r2["version"]["id"]
    assert r1["created"] is True
    assert r2["created"] is False


def test_version_immutable_after_material_change(client):
    ids = _id_map(client)
    payload = {"items": [
        {"material_id": ids["钾长石"], "amount": 60.0},
        {"material_id": ids["方解石"], "amount": 15.0},
        {"material_id": ids["石英"], "amount": 25.0},
    ]}
    v = _freeze(client, payload).json()["version"]
    frozen_seger = dict(v["result"]["seger"])
    frozen_snapshot_price = v["material_snapshot"][str(ids["钾长石"])]["price"]

    # 修改原料（价格、分析）
    client.patch(f"/materials/{ids['钾长石']}", json={"price": 999.0})
    got = client.get(f"/versions/{v['id']}").json()
    # 快照保持冻结时的值，结果不变
    assert got["result"]["seger"] == frozen_seger
    assert got["material_snapshot"][str(ids["钾长石"])]["price"] == frozen_snapshot_price
    assert got["result"]["cost"] == v["result"]["cost"]


def test_version_recompute_matches_compute(client):
    ids = _id_map(client)
    items = [
        {"material_id": ids["钾长石"], "amount": 33.0},
        {"material_id": ids["白云石"], "amount": 12.0},
        {"material_id": ids["高岭土"], "amount": 20.0},
        {"material_id": ids["石英"], "amount": 35.0},
    ]
    v = _freeze(client, {"items": items}).json()["version"]
    live = client.post("/compute", json={"items": items}).json()
    assert v["result"]["seger"] == live["seger"]
    assert v["result"]["fired_mass"] == live["fired_mass"]
    assert v["result"]["cost"] == live["cost"]


def test_freeze_from_search_result(client):
    """搜索候选 -> 连同搜索约束冻结 -> 版本存档约束与结果一致。"""
    ids = _id_map(client)
    search_body = {
        "batch_size": 100.0, "step": 0.5, "batch_tolerance": 0.5,
        "targets": {
            "K2O": {"low": 0.15, "high": 0.35},
            "CaO": {"low": 0.35, "high": 0.6},
            "Al2O3": {"low": 0.3, "high": 0.5},
            "SiO2": {"low": 2.5, "high": 3.5},
        },
        "forbidden": [ids["氧化锌"]],
    }
    sr = client.post("/search", json=search_body).json()
    assert sr["status"] == "optimal"
    best = sr["best"]

    resp = _freeze(client, {
        "items": best["items"],
        "search_constraints": sr["search_constraints"],
        "note": "搜索结果冻结",
    })
    assert resp.status_code == 201, resp.text
    v = resp.json()["version"]
    assert v["constraints"]["targets"]["K2O"]["low"] == 0.15
    assert v["constraints"]["forbidden"] == [ids["氧化锌"]]
    assert v["result"]["seger"] == client.post(
        "/compute", json={"items": best["items"]}
    ).json()["seger"]


def test_version_not_found(client):
    assert client.get("/versions/deadbeef").status_code == 404


def test_freeze_negative_amount(client):
    ids = _id_map(client)
    resp = _freeze(client, {"items": [
        {"material_id": ids["石英"], "amount": -1.0},
    ]})
    assert resp.status_code == 422


def test_freeze_snapshot_contains_analysis(client):
    ids = _id_map(client)
    v = _freeze(client, {"items": [
        {"material_id": ids["方解石"], "amount": 100.0},
    ]}).json()["version"]
    snap = v["material_snapshot"][str(ids["方解石"])]
    assert snap["oxides"]["CaO"] > 55.0
    assert snap["loi"] > 40.0
    # 约束存档（直接冻结模式）
    assert v["constraints"]["mode"] == "direct_freeze"
