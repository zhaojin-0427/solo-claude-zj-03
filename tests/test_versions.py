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


def test_version_freezes_full_constants_snapshot(client):
    """版本必须自带分子量与氧化物角色，可独立还原釉式。"""
    ids = _id_map(client)
    v = _freeze(client, {"items": [
        {"material_id": ids["钾长石"], "amount": 50.0},
        {"material_id": ids["方解石"], "amount": 20.0},
        {"material_id": ids["石英"], "amount": 30.0},
    ]}).json()["version"]

    snap = v["constants_snapshot"]
    assert snap["constants_version"] == v["constants_version"]
    assert snap["oxides"]["SiO2"]["molwt"]
    assert snap["oxides"]["K2O"]["role"] == "flux"
    assert snap["oxides"]["Al2O3"]["role"] == "amphoteric"
    assert snap["oxides"]["SiO2"]["role"] == "acid"
    # 用快照自带的分子量独立重算 SiO2 釉式，应与冻结结果一致
    items = {i["material_id"]: i["amount"] for i in v["items"]}
    moles = {}
    flux = 0.0
    for mid, amt in items.items():
        mat = v["material_snapshot"][str(mid)]
        for oxide, pct in mat["oxides"].items():
            n = amt * pct / 100.0 / snap["oxides"][oxide]["molwt"]
            moles[oxide] = moles.get(oxide, 0.0) + n
            if snap["oxides"][oxide]["role"] == "flux":
                flux += n
    import pytest
    assert moles["SiO2"] / flux == pytest.approx(
        v["result"]["seger"]["SiO2"], rel=1e-7
    )


def test_legacy_db_migration_backfills_constants_snapshot(client):
    """旧 schema（无 constants_snapshot 列）的版本在迁移后仍可读全量常量。"""
    import sqlite3
    from app import config, db

    path = config.settings.db_path
    # 直接以旧表结构插一条版本
    with sqlite3.connect(path) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS legacy_v (id TEXT)"""
        )
        # 删除新列重建旧结构不可行（SQLite 限制），改为手动建旧表并复制逻辑：
        # 这里直接验证 _migrate 对“缺列的真实旧库”的行为——新建一个隔离旧库。
    import tempfile
    old = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    old.close()
    old_path = old.name
    with sqlite3.connect(old_path) as conn:
        conn.executescript(
            """
            CREATE TABLE recipe_versions (
                id TEXT PRIMARY KEY, input_hash TEXT UNIQUE, items TEXT NOT NULL,
                note TEXT, material_snapshot TEXT NOT NULL, constraints TEXT NOT NULL,
                constants_version TEXT NOT NULL, result TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );
            INSERT INTO recipe_versions VALUES
              ('abc123','abc123','[]',NULL,'{}','{}','old-ver','{}',
               datetime('now'));
            CREATE TABLE materials (id INTEGER PRIMARY KEY);
            """
        )
    config.settings.db_path = old_path
    try:
        db.init_db()  # 触发迁移
        v = db.get_version("abc123")
        assert v["constants_version"] == "old-ver"
        # 回填：凭版本自身即可拿到分子量与角色
        assert v["constants_snapshot"]["oxides"]["SiO2"]["molwt"]
        assert v["constants_snapshot"]["oxides"]["K2O"]["role"] == "flux"
    finally:
        config.settings.db_path = path
        import os
        os.unlink(old_path)


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
