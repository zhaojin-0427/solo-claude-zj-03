"""釉浆调制批次：创建称量计划、台账守恒、读数闭合、回收浆、纠偏搜索与定稿冻结。

配方（100 kg 基准，三种原料）：
  钾长石 60 / 方解石 20 / 石英 20
批次按目标干料 10 kg、目标固含率 60~70%、比重 1.5~1.7、粉真密度 2.6 kg/L、
水温 20 °C（ρ_water≈0.99821 g/mL）、容器 20 L、添加剂 0.5% 创建。
"""
from __future__ import annotations

import pytest


def _ids(client):
    return {m["name"]: m["id"] for m in client.get("/materials").json()}


def _freeze(client, ids):
    resp = client.post("/versions", json={"items": [
        {"material_id": ids["钾长石"], "amount": 60.0},
        {"material_id": ids["方解石"], "amount": 20.0},
        {"material_id": ids["石英"], "amount": 20.0},
    ]})
    assert resp.status_code == 201, resp.text
    return resp.json()["version"]["id"]


def _create_payload(version_id, **overrides):
    payload = {
        "version_id": version_id,
        "target_dry_mass_kg": 10.0,
        "solids_low": 0.60,
        "solids_high": 0.70,
        "density_low": 1.50,
        "density_high": 1.70,
        "powder_true_density_kg_l": 2.6,
        "water_temp_c": 20.0,
        "container_capacity_l": 20.0,
        "additive_ratio": 0.005,
    }
    payload.update(overrides)
    return payload


def _make_batch(client, **overrides):
    ids = _ids(client)
    vid = _freeze(client, ids)
    resp = client.post("/slurry-batches", json=_create_payload(vid, **overrides))
    return resp, ids, vid


def _start(client, batch_id):
    resp = client.post(f"/slurry-batches/{batch_id}/start")
    assert resp.status_code == 201, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# 创建与初始称量
# ---------------------------------------------------------------------------

def test_create_slurry_batch_initial_plan(client):
    resp, ids, _ = _make_batch(client, note="首批")
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["status"] == "planned"
    assert body["note"] == "首批"
    plan = body["initial_plan"]
    # 各原料按配方份额给出：6000 / 2000 / 2000 g
    by_mid = {m["material_id"]: m for m in plan["materials"]}
    assert by_mid[ids["钾长石"]]["mass_g"] == pytest.approx(6000.0)
    assert by_mid[ids["方解石"]]["mass_g"] == pytest.approx(2000.0)
    assert by_mid[ids["石英"]]["mass_g"] == pytest.approx(2000.0)
    # 添加剂 = 10000 * 0.5% = 50 g
    assert plan["initial_additive_g"] == pytest.approx(50.0)
    # 初始水按固含率区间中点 65% 反推：10000/0.65 - 10000 - 50
    assert plan["initial_water_g"] == pytest.approx(10000 / 0.65 - 10050)
    # 初始点固含率恰为区间中点、比重在区间内、体积不超容器
    state = body["state"]
    assert state["dry_mass_g"] == 0.0  # 尚未开始登记
    assert plan["fits_container"] is True
    assert body["constants"]["water_density_g_ml"] == pytest.approx(0.99821)


def test_create_slurry_batch_validation_errors(client):
    _, _, vid = _make_batch(client)
    # 区间矛盾
    r = client.post("/slurry-batches", json=_create_payload(vid, solids_low=0.8, solids_high=0.6))
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "contradictory_bounds"
    # 水温越界
    r = client.post("/slurry-batches", json=_create_payload(vid, water_temp_c=120))
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "water_temp_out_of_range"
    # 添加剂比例过高导致初始水为负（固含率中点 65% 时添加剂上限约 53.8%）
    r = client.post("/slurry-batches", json=_create_payload(vid, additive_ratio=0.7))
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "additive_ratio_infeasible"
    # 来源版本不存在
    r = client.post("/slurry-batches", json=_create_payload("nope"))
    assert r.status_code == 404


def test_create_slurry_batch_over_capacity(client):
    # 容器只有 2 L，装不下 10 kg 干粉（仅粉体就占 3.85 L）
    resp, _, _ = _make_batch(client, container_capacity_l=2.0)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "planned_over_capacity"


def test_start_is_required_before_entries(client):
    resp, _, _ = _make_batch(client)
    bid = resp.json()["id"]
    r = client.post(f"/slurry-batches/{bid}/additions", json={
        "dry_materials": [], "water_g": 100.0,
    })
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_slurry_status"


# ---------------------------------------------------------------------------
# 台账登记与质量守恒
# ---------------------------------------------------------------------------

def test_additions_mass_balance_and_metrics(client):
    resp, ids, _ = _make_batch(client)
    bid = resp.json()["id"]
    _start(client, bid)
    r = client.post(f"/slurry-batches/{bid}/additions", json={
        "dry_materials": [
            {"material_id": ids["钾长石"], "mass_g": 6000.0},
            {"material_id": ids["方解石"], "mass_g": 2000.0},
            {"material_id": ids["石英"], "mass_g": 2000.0},
        ],
        "water_g": 5000.0,
        "additive_g": 50.0,
    })
    assert r.status_code == 201, r.text
    state = r.json()["state"]
    assert state["dry_mass_g"] == pytest.approx(10000.0)
    assert state["water_mass_g"] == pytest.approx(5000.0)
    assert state["additive_mass_g"] == pytest.approx(50.0)
    assert state["total_mass_g"] == pytest.approx(15050.0)
    # 固含率 = 10000 / 15050
    assert state["solids_fraction"] == pytest.approx(10000.0 / 15050.0)
    # 体积 = 10000/2.6 + 5050/0.99821
    expected_v = 10000.0 / 2.6 + 5050.0 / 0.99821
    assert state["occupied_volume_ml"] == pytest.approx(expected_v, rel=1e-6)
    assert state["theoretical_density_g_ml"] == pytest.approx(
        15050.0 / expected_v, rel=1e-6
    )
    assert state["target_status"]["all_targets_met"] is True
    assert state["breakdown"]["dry_g"]["direct"] == pytest.approx(10000.0)
    # 台账序号
    entries = r.json()["entries"]
    assert [e["seq"] for e in entries] == [1]
    assert r.json()["n_entries"] == 1


def test_addition_rejects_foreign_material_and_empty(client):
    resp, ids, _ = _make_batch(client)
    bid = resp.json()["id"]
    _start(client, bid)
    # 高岭土不在配方里
    r = client.post(f"/slurry-batches/{bid}/additions", json={
        "dry_materials": [{"material_id": ids["高岭土"], "mass_g": 100.0}],
    })
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "material_not_in_recipe"
    # 空登记
    r = client.post(f"/slurry-batches/{bid}/additions", json={})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "empty_addition"


def test_container_overfill_warning(client):
    # 默认 20 L 容器：20 kg 干粉（7.69 L）+ 13 kg 水（13.02 L）超过 20 L
    resp, ids, _ = _make_batch(client)
    bid = resp.json()["id"]
    _start(client, bid)
    r = client.post(f"/slurry-batches/{bid}/additions", json={
        "dry_materials": [{"material_id": ids["钾长石"], "mass_g": 20000.0}],
        "water_g": 13000.0,
    })
    assert r.status_code == 201, r.text
    codes = [w["code"] for w in r.json()["warnings"]]
    assert "container_overfill" in codes


# ---------------------------------------------------------------------------
# 比重杯读数
# ---------------------------------------------------------------------------

def test_reading_closed_when_matching_theory(client):
    resp, ids, _ = _make_batch(client)
    bid = resp.json()["id"]
    _start(client, bid)
    client.post(f"/slurry-batches/{bid}/additions", json={
        "dry_materials": [{"material_id": ids["钾长石"], "mass_g": 10000.0}],
        "water_g": 10000 / 0.65 - 10000 - 50,
        "additive_g": 50.0,
    })
    state = client.get(f"/slurry-batches/{bid}").json()["state"]
    rho = state["theoretical_density_g_ml"]
    # 100 mL 杯：满杯-空杯 = 100 * rho
    r = client.post(f"/slurry-batches/{bid}/readings", json={
        "empty_cup_mass_g": 80.0,
        "full_cup_mass_g": 80.0 + 100.0 * rho,
        "cup_volume_ml": 100.0,
    })
    assert r.status_code == 201, r.text
    reading = r.json()["entries"][-1]
    assert reading["state_after"]["closed"] is True
    assert reading["state_after"]["difference_g_ml"] <= 2e-9
    assert reading["state_after"]["implied_solids_fraction"] == pytest.approx(
        state["solids_fraction"], abs=1e-3
    )


def test_reading_not_closed_warning(client):
    resp, ids, _ = _make_batch(client)
    bid = resp.json()["id"]
    _start(client, bid)
    client.post(f"/slurry-batches/{bid}/additions", json={
        "dry_materials": [{"material_id": ids["钾长石"], "mass_g": 10000.0}],
        "water_g": 5300.0,
        "additive_g": 50.0,
    })
    r = client.post(f"/slurry-batches/{bid}/readings", json={
        "empty_cup_mass_g": 80.0,
        "full_cup_mass_g": 220.0,  # 实测比重 1.40
        "cup_volume_ml": 100.0,
    })
    assert r.status_code == 201, r.text
    reading = r.json()["entries"][-1]
    assert reading["state_after"]["closed"] is False
    assert reading["state_after"]["measured_density_g_ml"] == pytest.approx(1.4)
    codes = [w["code"] for w in r.json()["warnings"]]
    assert "reading_not_closed" in codes


def test_reading_inverted_rejected(client):
    resp, _, _ = _make_batch(client)
    bid = resp.json()["id"]
    _start(client, bid)
    r = client.post(f"/slurry-batches/{bid}/readings", json={
        "empty_cup_mass_g": 100.0, "full_cup_mass_g": 90.0, "cup_volume_ml": 100.0,
    })
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "cup_reading_inverted"


# ---------------------------------------------------------------------------
# 预混粉与回收浆
# ---------------------------------------------------------------------------

def _finalized_batch(client, ids, vid, water_g=5300.0, additive_g=50.0):
    r = client.post("/slurry-batches", json=_create_payload(vid))
    bid = r.json()["id"]
    _start(client, bid)
    client.post(f"/slurry-batches/{bid}/additions", json={
        "dry_materials": [
            {"material_id": ids["钾长石"], "mass_g": 6000.0},
            {"material_id": ids["方解石"], "mass_g": 2000.0},
            {"material_id": ids["石英"], "mass_g": 2000.0},
        ],
        "water_g": water_g,
        "additive_g": additive_g,
    })
    fr = client.post(f"/slurry-batches/{bid}/finalize")
    assert fr.status_code == 201, fr.text
    return bid, fr.json()["freeze"]


def test_premix_addition_and_breakdown(client):
    resp, ids, _ = _make_batch(client)
    bid = resp.json()["id"]
    _start(client, bid)
    r = client.post(f"/slurry-batches/{bid}/premix", json={
        "premix_mass_g": 1000.0, "water_g": 200.0,
    })
    assert r.status_code == 201, r.text
    entry = r.json()["entries"][-1]
    assert entry["entry_type"] == "premix"
    by_mid = {b["material_id"]: b for b in entry["payload"]["breakdown"]}
    assert by_mid[ids["钾长石"]]["mass_g"] == pytest.approx(600.0)
    assert by_mid[ids["石英"]]["mass_g"] == pytest.approx(200.0)
    state = r.json()["state"]
    assert state["dry_mass_g"] == pytest.approx(1000.0)
    assert state["breakdown"]["dry_g"]["premix"] == pytest.approx(1000.0)
    assert state["water_mass_g"] == pytest.approx(200.0)


def test_recycle_same_version_split(client):
    ids = _ids(client)
    vid = _freeze(client, ids)
    src_bid, _ = _finalized_batch(client, ids, vid)
    # 新批次加入 1505 g 回收浆（来源批次总质量 15350 g 的 1/10）
    r = client.post("/slurry-batches", json=_create_payload(vid))
    bid = r.json()["id"]
    _start(client, bid)
    rr = client.post(f"/slurry-batches/{bid}/recycles", json={
        "source_batch_id": src_bid, "slurry_mass_g": 1535.0,
    })
    assert rr.status_code == 201, rr.text
    state = rr.json()["state"]
    assert state["dry_mass_g"] == pytest.approx(1000.0)
    assert state["water_mass_g"] == pytest.approx(530.0)
    assert state["additive_mass_g"] == pytest.approx(5.0)
    assert state["breakdown"]["dry_g"]["recycle"] == pytest.approx(1000.0)
    entry = rr.json()["entries"][-1]
    assert entry["payload"]["source_version_id"] == vid


def test_recycle_version_mismatch_rejected(client):
    ids = _ids(client)
    vid_a = _freeze(client, ids)
    src_bid, _ = _finalized_batch(client, ids, vid_a)
    # 另一份配方（不同原料组成 -> 不同版本）
    vid_b = client.post("/versions", json={"items": [
        {"material_id": ids["钠长石"], "amount": 50.0},
        {"material_id": ids["高岭土"], "amount": 20.0},
        {"material_id": ids["石英"], "amount": 30.0},
    ]}).json()["version"]["id"]
    r = client.post("/slurry-batches", json=_create_payload(vid_b))
    bid = r.json()["id"]
    _start(client, bid)
    rr = client.post(f"/slurry-batches/{bid}/recycles", json={
        "source_batch_id": src_bid, "slurry_mass_g": 100.0,
    })
    assert rr.status_code == 422
    assert rr.json()["error"]["code"] == "recycle_version_mismatch"


def test_recycle_from_unfinalized_rejected(client):
    resp, _, _ = _make_batch(client)
    src = resp.json()["id"]
    _start(client, src)
    resp2, _, _ = _make_batch(client)
    bid = resp2.json()["id"]
    _start(client, bid)
    rr = client.post(f"/slurry-batches/{bid}/recycles", json={
        "source_batch_id": src, "slurry_mass_g": 100.0,
    })
    assert rr.status_code == 422
    assert rr.json()["error"]["code"] == "recycle_source_not_finalized"


# ---------------------------------------------------------------------------
# 纠偏搜索
# ---------------------------------------------------------------------------

def test_correction_search_thin_slurry_add_premix(client):
    # 浆液偏稀：固含率 0.50 低于 0.60，比重也偏低 -> 需要加预混粉
    resp, ids, _ = _make_batch(client)
    bid = resp.json()["id"]
    _start(client, bid)
    client.post(f"/slurry-batches/{bid}/additions", json={
        "dry_materials": [
            {"material_id": ids["钾长石"], "mass_g": 5000.0},
            {"material_id": ids["方解石"], "mass_g": 1000.0},
        ],
        "water_g": 6000.0,
    })
    r = client.post(f"/slurry-batches/{bid}/correction-search", json={
        "water_step_g": 100.0, "premix_step_g": 100.0,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["best"]["premix_g"] > 0.0
    plan = body["best"]
    # 最优方案结果落入目标区间
    assert 0.60 <= plan["resulting_solids_fraction"] <= 0.70
    assert 1.50 <= plan["resulting_density_g_ml"] <= 1.70
    # 排序：按固含率偏差、比重偏差、新增质量（可行方案偏差均为 0）
    masses = [p["added_mass_g"] for p in body["plans"]]
    assert masses == sorted(masses)


def test_correction_search_thick_slurry_add_water(client):
    # 浆液偏稠：固含率 0.80 高于 0.70 -> 需要加水
    resp, ids, _ = _make_batch(client)
    bid = resp.json()["id"]
    _start(client, bid)
    client.post(f"/slurry-batches/{bid}/additions", json={
        "dry_materials": [
            {"material_id": ids["钾长石"], "mass_g": 6000.0},
            {"material_id": ids["方解石"], "mass_g": 2000.0},
            {"material_id": ids["石英"], "mass_g": 2000.0},
        ],
        "water_g": 2450.0,
        "additive_g": 50.0,
    })
    r = client.post(f"/slurry-batches/{bid}/correction-search", json={
        "water_step_g": 50.0, "premix_step_g": 500.0,
        "remaining_capacity_ml": 3000.0,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["best"]["water_g"] > 0.0
    assert body["remaining_capacity_ml"] <= 3000.0 + 1e-9
    for p in body["plans"]:
        assert p["added_volume_ml"] <= 3000.0 + 1e-6


def test_correction_search_unreachable_returns_best_effort(client):
    # 浆液固含率 0.90 远超上限 0.70，只剩 10 mL 空余，加水纠正装不下
    resp, ids, _ = _make_batch(client)
    bid = resp.json()["id"]
    _start(client, bid)
    # 9000 g 粉占 3.462 L，水 1000 g 占 1.002 L，共约 4.46 L，余 ~15.5 L
    # 把剩余容量限定到 10 mL：需要约 3.86 kg 水才能把固含率拉回 0.70
    client.post(f"/slurry-batches/{bid}/additions", json={
        "dry_materials": [{"material_id": ids["钾长石"], "mass_g": 9000.0}],
        "water_g": 1000.0,
    })
    r = client.post(f"/slurry-batches/{bid}/correction-search", json={
        "water_step_g": 100.0, "premix_step_g": 100.0,
        "remaining_capacity_ml": 10.0,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "target_unreachable"
    assert body["best"] is None
    assert body["best_effort"] is not None
    # 10 mL 只能加约 9 g 水（步进 100 g），网格上唯一候选是零动作
    assert body["best_effort"]["water_g"] == 0.0
    assert body["best_effort"]["premix_g"] == 0.0


def test_correction_grid_too_fine_rejected(client):
    resp, ids, _ = _make_batch(client)
    bid = resp.json()["id"]
    _start(client, bid)
    client.post(f"/slurry-batches/{bid}/additions", json={
        "dry_materials": [{"material_id": ids["钾长石"], "mass_g": 100.0}],
        "water_g": 100.0,
    })
    r = client.post(f"/slurry-batches/{bid}/correction-search", json={
        "water_step_g": 0.001, "premix_step_g": 0.001,
    })
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "correction_grid_too_fine"


def test_adjustment_recorded_applies_correction(client):
    resp, ids, _ = _make_batch(client)
    bid = resp.json()["id"]
    _start(client, bid)
    client.post(f"/slurry-batches/{bid}/additions", json={
        "dry_materials": [
            {"material_id": ids["钾长石"], "mass_g": 5000.0},
            {"material_id": ids["方解石"], "mass_g": 1000.0},
        ],
        "water_g": 6000.0,
    })
    search = client.post(f"/slurry-batches/{bid}/correction-search", json={
        "water_step_g": 100.0, "premix_step_g": 100.0,
    }).json()
    plan = search["best"]
    r = client.post(f"/slurry-batches/{bid}/adjustments", json={
        "water_g": plan["water_g"], "premix_g": plan["premix_g"],
        "note": "按搜索 best 执行",
    })
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["state"]["target_status"]["all_targets_met"] is True
    entry = body["entries"][-1]
    assert entry["entry_type"] == "adjustment"
    assert entry["payload"]["note"] == "按搜索 best 执行"
    # 调整预混粉按配方份额拆分
    assert sum(b["mass_g"] for b in entry["payload"]["premix_breakdown"]) == \
        pytest.approx(plan["premix_g"])


def test_empty_adjustment_rejected(client):
    resp, _, _ = _make_batch(client)
    bid = resp.json()["id"]
    _start(client, bid)
    r = client.post(f"/slurry-batches/{bid}/adjustments", json={})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "empty_adjustment"


# ---------------------------------------------------------------------------
# 定稿冻结
# ---------------------------------------------------------------------------

def test_finalize_freezes_everything(client):
    ids = _ids(client)
    vid = _freeze(client, ids)
    bid, freeze = _finalized_batch(client, ids, vid)
    assert freeze["batch_id"] == bid
    snap = freeze["snapshot"]
    assert snap["version_id"] == vid
    assert len(snap["entries"]) == 1
    assert snap["constants"]["powder_true_density_g_ml"] == 2.6
    assert snap["constants"]["water_temp_c"] == 20.0
    assert snap["constants"]["density_closure_tolerance_g_ml"] == 0.02
    # 水密度表整体入档
    assert snap["constants"]["water_density_table"][1] == {
        "temp_c": 4.0, "density_g_ml": 1.0
    }
    assert snap["constants"]["segger_constants"]["constants_version"]
    fs = freeze["final_state"]
    assert fs["dry_mass_g"] == pytest.approx(10000.0)
    # 批次状态翻转
    view = client.get(f"/slurry-batches/{bid}").json()
    assert view["status"] == "finalized"
    assert view["freeze_id"] == freeze["id"]
    assert view["finalized_at"] is not None
    # GET 冻结记录
    got = client.get(f"/slurry-freezes/{freeze['id']}")
    assert got.status_code == 200
    assert got.json()["id"] == freeze["id"]


def test_finalize_idempotent_same_result(client):
    ids = _ids(client)
    vid = _freeze(client, ids)
    bid, freeze1 = _finalized_batch(client, ids, vid)
    r2 = client.post(f"/slurry-batches/{bid}/finalize")
    assert r2.status_code == 201
    body = r2.json()
    assert body["created"] is False
    assert body["freeze"]["id"] == freeze1["id"]
    # 带不同备注重复定稿仍返回同一冻结（幂等优先于备注）
    r3 = client.post(f"/slurry-batches/{bid}/finalize", json={"note": "又定稿一次"})
    assert r3.json()["freeze"]["id"] == freeze1["id"]


def test_finalize_rejects_empty_batch(client):
    resp, _, _ = _make_batch(client)
    bid = resp.json()["id"]
    _start(client, bid)
    r = client.post(f"/slurry-batches/{bid}/finalize")
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "empty_slurry_batch"


def test_finalized_batch_is_locked(client):
    ids = _ids(client)
    vid = _freeze(client, ids)
    bid, _ = _finalized_batch(client, ids, vid)
    r = client.post(f"/slurry-batches/{bid}/additions", json={"water_g": 1.0})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_slurry_status"
    r = client.post(f"/slurry-batches/{bid}/readings", json={
        "empty_cup_mass_g": 1.0, "full_cup_mass_g": 2.0, "cup_volume_ml": 1.0,
    })
    assert r.status_code == 422
    # 定稿后回收浆仍可用于其他批次（同版本）
    r = client.post("/slurry-batches", json=_create_payload(vid))
    bid2 = r.json()["id"]
    _start(client, bid2)
    rr = client.post(f"/slurry-batches/{bid2}/recycles", json={
        "source_batch_id": bid, "slurry_mass_g": 153.5,
    })
    assert rr.status_code == 201, rr.text
    assert rr.json()["state"]["dry_mass_g"] == pytest.approx(100.0)


def test_planned_cannot_finalize_and_start_idempotent_flow(client):
    resp, _, _ = _make_batch(client)
    bid = resp.json()["id"]
    # planned 直接定稿被拒
    r = client.post(f"/slurry-batches/{bid}/finalize")
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_slurry_status"
    # start 后再 start 被拒（只允许 planned）
    _start(client, bid)
    r = client.post(f"/slurry-batches/{bid}/start")
    assert r.status_code == 422


def test_list_slurry_batches(client):
    b1, _, _ = _make_batch(client)
    b2, _, _ = _make_batch(client)
    rows = client.get("/slurry-batches").json()
    assert len(rows) == 2
    assert {r["id"] for r in rows} == {b1.json()["id"], b2.json()["id"]}
