"""配方混合试验：试验创建校验、逐格舍入、母料拆分搜索与方案冻结。

两份线性来源（均为 100 kg 基准）：
  A = 钾长石 60 / 方解石 20 / 石英 20
  B = 钾长石 40 / 方解石 10 / 白云石 10 / 石英 40
第三份三元来源：
  C = 钠长石 50 / 高岭土 20 / 石英 30
三种原料为两来源共有，母料可同时覆盖钾长石/方解石/石英。
试验取单片干料 100 g、秤分度 0.5 g、最小称量 1 g，故百分比即克数。
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


def _linear_payload(a, b, **overrides):
    # 7 格完整梯度：两端 + 两个四分点 + 三个中点重复
    payload = {
        "sources": [a, b],
        "mode": "linear",
        "cells": [
            {"position": "A", "weights": [1.0, 0.0]},
            {"position": "B", "weights": [0.75, 0.25]},
            {"position": "C1", "weights": [0.5, 0.5]},
            {"position": "C2", "weights": [0.5, 0.5]},
            {"position": "C3", "weights": [0.5, 0.5]},
            {"position": "D", "weights": [0.25, 0.75]},
            {"position": "E", "weights": [0.0, 1.0]},
        ],
        "ratio_step": 0.25,
        "dry_mass_per_tile_g": 100.0,
        "scale_division_g": 0.5,
        "minimum_weighed_g": 1.0,
    }
    payload.update(overrides)
    return payload


def _interior_payload(a, b, **overrides):
    # 5 格内部布局：两种配方各重复（重复试片正是母料节省称量的场景）
    # B(75/25) 两片、C(50/50) 三片；共有料 K/方解石/白云石/石英逐格皆有
    payload = {
        "sources": [a, b],
        "mode": "linear",
        "cells": [
            {"position": "B1", "weights": [0.75, 0.25]},
            {"position": "B2", "weights": [0.75, 0.25]},
            {"position": "C1", "weights": [0.5, 0.5]},
            {"position": "C2", "weights": [0.5, 0.5]},
            {"position": "C3", "weights": [0.5, 0.5]},
        ],
        "ratio_step": 0.25,
        "dry_mass_per_tile_g": 100.0,
        "scale_division_g": 0.5,
        "minimum_weighed_g": 1.0,
    }
    payload.update(overrides)
    return payload


def _create_linear(client, **overrides):
    ids, a, b, c = _sources(client)
    resp = client.post(
        "/blend-experiments", json=_linear_payload(a, b, **overrides)
    )
    return resp, ids, (a, b, c)


def _create_interior(client, **overrides):
    ids, a, b, _ = _sources(client)
    resp = client.post(
        "/blend-experiments", json=_interior_payload(a, b, **overrides)
    )
    return resp, ids, (a, b)


# ---------------------------------------------------------------------------
# 创建试验
# ---------------------------------------------------------------------------

def test_create_blend_experiment_happy(client):
    resp, ids, _ = _create_linear(client, note="线性梯度一")
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["created"] is True
    exp = body["experiment"]
    assert exp["mode"] == "linear"
    assert exp["note"] == "线性梯度一"
    assert len(exp["sources"]) == 2
    assert exp["setup"]["scale_division_g"] == 0.5
    assert exp["constants_version"]

    result = exp["result"]
    cells = {c["position"]: c for c in result["cells"]}
    assert set(cells) == {"A", "B", "C1", "C2", "C3", "D", "E"}

    # 端点格即来源配方（百分比口径下克数 = 千克数）
    a_doses = {d["material_id"]: d for d in cells["A"]["doses"]}
    assert a_doses[ids["钾长石"]]["weighed_g"] == 60.0
    assert a_doses[ids["方解石"]]["weighed_g"] == 20.0
    assert a_doses[ids["石英"]]["weighed_g"] == 20.0
    e_doses = {d["material_id"]: d for d in cells["E"]["doses"]}
    assert e_doses[ids["钾长石"]]["weighed_g"] == 40.0
    assert e_doses[ids["方解石"]]["weighed_g"] == 10.0
    assert e_doses[ids["白云石"]]["weighed_g"] == 10.0
    assert e_doses[ids["石英"]]["weighed_g"] == 40.0

    # 中点合并：钾长石 50、方解石 15、白云石 5、石英 30（相同原料已合并）
    c_doses = {d["material_id"]: d for d in cells["C1"]["doses"]}
    assert c_doses[ids["钾长石"]]["theoretical_g"] == 50.0
    assert c_doses[ids["方解石"]]["theoretical_g"] == 15.0
    assert c_doses[ids["白云石"]]["theoretical_g"] == 5.0
    assert c_doses[ids["石英"]]["theoretical_g"] == 30.0

    # 每格实际称量恰为 100 g，且全部是分度整数倍
    for cell in result["cells"]:
        assert cell["total_weighed_g"] == pytest.approx(100.0)
        assert cell["total_rounding_error_g"] == pytest.approx(0.0, abs=1e-6)
        for d in cell["doses"]:
            assert d["units"] >= 2  # 最小称量 1 g = 2 单位
            assert d["weighed_g"] == d["units"] * 0.5
        assert cell["seger"]
        assert cell["fired_mass_g"] > 0
        assert cell["cost"] > 0
        assert "seger_shift_from_theoretical" in cell

    summary = result["summary"]
    assert summary["n_cells"] == 7
    assert summary["total_weighed_g"] == pytest.approx(700.0)
    assert summary["total_fired_mass_g"] > 0
    assert summary["total_cost"] > 0

    totals = {t["material_id"]: t for t in result["material_totals"]}
    # 各原料逐格舍入误差汇总
    assert set(totals) == {ids["钾长石"], ids["方解石"], ids["白云石"], ids["石英"]}
    assert totals[ids["钾长石"]]["weighed_g"] == pytest.approx(
        60 + 55 + 50 * 3 + 45 + 40
    )


def test_create_blend_idempotent_and_readable(client):
    resp1, _, _ = _create_linear(client)
    exp1 = resp1.json()["experiment"]
    resp2, _, (a, b, _) = _create_linear(client)
    body2 = resp2.json()
    assert body2["created"] is False
    assert body2["experiment"]["id"] == exp1["id"]

    got = client.get(f"/blend-experiments/{exp1['id']}")
    assert got.status_code == 200
    assert got.json()["result"] == exp1["result"]
    listing = client.get("/blend-experiments").json()
    assert any(e["id"] == exp1["id"] for e in listing)


def test_source_not_found(client):
    _, a, _, _ = _sources(client)
    payload = _linear_payload(a, "deadbeefdeadbeef")
    resp = client.post("/blend-experiments", json=payload)
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "version_not_found"


def test_duplicate_source_rejected(client):
    _, a, _, _ = _sources(client)
    payload = _linear_payload(a, a)
    resp = client.post("/blend-experiments", json=payload)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "duplicate_source"


def test_mode_source_count_mismatch(client):
    _, a, b, c = _sources(client)
    # 线性给了 3 份
    payload = _linear_payload(a, b)
    payload["sources"] = [a, b, c]
    resp = client.post("/blend-experiments", json=payload)
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["code"] == "mode_source_count_mismatch"
    assert err["details"]["expected"] == 2

    # 三元只给 2 份
    resp2 = client.post("/blend-experiments", json={
        "sources": [a, b], "mode": "ternary",
        "cells": [{"position": "X", "weights": [1.0, 0.0, 0.0]}],
        "dry_mass_per_tile_g": 100.0, "scale_division_g": 0.5,
        "minimum_weighed_g": 1.0,
    })
    assert resp2.status_code == 422
    assert resp2.json()["error"]["code"] == "mode_source_count_mismatch"


def test_weights_not_closed(client):
    resp, _, _ = _create_linear(client)
    payload = _linear_payload(*_sources(client)[1:3])
    payload["cells"][2]["weights"] = [0.5, 0.6]
    resp = client.post("/blend-experiments", json=payload)
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["code"] == "weights_not_closed"
    assert err["details"]["position"] == "C1"


def test_duplicate_cell_position(client):
    _, a, b, _ = _sources(client)
    payload = _linear_payload(a, b)
    payload["cells"][1]["position"] = "A"
    resp = client.post("/blend-experiments", json=payload)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "duplicate_cell_position"


def test_ratio_out_of_range(client):
    _, a, b, _ = _sources(client)
    payload = _linear_payload(a, b, ratio_low=0.2, ratio_high=0.8)
    resp = client.post("/blend-experiments", json=payload)
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["code"] == "ratio_out_of_range"
    assert err["details"]["position"] == "A"  # 端点 1.0 越界


def test_ratio_not_on_grid(client):
    _, a, b, _ = _sources(client)
    payload = _linear_payload(a, b)
    payload["cells"][1]["weights"] = [0.7, 0.3]  # 闭合但不在 0.25 网格
    resp = client.post("/blend-experiments", json=payload)
    assert resp.status_code == 422
    body = resp.json()["error"]
    assert body["code"] == "ratio_not_on_grid"
    assert body["details"]["position"] == "B"


def test_minimum_below_division(client):
    _, a, b, _ = _sources(client)
    payload = _linear_payload(a, b)
    payload["minimum_weighed_g"] = 0.1  # < 分度 0.5
    resp = client.post("/blend-experiments", json=payload)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "minimum_below_division"


def test_dry_mass_not_on_grid(client):
    _, a, b, _ = _sources(client)
    payload = _linear_payload(a, b)
    payload["dry_mass_per_tile_g"] = 100.1
    resp = client.post("/blend-experiments", json=payload)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "dry_mass_not_on_grid"


def test_cell_rounding_infeasible_minimum(client):
    """最小称量 5 g：白云石在 B/C/D 格分别为 2.5/5/7.5 g，B 格被拒。"""
    _, a, b, _ = _sources(client)
    payload = _linear_payload(a, b, minimum_weighed_g=5.0)
    resp = client.post("/blend-experiments", json=payload)
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["code"] == "cell_rounding_infeasible"
    positions = {c["position"] for c in err["details"]["cells"]}
    # B 格白云石理论 2.5 g < 5 g；A/E 不含微量料，C1..C3 白云石恰为 5 g
    assert "B" in positions
    assert "A" not in positions and "E" not in positions
    reasons = {
        (c["position"], i["reason"])
        for c in err["details"]["cells"] for i in c["issues"]
    }
    assert any(r == "below_minimum_weighed" for _, r in reasons)


def test_cell_rounding_distributes_remainder(client):
    """不成整除的理论量：最大余数法补发后每格恰好 100 g，无 below-min。"""
    ids = _id_map(client)
    a = _freeze(client, [
        {"material_id": ids["钾长石"], "amount": 33.34},
        {"material_id": ids["方解石"], "amount": 33.33},
        {"material_id": ids["石英"], "amount": 33.33},
    ])
    b = _freeze(client, [
        {"material_id": ids["钾长石"], "amount": 100.0},
    ])
    resp = client.post("/blend-experiments", json={
        "sources": [a, b], "mode": "linear",
        "cells": [{"position": "X", "weights": [0.5, 0.5]}],
        "ratio_step": 0.5,
        "dry_mass_per_tile_g": 100.0, "scale_division_g": 0.5,
        "minimum_weighed_g": 1.0,
    })
    assert resp.status_code == 201, resp.text
    cell = resp.json()["experiment"]["result"]["cells"][0]
    assert cell["total_weighed_g"] == pytest.approx(100.0)
    # 方解石/石英理论 16.665 g → 33.33 单位，舍入误差在 ±0.5 g 内
    calcite = next(
        d for d in cell["doses"] if d["material_id"] == ids["方解石"]
    )
    assert abs(calcite["rounding_error_g"]) <= 0.5 + 1e-9


def test_ternary_triangle_happy(client):
    _, a, b, c = _sources(client)

    def ternary_cells(step=0.25):
        cells = []
        n = round(1.0 / step)
        label = 0
        for i in range(n + 1):
            for j in range(n + 1 - i):
                k = n - i - j
                label += 1
                cells.append({
                    "position": f"T{label}",
                    "weights": [i / n, j / n, k / n],
                })
        return cells

    resp = client.post("/blend-experiments", json={
        "sources": [a, b, c], "mode": "ternary",
        "cells": ternary_cells(),
        "ratio_step": 0.25,
        "dry_mass_per_tile_g": 100.0, "scale_division_g": 0.5,
        "minimum_weighed_g": 1.0,
    })
    assert resp.status_code == 201, resp.text
    exp = resp.json()["experiment"]
    # 三角网格 0/0.25/.../1 共 15 格
    assert len(exp["result"]["cells"]) == 15
    for cell in exp["result"]["cells"]:
        assert cell["total_weighed_g"] == pytest.approx(100.0)
        assert len(cell["weights"]) == 3


# ---------------------------------------------------------------------------
# 母料拆分搜索
# ---------------------------------------------------------------------------

def _experiment(client):
    """母料测试使用内部布局（含重复中点），共有料在每格都出现。"""
    resp, ids, _ = _create_interior(client)
    assert resp.status_code == 201, resp.text
    return resp.json()["experiment"], ids


def test_master_plans_default_exact_split(client):
    exp, ids = _experiment(client)
    resp = client.post(
        f"/blend-experiments/{exp['id']}/master-plans", json={}
    )
    assert resp.status_code == 200, resp.text
    r = resp.json()
    assert r["parameters"]["fixed_batch"] is False
    plans = r["plans"]
    assert plans  # 至少有基线
    best = r["best"]

    # 等分方案釉式与冻结逐格完全一致、无剩余料
    assert best["max_abs_seger_shift"] == 0.0
    assert best["leftover_g"] == pytest.approx(0.0)
    assert best["master_batch_prepared_g"] == pytest.approx(
        best["master_consumed_g"]
    )
    # 钾长石逐格最小 50 g、方解石 15 g、白云石 2.5 g、石英 25 g
    # → 四种料全部全量入母料
    assert {ids["钾长石"], ids["方解石"], ids["白云石"], ids["石英"]} <= set(
        best["included_material_ids"]
    )
    bulk = {b["material_id"]: b for b in best["bulk"]}
    assert bulk[ids["钾长石"]]["aliquot_g"] == 50.0
    assert bulk[ids["钾长石"]]["prepared_g"] == 250.0  # 50 g × 5 格
    assert bulk[ids["石英"]]["prepared_g"] == 125.0
    assert best["aliquot_total_g"] == pytest.approx(92.5)

    # 母料方案称量次数严格少于逐格基线（20 → 19）
    baseline = next(p for p in plans if p["signature"] == "none")
    assert best["total_weighings"] < baseline["total_weighings"]
    # 排序：偏差(全0) → 称量次数 → 余量 → 剩余
    keys = [
        (p["max_abs_seger_shift"], p["total_weighings"],
         -p["min_weighing_margin_g"], p["leftover_g"])
        for p in plans
    ]
    assert keys == sorted(keys)

    # 称量顺序：先母料配料，再逐格取一次混合料等分、逐料补料
    stages = [s["stage"] for s in best["weighing_order"]]
    assert stages[0] == "bulk"
    assert "aliquot" in stages and "topup" in stages
    aliquot_steps = [s for s in best["weighing_order"] if s["stage"] == "aliquot"]
    # 母料是均质混合料：每格只称 1 次等分，按格位顺序排列
    assert [s["position"] for s in aliquot_steps] == [
        "B1", "B2", "C1", "C2", "C3"
    ]
    # 等分含钾长石 50 + 方解石 15 + 白云石 2.5 + 石英 25 = 92.5 g
    first = aliquot_steps[0]
    assert first["weighed_g"] == pytest.approx(92.5)
    assert {c["material_id"] for c in first["components"]} == {
        ids["钾长石"], ids["方解石"], ids["白云石"], ids["石英"]
    }
    step_numbers = [s["step"] for s in best["weighing_order"]]
    assert step_numbers == list(range(1, len(step_numbers) + 1))

    # 每格：等分混合料各成分 + 补料的总量必须等于冻结方案
    for cp, frozen in zip(best["cells"], exp["result"]["cells"]):
        frozen_units = {d["material_id"]: d["units"] for d in frozen["doses"]}
        plan_units: dict[int, int] = {}
        parts = cp["master_aliquot"]["components"] + cp["topup_doses"]
        for d in parts:
            plan_units[d["material_id"]] = (
                plan_units.get(d["material_id"], 0) + d["units"]
            )
        assert plan_units == frozen_units


def test_master_plans_fixed_batch_with_leftover(client):
    exp, ids = _experiment(client)
    # 全量入母料需求：92.5 g × 5 格 = 462.5 g；配 465 g → 剩余 2.5 g
    resp = client.post(
        f"/blend-experiments/{exp['id']}/master-plans",
        json={"master_batch_g": 465.0, "allowed_leftover_g": 2.5},
    )
    assert resp.status_code == 200, resp.text
    best = resp.json()["best"]
    assert best["master_batch_prepared_g"] == pytest.approx(465.0)
    assert best["master_consumed_g"] == pytest.approx(462.5)
    assert best["leftover_g"] == pytest.approx(2.5)
    # 需求单位 500/150/25/250（共 925），最大余数法分配 930 单位
    bulk = {b["material_id"]: b for b in best["bulk"]}
    total_prepared = sum(b["prepared_units"] for b in bulk.values())
    assert total_prepared == 930
    assert bulk[ids["钾长石"]]["prepared_units"] == 503
    assert bulk[ids["方解石"]]["prepared_units"] == 151
    assert bulk[ids["白云石"]]["prepared_units"] == 25
    assert bulk[ids["石英"]]["prepared_units"] == 251

    # 每格取 92.5 g 混合料，其中各料按实际配料组成折算（允许小数分量），
    # 整数补料闭合每格总量；C 格釉式偏移最大，必须如实计入而不是 0
    for cp in best["cells"]:
        comp = {c["material_id"]: c for c in cp["master_aliquot"]["components"]}
        assert comp[ids["钾长石"]]["units"] == pytest.approx(503 * 185 / 930)
        top_total = sum(d["units"] for d in cp["topup_doses"])
        assert top_total + cp["master_aliquot"]["total_units"] == 200
    assert best["max_abs_seger_shift"] == pytest.approx(0.002039628, abs=1e-8)
    shifts = [cp["max_abs_seger_shift"] for cp in best["cells"]]
    assert max(shifts) == pytest.approx(0.002039628, abs=1e-8)
    # 称量顺序中的补料与方案逐格补料一致
    for cp in best["cells"]:
        order_top = [
            s for s in best["weighing_order"]
            if s["stage"] == "topup" and s["position"] == cp["position"]
        ]
        assert [s["material_id"] for s in order_top] == [
            d["material_id"] for d in cp["topup_doses"]
        ]


def test_master_plans_leftover_cap_too_tight(client):
    exp, _ = _experiment(client)
    resp = client.post(
        f"/blend-experiments/{exp['id']}/master-plans",
        json={"master_batch_g": 500.0, "allowed_leftover_g": 5.0},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "no_master_plan"


def test_master_plans_small_batch_zero_leftover_aliquot(client):
    """300 g 批量、零剩余：每格取 T/n=120 单位（60 g）正好耗尽母料。

    该粒度（60 g）远低于全量需求等分（92.5 g），旧搜索只在需求量附近
    向下看 n+1 个分度而漏掉，误报 no_master_plan。
    """
    exp, ids = _experiment(client)
    resp = client.post(
        f"/blend-experiments/{exp['id']}/master-plans",
        json={"master_batch_g": 300.0, "allowed_leftover_g": 0.0,
              "max_candidates": 50},
    )
    assert resp.status_code == 200, resp.text
    best = resp.json()["best"]
    assert best["master_batch_prepared_g"] == pytest.approx(300.0)
    assert best["master_consumed_g"] == pytest.approx(300.0)
    assert best["leftover_g"] == pytest.approx(0.0)
    assert best["aliquot_total_g"] == pytest.approx(60.0)  # 300/5
    # 每格都取 120 个分度单位的混合料
    for cp in best["cells"]:
        assert cp["master_aliquot"]["total_units"] == 120
        total = cp["master_aliquot"]["total_units"] + sum(
            d["units"] for d in cp["topup_doses"]
        )
        assert total == 200  # 每格 100 g 闭合
    # 零剩余方案可以冻结
    fr = client.post(
        f"/blend-experiments/{exp['id']}/freeze",
        json={"plan_index": 0, "master_batch_g": 300.0,
              "allowed_leftover_g": 0.0, "max_candidates": 50},
    )
    assert fr.status_code == 201, fr.text
    freeze = fr.json()["freeze"]
    assert freeze["plan"]["aliquot_total_g"] == pytest.approx(60.0)
    assert freeze["plan"]["leftover_g"] == pytest.approx(0.0)


def test_master_plans_leftover_cap_enforced_in_grams(client):
    """余量上限严格按克数：2.5 g 剩余不得通过 2.4 g 上限（旧逻辑四舍五入放行）。"""
    exp, _ = _experiment(client)
    # 需要剩余 2.5 g 的拆分（配 465 g、耗 462.5 g）
    ok = client.post(
        f"/blend-experiments/{exp['id']}/master-plans",
        json={"master_batch_g": 465.0, "allowed_leftover_g": 2.5},
    )
    assert ok.status_code == 200
    assert ok.json()["best"]["leftover_g"] == pytest.approx(2.5)

    # 同样拆分在 2.4 g 上限下必须被拒绝，不能进入排序/冻结
    tight = client.post(
        f"/blend-experiments/{exp['id']}/master-plans",
        json={"master_batch_g": 465.0, "allowed_leftover_g": 2.4},
    )
    assert tight.status_code == 422
    assert tight.json()["error"]["code"] == "no_master_plan"

    fr = client.post(
        f"/blend-experiments/{exp['id']}/freeze",
        json={"plan_index": 0, "master_batch_g": 465.0,
              "allowed_leftover_g": 2.4},
    )
    assert fr.status_code == 422
    assert fr.json()["error"]["code"] == "no_master_plan"


def test_master_plans_batch_not_on_grid(client):
    exp, _ = _experiment(client)
    resp = client.post(
        f"/blend-experiments/{exp['id']}/master-plans",
        json={"master_batch_g": 465.2},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "master_batch_not_on_grid"


def test_master_plans_leftover_without_batch(client):
    exp, _ = _experiment(client)
    resp = client.post(
        f"/blend-experiments/{exp['id']}/master-plans",
        json={"allowed_leftover_g": 5.0},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "leftover_without_batch"


def test_master_plans_minimum_below_division(client):
    exp, _ = _experiment(client)
    resp = client.post(
        f"/blend-experiments/{exp['id']}/master-plans",
        json={"master_minimum_weighed_g": 0.1},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "minimum_below_division"


def test_master_plans_experiment_not_found(client):
    resp = client.post(
        "/blend-experiments/deadbeef/master-plans", json={}
    )
    assert resp.status_code == 404


def _two_common_sources(client):
    """两份配方：钾长石/石英为两格恒量共有料，方解石/白云石各差 1 g。

    A = 钾长石 40 / 方解石 19.5 / 白云石 10.5 / 石英 30
    B = 钾长石 40 / 方解石 18.5 / 白云石 11.5 / 石英 30
    布局 A、A、B；最小称量 2 g（4 个分度），故 1 g 差量不能单独补称，
    把方解石/白云石纳入母料的方案全部不可行，只有钾长石/石英可入母料。
    """
    ids = _id_map(client)
    a = _freeze(client, [
        {"material_id": ids["钾长石"], "amount": 40.0},
        {"material_id": ids["方解石"], "amount": 19.5},
        {"material_id": ids["白云石"], "amount": 10.5},
        {"material_id": ids["石英"], "amount": 30.0},
    ])
    b = _freeze(client, [
        {"material_id": ids["钾长石"], "amount": 40.0},
        {"material_id": ids["方解石"], "amount": 18.5},
        {"material_id": ids["白云石"], "amount": 11.5},
        {"material_id": ids["石英"], "amount": 30.0},
    ])
    return ids, a, b


def test_master_plans_covers_all_common_subsets(client):
    """四种共有料、三格、最小称量 2 g：必须找到 11 次的二原料方案。"""
    ids, a, b = _two_common_sources(client)
    resp = client.post("/blend-experiments", json={
        "sources": [a, b], "mode": "linear",
        "cells": [
            {"position": "A1", "weights": [1.0, 0.0]},
            {"position": "A2", "weights": [1.0, 0.0]},
            {"position": "B1", "weights": [0.0, 1.0]},
        ],
        "dry_mass_per_tile_g": 100.0,
        "scale_division_g": 0.5,
        "minimum_weighed_g": 2.0,
    })
    assert resp.status_code == 201, resp.text
    exp = resp.json()["experiment"]

    plans = client.post(
        f"/blend-experiments/{exp['id']}/master-plans",
        json={"max_candidates": 20},
    ).json()
    # 无母料基线：4 料 × 3 格 = 12 次
    baseline = next(p for p in plans["plans"] if p["signature"] == "none")
    assert baseline["total_weighings"] == 12
    # {钾长石,石英} 母料：2 配料 + 3 等分 + 6 补料 = 11 次
    pair = next(
        p for p in plans["plans"]
        if set(p["included_material_ids"]) == {ids["钾长石"], ids["石英"]}
    )
    assert pair["total_weighings"] == 11
    # 方解石/白云石差量仅 1 g < 最小称量 2 g：凡纳入它们的方案均不可行
    for p in plans["plans"]:
        assert ids["方解石"] not in p["included_material_ids"]
        assert ids["白云石"] not in p["included_material_ids"]
    # best 即二原料方案（旧启发式会漏掉它，把 12 次基线排最前）
    assert plans["best"]["signature"] == pair["signature"]
    assert plans["best"]["total_weighings"] == 11


def test_master_plans_subset_weighing_order_consistent(client):
    ids, a, b = _two_common_sources(client)
    exp = client.post("/blend-experiments", json={
        "sources": [a, b], "mode": "linear",
        "cells": [
            {"position": "A1", "weights": [1.0, 0.0]},
            {"position": "A2", "weights": [1.0, 0.0]},
            {"position": "B1", "weights": [0.0, 1.0]},
        ],
        "dry_mass_per_tile_g": 100.0, "scale_division_g": 0.5,
        "minimum_weighed_g": 2.0,
    }).json()["experiment"]
    best = client.post(
        f"/blend-experiments/{exp['id']}/master-plans", json={}
    ).json()["best"]
    # 每个格位：等分混合料分量 + 补料必须还原冻结用量
    frozen = {c["position"]: {d["material_id"]: d["units"]
                              for d in c["doses"]}
              for c in exp["result"]["cells"]}
    for cp in best["cells"]:
        got: dict[int, float] = {}
        for comp in cp["master_aliquot"]["components"]:
            got[comp["material_id"]] = got.get(comp["material_id"], 0) + comp["units"]
        for d in cp["topup_doses"]:
            got[d["material_id"]] = got.get(d["material_id"], 0) + d["units"]
        assert got == frozen[cp["position"]]


def test_blend_cost_weighted_and_order_invariant(client):
    """同料异价：中点成本按各来源份额加权，交换来源顺序结果不变。"""
    ids = _id_map(client)
    # 钾长石在 A 冻结后改价，再冻结 B：同料两价
    a = _freeze(client, [
        {"material_id": ids["钾长石"], "amount": 50.0},
        {"material_id": ids["方解石"], "amount": 20.0},
        {"material_id": ids["石英"], "amount": 30.0},
    ])
    client.patch(f"/materials/{ids['钾长石']}", json={"price": 12.0})
    b = _freeze(client, [
        {"material_id": ids["钾长石"], "amount": 50.0},
        {"material_id": ids["方解石"], "amount": 20.0},
        {"material_id": ids["石英"], "amount": 30.0},
    ])
    assert a != b  # 不同价格 → 不同版本哈希

    def midpoint(source_ids):
        return client.post("/blend-experiments", json={
            "sources": source_ids, "mode": "linear",
            "cells": [{"position": "M", "weights": [0.5, 0.5]}],
            "dry_mass_per_tile_g": 100.0, "scale_division_g": 0.5,
            "minimum_weighed_g": 1.0,
        }).json()["experiment"]["result"]["cells"][0]

    cell_ab = midpoint([a, b])
    cell_ba = midpoint([b, a])
    # 中点 50 g 钾长石：25 g 来自 2.4 元/kg、25 g 来自 12 元/kg
    expected_k = 0.025 * 2.4 + 0.025 * 12.0
    k_dose = next(d for d in cell_ab["doses"] if d["material_id"] == ids["钾长石"])
    assert k_dose["effective_price_per_kg"] == pytest.approx(7.2)
    assert k_dose["cost"] == pytest.approx(expected_k)
    assert cell_ab["cost"] == pytest.approx(cell_ba["cost"])
    # 交换顺序前后逐格有效单价一致
    assert {d["material_id"]: d["effective_price_per_kg"] for d in cell_ab["doses"]} == \
        {d["material_id"]: d["effective_price_per_kg"] for d in cell_ba["doses"]}
    # 端点格仍取对应来源的真实价格，而不是快照首现价
    endpoint = client.post("/blend-experiments", json={
        "sources": [a, b], "mode": "linear",
        "cells": [
            {"position": "P", "weights": [1.0, 0.0]},
            {"position": "Q", "weights": [0.0, 1.0]},
        ],
        "dry_mass_per_tile_g": 100.0, "scale_division_g": 0.5,
        "minimum_weighed_g": 1.0,
    }).json()["experiment"]["result"]["cells"]
    pk = next(d for d in endpoint[0]["doses"] if d["material_id"] == ids["钾长石"])
    qk = next(d for d in endpoint[1]["doses"] if d["material_id"] == ids["钾长石"])
    assert pk["effective_price_per_kg"] == pytest.approx(2.4)
    assert qk["effective_price_per_kg"] == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# 方案冻结
# ---------------------------------------------------------------------------

def test_freeze_blend_plan(client):
    exp, ids = _experiment(client)
    search = client.post(
        f"/blend-experiments/{exp['id']}/master-plans", json={}
    ).json()
    best = search["best"]

    resp = client.post(
        f"/blend-experiments/{exp['id']}/freeze",
        json={"plan_index": 0, "note": "母料方案 v1"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["created"] is True
    freeze = body["freeze"]
    assert freeze["experiment_id"] == exp["id"]
    assert freeze["note"] == "母料方案 v1"
    assert freeze["plan"]["signature"] == best["signature"]
    assert freeze["plan"]["weighing_order"] == best["weighing_order"]

    # 来源配方、布局、母料拆分与计算常量整体存档
    snap = freeze["source_snapshot"]
    assert snap["experiment_id"] == exp["id"]
    assert [s["version_id"] for s in snap["sources"]] == list(
        s["version_id"] for s in exp["sources"]
    )
    assert len(snap["layout"]) == 5
    assert snap["setup"]["scale_division_g"] == 0.5
    assert snap["constants_snapshot"]["oxides"]["SiO2"]["molwt"]
    assert len(snap["cells"]) == 5

    # 幂等
    resp2 = client.post(
        f"/blend-experiments/{exp['id']}/freeze",
        json={"plan_index": 0, "note": "母料方案 v1"},
    )
    assert resp2.json()["created"] is False
    assert resp2.json()["freeze"]["id"] == freeze["id"]

    got = client.get(f"/blend-versions/{freeze['id']}")
    assert got.status_code == 200
    assert got.json()["plan"]["signature"] == best["signature"]
    listing = client.get(
        f"/blend-experiments/{exp['id']}/freezes"
    ).json()
    assert any(f["id"] == freeze["id"] for f in listing)


def test_freeze_blend_plan_index_out_of_range(client):
    exp, _ = _experiment(client)
    resp = client.post(
        f"/blend-experiments/{exp['id']}/freeze",
        json={"plan_index": 50},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "plan_index_out_of_range"


def test_freeze_blend_experiment_not_found(client):
    resp = client.post(
        "/blend-experiments/deadbeef/freeze", json={"plan_index": 0}
    )
    assert resp.status_code == 404


def test_blend_versions_not_found(client):
    assert client.get("/blend-versions/deadbeef").status_code == 404
    assert client.get("/blend-experiments/deadbeef/freezes").status_code == 404
