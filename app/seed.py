"""原料库种子数据：典型陶艺原料的化学分析（百分数口径）。

数值按理论矿物组成折算到氧化物 + LOI = 100；
价格为示例单价（元/kg），库存为示例可用量。
重复执行不会插入重复记录（按名称判断）。
"""
from __future__ import annotations

from . import db
from .schemas import MaterialCreate

SEED_MATERIALS = [
    {
        "name": "钾长石",
        "oxides": {"K2O": 16.92, "Al2O3": 18.32, "SiO2": 64.76},
        "loi": 0.0,
        "price": 2.4,
        "available": 500.0,
    },
    {
        "name": "钠长石",
        "oxides": {"Na2O": 11.82, "Al2O3": 19.45, "SiO2": 68.73},
        "loi": 0.0,
        "price": 2.2,
        "available": 500.0,
    },
    {
        "name": "方解石",
        "oxides": {"CaO": 56.03},
        "loi": 43.97,
        "price": 0.8,
        "available": 800.0,
    },
    {
        "name": "白云石",
        "oxides": {"CaO": 30.41, "MgO": 21.86},
        "loi": 47.73,
        "price": 0.9,
        "available": 600.0,
    },
    {
        "name": "高岭土",
        "oxides": {"Al2O3": 39.50, "SiO2": 46.54},
        "loi": 13.96,
        "price": 1.5,
        "available": 700.0,
    },
    {
        "name": "石英",
        "oxides": {"SiO2": 100.0},
        "loi": 0.0,
        "price": 1.0,
        "available": 1000.0,
    },
    {
        "name": "滑石",
        "oxides": {"MgO": 31.88, "SiO2": 63.37},
        "loi": 4.75,
        "price": 1.8,
        "available": 300.0,
    },
    {
        "name": "氧化锌",
        "oxides": {"ZnO": 100.0},
        "loi": 0.0,
        "price": 18.0,
        "available": 80.0,
    },
    {
        "name": "碳酸钡",
        "oxides": {"BaO": 77.70},
        "loi": 22.30,
        "price": 6.5,
        "available": 120.0,
    },
    {
        "name": "硼砂",
        "oxides": {"Na2O": 16.27, "B2O3": 36.51},
        "loi": 47.22,
        "price": 5.0,
        "available": 100.0,
    },
    {
        "name": "氧化铁红",
        "oxides": {"Fe2O3": 100.0},
        "loi": 0.0,
        "price": 9.0,
        "available": 60.0,
    },
    {
        "name": "金红石",
        "oxides": {"TiO2": 100.0},
        "loi": 0.0,
        "price": 12.0,
        "available": 50.0,
    },
]


def seed() -> int:
    db.init_db()
    existing = {m.name for m in db.list_materials()}
    count = 0
    for data in SEED_MATERIALS:
        if data["name"] in existing:
            continue
        payload = MaterialCreate(**data)
        from .chemistry import validate_analysis

        oxides, loi = validate_analysis(
            payload.oxides, payload.loi, payload.analysis_tolerance
        )
        db.create_material(payload, oxides, loi)
        count += 1
    return count


if __name__ == "__main__":  # pragma: no cover
    inserted = seed()
    print(f"插入 {inserted} 种原料，库中共 {len(db.list_materials())} 种。")
