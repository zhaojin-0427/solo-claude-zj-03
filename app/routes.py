"""FastAPI 路由：原料库、直接计算、搜索重配、不可变版本。"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query
from pydantic import ValidationError

from . import db, service
from .errors import GlazeError
from .schemas import (
    BatchRequest,
    BlendExperimentRequest,
    BlendFreezeRequest,
    FreezeRequest,
    MasterPlanRequest,
    MaterialCreate,
    MaterialUpdate,
    RobustFreezeRequest,
    RobustSearchRequest,
    SearchRequest,
    StudyRequest,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# 健康检查 / 常量
# ---------------------------------------------------------------------------

@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/constants")
def constants() -> dict[str, Any]:
    from .config import CONSTANTS_VERSION, OXIDE_CATALOG

    return {
        "constants_version": CONSTANTS_VERSION,
        "oxides": {
            name: {"molwt": info.molwt, "role": info.role}
            for name, info in OXIDE_CATALOG.items()
        },
    }


# ---------------------------------------------------------------------------
# 原料库
# ---------------------------------------------------------------------------

@router.get("/materials")
def list_materials() -> list[dict]:
    return [m.model_dump() for m in db.list_materials()]


@router.post("/materials", status_code=201)
def create_material(payload: MaterialCreate) -> dict:
    return service.create_material(payload).model_dump()


@router.get("/materials/{material_id}")
def get_material(material_id: int) -> dict:
    return db.get_material(material_id).model_dump()


@router.patch("/materials/{material_id}")
def patch_material(material_id: int, payload: MaterialUpdate) -> dict:
    return service.update_material(material_id, payload).model_dump()


@router.delete("/materials/{material_id}", status_code=204)
def remove_material(material_id: int) -> None:
    db.delete_material(material_id)


# ---------------------------------------------------------------------------
# 直接计算 / 搜索
# ---------------------------------------------------------------------------

@router.post("/compute")
def compute(req: BatchRequest) -> dict[str, Any]:
    return service.compute_batch(req)


@router.post("/search")
def search(req: SearchRequest) -> dict[str, Any]:
    result = service.search_recipes(req)
    # 回传规范化后的约束，便于客户端把所选候选连同约束一起冻结
    result["search_constraints"] = req.model_dump()
    return result


# ---------------------------------------------------------------------------
# 不可变配方版本
# ---------------------------------------------------------------------------

@router.post("/versions", status_code=201)
def freeze_version(payload: dict) -> dict[str, Any]:
    """冻结版本。

    请求体::

        {"items": [{"material_id": 1, "amount": 12.5}, ...],
         "note": "...", "search_constraints": {...可选，原样存档...}}
    """
    try:
        req = FreezeRequest.model_validate(payload)
    except ValidationError as exc:
        raise GlazeError(
            "冻结请求校验失败", "validation_failed", exc.errors()
        )
    constraints = payload.get("search_constraints")
    stored, created = service.freeze_version(req, constraints)
    return {"created": created, "version": stored}


@router.get("/versions")
def list_versions(limit: int = Query(50, ge=1, le=500)) -> list[dict]:
    return db.list_versions(limit)


@router.get("/versions/{version_id}")
def get_version(version_id: str) -> dict[str, Any]:
    return db.get_version(version_id)


# ---------------------------------------------------------------------------
# 原料批次波动研究（不可变研究快照 + 稳健配方冻结）
# ---------------------------------------------------------------------------

@router.post("/studies", status_code=201)
def create_study(req: StudyRequest) -> dict[str, Any]:
    """以一份冻结配方和一组化验批次创建独立的波动研究版本。"""
    stored, created = service.create_study(req)
    return {"created": created, "study": stored}


@router.get("/studies")
def list_studies(limit: int = Query(50, ge=1, le=500)) -> list[dict]:
    return db.list_studies(limit)


@router.get("/studies/{study_id}")
def get_study(study_id: str) -> dict[str, Any]:
    return db.get_study(study_id)


@router.post("/studies/{study_id}/robust-search")
def robust_search(study_id: str, req: RobustSearchRequest) -> dict[str, Any]:
    """在库存、步进与最大改动量内搜索稳健配方并排序。"""
    return service.robust_search(study_id, req)


@router.post("/studies/{study_id}/freeze", status_code=201)
def freeze_robust(study_id: str, req: RobustFreezeRequest) -> dict[str, Any]:
    """冻结选定的稳健配方：来源配方、化验数据、抽样规则与种子整体存档。"""
    stored, created = service.freeze_robust(study_id, req)
    return {"created": created, "freeze": stored}


@router.get("/studies/{study_id}/freezes")
def list_robust_freezes(
    study_id: str, limit: int = Query(50, ge=1, le=500)
) -> list[dict]:
    db.get_study(study_id)  # 404: 研究不存在
    return db.list_robust_versions(study_id, limit)


@router.get("/robust-versions/{freeze_id}")
def get_robust_version(freeze_id: str) -> dict[str, Any]:
    return db.get_robust_version(freeze_id)


# ---------------------------------------------------------------------------
# 配方混合试验（冻结配方混料 + 试片布局 + 母料拆分）
# ---------------------------------------------------------------------------

@router.post("/blend-experiments", status_code=201)
def create_blend_experiment(req: BlendExperimentRequest) -> dict[str, Any]:
    """把两至三份冻结配方与一张试片布局保存为独立混合试验版本。"""
    stored, created = service.create_blend_experiment(req)
    return {"created": created, "experiment": stored}


@router.get("/blend-experiments")
def list_blend_experiments(limit: int = Query(50, ge=1, le=500)) -> list[dict]:
    return db.list_blend_experiments(limit)


@router.get("/blend-experiments/{experiment_id}")
def get_blend_experiment(experiment_id: str) -> dict[str, Any]:
    return db.get_blend_experiment(experiment_id)


@router.post("/blend-experiments/{experiment_id}/master-plans")
def master_plans(experiment_id: str, req: MasterPlanRequest) -> dict[str, Any]:
    """提取各格共有用量搜索"母料+逐格补料"方案并排序。"""
    return service.master_plan_search(experiment_id, req)


@router.post("/blend-experiments/{experiment_id}/freeze", status_code=201)
def freeze_blend_plan(experiment_id: str, req: BlendFreezeRequest) -> dict[str, Any]:
    """冻结选定方案：来源配方、布局、母料拆分及计算常量整体存档。"""
    stored, created = service.freeze_blend_plan(experiment_id, req)
    return {"created": created, "freeze": stored}


@router.get("/blend-experiments/{experiment_id}/freezes")
def list_blend_freezes(
    experiment_id: str, limit: int = Query(50, ge=1, le=500)
) -> list[dict]:
    db.get_blend_experiment(experiment_id)  # 404: 试验不存在
    return db.list_blend_versions(experiment_id, limit)


@router.get("/blend-versions/{freeze_id}")
def get_blend_version(freeze_id: str) -> dict[str, Any]:
    return db.get_blend_version(freeze_id)
