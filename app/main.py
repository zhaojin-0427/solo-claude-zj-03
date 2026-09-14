"""FastAPI 应用入口与异常映射。"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import db
from .errors import GlazeError, NotFoundError
from .routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="陶艺釉式计算与重配服务",
        version="1.0.0",
        description=(
            "原料库管理、Seger 釉式换算、库存/步进约束下的替代配方搜索、"
            "不可变配方版本冻结，原料批次波动研究与稳健配方冻结，"
            "配方混合试片与母料称量，以及釉浆调制批次的台账、"
            "比重杯闭合与纠偏方案搜索。"
        ),
    )
    db.init_db()
    app.include_router(router)

    @app.exception_handler(GlazeError)
    async def _glaze_error(request: Request, exc: GlazeError):
        return JSONResponse(
            status_code=422,
            content={
                "error": {"code": exc.code, "message": exc.message, "details": exc.details}
            },
        )

    @app.exception_handler(NotFoundError)
    async def _not_found(request: Request, exc: NotFoundError):
        return JSONResponse(
            status_code=404,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        # Pydantic 校验器中抛出的 GlazeError 藏在 ctx['error'] 中，
        # 提取其结构化 code/details；否则返回标准校验错误。
        for err in exc.errors():
            ctx = err.get("ctx") or {}
            inner = ctx.get("error")
            if isinstance(inner, GlazeError):
                return JSONResponse(
                    status_code=422,
                    content={
                        "error": {
                            "code": inner.code,
                            "message": inner.message,
                            "details": inner.details,
                        }
                    },
                )
        return JSONResponse(status_code=422, content={"error": {
            "code": "validation_error",
            "message": "请求数据校验失败",
            "details": exc.errors(),
        }})

    return app


app = create_app()
