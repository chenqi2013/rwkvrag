from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .admin_service import (
    AdminConflictError,
    AdminNotFoundError,
    AdminService,
    AdminValidationError,
)
from .components import create_index
from .config import get_settings
from .qdrant_admin import QdrantAdmin, QdrantUnavailableError
from .repository import MongoRepository, RepositoryConflictError
from .routers.admin import router as admin_router
from .routers.public import router as public_router
from .service import SearchService, create_reranker
from .tasks import TaskManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    repository = MongoRepository(settings.mongo_url, settings.mongo_database)
    await repository.connect()
    qdrant = QdrantAdmin(settings)
    task_manager = TaskManager(settings, repository, qdrant)
    search = SearchService(
        settings=settings,
        index=create_index(settings),
        reranker=create_reranker(settings),
    )
    admin = AdminService(settings, repository, qdrant, task_manager)
    app.state.repository = repository
    app.state.qdrant_admin = qdrant
    app.state.task_manager = task_manager
    app.state.search_service = search
    app.state.admin_service = admin
    await task_manager.start()
    yield
    await task_manager.shutdown()
    await repository.close()


app = FastAPI(
    title="RWKVRAG LlamaIndex Retrieval API",
    version="0.2.0",
    lifespan=lifespan,
)
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(public_router)
app.include_router(admin_router)


@app.exception_handler(AdminNotFoundError)
async def not_found_handler(_: Request, error: AdminNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(error)})


@app.exception_handler(AdminConflictError)
@app.exception_handler(RepositoryConflictError)
async def conflict_handler(_: Request, error: Exception) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(error)})


@app.exception_handler(AdminValidationError)
async def validation_handler(_: Request, error: AdminValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(error)})


@app.exception_handler(QdrantUnavailableError)
async def qdrant_unavailable_handler(
    _: Request,
    error: QdrantUnavailableError,
) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(error)})


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse("/admin/")


static_directory = Path(__file__).resolve().parent / "static" / "admin"
app.mount(
    "/admin",
    StaticFiles(directory=static_directory, html=True, check_dir=False),
    name="admin",
)
