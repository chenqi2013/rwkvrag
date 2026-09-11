from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from .admin_service import (
    AdminConflictError,
    AdminNotFoundError,
    AdminService,
    AdminValidationError,
)
from .active_retrieval import ActiveRetrievalAgent
from .config import get_settings
from .generation import EvidenceAnswerGenerator
from .evidence_extraction import LanguageModelEvidenceExtractor
from .document_reranking import LanguageModelDocumentReranker
from .lexical_index import LexicalIndex
from .repository import MongoRepository, RepositoryConflictError
from .routers.admin import router as admin_router
from .routers.public import router as public_router
from .service import SearchService
from .semantic_query_planning import LanguageModelQueryPlanner
from .tasks import TaskManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    repository = MongoRepository(settings.mongo_url, settings.mongo_database)
    await repository.connect()
    lexical_index = LexicalIndex(settings)
    task_manager = TaskManager(settings, repository, lexical_index)
    search = SearchService(
        settings=settings,
        index=lexical_index,
        generator=EvidenceAnswerGenerator(settings),
        evidence_extractor=LanguageModelEvidenceExtractor(settings),
        query_planner=LanguageModelQueryPlanner(settings),
        retrieval_agent=ActiveRetrievalAgent(settings),
        document_reranker=LanguageModelDocumentReranker(settings),
        native_recorder=repository.record_model_http,
    )
    admin = AdminService(settings, repository, task_manager, lexical_index)
    app.state.repository = repository
    app.state.lexical_index = lexical_index
    app.state.task_manager = task_manager
    app.state.search_service = search
    app.state.admin_service = admin
    await task_manager.start()
    yield
    await task_manager.shutdown()
    await search.aclose()
    lexical_index.close()
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


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse("/docs")
