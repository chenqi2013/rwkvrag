from fastapi import Request

from .admin_service import AdminService
from .lexical_index import LexicalIndex
from .repository import MongoRepository
from .service import SearchService
from .tasks import TaskManager


def search_service(request: Request) -> SearchService:
    return request.app.state.search_service


def admin_service(request: Request) -> AdminService:
    return request.app.state.admin_service


def repository(request: Request) -> MongoRepository:
    return request.app.state.repository


def lexical_index(request: Request) -> LexicalIndex:
    return request.app.state.lexical_index


def task_manager(request: Request) -> TaskManager:
    return request.app.state.task_manager
