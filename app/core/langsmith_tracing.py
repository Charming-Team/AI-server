import os

from app.core.config import Settings


def configure_langsmith_tracing_from_settings(settings: Settings) -> None:
    """
    Configure LangSmith environment variables for code paths that do not go
    through LangChain/LangGraph setup directly.
    """
    if not settings.langsmith_tracing:
        return

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    if settings.langsmith_project:
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    if settings.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
