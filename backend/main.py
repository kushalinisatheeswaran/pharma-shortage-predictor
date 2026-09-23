"""
MedCascade FastAPI Backend Application Entry Point
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from backend.config import settings
from backend.services.state import state
from backend.routers import health, drugs, model, graph, scenarios

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan event handler. Loads expensive artifacts ONCE at startup.
    """
    state.load_artifacts()
    yield

app = FastAPI(
    title=settings.PROJECT_TITLE,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.PROJECT_VERSION,
    lifespan=lifespan
)

# Configure CORS for Next.js frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(health.router)
app.include_router(drugs.router)
app.include_router(model.router)
app.include_router(graph.router)
app.include_router(scenarios.router)

# Controlled Error Handlers (Do not expose internal stack traces)
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Please verify backend logs."}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
