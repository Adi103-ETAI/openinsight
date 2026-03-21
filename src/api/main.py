from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import query as query_router

app = FastAPI(
    title="OpenMed API",
    description="AI Clinical Decision Support for Indian Physicians — SentArc Labs",
    version="0.1.0",
    redirect_slashes=False,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "openmed-api"}


app.include_router(query_router.router, prefix="/query", tags=["Query"])
