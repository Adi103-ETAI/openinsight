from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="OpenMed API",
    description="AI Clinical Decision Support for Indian Physicians — SentArc Labs",
    version="0.1.0",
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


# Routers will be added here as we build them
# from src.api.routes import query, ingest
# app.include_router(query.router, prefix="/query")
# app.include_router(ingest.router, prefix="/ingest")
