import os
import json
from datetime import datetime, timezone
from typing import Any, Optional, List, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import (
    create_engine, Column, String, Text, DateTime, JSON, func
)
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is required")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class Record(Base):
    __tablename__ = "records"
    id = Column(String, primary_key=True)
    ts = Column(DateTime(timezone=True), nullable=False)
    user_text = Column(Text, nullable=False)
    assistant_text = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    tags = Column(JSON, nullable=False, default=list)

class TagItem(BaseModel):
    name: str
    confidence: float = Field(ge=0, le=1)

class RecordIn(BaseModel):
    id: str
    ts: int  # unix seconds
    user_text: str
    assistant_text: str
    summary: Optional[str] = ""
    tags: List[TagItem] = []

app = FastAPI()

@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)

@app.get("/")
def root():
    return {"ok": True, "service": "ssb-backend"}

@app.get("/healthz")
def healthz():
    # DB ping
    try:
        with engine.connect() as conn:
            conn.execute(func.now().select())  # lightweight
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"db_error: {e}")
    return {"ok": True}

@app.post("/v1/records/ingest")
def ingest(rec: RecordIn):
    dt = datetime.fromtimestamp(rec.ts, tz=timezone.utc)
    with SessionLocal() as db:
        existing = db.get(Record, rec.id)
        if existing:
            existing.ts = dt
            existing.user_text = rec.user_text
            existing.assistant_text = rec.assistant_text
            existing.summary = rec.summary or ""
            existing.tags = [t.model_dump() for t in rec.tags]
            db.add(existing)
        else:
            row = Record(
                id=rec.id,
                ts=dt,
                user_text=rec.user_text,
                assistant_text=rec.assistant_text,
                summary=rec.summary or "",
                tags=[t.model_dump() for t in rec.tags],
            )
            db.add(row)
        db.commit()
    return {"id": rec.id}

@app.get("/v1/records")
def list_records(limit: int = 20):
    limit = max(1, min(limit, 100))
    with SessionLocal() as db:
        rows = db.query(Record).order_by(Record.ts.desc()).limit(limit).all()
        items = []
        for r in rows:
            items.append({
                "id": r.id,
                "ts": int(r.ts.timestamp()),
                "user_text": r.user_text,
                "assistant_text": r.assistant_text,
                "summary": r.summary or "",
                "tags": r.tags or [],
            })
    return {"items": items}
