import os
import json
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional, List, Dict, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
from sqlalchemy import (
    create_engine, Column, String, Text, DateTime, JSON, func, text
)
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is required")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

# --- ModelScope LLM client config ---
MODELSCOPE_API_KEY = os.getenv("MODELSCOPE_API_KEY", "").strip()
MODELSCOPE_BASE_URL = os.getenv("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1").strip()
CHAT_MODEL = os.getenv("CHAT_MODEL", "").strip()
TAG_MODEL = os.getenv("TAG_MODEL", CHAT_MODEL).strip()

if not MODELSCOPE_API_KEY or not CHAT_MODEL:
    # Keep server booting if you prefer; but for MVP it's better to fail fast.
    raise RuntimeError("MODELSCOPE_API_KEY and CHAT_MODEL are required in env (.env)")

llm_client = OpenAI(api_key=MODELSCOPE_API_KEY, base_url=MODELSCOPE_BASE_URL)

class Record(Base):
    __tablename__ = "records"
    id = Column(String, primary_key=True)
    project_id = Column(String, nullable=True, default="default")
    source = Column(String, nullable=True, default="unknown")
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
    project_id: Optional[str] = "default"
    source: Optional[str] = "unknown"
    ts: int  # unix seconds
    user_text: str
    assistant_text: str
    summary: Optional[str] = ""
    tags: List[TagItem] = []

# --- prompts ---
CHAT_SYS = "You are a helpful assistant."
TAG_SYS = """
你是一个"对话总结与打标签"引擎。
请根据给定文本输出严格 JSON（不要 markdown，不要解释）：

{
  "summary": "1-2句中文总结",
  "tags": [
    {"name": "标签1", "confidence": 0.0},
    {"name": "标签2", "confidence": 0.0}
  ]
}

规则：
- tags 最多 8 个，按重要性排序
- confidence 是 0~1 的小数
- 标签用中文名词短语，尽量短
"""

# --- request/response models for chat ---
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    user_text: str = Field(min_length=1)
    project_id: str = "default"
    history: Optional[List[ChatMessage]] = None  # optional

class ChatResponse(BaseModel):
    record: dict  # keep flexible for now

def _safe_parse_tags_obj(obj: dict) -> dict:
    summary = (obj.get("summary") or "").strip()
    tags = obj.get("tags") or []
    out_tags = []
    if isinstance(tags, list):
        for t in tags[:8]:
            if not isinstance(t, dict):
                continue
            name = (t.get("name") or "").strip()
            if not name:
                continue
            try:
                conf = float(t.get("confidence", 0.0) or 0.0)
            except Exception:
                conf = 0.0
            conf = max(0.0, min(1.0, conf))
            out_tags.append({"name": name, "confidence": conf})
    return {"summary": summary, "tags": out_tags}

def extract_summary_tags(text: str) -> dict:
    try:
        r = llm_client.chat.completions.create(
            model=TAG_MODEL,
            messages=[
                {"role": "system", "content": TAG_SYS},
                {"role": "user", "content": f"文本如下：\n{text}"},
            ],
            temperature=0.1,
            stream=False,
        )
        content = (r.choices[0].message.content or "").strip()
        m = re.search(r"\{.*\}", content, flags=re.S)
        if not m:
            return {"summary": "", "tags": []}
        obj = json.loads(m.group(0))
        if not isinstance(obj, dict):
            return {"summary": "", "tags": []}
        return _safe_parse_tags_obj(obj)
    except Exception:
        return {"summary": "", "tags": []}

def run_chat(user_text: str, history: Optional[List[ChatMessage]] = None) -> str:
    msgs = [{"role": "system", "content": CHAT_SYS}]
    if history:
        msgs += [{"role": m.role, "content": m.content} for m in history]
    msgs.append({"role": "user", "content": user_text})

    try:
        r = llm_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=msgs,
            temperature=0.7,
            stream=False,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"llm_error: {e}")

app = FastAPI()

@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)

@app.get("/")
def root():
    return {"ok": True, "service": "ssb-backend"}

@app.get("/healthz")
def healthz():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"db_error: {e}")
    return {"ok": True}

@app.post("/v1/records/ingest")
def ingest(rec: RecordIn):
    dt = datetime.fromtimestamp(rec.ts, tz=timezone.utc)
    with SessionLocal() as db:
        existing = db.get(Record, rec.id)
        if existing:
            existing.project_id = rec.project_id or "default"
            existing.source = rec.source or "unknown"
            existing.ts = dt
            existing.user_text = rec.user_text
            existing.assistant_text = rec.assistant_text
            existing.summary = rec.summary or ""
            existing.tags = [t.model_dump() for t in rec.tags]
            db.add(existing)
        else:
            row = Record(
                id=rec.id,
                project_id=rec.project_id or "default",
                source=rec.source or "unknown",
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
def list_records(
    limit: int = 20,
    project_id: Optional[str] = None,
    source: Optional[str] = None,
    before_ts: Optional[int] = None,  # unix seconds, for pagination
):
    limit = max(1, min(limit, 100))

    with SessionLocal() as db:
        q = db.query(Record)

        if project_id:
            q = q.filter(Record.project_id == project_id)

        if source:
            q = q.filter(Record.source == source)

        if before_ts:
            dt = datetime.fromtimestamp(before_ts, tz=timezone.utc)
            q = q.filter(Record.ts < dt)

        rows = q.order_by(Record.ts.desc()).limit(limit).all()

        items = []
        for r in rows:
            items.append({
                "id": r.id,
                "project_id": r.project_id or "default",
                "source": r.source or "unknown",
                "ts": int(r.ts.timestamp()),
                "user_text": r.user_text,
                "assistant_text": r.assistant_text,
                "summary": r.summary or "",
                "tags": r.tags or [],
            })

    return {"items": items}

# --- endpoint: POST /v1/chat ---
@app.post("/v1/chat", response_model=ChatResponse)
def v1_chat(req: ChatRequest):
    assistant_text = run_chat(req.user_text, req.history)
    meta = extract_summary_tags(assistant_text)

    rec_id = str(uuid.uuid4())
    ts = int(time.time())
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)

    try:
        with SessionLocal() as db:
            row = Record(
                id=rec_id,
                project_id=req.project_id,
                source="chat",
                ts=dt,
                user_text=req.user_text,
                assistant_text=assistant_text,
                summary=meta.get("summary", ""),
                tags=meta.get("tags", []),
            )
            db.add(row)
            db.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"db_write_error: {e}")

    record_out = {
        "id": rec_id,
        "ts": ts,
        "project_id": req.project_id,
        "source": "chat",
        "user_text": req.user_text,
        "assistant_text": assistant_text,
        "summary": meta.get("summary", ""),
        "tags": meta.get("tags", []),
    }
    return {"record": record_out}
