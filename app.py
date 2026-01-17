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

# --- recommendation prompts ---
RECO_SYS = """
You are a prompt recommendation engine.
Return STRICT JSON only (no markdown, no explanation):

{
  "prompts": [
    {
      "category": "string",
      "title": "string",
      "prompt": "string",
      "why": "string"
    }
  ],
  "meta": {
    "project_id": "string",
    "head_id": "string",
    "tag_hint": "string",
    "summary": "string"
  }
}

Rules:
- prompts: 1 to N items (N <= 6), ordered by usefulness.
- category/title/prompt/why should be in Chinese.
"""

# --- Time Machine prompts ---
TM_SYS = """
你是知识时间线判定引擎。
给你 current 和 candidates（历史知识），判断关系：

返回严格 JSON：
{
  "relation": "new|update|duplicate|related",
  "parent_id": "uuid or empty",
  "confidence": 0-1,
  "reason": "一句话",
  "delta": {
    "added": [],
    "changed": [],
    "conflicts": []
  },
  "merged_summary": "string"
}
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

class RecommendationRequest(BaseModel):
    project_id: str = "default"
    head_id: Optional[str] = None
    max_q: int = 6
    persist: bool = True

class TimeMachineRequest(BaseModel):
    project_id: str = "default"
    current_record_id: str
    limit_candidates: int = 30
    persist_tags: bool = True  # write __tm_* tags back to current record

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

def _extract_json_obj(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", flags=re.S)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

def _tag_hint_from_tags(tags: list) -> str:
    names = []
    for t in tags or []:
        if isinstance(t, dict) and t.get("name"):
            names.append(str(t["name"]).strip())
        if len(names) >= 4:
            break
    return "、".join([n for n in names if n]) or "当前主题"

def _get_head_record(db, project_id: str, head_id: Optional[str]) -> Optional[Record]:
    if head_id:
        r = db.get(Record, head_id)
        if r and (r.project_id or "default") == project_id:
            return r
        return None

    # Default: latest chat record for this project
    return (
        db.query(Record)
        .filter(Record.project_id == project_id)
        .filter(Record.source == "chat")
        .order_by(Record.ts.desc())
        .first()
    )

def _tm_tags(tm: dict) -> list:
    # Encode tm result into tags for timeline backtracking
    rel = (tm.get("relation") or "").strip()
    pid = (tm.get("parent_id") or "").strip()
    out = []
    if rel in ("update", "duplicate") and pid:
        out.append({"name": f"__tm_rel:{rel}__", "confidence": 1.0})
        out.append({"name": f"__tm_parent:{pid}__", "confidence": 1.0})
    return out

def _run_time_machine_llm(current: dict, candidates: list) -> dict:
    payload = {"current": current, "candidates": candidates}
    try:
        r = llm_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": TM_SYS},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.1,
            stream=False,
        )
        raw = (r.choices[0].message.content or "").strip()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"llm_error: {e}")

    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        return {"relation": "new", "parent_id": "", "confidence": 0.0, "reason": "no_json", "delta": {"added": [], "changed": [], "conflicts": []}, "merged_summary": ""}

    try:
        obj = json.loads(m.group(0))
        if not isinstance(obj, dict):
            raise ValueError("not_dict")
    except Exception:
        return {"relation": "new", "parent_id": "", "confidence": 0.0, "reason": "json_parse_error", "delta": {"added": [], "changed": [], "conflicts": []}, "merged_summary": ""}

    # Normalize fields
    rel = str(obj.get("relation") or "new").strip()
    if rel not in ("new", "update", "duplicate", "related"):
        rel = "new"
    pid = str(obj.get("parent_id") or "").strip()
    try:
        conf = float(obj.get("confidence", 0.0) or 0.0)
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    reason = str(obj.get("reason") or "").strip()
    delta = obj.get("delta") if isinstance(obj.get("delta"), dict) else {"added": [], "changed": [], "conflicts": []}
    merged = str(obj.get("merged_summary") or "").strip()

    # Ensure delta lists
    for k in ("added", "changed", "conflicts"):
        v = delta.get(k)
        if not isinstance(v, list):
            delta[k] = []

    return {
        "relation": rel,
        "parent_id": pid,
        "confidence": conf,
        "reason": reason,
        "delta": delta,
        "merged_summary": merged,
    }

# --- Timeline build helpers ---

def _extract_tm_from_tags(tags: list) -> tuple[Optional[str], Optional[str]]:
    rel, pid = None, None
    for t in tags or []:
        if not isinstance(t, dict):
            continue
        name = (t.get("name") or "").strip()
        if name.startswith("__tm_rel:") and name.endswith("__"):
            rel = name[len("__tm_rel:"):-2]
        elif name.startswith("__tm_parent:") and name.endswith("__"):
            pid = name[len("__tm_parent:"):-2]
    return rel, pid

def _build_timeline_from_rows(head_id: str, rows: list, limit_chain: int = 200) -> dict:
    # Build nodes map
    nodes: dict[str, dict] = {}
    parent_of: dict[str, str] = {}
    edge_meta: dict[str, dict] = {}

    for r in rows:
        rid = r.id
        nodes[rid] = {
            "id": rid,
            "project_id": r.project_id or "default",
            "source": r.source or "unknown",
            "ts": int(r.ts.timestamp()),
            "user_text": r.user_text,
            "assistant_text": r.assistant_text,
            "summary": r.summary or "",
            "tags": r.tags or [],
        }

    # Parent pointers from __tm_* tags
    for rid, node in nodes.items():
        rel, pid = _extract_tm_from_tags(node.get("tags", []))
        if rel in ("update", "duplicate") and pid and pid in nodes:
            parent_of[rid] = pid
            edge_meta[rid] = {
                "type": rel,
                "confidence": 1.0,
                "reason": "from __tm_* tags",
                "delta": {"added": [], "changed": [], "conflicts": []},
            }

    if head_id not in nodes:
        raise HTTPException(status_code=404, detail="head_id_not_found_in_project")

    # Backtrack chain: head -> parent -> ...
    chain: list[str] = []
    cur = head_id
    seen = set()
    while cur and cur in nodes and cur not in seen and len(chain) < limit_chain:
        seen.add(cur)
        chain.append(cur)
        cur = parent_of.get(cur)

    root_id = chain[-1] if chain else head_id

    # Timeline output: root -> head
    ordered = list(reversed(chain))

    timeline = {
        "timeline_id": str(uuid.uuid4()),
        "project_id": nodes[head_id]["project_id"],
        "root_id": root_id,
        "head_id": head_id,
        "title": (nodes[head_id]["summary"][:24] if nodes[head_id]["summary"] else "timeline"),
        "nodes": [],
        "edges": [],
        "events": [],
    }

    total = len(ordered)
    for idx, rid in enumerate(ordered, start=1):
        n = dict(nodes[rid])
        n["version_index"] = idx
        n["version_count"] = total
        timeline["nodes"].append(n)

    for child in chain:
        parent = parent_of.get(child)
        if parent:
            meta = edge_meta.get(child, {})
            timeline["edges"].append({
                "from": child,
                "to": parent,
                "type": meta.get("type", "update"),
                "confidence": meta.get("confidence", 1.0),
                "delta": meta.get("delta", {"added": [], "changed": [], "conflicts": []}),
                "reason": meta.get("reason", ""),
            })

    for n in timeline["nodes"]:
        timeline["events"].append({
            "ts": n.get("ts"),
            "type": "created" if n["id"] == root_id else "updated",
            "node_id": n["id"],
            "message": "created" if n["id"] == root_id else "updated",
        })

    return timeline

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

@app.post("/v1/recommendations/generate")
def generate_recommendations(req: RecommendationRequest):
    max_q = max(1, min(int(req.max_q or 6), 6))

    with SessionLocal() as db:
        head = _get_head_record(db, req.project_id, req.head_id)
        if not head:
            raise HTTPException(status_code=404, detail="head_record_not_found")

        tag_hint = _tag_hint_from_tags(head.tags or [])
        context = {
            "project_id": req.project_id,
            "head_id": head.id,
            "summary": head.summary or "",
            "tags": head.tags or [],
            "user_text": head.user_text,
            "assistant_text": head.assistant_text,
            "tag_hint": tag_hint,
            "max_q": max_q,
        }

    # LLM call (non-stream)
    try:
        r = llm_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": RECO_SYS},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
            ],
            temperature=0.3,
            stream=False,
        )
        raw = (r.choices[0].message.content or "").strip()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"llm_error: {e}")

    obj = _extract_json_obj(raw)
    prompts = obj.get("prompts") if isinstance(obj.get("prompts"), list) else []
    prompts = prompts[:max_q]

    # Normalize output
    clean_prompts = []
    for p in prompts:
        if not isinstance(p, dict):
            continue
        clean_prompts.append({
            "category": str(p.get("category", "") or ""),
            "title": str(p.get("title", "") or ""),
            "prompt": str(p.get("prompt", "") or ""),
            "why": str(p.get("why", "") or ""),
        })

    out = {
        "prompts": clean_prompts,
        "meta": {
            "project_id": req.project_id,
            "head_id": req.head_id or "",
            "tag_hint": obj.get("meta", {}).get("tag_hint", "") if isinstance(obj.get("meta"), dict) else "",
            "summary": obj.get("meta", {}).get("summary", "") if isinstance(obj.get("meta"), dict) else "",
        }
    }

    # Persist as a record if requested
    if req.persist:
        rec_id = str(uuid.uuid4())
        ts = int(time.time())
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)

        with SessionLocal() as db:
            head = _get_head_record(db, req.project_id, req.head_id)
            base_tags = (head.tags or []) if head else []
            tags_for_rec = list(base_tags) + [{"name": "__kind:recommendation__", "confidence": 1.0}]

            row = Record(
                id=rec_id,
                project_id=req.project_id,
                source="recommendation",
                ts=dt,
                user_text="",
                assistant_text=json.dumps(out, ensure_ascii=False),
                summary="prompt recommendations",
                tags=tags_for_rec,
            )
            db.add(row)
            db.commit()

        return {"recommendations": out, "record_id": rec_id}

    return {"recommendations": out}

@app.post("/v1/time-machine/resolve")
def time_machine_resolve(req: TimeMachineRequest):
    lim = max(5, min(int(req.limit_candidates or 30), 100))

    with SessionLocal() as db:
        cur = db.get(Record, req.current_record_id)
        if not cur:
            raise HTTPException(status_code=404, detail="current_record_not_found")
        if (cur.project_id or "default") != req.project_id:
            raise HTTPException(status_code=400, detail="project_id_mismatch")

        # Candidates: same project, exclude current
        cand_rows = (
            db.query(Record)
            .filter(Record.project_id == req.project_id)
            .filter(Record.id != req.current_record_id)
            .order_by(Record.ts.desc())
            .limit(lim)
            .all()
        )

        current_obj = {
            "id": cur.id,
            "ts": int(cur.ts.timestamp()),
            "project_id": cur.project_id or "default",
            "source": cur.source or "unknown",
            "user_text": cur.user_text,
            "assistant_text": cur.assistant_text,
            "summary": cur.summary or "",
            "tags": cur.tags or [],
        }

        candidates = []
        for r in cand_rows:
            candidates.append({
                "id": r.id,
                "ts": int(r.ts.timestamp()),
                "project_id": r.project_id or "default",
                "source": r.source or "unknown",
                "summary": r.summary or "",
                "tags": r.tags or [],
                # Keep payload small; summary+tags is usually enough
            })

    tm = _run_time_machine_llm(current_obj, candidates)
    tag_patch = _tm_tags(tm)

    updated_record = None
    if req.persist_tags and tag_patch:
        with SessionLocal() as db:
            cur = db.get(Record, req.current_record_id)
            if not cur:
                raise HTTPException(status_code=404, detail="current_record_not_found")
            old_tags = cur.tags or []
            # Dedup by name
            existing = set([t.get("name") for t in old_tags if isinstance(t, dict)])
            for t in tag_patch:
                if t["name"] not in existing:
                    old_tags.append(t)
            cur.tags = old_tags
            db.add(cur)
            db.commit()

            updated_record = {
                "id": cur.id,
                "project_id": cur.project_id or "default",
                "source": cur.source or "unknown",
                "ts": int(cur.ts.timestamp()),
                "user_text": cur.user_text,
                "assistant_text": cur.assistant_text,
                "summary": cur.summary or "",
                "tags": cur.tags or [],
            }

    return {
        "time_machine": tm,
        "tag_patch": tag_patch,
        "record": updated_record or current_obj,
    }

@app.get("/v1/timeline")
def get_timeline(project_id: str, head_id: str, limit: int = 400):
    limit = max(10, min(int(limit or 400), 2000))
    with SessionLocal() as db:
        rows = (
            db.query(Record)
            .filter(Record.project_id == project_id)
            .order_by(Record.ts.desc())
            .limit(limit)
            .all()
        )
    return _build_timeline_from_rows(head_id=head_id, rows=rows)
