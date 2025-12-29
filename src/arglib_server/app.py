"""ArgLib server application."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict
from typing import Any, Literal
from uuid import uuid4

from arglib.core import ArgumentGraph, EvidenceCard, SupportingDocument
from arglib.io import dumps, validate_graph_payload
from arglib.reasoning import compute_credibility
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

app = FastAPI(title="ArgLib Server", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GraphPayload(BaseModel):
    payload: dict[str, Any] = Field(..., description="ArgumentGraph JSON payload")
    validate: bool = True


class GraphResponse(BaseModel):
    id: str
    payload: dict[str, Any]


class MiningRequest(BaseModel):
    text: str
    doc_id: str | None = None


class ExportRequest(BaseModel):
    format: Literal["json", "dot"] = "json"


class EvidenceCardPayload(BaseModel):
    payload: dict[str, Any]


class SupportingDocumentPayload(BaseModel):
    payload: dict[str, Any]


class GraphStore:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS graphs (id TEXT PRIMARY KEY, payload TEXT)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def create(self, payload: dict[str, Any], *, validate: bool = True) -> str:
        if validate:
            validate_graph_payload(payload)
        graph = ArgumentGraph.from_dict(payload)
        graph_id = uuid4().hex
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO graphs (id, payload) VALUES (?, ?)",
                (graph_id, json.dumps(graph.to_dict())),
            )
        return graph_id

    def get(self, graph_id: str) -> ArgumentGraph:
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT payload FROM graphs WHERE id = ?", (graph_id,)
            )
            row = cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="graph not found")
        payload = json.loads(row[0])
        return ArgumentGraph.from_dict(payload)

    def update(self, graph_id: str, payload: dict[str, Any], *, validate: bool) -> None:
        if validate:
            validate_graph_payload(payload)
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE graphs SET payload = ? WHERE id = ?",
                (json.dumps(payload), graph_id),
            )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="graph not found")

    def delete(self, graph_id: str) -> None:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM graphs WHERE id = ?", (graph_id,))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="graph not found")


store = GraphStore(os.getenv("ARGLIB_SERVER_DB", "arglib.db"))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/graphs", response_model=GraphResponse)
def create_graph(request: GraphPayload) -> GraphResponse:
    graph_id = store.create(request.payload, validate=request.validate)
    return GraphResponse(id=graph_id, payload=request.payload)


@app.get("/graphs/{graph_id}", response_model=GraphResponse)
def get_graph(graph_id: str) -> GraphResponse:
    graph = store.get(graph_id)
    return GraphResponse(id=graph_id, payload=graph.to_dict())


@app.put("/graphs/{graph_id}")
def update_graph(graph_id: str, request: GraphPayload) -> dict[str, str]:
    store.update(graph_id, request.payload, validate=request.validate)
    return {"status": "ok"}


@app.delete("/graphs/{graph_id}")
def delete_graph(graph_id: str) -> dict[str, str]:
    store.delete(graph_id)
    return {"status": "ok"}


@app.post("/graphs/{graph_id}/diagnostics")
def diagnostics(graph_id: str) -> dict[str, Any]:
    graph = store.get(graph_id)
    return graph.diagnostics()


@app.post("/graphs/{graph_id}/credibility")
def credibility(graph_id: str) -> dict[str, Any]:
    graph = store.get(graph_id)
    result = compute_credibility(graph)
    return asdict(result)


@app.post("/graphs/{graph_id}/export")
def export_graph(graph_id: str, request: ExportRequest) -> dict[str, str]:
    graph = store.get(graph_id)
    if request.format == "dot":
        from arglib.viz import to_dot

        return {"format": "dot", "content": to_dot(graph)}
    return {"format": "json", "content": dumps(graph)}


@app.get("/graphs/{graph_id}/supporting-documents")
def list_supporting_documents(graph_id: str) -> dict[str, Any]:
    graph = store.get(graph_id)
    return graph.supporting_documents


@app.post("/graphs/{graph_id}/supporting-documents")
def add_supporting_document(
    graph_id: str, request: SupportingDocumentPayload
) -> dict[str, str]:
    graph = store.get(graph_id)
    document = SupportingDocument.from_dict(request.payload)
    graph.add_supporting_document(document, overwrite=True)
    store.update(graph_id, graph.to_dict(), validate=False)
    return {"status": "ok"}


@app.get("/graphs/{graph_id}/evidence-cards")
def list_evidence_cards(graph_id: str) -> dict[str, Any]:
    graph = store.get(graph_id)
    return graph.evidence_cards


@app.post("/graphs/{graph_id}/evidence-cards")
def add_evidence_card(
    graph_id: str, request: EvidenceCardPayload
) -> dict[str, str]:
    graph = store.get(graph_id)
    card = EvidenceCard.from_dict(request.payload)
    graph.add_evidence_card(card, overwrite=True)
    store.update(graph_id, graph.to_dict(), validate=False)
    return {"status": "ok"}


@app.post("/mining/parse")
def mining_parse(request: MiningRequest) -> GraphResponse:
    from arglib.ai import LongDocumentMiner, SimpleArgumentMiner

    miner = LongDocumentMiner(miner=SimpleArgumentMiner())
    graph = miner.parse(request.text, doc_id=request.doc_id)
    graph_id = store.create(graph.to_dict(), validate=False)
    return GraphResponse(id=graph_id, payload=graph.to_dict())


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8000)
