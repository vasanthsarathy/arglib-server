"""ArgLib server application."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal
from uuid import uuid4

from arglib.core import ArgumentGraph
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


class GraphStore:
    def __init__(self) -> None:
        self._graphs: dict[str, ArgumentGraph] = {}

    def create(self, payload: dict[str, Any], *, validate: bool = True) -> str:
        if validate:
            validate_graph_payload(payload)
        graph = ArgumentGraph.from_dict(payload)
        graph_id = uuid4().hex
        self._graphs[graph_id] = graph
        return graph_id

    def get(self, graph_id: str) -> ArgumentGraph:
        try:
            return self._graphs[graph_id]
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="graph not found") from exc

    def update(self, graph_id: str, payload: dict[str, Any], *, validate: bool) -> None:
        if validate:
            validate_graph_payload(payload)
        self._graphs[graph_id] = ArgumentGraph.from_dict(payload)

    def delete(self, graph_id: str) -> None:
        if graph_id not in self._graphs:
            raise HTTPException(status_code=404, detail="graph not found")
        del self._graphs[graph_id]


store = GraphStore()


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


@app.post("/mining/parse")
def mining_parse(request: MiningRequest) -> GraphResponse:
    from arglib.ai import LongDocumentMiner, SimpleArgumentMiner

    miner = LongDocumentMiner(miner=SimpleArgumentMiner())
    graph = miner.parse(request.text, doc_id=request.doc_id)
    graph_id = uuid4().hex
    store._graphs[graph_id] = graph
    return GraphResponse(id=graph_id, payload=graph.to_dict())


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8000)
