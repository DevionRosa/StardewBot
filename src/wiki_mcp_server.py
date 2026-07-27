"""MCP server for the deterministic Stardew wiki index.

Speaks JSON-RPC over stdio (the Model Context Protocol). Tools:

* ``search_wiki`` / ``get_wiki_context`` / ``answer_from_wiki`` — legacy RAG.
* ``lookup_entity`` — name → structured ``stardew_entity`` row.
* ``compose_answer`` — entity-driven deterministic answer (no LLM).
* ``get_vocabulary`` — Stardew proper-noun vocab for external fuzzy matchers.
* ``fuzzy_correct_transcript`` — apply the local vocabulary corrector.
* ``get_structured_answer`` — convenience: lookup + compose in one call.
"""
from __future__ import annotations

import json
import sys
from typing import Any

from .wiki_index import (
    compose_stardew_answer,
    get_stardew_vocabulary,
    get_wiki_context,
    get_wiki_index,
    lookup_stardew_body,
    lookup_stardew_entity,
    search_stardew_entities,
)

SERVER_INFO = {"name": "stardew-wiki-mcp", "version": "1.1.0"}


def _text_content(text: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text}]


def _ok(result: Any, request_id: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def _tool_schemas() -> dict[str, Any]:
    return {
        "tools": [
            {
                "name": "search_wiki",
                "description": "Search the Stardew wiki cache and return grounded passages.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_wiki_context",
                "description": "Return compact wiki passages for a Stardew question.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "answer_from_wiki",
                "description": "Return a direct wiki-grounded answer string for the query.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "refresh_wiki_cache",
                "description": "Return ranked wiki passages — useful to confirm cache freshness.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "lookup_entity",
                "description": "Resolve a free-form query to its structured stardew_entity row.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            {
                "name": "compose_answer",
                "description": "Build a deterministic, no-LLM answer from a matched entity.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            {
                "name": "get_vocabulary",
                "description": "Return the local Stardew vocabulary as a JSON list of words.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "fuzzy_correct_transcript",
                "description": "Apply vocab-aware fuzzy correction to an LLM/ASR transcript.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"transcript": {"type": "string"}},
                    "required": ["transcript"],
                },
            },
            {
                "name": "get_structured_answer",
                "description": "Try lookup → compose, then fall back to wiki passage text.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            {
                "name": "get_corpus_stats",
                "description": "Return counts of cached wiki pages, entities, and vocabulary.",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]
    }


def _call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    query = str(arguments.get("query", "")).strip() if "query" in arguments else ""
    limit = int(arguments.get("limit", 5) or 5)

    if name == "search_wiki":
        passages = get_wiki_index().search(query, limit=limit)
        text = "\n\n".join(f"[{passage.title}] {passage.extract}" for passage in passages) or "No wiki results found."
        return {"content": _text_content(text), "isError": False}

    if name == "get_wiki_context":
        text = get_wiki_context(query, limit=limit) or "No wiki context found."
        return {"content": _text_content(text), "isError": False}

    if name == "answer_from_wiki":
        text = get_wiki_index().answer_from_context(query, limit=limit) or "No wiki answer found."
        return {"content": _text_content(text), "isError": False}

    if name == "refresh_wiki_cache":
        passages = get_wiki_index().search(query, limit=limit)
        text = "\n\n".join(f"[{passage.title}] {passage.extract}" for passage in passages) or "No wiki results found."
        return {"content": _text_content(text), "isError": False}

    if name == "lookup_entity":
        entity = lookup_stardew_entity(query)
        if not entity:
            return {"content": _text_content("No structured entity matched."), "isError": False}
        payload = {
            "canonical_name": entity.canonical_name,
            "display_name": entity.display_name,
            "kind": entity.kind,
            "aliases": list(entity.aliases),
            "attributes": entity.attributes,
        }
        return {"content": _text_content(json.dumps(payload, ensure_ascii=False, indent=2)), "isError": False}

    if name == "compose_answer":
        answer = compose_stardew_answer(query) or "No structured answer available for that query."
        return {"content": _text_content(answer), "isError": False}

    if name == "get_vocabulary":
        words = sorted(get_stardew_vocabulary())
        return {"content": _text_content(json.dumps(words, ensure_ascii=False)), "isError": False}

    if name == "fuzzy_correct_transcript":
        transcript = str(arguments.get("transcript", ""))
        from .transcribe import fuzzy_correct_transcript
        corrected = fuzzy_correct_transcript(transcript)
        return {"content": _text_content(json.dumps({"original": transcript, "corrected": corrected})), "isError": False}

    if name == "get_structured_answer":
        composed = compose_stardew_answer(query)
        if composed:
            return {"content": _text_content(composed), "isError": False}
        entities = search_stardew_entities(query, limit=3)
        if entities:
            text = "\n".join(entity.display_name + ": " + ", ".join(entity.attributes.keys()) for entity in entities)
            return {"content": _text_content(text), "isError": False}
        body = lookup_stardew_body(query)
        if body:
            return {"content": _text_content(body[:400]), "isError": False}
        return {"content": _text_content("No structured answer available."), "isError": False}

    if name == "get_corpus_stats":
        return {"content": _text_content(json.dumps(get_wiki_index().corpus_stats())), "isError": False}

    return {"content": _text_content(f"Unknown tool: {name}"), "isError": True}


def handle_request(payload: dict[str, Any]) -> dict[str, Any] | None:
    method = payload.get("method")
    request_id = payload.get("id")
    params = payload.get("params") or {}

    if method == "initialize":
        return _ok({"protocolVersion": "2024-11-05", "serverInfo": SERVER_INFO, "capabilities": {"tools": {}}}, request_id)

    if method == "initialized":
        return None

    if method == "ping":
        return _ok({}, request_id)

    if method == "tools/list":
        return _ok(_tool_schemas(), request_id)

    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not name:
            return _error(request_id, -32602, "Missing tool name")
        return _ok(_call_tool(name, arguments), request_id)

    if request_id is None:
        return None
    return _error(request_id, -32601, f"Method not found: {method}")


def main() -> None:
    for line in sys.stdin:
        raw = line.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            response = _error(None, -32700, f"Parse error: {exc}")
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
            continue
        try:
            response = handle_request(payload)
        except Exception as exc:
            response = _error(payload.get("id"), -32603, f"Internal error: {exc}")
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    get_wiki_index()  # warm the singleton before serving
    main()
