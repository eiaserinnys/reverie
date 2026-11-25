"""REST API routes for Memory Explorer.

Registers custom HTTP endpoints on the FastMCP server for a
dashboard Memory Explorer interface.

MemoryLink v2 only - uses bipartite graph (Entity↔Passage + Passage↔Passage).

Endpoints:
    POST /api/memory/search        - Semantic entity search + 1-hop passages
    POST /api/memory/retrieve      - Full retrieval pipeline (memorylink-v2)
    GET  /api/memory/entity/{id}/passages  - Passages containing entity
    GET  /api/memory/passage/{id}/entities - Entities in passage
    GET  /api/memory/v2/node/{type}/{id}/neighbors - Bipartite graph traversal
    GET  /api/memory/status        - DB statistics

Note: These endpoints are public (no MCP auth) and intended to be accessed
only via the dashboard backend proxy on localhost.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from starlette.requests import Request
from starlette.responses import JSONResponse

# Load .env file from project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from reverie.db.reverie_db import ReverieDB

logger = logging.getLogger(__name__)


def _entity_to_dict(entity_row) -> dict:
    """Convert an EntityRow to a JSON-serializable dict."""
    return {
        "id": entity_row.id,
        "name": entity_row.name,
        "created_at": entity_row.created_at.isoformat() if entity_row.created_at else None,
    }


def _parse_positive_int(value, default: int, max_value: int) -> int | None:
    """Parse a positive integer with bounds. Returns None on invalid input."""
    try:
        n = int(value) if value is not None else default
        return max(1, min(n, max_value))
    except (ValueError, TypeError):
        return None


def register_routes(mcp, get_db, get_stats, get_pipeline):
    """Register REST API routes on the FastMCP server.

    Args:
        mcp: FastMCP server instance.
        get_db: Callable returning a ReverieDB instance.
        get_stats: Callable returning DB statistics dict.
        get_pipeline: Callable returning a MemoryLinkV2Pipeline.
    """
    def _cors_headers() -> dict:
        # Default to "*" for setup mode, use configured value after setup
        allowed_origin = os.environ.get("REVERIE_CORS_ORIGIN", "*")
        return {
            "Access-Control-Allow-Origin": allowed_origin,
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }

    def _json_response(data: dict, status_code: int = 200) -> JSONResponse:
        return JSONResponse(data, status_code=status_code, headers=_cors_headers())

    def _error_response(message: str, status_code: int = 500) -> JSONResponse:
        return _json_response({"error": message}, status_code=status_code)

    # --- OPTIONS preflight for all /api/memory/* ---

    @mcp.custom_route("/api/memory/{rest:path}", methods=["OPTIONS"])
    async def memory_options(request: Request):
        """Handle CORS preflight for all memory API routes."""
        from starlette.responses import Response
        return Response(status_code=204, headers=_cors_headers())

    # --- POST /api/memory/search ---

    @mcp.custom_route("/api/memory/search", methods=["POST"])
    async def memory_search(request: Request):
        """Search entities by semantic similarity and return with linked passages.

        Uses v2 bipartite graph: returns passages containing the matched entities.
        """
        try:
            body = await request.json()
        except Exception:
            return _error_response("Invalid JSON body", 400)

        query = body.get("query", "").strip()
        if not query:
            return _error_response("'query' is required", 400)

        top_k = _parse_positive_int(body.get("top_k"), default=10, max_value=50)
        if top_k is None:
            return _error_response("'top_k' must be a valid positive integer", 400)

        try:
            pipeline = get_pipeline()
            db = get_db()

            # Embed the query
            query_embedding = pipeline.embedding_client.embed_text(query)

            # Search entities by embedding similarity
            search_results = db.search_entities_by_embedding(
                query_embedding, top_k=top_k
            )

            if not search_results:
                return _json_response({"results": [], "neighbors": {}})

            # Build results list
            results = []
            result_entity_ids = []
            for entity, distance in search_results:
                results.append({
                    "entity_id": entity.id,
                    "name": entity.name,
                    "distance": round(distance, 4),
                })
                result_entity_ids.append(entity.id)

            # Get passages for each entity via ReverieDB
            neighbors = {}
            for entity_id in result_entity_ids:
                passages_data = db.get_entity_passages(entity_id, limit=5)
                passages = []
                for p in passages_data:
                    passages.append({
                        "passage_id": p["id"],
                        "session_id": p["session_id"],
                        "turn_index": p["turn_index"],
                        "text": p["text"][:500] if p.get("text") else None,
                    })
                neighbors[str(entity_id)] = {"passages": passages}

            return _json_response({
                "results": results,
                "neighbors": neighbors,
            })

        except Exception:
            logger.exception("memory_search error")
            return _error_response("Internal server error")

    # --- GET /api/memory/entity/{id}/passages ---

    @mcp.custom_route("/api/memory/entity/{entity_id:int}/passages", methods=["GET"])
    async def memory_entity_passages(request: Request):
        """Get passages containing an entity (via entity_passages table)."""
        entity_id = request.path_params["entity_id"]

        limit = _parse_positive_int(
            request.query_params.get("limit"), default=20, max_value=50
        )
        if limit is None:
            return _error_response("'limit' must be a valid positive integer", 400)

        try:
            db = get_db()

            entity = db.get_entity_by_id(entity_id)
            if not entity:
                return _error_response(f"Entity {entity_id} not found", 404)

            # Get passages via ReverieDB
            passage_list = db.get_entity_passages(entity_id, limit=limit)

            return _json_response({
                "entity_id": entity_id,
                "passages": passage_list,
            })

        except Exception:
            logger.exception("memory_entity_passages error")
            return _error_response("Internal server error")

    # --- GET /api/memory/passage/{id}/entities ---

    @mcp.custom_route("/api/memory/passage/{passage_id:int}/entities", methods=["GET"])
    async def memory_passage_entities(request: Request):
        """Get entities in a passage via entity_passages table."""
        passage_id = request.path_params["passage_id"]

        try:
            db = get_db()

            # Verify passage exists
            passage = db.get_passage_by_id(passage_id)
            if not passage:
                return _error_response(f"Passage {passage_id} not found", 404)

            # Get entities linked to this passage via ReverieDB
            entities = db.get_passage_entities(passage_id)

            return _json_response({
                "passage_id": passage_id,
                "session_id": passage.session_id,
                "turn_index": passage.turn_index,
                "text": passage.text,
                "entities": entities,
            })

        except Exception:
            logger.exception("memory_passage_entities error")
            return _error_response("Internal server error")

    # --- POST /api/memory/retrieve ---

    @mcp.custom_route("/api/memory/retrieve", methods=["POST"])
    async def memory_retrieve(request: Request):
        """Run MemoryLink v2 retrieval pipeline (Fully Lazy).

        Request body:
            query: str - The search query
            top_k: int - Max passages to return (default: 10, max: 50)
            tau: float - Entity activation threshold (default: from config)
            damping: float - PPR damping factor (default: from config)
            beta: float - PPR/KNN hybrid weight (default: from config)
            ppr_min_score: float - PPR node pruning threshold (default: 0.0001)
                Lower values (1e-4 to 1e-7) give richer results but slower PPR.
                Higher values (1e-3) give faster PPR but less diversity.
            score_threshold: float - Min PPR score for graph nodes (default: 0.001)
            half_life_days: float - Time decay half-life in days (default: None = no decay)
                When set, older passages get exponentially lower scores.
                Range: 7-1000 days. 365 = score halves after 1 year.

        Response:
            results: list of {passage_id, text, score, source, entity_ids, ...}
            entities: list of {entity_id, name}
            seed_entities: list of {entity_id, name, score}
            graph_nodes: list of graph nodes for visualization
            graph_edges: list of graph edges for visualization
            algorithm: str - Always "memorylink-v2"
            elapsed_ms: float
            timing: Stage-by-stage timing breakdown:
                total_ms: float - Total pipeline time
                stages: dict - Per-stage durations (embed_query, knn_search, etc.)
                cache_hit: bool - Always true (fully lazy, no graph cache)
        """
        try:
            body = await request.json()
        except Exception:
            return _error_response("Invalid JSON body", 400)

        query = body.get("query", "").strip()
        if not query:
            return _error_response("'query' is required", 400)

        top_k = _parse_positive_int(body.get("top_k"), default=10, max_value=50)
        if top_k is None:
            return _error_response("'top_k' must be a valid positive integer", 400)

        # Parse v2 parameters (use config defaults if not provided)
        score_threshold = body.get("score_threshold", 0.001)

        try:
            import time
            request_start = time.time()
            api_stages = {}  # Track all API-level timing

            from reverie.config import get_settings
            from reverie.retriever.pipeline_v2 import MemoryLinkV2Pipeline

            settings = get_settings()
            db = get_db()

            # Get the pipeline (uses shared embedding client)
            pipeline = get_pipeline()

            # Override parameters if provided
            tau = body.get("tau", pipeline._tau)
            damping = body.get("damping", pipeline._damping)
            beta = body.get("beta", pipeline._beta)

            # PPR min_score parameter (1e-7 to 1e-3 range)
            ppr_min_score = body.get("ppr_min_score", None)
            if ppr_min_score is not None:
                try:
                    ppr_min_score = float(ppr_min_score)
                    if ppr_min_score < 1e-7 or ppr_min_score > 1e-2:
                        return _error_response("'ppr_min_score' must be between 1e-7 and 1e-2", 400)
                except (ValueError, TypeError):
                    return _error_response("'ppr_min_score' must be a valid number", 400)

            # PPR max_iter parameter (10-50 range)
            ppr_max_iter = body.get("ppr_max_iter", None)
            if ppr_max_iter is not None:
                try:
                    ppr_max_iter = int(ppr_max_iter)
                    if ppr_max_iter < 10 or ppr_max_iter > 50:
                        return _error_response("'ppr_max_iter' must be between 10 and 50", 400)
                except (ValueError, TypeError):
                    return _error_response("'ppr_max_iter' must be a valid integer", 400)

            # Time decay parameter (None = disabled, 7-1000 days)
            half_life_days = body.get("half_life_days", None)
            if half_life_days is not None:
                try:
                    half_life_days = float(half_life_days)
                    if half_life_days < 7 or half_life_days > 1000:
                        return _error_response("'half_life_days' must be between 7 and 1000", 400)
                except (ValueError, TypeError):
                    return _error_response("'half_life_days' must be a valid number", 400)

            # Update pipeline parameters
            pipeline._tau = tau
            pipeline._damping = damping
            pipeline._beta = beta
            pipeline._half_life_days = half_life_days

            # Run retrieval with timing (ppr_min_score, ppr_max_iter passed per-query)
            t0 = time.time()
            retrieval_response = pipeline.retrieve_with_timing(
                query=query,
                top_k=top_k,
                ppr_min_score=ppr_min_score,
                ppr_max_iter=ppr_max_iter,
            )
            results = retrieval_response.results
            timing_data = retrieval_response.timing
            api_stages["pipeline"] = round((time.time() - t0) * 1000, 1)

            # Get seed entities (activated in Stage 1)
            t0 = time.time()
            seed_entities = pipeline.get_seed_entities(top_k=20)
            api_stages["get_seed_entities"] = round((time.time() - t0) * 1000, 1)

            # Build response with entities included (query DB directly, no graph)
            t0 = time.time()
            result_list = []
            all_entity_ids = set()

            # Batch fetch entities for all result passages
            result_passage_ids = [r.passage_id for r in results]
            passage_entities_map = db.get_passage_entity_ids(result_passage_ids)
            for entity_ids_set in passage_entities_map.values():
                all_entity_ids.update(entity_ids_set)

            for r in results:
                passage_entity_ids = passage_entities_map.get(r.passage_id, set())

                result_item = {
                    "passage_id": r.passage_id,
                    "text": r.text[:2000],
                    "score": round(r.score, 6),
                    "source": r.source,
                    "entity_ids": list(passage_entity_ids),
                    "session_seq": r.session_seq,
                    "turn_index": r.turn_index,
                    "session_id": r.session_id,
                }
                # Add time decay info if present
                if r.original_score is not None:
                    result_item["original_score"] = round(r.original_score, 6)
                if r.decay_weight is not None:
                    result_item["decay_weight"] = round(r.decay_weight, 4)
                if r.created_at is not None:
                    result_item["created_at"] = r.created_at.isoformat()
                result_list.append(result_item)
            api_stages["build_result_list"] = round((time.time() - t0) * 1000, 1)

            # Filter seed entities to only those that appear in result passages
            t0 = time.time()
            seed_entity_ids = {eid for eid, _ in seed_entities}
            relevant_seed_ids = seed_entity_ids & all_entity_ids

            # If no overlap, include top seeds anyway for context
            if not relevant_seed_ids:
                relevant_seed_ids = seed_entity_ids

            all_entity_ids.update(relevant_seed_ids)

            entities_map = {}
            if all_entity_ids:
                entities_db = db.get_entities_by_ids(list(all_entity_ids))
                for e in entities_db:
                    entities_map[e.id] = {"entity_id": e.id, "name": e.name}

            # Build seed entities with scores (only relevant ones)
            seed_entities_list = [
                {"entity_id": eid, "name": entities_map[eid]["name"], "score": round(score, 6)}
                for eid, score in seed_entities
                if eid in relevant_seed_ids and eid in entities_map
            ]
            api_stages["fetch_result_entities"] = round((time.time() - t0) * 1000, 1)

            # Get full traversal graph for visualization
            t0 = time.time()
            traversal_graph = pipeline.get_traversal_graph(score_threshold=score_threshold)
            api_stages["get_traversal_graph"] = round((time.time() - t0) * 1000, 1)

            # Enrich graph nodes with entity names and passage metadata
            t0 = time.time()
            graph_nodes = []
            graph_edges = []
            if traversal_graph:
                # Fetch all entity names for graph nodes
                graph_entity_ids = [
                    n["entity_id"] for n in traversal_graph["graph_nodes"]
                    if n["type"] == "entity"
                ]
                if graph_entity_ids:
                    graph_entities_db = db.get_entities_by_ids(graph_entity_ids)
                    graph_entity_names = {e.id: e.name for e in graph_entities_db}
                else:
                    graph_entity_names = {}

                # Fetch passage metadata for graph nodes
                graph_passage_ids = [
                    n["passage_id"] for n in traversal_graph["graph_nodes"]
                    if n["type"] == "passage"
                ]
                graph_passage_meta = {}
                if graph_passage_ids:
                    passages_db = db.get_passages_by_ids(graph_passage_ids)
                    for p in passages_db:
                        graph_passage_meta[p.id] = {
                            "session_seq": p.session_seq,
                            "turn_index": p.turn_index,
                            "session_id": p.session_id,
                        }

                # Build enriched graph nodes
                for node in traversal_graph["graph_nodes"]:
                    if node["type"] == "entity":
                        graph_nodes.append({
                            "id": node["id"],
                            "type": "entity",
                            "entity_id": node["entity_id"],
                            "name": graph_entity_names.get(node["entity_id"], ""),
                            "score": round(node["score"], 6),
                        })
                    else:
                        meta = graph_passage_meta.get(node["passage_id"], {})
                        graph_nodes.append({
                            "id": node["id"],
                            "type": "passage",
                            "passage_id": node["passage_id"],
                            "score": round(node["score"], 6),
                            "session_seq": meta.get("session_seq"),
                            "turn_index": meta.get("turn_index"),
                            "session_id": meta.get("session_id"),
                        })

                graph_edges = traversal_graph["graph_edges"]
            api_stages["enrich_graph_nodes"] = round((time.time() - t0) * 1000, 1)

            # Get query entities extracted via NER (may be None if NER failed/not used)
            query_entities = pipeline.get_query_entities()

            # Build timing dict with all stages
            timing_dict = timing_data.to_dict()
            timing_dict["api_stages"] = api_stages

            # Calculate total elapsed
            elapsed_ms = (time.time() - request_start) * 1000

            response_data = {
                "results": result_list,
                "entities": list(entities_map.values()),
                "seed_entities": seed_entities_list,
                "query_entities": query_entities,  # NER-extracted entities from query
                "graph_nodes": graph_nodes,
                "graph_edges": graph_edges,
                "algorithm": "memorylink-v2",
                "query": query,
                "elapsed_ms": round(elapsed_ms, 1),
                "timing": timing_dict,
                "params": {
                    "tau": tau,
                    "damping": damping,
                    "beta": beta,
                    "ppr_min_score": ppr_min_score or pipeline._ppr_min_score,
                    "score_threshold": score_threshold,
                    "half_life_days": half_life_days,
                },
            }

            return _json_response(response_data)

        except Exception:
            logger.exception("memory_retrieve error")
            return _error_response("Internal server error")

    # --- GET /api/memory/v2/node/{type}/{id}/neighbors ---

    @mcp.custom_route("/api/memory/v2/node/{node_type}/{node_id:int}/neighbors", methods=["GET"])
    async def memory_v2_node_neighbors(request: Request):
        """Get neighbors of a node (entity or passage) via direct DB queries.

        Uses simple SQL queries instead of building the full graph.
        - Entity → Passage: via entity_passages table
        - Passage → Entity: via entity_passages table
        - Passage → Passage: via passage_adjacency table

        Args:
            node_type: "entity" or "passage"
            node_id: DB ID of the node

        Response:
            node: The source node info
            entities: List of connected entities (for passage nodes)
            passages: List of connected passages
        """
        node_type = request.path_params["node_type"]
        node_id = request.path_params["node_id"]

        if node_type not in ("entity", "passage"):
            return _error_response("node_type must be 'entity' or 'passage'", 400)

        try:
            db = get_db()

            if node_type == "entity":
                # Get entity info
                entity = db.get_entity_by_id(node_id)
                if not entity:
                    return _error_response(f"Entity {node_id} not found", 404)

                # Get passages containing this entity via ReverieDB
                passages_data = db.get_entity_passages(node_id, limit=50)
                passages_list = [
                    {
                        "passage_id": p["id"],
                        "session_id": p["session_id"],
                        "turn_index": p["turn_index"],
                        "session_seq": p["session_seq"],
                        "text": p["text"][:200] if p.get("text") else None,
                    }
                    for p in passages_data
                ]

                return _json_response({
                    "node": {
                        "type": "entity",
                        "id": entity.id,
                        "name": entity.name,
                    },
                    "entities": [],  # Entities don't connect to other entities in v2
                    "passages": passages_list,
                })

            else:  # passage
                # Get passage info
                passage = db.get_passage_by_id(node_id)
                if not passage:
                    return _error_response(f"Passage {node_id} not found", 404)

                # Get entities in this passage via ReverieDB
                entities_list = db.get_passage_entities(node_id)

                # Get adjacent passages via ReverieDB
                passages_list = db.get_adjacent_passages(node_id)

                return _json_response({
                    "node": {
                        "type": "passage",
                        "id": passage.id,
                        "session_id": passage.session_id,
                        "turn_index": passage.turn_index,
                        "session_seq": passage.session_seq,
                        "text": passage.text[:200] if passage.text else None,
                    },
                    "entities": entities_list,
                    "passages": passages_list,
                })

        except Exception:
            logger.exception("memory_v2_node_neighbors error")
            return _error_response("Internal server error")

    # --- Legacy API stubs (return deprecation warnings) ---

    @mcp.custom_route("/api/memory/entity/{entity_id:int}/neighbors", methods=["GET"])
    async def memory_entity_neighbors_deprecated(request: Request):
        """Deprecated: Use /api/memory/v2/node/entity/{id}/neighbors instead."""
        return _error_response(
            "This endpoint is deprecated. Use /api/memory/v2/node/entity/{id}/neighbors",
            410  # Gone
        )

    @mcp.custom_route("/api/memory/enrich/stream", methods=["GET"])
    async def memory_enrich_stream(request: Request):
        """SSE endpoint for enrich with progress updates.

        Query params:
            query: str - The search query
            top_k: int - Max passages to retrieve (default: 25)
        """
        from starlette.responses import StreamingResponse
        import asyncio
        import json as json_module
        import time

        query = request.query_params.get("query", "").strip()
        if not query:
            async def error_gen():
                yield f"data: {json_module.dumps({'type': 'error', 'error': 'query required'})}\n\n"
            return StreamingResponse(error_gen(), media_type="text/event-stream", headers=_cors_headers())

        top_k = _parse_positive_int(request.query_params.get("top_k"), default=25, max_value=50)

        async def event_generator():
            try:
                from reverie.retriever.enricher import EnricherWithProgress

                request_start = time.time()
                pipeline = get_pipeline()

                # Run retrieval
                retrieval_response = pipeline.retrieve_with_timing(
                    query=query,
                    top_k=top_k,
                )
                results = retrieval_response.results

                passages = [{
                    "text": r.text,
                    "score": r.score,
                    "session_seq": r.session_seq,
                    "turn_index": r.turn_index,
                    "session_id": r.session_id,
                    "passage_id": r.passage_id,
                } for r in results]

                # Create enricher with progress callback
                enricher = EnricherWithProgress()

                def progress_callback(stage: str, current: int, total: int, message: str = ""):
                    # We'll collect progress updates in a queue
                    enricher._progress_queue.append({
                        "stage": stage,
                        "current": current,
                        "total": total,
                        "message": message,
                    })

                enricher.set_progress_callback(progress_callback)

                # Run enricher in background thread
                loop = asyncio.get_event_loop()
                import concurrent.futures
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                future = loop.run_in_executor(executor, enricher.enrich, query, passages)

                # Stream progress updates while waiting
                while not future.done():
                    # Drain progress queue
                    while enricher._progress_queue:
                        progress = enricher._progress_queue.pop(0)
                        yield f"data: {json_module.dumps({'type': 'progress', **progress})}\n\n"
                    await asyncio.sleep(0.1)

                # Get final result
                facts = await future

                elapsed_ms = (time.time() - request_start) * 1000

                retrieve_results = [{
                    "passage_id": r.passage_id,
                    "text": r.text[:500],
                    "score": round(r.score, 6),
                    "session_seq": r.session_seq,
                    "turn_index": r.turn_index,
                } for r in results]

                result = {
                    "query": query,
                    "facts": facts,
                    "passage_count": len(passages),
                    "retrieve_results": retrieve_results,
                    "elapsed_ms": round(elapsed_ms, 1),
                }

                yield f"data: {json_module.dumps({'type': 'result', 'result': result})}\n\n"

            except Exception as e:
                logger.exception("enrich stream error")
                yield f"data: {json_module.dumps({'type': 'error', 'error': str(e)})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                **_cors_headers(),
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    @mcp.custom_route("/api/memory/enrich", methods=["POST"])
    async def memory_enrich(request: Request):
        """Enrich retrieved passages into scored atomic facts.

        This is the final stage of the retrieval pipeline:
        1. First calls /api/memory/retrieve to get relevant passages
        2. Then uses LLM (gpt-5-mini) to extract scored facts

        Request body:
            query: str - The search query
            top_k: int - Max passages to retrieve (default: 20)
            (All other params are passed to retrieve endpoint)

        Response:
            query: str - Original query
            facts: list[{score: int, text: str}] - Scored facts (1-10)
            passage_count: int - Number of passages retrieved
            retrieve_results: list - Raw retrieve results (for debugging)
            elapsed_ms: float
        """
        try:
            body = await request.json()
        except Exception:
            return _error_response("Invalid JSON body", 400)

        query = body.get("query", "").strip()
        if not query:
            return _error_response("'query' is required", 400)

        top_k = _parse_positive_int(body.get("top_k"), default=20, max_value=50)
        if top_k is None:
            return _error_response("'top_k' must be a valid positive integer", 400)

        try:
            import asyncio
            import time
            request_start = time.time()

            from reverie.retriever.enricher import Enricher

            # Get the pipeline for retrieval
            pipeline = get_pipeline()

            # Run retrieval first (blocking but fast)
            retrieval_response = pipeline.retrieve_with_timing(
                query=query,
                top_k=top_k,
            )
            results = retrieval_response.results

            # Prepare passages for enricher
            passages = []
            for r in results:
                passages.append({
                    "text": r.text,
                    "score": r.score,
                    "session_seq": r.session_seq,
                    "turn_index": r.turn_index,
                    "session_id": r.session_id,
                    "passage_id": r.passage_id,
                })

            # Run enricher in executor to avoid blocking event loop
            # This is critical - LLM calls can take 30-60 seconds
            loop = asyncio.get_event_loop()
            enricher = Enricher()
            facts = await loop.run_in_executor(
                None,  # Use default executor
                enricher.enrich,
                query,
                passages,
            )

            elapsed_ms = (time.time() - request_start) * 1000

            # Build retrieve_results for debugging
            retrieve_results = [
                {
                    "passage_id": r.passage_id,
                    "text": r.text[:500],  # Truncate for response
                    "score": round(r.score, 6),
                    "session_seq": r.session_seq,
                    "turn_index": r.turn_index,
                }
                for r in results
            ]

            return _json_response({
                "query": query,
                "facts": facts,
                "passage_count": len(passages),
                "retrieve_results": retrieve_results,
                "elapsed_ms": round(elapsed_ms, 1),
            })

        except Exception:
            logger.exception("memory_enrich error")
            return _error_response("Internal server error")

    @mcp.custom_route("/api/memory/cooccurrence/entity/{entity_id:int}/neighbors", methods=["GET"])
    async def memory_cooccurrence_neighbors_deprecated(request: Request):
        """Deprecated: Co-occurrence graph removed in v2."""
        return _error_response(
            "Co-occurrence graph is deprecated. Use /api/memory/v2/node/entity/{id}/neighbors",
            410
        )

    @mcp.custom_route("/api/memory/cooccurrence/node/{node_type}/{node_id:int}/neighbors", methods=["GET"])
    async def memory_cooccurrence_node_neighbors_deprecated(request: Request):
        """Deprecated: Co-occurrence graph removed in v2."""
        return _error_response(
            "Co-occurrence graph is deprecated. Use /api/memory/v2/node/{type}/{id}/neighbors",
            410
        )

    @mcp.custom_route("/api/memory/cooccurrence/edges", methods=["POST"])
    async def memory_cooccurrence_edges_deprecated(request: Request):
        """Deprecated: Co-occurrence graph removed in v2."""
        return _error_response(
            "Co-occurrence graph is deprecated in MemoryLink v2",
            410
        )

    @mcp.custom_route("/api/memory/entity-paths", methods=["POST"])
    async def memory_entity_paths_deprecated(request: Request):
        """Deprecated: Entity paths (Dijkstra) removed in v2."""
        return _error_response(
            "Entity paths endpoint is deprecated in MemoryLink v2",
            410
        )

    # --- GET /api/memory/status ---

    @mcp.custom_route("/api/memory/status", methods=["GET"])
    async def memory_status_api(request: Request):
        """Return DB statistics as JSON."""
        try:
            stats = get_stats()
            return _json_response(stats)
        except Exception:
            logger.exception("memory_status_api error")
            return _error_response("Internal server error")

    # --- GET /api/memory/entities ---

    @mcp.custom_route("/api/memory/entities", methods=["GET"])
    async def memory_entities_list(request: Request):
        """List entities with pagination.

        Query params:
            offset: int - Starting offset (default: 0)
            limit: int - Max items to return (default: 50, max: 200)
            sort: str - Sort field: 'id', 'name', 'created_at' (default: 'id')
            order: str - Sort order: 'asc', 'desc' (default: 'desc')

        Response:
            entities: list of {id, name, created_at, passage_count}
            total: int - Total entity count
            offset: int
            limit: int
        """
        offset = _parse_positive_int(request.query_params.get("offset"), default=0, max_value=100000)
        if offset is None:
            offset = 0
        limit = _parse_positive_int(request.query_params.get("limit"), default=50, max_value=200)
        if limit is None:
            limit = 50

        sort_field = request.query_params.get("sort", "id")
        order = request.query_params.get("order", "desc")

        try:
            db = get_db()
            entities, total = db.list_entities(
                limit=limit,
                offset=offset,
                sort_field=sort_field,
                sort_order=order,
            )

            return _json_response({
                "entities": entities,
                "total": total,
                "offset": offset,
                "limit": limit,
            })

        except Exception:
            logger.exception("memory_entities_list error")
            return _error_response("Internal server error")

    # --- GET /api/memory/passages ---

    @mcp.custom_route("/api/memory/passages", methods=["GET"])
    async def memory_passages_list(request: Request):
        """List passages with pagination.

        Query params:
            offset: int - Starting offset (default: 0)
            limit: int - Max items to return (default: 50, max: 200)
            sort: str - Sort field: 'id', 'session_seq', 'turn_index' (default: 'id')
            order: str - Sort order: 'asc', 'desc' (default: 'desc')

        Response:
            passages: list of {id, session_id, session_seq, turn_index, text, entity_count}
            total: int - Total passage count
            offset: int
            limit: int
        """
        offset = _parse_positive_int(request.query_params.get("offset"), default=0, max_value=100000)
        if offset is None:
            offset = 0
        limit = _parse_positive_int(request.query_params.get("limit"), default=50, max_value=200)
        if limit is None:
            limit = 50

        sort_field = request.query_params.get("sort", "id")
        order = request.query_params.get("order", "desc")

        try:
            db = get_db()
            passages, total = db.list_passages(
                limit=limit,
                offset=offset,
                sort_field=sort_field,
                sort_order=order,
            )

            return _json_response({
                "passages": passages,
                "total": total,
                "offset": offset,
                "limit": limit,
            })

        except Exception:
            logger.exception("memory_passages_list error")
            return _error_response("Internal server error")

    logger.info("Memory REST API routes registered (v2 only)")


def register_index_routes(mcp, get_scheduler):
    """Register indexing API routes on the FastMCP server.

    Args:
        mcp: FastMCP server instance.
        get_scheduler: Callable returning the IndexScheduler instance.
    """
    def _cors_headers() -> dict:
        # Default to "*" for setup mode, use configured value after setup
        allowed_origin = os.environ.get("REVERIE_CORS_ORIGIN", "*")
        return {
            "Access-Control-Allow-Origin": allowed_origin,
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }

    def _json_response(data: dict, status_code: int = 200) -> JSONResponse:
        return JSONResponse(data, status_code=status_code, headers=_cors_headers())

    def _error_response(message: str, status_code: int = 500) -> JSONResponse:
        return _json_response({"error": message}, status_code=status_code)

    # --- OPTIONS preflight for all /api/index/* ---

    @mcp.custom_route("/api/index/{rest:path}", methods=["OPTIONS"])
    async def index_options(request: Request):
        """Handle CORS preflight for all index API routes."""
        from starlette.responses import Response
        return Response(status_code=204, headers=_cors_headers())

    # --- POST /api/index/session ---

    @mcp.custom_route("/api/index/session", methods=["POST"])
    async def index_session(request: Request):
        """Trigger indexing for a specific session.

        Request body:
            profile_id: str - Profile identifier
            session_id: str - Session identifier

        Response:
            status: "queued" | "already_processing" | "retry_scheduled"
            profile_id: str
            session_id: str
        """
        try:
            body = await request.json()
        except Exception:
            return _error_response("Invalid JSON body", 400)

        profile_id = body.get("profile_id", "").strip()
        session_id = body.get("session_id", "").strip()

        if not profile_id or not session_id:
            return _error_response("'profile_id' and 'session_id' are required", 400)

        try:
            scheduler = get_scheduler()
            if scheduler is None:
                return _error_response("Index scheduler not running", 503)

            immediate = await scheduler.trigger_session(profile_id, session_id)

            return _json_response({
                "status": "queued" if immediate else "retry_scheduled",
                "profile_id": profile_id,
                "session_id": session_id,
            })

        except Exception:
            logger.exception("index_session error")
            return _error_response("Internal server error")

    # --- POST /api/index/reindex ---

    @mcp.custom_route("/api/index/reindex", methods=["POST"])
    async def index_reindex(request: Request):
        """Trigger reindexing for a profile or specific sessions.

        Request body:
            profile_id: str - Profile to reindex (optional, all if not provided)
            session_list: list[str] - Specific sessions to reindex (optional)
            confirm: bool - Must be true to proceed
            dry_run: bool - If true, just count sessions without queueing

        Response:
            sessions_queued: number
            profile_id: str | null
        """
        try:
            body = await request.json()
        except Exception:
            return _error_response("Invalid JSON body", 400)

        profile_id = body.get("profile_id", "").strip() or None
        session_list = body.get("session_list")
        confirm = body.get("confirm", False)
        dry_run = body.get("dry_run", False)

        if not confirm and not dry_run:
            return _error_response("'confirm' must be true to reindex", 400)

        try:
            scheduler = get_scheduler()
            if scheduler is None:
                return _error_response("Index scheduler not running", 503)

            if dry_run:
                # Just count sessions that would be queued
                from reverie.config import get_profile, get_profiles
                from pathlib import Path

                if profile_id:
                    profile = get_profile(profile_id)
                    if not profile:
                        return _error_response(f"Unknown profile: {profile_id}", 400)
                    sessions_dir = profile.sessions_path
                    count = len(list(sessions_dir.glob("*.jsonl"))) if sessions_dir.exists() else 0
                else:
                    count = 0
                    for p in get_profiles().values():
                        if p.sessions_path.exists():
                            count += len(list(p.sessions_path.glob("*.jsonl")))

                return _json_response({
                    "sessions_queued": count,
                    "profile_id": profile_id,
                    "dry_run": True,
                })

            count = await scheduler.trigger_reindex(
                profile_id=profile_id,
                session_ids=session_list,
            )

            return _json_response({
                "sessions_queued": count,
                "profile_id": profile_id,
            })

        except Exception:
            logger.exception("index_reindex error")
            return _error_response("Internal server error")

    # --- GET /api/index/status ---

    @mcp.custom_route("/api/index/status", methods=["GET"])
    async def index_status(request: Request):
        """Get indexing status and statistics.

        Query params:
            profile_id: str - Filter by profile (optional)

        Response:
            running: bool
            queue_size: int
            workers: dict
            profiles: dict[profile_id, stats]
        """
        profile_id = request.query_params.get("profile_id")

        try:
            scheduler = get_scheduler()

            if scheduler is None:
                return _json_response({
                    "running": False,
                    "queue_size": 0,
                    "workers": {},
                    "profiles": {},
                })

            stats = scheduler.stats

            # Get per-profile stats from DB
            from reverie.config import get_profile, get_profiles
            from reverie.db.connection import get_connection
            from reverie.db.repositories import SessionIndexStateRepository

            profile_stats = {}
            profiles = get_profiles()

            if profile_id:
                profiles = {profile_id: profiles.get(profile_id)} if profile_id in profiles else {}

            with get_connection() as conn:
                state_repo = SessionIndexStateRepository(conn)
                for pid in profiles:
                    profile_stats[pid] = state_repo.get_stats(pid)

            return _json_response({
                "running": stats["running"],
                "queue": stats["queue"],
                "pool": stats["pool"],
                "watcher": stats["watcher"],
                "sdk": stats["sdk"],
                "profiles": profile_stats,
            })

        except Exception:
            logger.exception("index_status error")
            return _error_response("Internal server error")

    # --- GET /api/index/session/{profile_id}/{session_id} ---

    @mcp.custom_route("/api/index/session/{profile_id}/{session_id}", methods=["GET"])
    async def index_session_status(request: Request):
        """Get indexing status for a specific session.

        Response:
            profile_id: str
            session_id: str
            session_seq: int
            status: "pending" | "in_progress" | "completed" | "failed"
            last_indexed_turn: int
            total_passages: int
            total_dropped: int
            error_message: str | null
            history: list[{turn_index, action, drop_reason, processed_at}]
        """
        profile_id = request.path_params["profile_id"]
        session_id = request.path_params["session_id"]

        try:
            scheduler = get_scheduler()
            if scheduler is None:
                return _error_response("Index scheduler not running", 503)

            status = scheduler.get_session_status(profile_id, session_id)
            if status is None:
                return _error_response(f"Session {profile_id}/{session_id} not found", 404)

            return _json_response(status)

        except Exception:
            logger.exception("index_session_status error")
            return _error_response("Internal server error")

    logger.info("Index REST API routes registered")
