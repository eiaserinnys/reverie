/** Types for the Memory Explorer graph visualization. */

export interface LinkedEntity {
  entity_id: number;
  name: string;
  importance: number;
}

export interface GraphNode {
  id: string;
  name: string;
  group: 'query' | 'search-result' | 'neighbor' | 'passage' | 'entity' | 'seed-entity' | 'query-entity';
  distance?: number;
  entityId?: number;
  // For passage nodes (PPR/MemoryLink results)
  passageId?: number;
  score?: number;
  text?: string;
  source?: string;
  sessionId?: string;
  turnIndex?: number;
  sessionSeq?: number;  // Session sequence for S{seq}-T{turn} display
  // For passage nodes - linked entities (fetched on click)
  linkedEntities?: LinkedEntity[];
}

export interface GraphLink {
  source: string;
  target: string;
  label: string;
  factId: number;
}

export interface SearchResult {
  entity_id: number;
  name: string;
  distance: number;
}

export interface FactData {
  id: number;
  subject: { id: number; name: string };
  predicate: string;
  object: { id: number; name: string };
}

export interface SearchResponse {
  results: SearchResult[];
  neighbors: Record<string, { facts: FactData[] }>;
}

export interface NeighborResponse {
  entity: { id: number; name: string; created_at: string | null };
  facts: FactData[];
  connected_entities: { id: number; name: string; created_at: string | null }[];
}

export interface PassageData {
  id: number;
  session_id: string;
  turn_index: number;
  session_seq?: number;
  text: string;
  created_at: string | null;
}

export interface PassageResponse {
  entity_id: number;
  passages: PassageData[];
}

export interface MemoryStatus {
  entities: number;
  facts?: number;  // Optional: removed in MemoryLink v2
  passages: number;
  sessions: number;
  last_indexed: string;
  entity_passages?: number;
  passage_adjacency?: number;
  process_memory_mb?: number;
}

/** Sigma.js node attributes stored in graphology. */
export interface SigmaNodeAttributes {
  label: string;
  size: number;
  color: string;
  group: 'query' | 'search-result' | 'neighbor' | 'passage' | 'entity' | 'seed-entity' | 'query-entity';
  entityId?: number;
  distance?: number;
  // For passage nodes (PPR/MemoryLink results)
  passageId?: number;
  score?: number;
  text?: string;
  source?: string;
  sessionId?: string;
  turnIndex?: number;
  sessionSeq?: number;  // Session sequence for S{seq}-T{turn} display
  x: number;
  y: number;
  // Border styling (for search result nodes)
  borderColor?: string;
  borderSize?: number;
}

/** Sigma.js edge attributes stored in graphology. */
export interface SigmaEdgeAttributes {
  label: string;
  size: number;
  color: string;
  factId: number;
  weight?: number;
}

/** Retrieval result from PPR, MemoryLink, or MemoryLink v2. */
export interface RetrievalResult {
  passage_id: number;
  text: string;
  score: number;
  source: 'ppr' | 'memorylink' | 'memorylink-v2' | 'knn';
  entity_ids?: number[];  // Entity IDs linked to this passage (v2 only)
  session_seq?: number;   // Session sequence for S{seq}-T{turn} display
  turn_index?: number;    // Turn index within session
  session_id?: string;    // Session UUID
}

/** Entity info returned in v2 retrieve response. */
export interface RetrieveEntity {
  entity_id: number;
  name: string;
}

/** Seed entity with activation score (v2 only). */
export interface SeedEntity {
  entity_id: number;
  name: string;
  score: number;
}

/** Graph node from PPR traversal (v2 only). */
export interface GraphTraversalNode {
  id: string;  // "entity:{id}" or "passage:{id}"
  type: 'entity' | 'passage';
  entity_id?: number;
  passage_id?: number;
  name?: string;  // Only for entities
  score: number;
}

/** Graph edge from PPR traversal (v2 only). */
export interface GraphTraversalEdge {
  source: string;
  target: string;
  type: 'contain' | 'adjacency';
}

/** Stage timing breakdown from pipeline profiling. */
export interface StageTiming {
  /** Total pipeline time in ms */
  total_ms: number;
  /** Per-stage durations in ms (pipeline internal stages) */
  stages: Record<string, number>;
  /** Graph build time (null if cache hit) */
  graph_build_ms: number | null;
  /** Whether graph cache was hit */
  cache_hit: boolean | null;
  /** API-level stage timings (outside pipeline) */
  api_stages?: Record<string, number>;
  /** Detailed timing for seed selection (entity mode) */
  seed_selection_details?: {
    mode?: string;  // "query" or "entity"
    total_ms?: number;
    num_query_entities?: number;
    num_seeds_found?: number;
    total_candidates?: number;
    per_entity_avg_ms?: number;
    per_entity_max_ms?: number;
    per_entity_min_ms?: number;
    [key: string]: string | number | undefined;  // entity_0_ms, entity_1_ms, etc.
  };
  /** Detailed timing for PPR execution */
  ppr_details?: {
    iterations?: number;
    db_query_ms?: number;
    propagation_ms?: number;
    per_iter_avg_ms?: number;
    total_ms?: number;
    server_side?: boolean;  // true if using PostgreSQL server-side PPR
  };
}

/** Response from /api/memory/retrieve endpoint. */
export interface RetrieveResponse {
  results: RetrievalResult[];
  entities?: RetrieveEntity[];  // All entities referenced by results (v2 only)
  seed_entities?: SeedEntity[];  // Activated seed entities from Stage 1 (v2 only)
  query_entities?: string[] | null;  // NER-extracted entities from query (v2 only, null if NER failed)
  graph_nodes?: GraphTraversalNode[];  // PPR traversal graph nodes (v2 only)
  graph_edges?: GraphTraversalEdge[];  // PPR traversal graph edges (v2 only)
  algorithm: 'ppr' | 'memorylink' | 'memorylink-v2';
  query: string;
  elapsed_ms: number;
  /** Stage-by-stage timing breakdown (v2 only) */
  timing?: StageTiming;
}

/** Path node in Dijkstra path visualization. */
export interface PathNode {
  type: 'entity' | 'passage';
  id: number;
  name: string;
}

/** Path result from MemoryLink Dijkstra traversal. */
export interface PathResult {
  passage_id: number;
  score: number;
  distance: number;
  source_entity_id: number;
  path: PathNode[];
  text: string;
}

/** Response from /api/memory/entity-paths endpoint. */
export interface EntityPathsResponse {
  results: PathResult[];
  entity_names: Record<number, string>;
  elapsed_ms: number;
}

/** Co-occurrence neighbor from MemoryLink graph. */
export interface CoocNeighbor {
  entity_id: number;
  name: string;
  count: number;
  weight: number;
  depth: number;
  connected_from: number;
  importance: number;
}

/** Scored entity from Enrich pipeline. */
export interface EnrichEntity {
  entity_id: number;
  name: string;
  importance: number;
  combined_score: number;
  hop_distance: number;
}

/** Scored fact from Enrich pipeline. */
export interface EnrichFact {
  fact_id: number;
  subject_name: string;
  predicate: string;
  object_name: string;
  combined_score: number;
}

/** Scored fact from LLM enricher. */
export interface EnrichedFact {
  score: number;  // 1-10 importance
  confidence: number;  // 1-10 reliability
  text: string;
  passage_id: number | null;  // Source passage index (1-based)
  reason: string;  // Why this fact is relevant
}

/** Progress update during enrichment. */
export interface EnrichProgress {
  stage: 'stage1' | 'stage2';
  current: number;
  total: number;
  message?: string;
}

/** Response from /api/memory/enrich endpoint (v2 - LLM-based). */
export interface EnrichResponse {
  query: string;
  facts: EnrichedFact[];
  passage_count: number;
  retrieve_results: {
    passage_id: number;
    text: string;
    score: number;
    session_seq?: number;
    turn_index?: number;
  }[];
  elapsed_ms: number;
}

/** Legacy EnrichResponse (deprecated, kept for type compatibility). */
export interface LegacyEnrichResponse {
  query: string;
  seed_entity_ids: number[];
  entities: EnrichEntity[];
  facts: EnrichFact[];
  prompt_preview: string;
  elapsed_ms: number;
}

/** Passage linked from co-occurrence graph. */
export interface CoocPassage {
  passage_id: number;
  connected_from: number;
  text?: string;
  session_id?: string;
  turn_index?: number;
}

/** Response from /api/memory/cooccurrence/entity/{id}/neighbors endpoint. */
export interface CoocNeighborsResponse {
  entity: { id: number; name: string };
  neighbors: CoocNeighbor[];
  passages: CoocPassage[];
}

/** Connected entity in unified node neighbors API. */
export interface NodeNeighborEntity {
  entity_id: number;
  name: string;
  weight: number;
  importance: number;
}

/** Connected passage in unified node neighbors API. */
export interface NodeNeighborPassage {
  passage_id: number;
  weight: number;
  text?: string;
  session_id?: string;
  turn_index?: number;
}

/** Response from /api/memory/cooccurrence/node/{type}/{id}/neighbors endpoint. */
export interface NodeNeighborsResponse {
  node: {
    type: 'entity' | 'passage';
    id: number;
    name?: string;
    session_id?: string;
    turn_index?: number;
    text?: string;
  };
  entities: NodeNeighborEntity[];
  passages: NodeNeighborPassage[];
}

/** Edge between entities from co-occurrence graph. */
export interface CoocEdge {
  source_id: number;
  target_id: number;
  weight: number;
  count: number;
}

/** Response from /api/memory/cooccurrence/edges endpoint. */
export interface CoocEdgesResponse {
  edges: CoocEdge[];
}
