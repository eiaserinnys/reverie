import { useState, useCallback, useEffect } from 'react';
import { MultiGraph } from 'graphology';
import { SearchBar } from './SearchBar';
import { GraphCanvas } from './GraphCanvas';
import { NodeDetail } from './NodeDetail';
import { EnrichResult } from './EnrichResult';
import { QuerySettings, type QueryParams } from './QuerySettings';
import { ResultList } from './ResultList';
import { TimingBreakdown } from './TimingBreakdown';
import { ResizablePanel } from '../ui/resizable-panel';
import { ResizableVertical } from '../ui/resizable-vertical';
import type {
  GraphNode,
  SigmaNodeAttributes,
  SigmaEdgeAttributes,
  FactData,
  MemoryStatus,
  RetrieveResponse,
  EnrichResponse,
  EnrichProgress,
} from './types';

/** Convert entity ID to a graph node ID string. */
function entityNodeId(entityId: number): string {
  return `entity:${entityId}`;
}

/** Convert passage ID to a graph node ID string. */
function passageNodeId(passageId: number): string {
  return `passage:${passageId}`;
}

/** Build a graphology MultiGraph from retrieval (MemoryLink v2) response data.
 *
 * Visualizes the full PPR traversal graph:
 * - All nodes (entities + passages) that PPR explored
 * - All edges (contain + adjacency) between explored nodes
 * - Node size based on PPR score
 */
function buildGraphFromRetrieval(
  query: string,
  data: RetrieveResponse,
): MultiGraph<SigmaNodeAttributes, SigmaEdgeAttributes> {
  const graph = new MultiGraph<SigmaNodeAttributes, SigmaEdgeAttributes>();

  // Colors - distinct for each element type
  const queryColor = '#ef4444';     // Red for query node
  const passageColor = '#22c55e';   // Green for passages
  const entityColor = '#a855f7';    // Brighter purple for entity nodes
  const queryEntityColor = '#f97316';  // Orange for NER-extracted query entities
  const containEdgeColor = '#6366f1';  // Indigo for contain edges (entity→passage)
  const adjacencyEdgeColor = '#14b8a6';  // Teal for adjacency edges (passage→passage)

  // Query node at center
  const queryId = 'query:0';
  graph.addNode(queryId, {
    label: query,
    size: 14,
    color: queryColor,
    group: 'query',
    x: 0,
    y: 0,
  });

  // Add query entities (NER-extracted) in a small ring around query
  const queryEntities = data.query_entities || [];
  const queryEntityRadius = 25;
  for (let i = 0; i < queryEntities.length; i++) {
    const entityName = queryEntities[i];
    const nodeId = `query-entity:${i}`;
    const angle = (2 * Math.PI * i) / Math.max(queryEntities.length, 1);

    graph.addNode(nodeId, {
      label: entityName,
      size: 8,
      color: queryEntityColor,
      group: 'query-entity',
      x: Math.cos(angle) * queryEntityRadius,
      y: Math.sin(angle) * queryEntityRadius,
    });

    // Connect query to query-entity
    graph.addEdge(queryId, nodeId, {
      label: '',
      size: 2,
      color: `${queryEntityColor}88`,
      factId: -1,
    });
  }

  // Use PPR traversal graph if available
  const graphNodes = data.graph_nodes || [];
  const graphEdges = data.graph_edges || [];

  if (graphNodes.length === 0) {
    // Fallback: just show results if no graph data
    const resultCount = data.results.length;
    for (let i = 0; i < resultCount; i++) {
      const result = data.results[i];
      const pNodeId = passageNodeId(result.passage_id);
      const angle = (2 * Math.PI * i) / resultCount;
      const radius = 60;

      // Use S{seq}-T{turn} format if available, otherwise P{id}
      const resultAny = result as any;
      const label = resultAny.session_seq !== undefined && resultAny.session_seq !== null
        ? `S${resultAny.session_seq}-T${resultAny.turn_index}`
        : `P${result.passage_id}`;

      graph.addNode(pNodeId, {
        label,
        size: 8,
        color: passageColor,
        group: 'search-result',
        passageId: result.passage_id,
        score: result.score,
        text: result.text,
        sessionSeq: resultAny.session_seq,
        turnIndex: resultAny.turn_index,
        sessionId: resultAny.session_id,
        x: Math.cos(angle) * radius,
        y: Math.sin(angle) * radius,
        borderColor: '#ef4444',
        borderSize: 3,
      });

      graph.addEdge(queryId, pNodeId, {
        label: '',
        size: 1,
        color: passageColor,
        factId: -1,
      });
    }
    return graph;
  }

  // Calculate max scores for normalization
  const entityNodes = graphNodes.filter(n => n.type === 'entity');
  const passageNodes = graphNodes.filter(n => n.type === 'passage');
  const maxEntityScore = Math.max(...entityNodes.map(n => n.score), 0.0001);
  const maxPassageScore = Math.max(...passageNodes.map(n => n.score), 0.0001);

  // Get result passage IDs for highlighting
  const resultPassageIds = new Set(data.results.map(r => r.passage_id));

  // Calculate radius based on node density
  const nodeSpacing = 18;
  const entityRadius = Math.max(40, (entityNodes.length * nodeSpacing) / (2 * Math.PI));
  const passageRadius = Math.max(entityRadius + 50, (passageNodes.length * nodeSpacing) / (2 * Math.PI));

  // Add entity nodes
  for (let i = 0; i < entityNodes.length; i++) {
    const node = entityNodes[i];
    const angle = (2 * Math.PI * i) / Math.max(entityNodes.length, 1);
    const normalizedScore = node.score / maxEntityScore;
    const size = 5 + normalizedScore * 8;

    graph.addNode(node.id, {
      label: node.name || `Entity ${node.entity_id}`,
      size,
      color: entityColor,
      group: 'entity',
      entityId: node.entity_id,
      score: node.score,
      x: Math.cos(angle) * entityRadius,
      y: Math.sin(angle) * entityRadius,
    });

    // Connect high-scoring entities to query (or query-entity if available)
    if (normalizedScore > 0.5) {
      const entityName = node.name || '';
      const matchingQueryEntityIdx = queryEntities.findIndex(
        qe => qe.toLowerCase() === entityName.toLowerCase()
      );

      if (matchingQueryEntityIdx >= 0) {
        const queryEntityNodeId = `query-entity:${matchingQueryEntityIdx}`;
        graph.addEdge(queryEntityNodeId, node.id, {
          label: '',
          size: 1 + normalizedScore * 2,
          color: `${queryEntityColor}66`,
          factId: -1,
        });
      } else {
        graph.addEdge(queryId, node.id, {
          label: '',
          size: 1 + normalizedScore * 2,
          color: `${entityColor}66`,
          factId: -1,
        });
      }
    }
  }

  // Add passage nodes
  for (let i = 0; i < passageNodes.length; i++) {
    const node = passageNodes[i] as any;
    const angle = (2 * Math.PI * i) / Math.max(passageNodes.length, 1);
    const normalizedScore = node.score / maxPassageScore;
    const isResult = resultPassageIds.has(node.passage_id!);

    const size = isResult ? 8 + normalizedScore * 6 : 4 + normalizedScore * 4;
    const color = isResult ? passageColor : `${passageColor}88`;

    const resultInfo = data.results.find(r => r.passage_id === node.passage_id);

    const label = node.session_seq !== undefined && node.session_seq !== null
      ? `S${node.session_seq}-T${node.turn_index}`
      : `P${node.passage_id}`;

    graph.addNode(node.id, {
      label,
      size,
      color,
      group: isResult ? 'search-result' : 'passage',
      passageId: node.passage_id,
      score: node.score,
      text: resultInfo?.text,
      sessionSeq: node.session_seq,
      turnIndex: node.turn_index,
      sessionId: node.session_id,
      x: Math.cos(angle) * passageRadius,
      y: Math.sin(angle) * passageRadius,
      borderColor: isResult ? '#ef4444' : undefined,
      borderSize: isResult ? 3 : undefined,
    });
  }

  // Add edges from traversal graph
  for (const edge of graphEdges) {
    if (graph.hasNode(edge.source) && graph.hasNode(edge.target)) {
      const edgeId = `${edge.type}:${edge.source}-${edge.target}`;
      if (!graph.hasEdge(edgeId)) {
        graph.addEdgeWithKey(edgeId, edge.source, edge.target, {
          label: '',
          size: edge.type === 'contain' ? 1.5 : 1,
          color: edge.type === 'contain' ? containEdgeColor : adjacencyEdgeColor,
          factId: 0,
          weight: 1.0,
        });
      }
    }
  }

  return graph;
}

// Bottom section tab type (Content or Advanced)
type BottomTab = 'content' | 'advanced';

export function MemoryExplorerTab() {
  const [graph, setGraph] = useState(
    () => new MultiGraph<SigmaNodeAttributes, SigmaEdgeAttributes>(),
  );
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [selectedFacts, setSelectedFacts] = useState<FactData[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState<MemoryStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastResponse, setLastResponse] = useState<RetrieveResponse | null>(null);
  const [enrichResponse, setEnrichResponse] = useState<EnrichResponse | null>(null);
  const [enrichLoading, setEnrichLoading] = useState(false);
  const [enrichProgress, setEnrichProgress] = useState<EnrichProgress | null>(null);
  const [bottomTab, setBottomTab] = useState<BottomTab>('content');
  const [queryParams, setQueryParams] = useState<QueryParams>({
    pprMinScore: 0.0001,
    pprMaxIter: 20,
    tau: 0.5,
    damping: 0.85,
    beta: 1.0,
    topK: 25,
    halfLifeDays: 365,
  });

  // Fetch memory status on mount
  useEffect(() => {
    fetch('/api/memory/status')
      .then((r) => r.json())
      .then((data) => {
        if (!data.error) setStatus(data);
      })
      .catch(() => {});
  }, []);

  // Fallback to POST endpoint when SSE fails
  const fetchEnrichPost = useCallback(async (query: string) => {
    try {
      const response = await fetch('/api/memory/enrich', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, top_k: 25 }),
      });
      if (response.ok) {
        const data = await response.json();
        setEnrichResponse(data);
      }
    } catch (err) {
      console.error('Enrich POST fallback failed:', err);
    } finally {
      setEnrichProgress(null);
      setEnrichLoading(false);
    }
  }, []);

  // Fetch enrich with SSE progress updates
  const fetchEnrich = useCallback(async (query: string) => {
    setEnrichLoading(true);
    setEnrichProgress(null);

    try {
      // Use SSE endpoint for progress updates
      const eventSource = new EventSource(
        `/api/memory/enrich/stream?query=${encodeURIComponent(query)}&top_k=25`
      );

      eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === 'progress') {
          setEnrichProgress({
            stage: data.stage,
            current: data.current,
            total: data.total,
            message: data.message,
          });
        } else if (data.type === 'result') {
          setEnrichResponse(data.result);
          setEnrichProgress(null);
          setEnrichLoading(false);
          eventSource.close();
        } else if (data.type === 'error') {
          console.error('Enrich error:', data.error);
          setEnrichProgress(null);
          setEnrichLoading(false);
          eventSource.close();
        }
      };

      eventSource.onerror = () => {
        console.error('Enrich SSE failed, falling back to POST');
        eventSource.close();
        // Fallback to POST endpoint
        fetchEnrichPost(query);
      };
    } catch (err) {
      console.error('Enrich fetch failed:', err);
      fetchEnrichPost(query);
    }
  }, [fetchEnrichPost]);

  const handleSearch = useCallback(async (query: string) => {
    setIsLoading(true);
    setError(null);
    setSelectedNode(null);
    setSelectedFacts([]);
    setLastResponse(null);
    setEnrichResponse(null);

    try {
      // Run retrieve (MemoryLink v2)
      const response = await fetch('/api/memory/retrieve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          algorithm: 'memorylink-v2',
          top_k: queryParams.topK,
          ppr_min_score: queryParams.pprMinScore,
          ppr_max_iter: queryParams.pprMaxIter,
          tau: queryParams.tau,
          damping: queryParams.damping,
          beta: queryParams.beta,
          half_life_days: queryParams.halfLifeDays,
        }),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error || `HTTP ${response.status}`);
      }

      const data: RetrieveResponse = await response.json();
      setLastResponse(data);
      const newGraph = buildGraphFromRetrieval(query, data);
      setGraph(newGraph);

      // Trigger enrich in parallel (non-blocking)
      fetchEnrich(query);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed');
      setGraph(new MultiGraph<SigmaNodeAttributes, SigmaEdgeAttributes>());
    } finally {
      setIsLoading(false);
    }
  }, [queryParams, fetchEnrich]);

  const handleNodeClick = useCallback(
    async (nodeId: string) => {
      if (!graph.hasNode(nodeId)) return;

      const attrs = graph.getNodeAttributes(nodeId);
      const node: GraphNode = {
        id: nodeId,
        name: attrs.label,
        group: attrs.group,
        distance: attrs.distance,
        entityId: attrs.entityId,
        passageId: attrs.passageId,
        score: attrs.score,
        text: attrs.text,
        source: attrs.source,
        sessionId: attrs.sessionId,
        turnIndex: attrs.turnIndex,
      };
      setSelectedNode(node);

      // Switch to content tab when clicking a node
      setBottomTab('content');

      // For passage nodes, fetch passage text and linked entities
      if (attrs.passageId) {
        try {
          const response = await fetch(
            `/api/memory/v2/node/passage/${attrs.passageId}/neighbors`,
          );
          if (response.ok) {
            const data = await response.json();
            setSelectedNode(prev => prev ? {
              ...prev,
              text: prev.text || data.node?.text,
              sessionId: prev.sessionId || data.node?.session_id,
              turnIndex: prev.turnIndex ?? data.node?.turn_index,
              sessionSeq: prev.sessionSeq ?? data.node?.session_seq,
              linkedEntities: data.entities?.map((e: { entity_id: number; name: string }) => ({
                entity_id: e.entity_id,
                name: e.name,
                importance: 1.0,
              })) || [],
            } : null);
          }
        } catch {
          // Silently fail
        }
      }

      setSelectedFacts([]);
    },
    [graph],
  );

  const handleEntityClick = useCallback(
    (entityId: number) => {
      const nodeId = entityNodeId(entityId);
      if (graph.hasNode(nodeId)) {
        handleNodeClick(nodeId);
      }
    },
    [graph, handleNodeClick],
  );

  const handleDetailClose = useCallback(() => {
    setSelectedNode(null);
    setSelectedFacts([]);
  }, []);

  return (
    <div className="flex flex-col flex-1 overflow-hidden min-w-0 min-h-0">
      {/* Search bar */}
      <div className="px-4 py-3 border-b border-border bg-card shrink-0">
        <SearchBar
          onSearch={handleSearch}
          isLoading={isLoading}
        />
        {/* Timing breakdown for memorylink-v2 */}
        {lastResponse?.timing && (
          <div className="mt-2">
            <TimingBreakdown timing={lastResponse.timing} expanded={false} />
          </div>
        )}
      </div>

      {/* Error message */}
      {error && (
        <div className="px-4 py-2 bg-destructive/10 text-destructive text-sm shrink-0">
          {error}
        </div>
      )}

      {/* ResultList + Graph + Right panel */}
      <div className="flex flex-1 overflow-hidden min-h-0">
        {/* Left panel: Result list (resizable) */}
        <ResizablePanel
          defaultWidth={17.5}
          minWidth={10}
          maxWidth={35}
          side="left"
          className="hidden md:flex flex-col bg-card overflow-hidden"
        >
          <ResultList
            results={lastResponse?.results || []}
            selectedPassageId={selectedNode?.passageId ?? null}
            onSelect={(passageId) => {
              const nodeId = passageNodeId(passageId);
              handleNodeClick(nodeId);
            }}
            fullWidth={true}
          />
        </ResizablePanel>

        {/* Center: Graph canvas */}
        <div className="flex-1 min-w-0 overflow-hidden bg-background relative">
          <GraphCanvas
            graph={graph}
            onNodeClick={handleNodeClick}
            selectedNodeId={selectedNode?.id ?? null}
          />

          {/* Stats overlay at bottom center */}
          {status && (
            <div className="absolute bottom-4 left-1/2 -translate-x-1/2 px-3 py-1.5 bg-card/90 border border-border rounded-full text-xs text-muted-foreground">
              {status.entities.toLocaleString()} entities / {status.passages.toLocaleString()} passages / {status.sessions.toLocaleString()} sessions
            </div>
          )}
        </div>

        {/* Right panel: Split layout with Enrich on top, Content/Advanced tabs on bottom */}
        <ResizablePanel
          defaultWidth={17.5}
          minWidth={10}
          maxWidth={35}
          side="right"
          className="hidden md:flex flex-col bg-card overflow-hidden"
        >
          <ResizableVertical
            defaultTopHeight={50}
            minTopHeight={25}
            maxTopHeight={75}
            topContent={
              <div className="flex flex-col h-full overflow-hidden">
                <div className="px-3 py-2 border-b border-border shrink-0 flex items-center gap-2">
                  <span className="text-sm font-medium">Enrich</span>
                  {enrichLoading && <span className="text-xs animate-pulse text-muted-foreground">...</span>}
                </div>
                <div className="flex-1 overflow-hidden">
                  <EnrichResult
                    data={enrichResponse}
                    isLoading={enrichLoading}
                    progress={enrichProgress}
                    onPassageClick={(passageId) => {
                      const nodeId = passageNodeId(passageId);
                      handleNodeClick(nodeId);
                    }}
                  />
                </div>
              </div>
            }
            bottomContent={
              <div className="flex flex-col h-full overflow-hidden">
                {/* Tab headers */}
                <div className="flex border-b border-border shrink-0">
                  <button
                    onClick={() => setBottomTab('content')}
                    className={`flex-1 px-3 py-2 text-sm font-medium transition-colors ${
                      bottomTab === 'content'
                        ? 'text-primary border-b-2 border-primary'
                        : 'text-muted-foreground hover:text-foreground'
                    }`}
                  >
                    Content
                  </button>
                  <button
                    onClick={() => setBottomTab('advanced')}
                    className={`flex-1 px-3 py-2 text-sm font-medium transition-colors ${
                      bottomTab === 'advanced'
                        ? 'text-primary border-b-2 border-primary'
                        : 'text-muted-foreground hover:text-foreground'
                    }`}
                  >
                    Advanced
                  </button>
                </div>

                {/* Tab content */}
                <div className="flex-1 overflow-hidden min-w-0 min-h-0">
                  {bottomTab === 'content' && (
                    <NodeDetail
                      node={selectedNode}
                      facts={selectedFacts}
                      onClose={handleDetailClose}
                      onEntityClick={handleEntityClick}
                      embedded={true}
                    />
                  )}
                  {bottomTab === 'advanced' && (
                    <div className="p-3 overflow-y-auto h-full">
                      <QuerySettings
                        params={queryParams}
                        onParamsChange={setQueryParams}
                        visible={true}
                        embedded={true}
                      />
                    </div>
                  )}
                </div>
              </div>
            }
          />
        </ResizablePanel>
      </div>
    </div>
  );
}
