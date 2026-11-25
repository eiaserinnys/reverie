import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { SigmaContainer, useSigma, useRegisterEvents, useSetSettings } from '@react-sigma/core';
import '@react-sigma/core/lib/style.css';
import { useWorkerLayoutForceAtlas2 } from '@react-sigma/layout-forceatlas2';
import { createNodeBorderProgram } from '@sigma/node-border';
import { ZoomIn, ZoomOut, Maximize2 } from 'lucide-react';
import { MultiGraph } from 'graphology';
import type { AbstractGraph, Attributes } from 'graphology-types';
import type { SigmaNodeAttributes, SigmaEdgeAttributes } from './types';

// Node program with border support - reads borderColor/borderSize from node attributes
// Borders are rendered from outside to inside: first border (outer), then fill (inner)
const NodeWithBorderProgram = createNodeBorderProgram({
  borders: [
    // Outer border (red for search results, transparent for others)
    {
      color: { attribute: 'borderColor', defaultValue: 'transparent' },
      size: { attribute: 'borderSize', defaultValue: 0, mode: 'pixels' },
    },
    // Inner fill uses the node's color attribute
    {
      color: { attribute: 'color' },
      size: { fill: true },
    },
  ],
});

// Stable reference for SigmaContainer's graph prop — created once, never changes.
// SigmaContainer recreates Sigma when this prop changes, so it MUST be stable.
const EMPTY_GRAPH = new MultiGraph<SigmaNodeAttributes, SigmaEdgeAttributes>();

interface GraphCanvasProps {
  graph: MultiGraph<SigmaNodeAttributes, SigmaEdgeAttributes>;
  onNodeClick: (nodeId: string) => void;
  selectedNodeId?: string | null;
}

/** Loads external graph data into Sigma's internal MultiGraph whenever it changes. */
function GraphLoader({ graph }: { graph: MultiGraph<SigmaNodeAttributes, SigmaEdgeAttributes> }) {
  const sigma = useSigma();

  useEffect(() => {
    const internal = sigma.getGraph();
    // Clear then manually copy nodes/edges to avoid Graph.import() type mismatch
    internal.clear();

    graph.forEachNode((node, attrs) => {
      internal.addNode(node, { ...attrs });
    });
    graph.forEachEdge((edge, attrs, source, target) => {
      if (!internal.hasNode(source) || !internal.hasNode(target)) return;
      internal.addEdgeWithKey(edge, source, target, { ...attrs });
    });

    requestAnimationFrame(() => {
      sigma.refresh();
    });
  }, [graph, sigma]);

  return null;
}

/** Handles click/hover events via Sigma's native event system. */
function GraphEventHandler({
  onNodeClick,
  onHoverChange,
}: {
  onNodeClick: (nodeId: string) => void;
  onHoverChange: (nodeId: string | null) => void;
}) {
  const sigma = useSigma();
  const registerEvents = useRegisterEvents();

  useEffect(() => {
    registerEvents({
      clickNode: (event) => {
        onNodeClick(event.node);
      },
      enterNode: (event) => {
        const container = sigma.getContainer();
        container.style.cursor = 'pointer';
        onHoverChange(event.node);
      },
      leaveNode: () => {
        const container = sigma.getContainer();
        container.style.cursor = 'default';
        onHoverChange(null);
      },
    });
  }, [registerEvents, onNodeClick, onHoverChange, sigma]);

  return null;
}

/** Breathing animation - nodes gently pulse toward/away from center. */
function BreathingAnimation({ graphOrder }: { graphOrder: number }) {
  const sigma = useSigma();

  useEffect(() => {
    if (graphOrder === 0) return;

    const graph = sigma.getGraph();
    const breathingAmplitude = 0.06; // 6% expansion/contraction
    const breathingSpeed = 0.001; // ~6 second cycle
    let animationId: number;
    let startTime: number;
    let originalPositions = new Map<string, { x: number; y: number }>();
    let cx = 0, cy = 0;

    const animate = (time: number) => {
      const elapsed = time - startTime;
      const breathFactor = Math.sin(elapsed * breathingSpeed) * breathingAmplitude;

      graph.forEachNode((node) => {
        const orig = originalPositions.get(node);
        if (!orig) return;

        const dx = orig.x - cx;
        const dy = orig.y - cy;

        graph.setNodeAttribute(node, 'x', orig.x + dx * breathFactor);
        graph.setNodeAttribute(node, 'y', orig.y + dy * breathFactor);
      });

      sigma.refresh();
      animationId = requestAnimationFrame(animate);
    };

    // Delay start to let ForceAtlas2 settle first
    const timeoutId = setTimeout(() => {
      // Capture positions after layout settles
      originalPositions.clear();
      graph.forEachNode((node, attrs) => {
        originalPositions.set(node, { x: attrs.x, y: attrs.y });
      });

      if (originalPositions.size === 0) return;

      // Calculate center of mass
      cx = 0; cy = 0;
      originalPositions.forEach(({ x, y }) => {
        cx += x;
        cy += y;
      });
      cx /= originalPositions.size;
      cy /= originalPositions.size;

      startTime = performance.now();
      animationId = requestAnimationFrame(animate);
    }, 3500); // Wait for ForceAtlas2 to finish (~2.5s)

    return () => {
      clearTimeout(timeoutId);
      if (animationId) cancelAnimationFrame(animationId);
    };
  }, [sigma, graphOrder]);

  return null;
}

/** Runs ForceAtlas2 layout for ~2 seconds then stops. */
function LayoutController({ graphOrder }: { graphOrder: number }) {
  const { start, stop, isRunning } = useWorkerLayoutForceAtlas2({
    settings: {
      slowDown: 20,        // Higher = more stable, less oscillation
      gravity: 2,          // Higher = nodes pulled toward center more
      scalingRatio: 1.5,   // Lower = less repulsion between nodes
      barnesHutOptimize: graphOrder > 50, // Only use BH for larger graphs
      strongGravityMode: true, // Prevents nodes from flying away
    },
  });

  useEffect(() => {
    if (graphOrder === 0) return;
    start();
    const timer = setTimeout(() => {
      if (isRunning) stop();
    }, 2500); // Slightly longer to allow convergence
    return () => {
      clearTimeout(timer);
      if (isRunning) stop();
    };
  }, [graphOrder, start, stop, isRunning]);

  return null;
}

/** Computes BFS distances from a source node. */
function computeDistances(
  graph: AbstractGraph<Attributes, Attributes, Attributes>,
  sourceId: string,
  maxDistance: number = 5,
): Map<string, number> {
  const distances = new Map<string, number>();
  if (!graph.hasNode(sourceId)) return distances;

  distances.set(sourceId, 0);
  const queue: string[] = [sourceId];
  let head = 0;

  while (head < queue.length) {
    const current = queue[head++];
    const currentDist = distances.get(current)!;
    if (currentDist >= maxDistance) continue;

    graph.forEachNeighbor(current, (neighbor) => {
      if (!distances.has(neighbor)) {
        distances.set(neighbor, currentDist + 1);
        queue.push(neighbor);
      }
    });
  }

  return distances;
}

/** Interpolates between two hex colors based on factor (0-1). */
function interpolateColor(color1: string, color2: string, factor: number): string {
  const hex1 = color1.replace('#', '');
  const hex2 = color2.replace('#', '');

  const r1 = parseInt(hex1.substring(0, 2), 16);
  const g1 = parseInt(hex1.substring(2, 4), 16);
  const b1 = parseInt(hex1.substring(4, 6), 16);

  const r2 = parseInt(hex2.substring(0, 2), 16);
  const g2 = parseInt(hex2.substring(2, 4), 16);
  const b2 = parseInt(hex2.substring(4, 6), 16);

  const r = Math.round(r1 + (r2 - r1) * factor);
  const g = Math.round(g1 + (g2 - g1) * factor);
  const b = Math.round(b1 + (b2 - b1) * factor);

  return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
}

/** Dims non-selected nodes/edges when a node is selected.
 *  Nodes fade to gray gradually based on BFS distance from selected node.
 *  Uses Sigma's live internal graph inside reducers (not a stale closure)
 *  so it stays in sync even after GraphLoader rebuilds the graph. */
function NodeHighlighter({
  selectedNodeId,
  hoveredNodeId,
}: {
  selectedNodeId?: string | null;
  hoveredNodeId?: string | null;
}) {
  const sigma = useSigma();
  const setSettings = useSetSettings();

  useEffect(() => {
    // Precompute distances when selection changes
    let distances: Map<string, number> | null = null;
    if (selectedNodeId) {
      const g = sigma.getGraph();
      distances = computeDistances(g, selectedNodeId, 5);
    }

    const maxFadeDistance = 4; // Nodes beyond this are fully dimmed
    const dimColor = '#1a1a1a'; // Dark gray for far nodes
    const dimLabelColor = '#333333';

    setSettings({
      nodeReducer: (node, data) => {
        // Check if this is a query node
        const isQueryNode = node.startsWith('query:');

        // Hovered node: show label with black text on white background
        if (node === hoveredNodeId) {
          return {
            ...data,
            highlighted: true,
            forceLabel: true,
            labelColor: '#000000',
          };
        }

        // Selected node handling
        if (selectedNodeId && distances) {
          if (node === selectedNodeId) {
            // Query node stays red when selected
            if (isQueryNode) {
              return {
                ...data,
                zIndex: 1,
                forceLabel: true,
                size: (data.size || 7) * 1.5,
                color: '#ef4444', // Keep red for query node
                labelColor: '#ffffff',
              };
            }
            // Other selected nodes: yellow color
            return {
              ...data,
              zIndex: 1,
              forceLabel: true,
              size: (data.size || 7) * 1.5,
              color: '#f59e0b',
              labelColor: '#ffffff',
            };
          }

          const distance = distances.get(node);
          if (distance === undefined) {
            // Not reachable: fully dimmed
            return { ...data, color: dimColor, labelColor: dimLabelColor };
          }

          // Fade based on distance (0 = selected, 1 = neighbor, etc.)
          // distance 1: 100% original, distance 4+: fully dimmed
          const fadeFactor = Math.min(1, (distance - 1) / (maxFadeDistance - 1));
          const originalColor = data.color || '#8b5cf6';
          const fadedColor = interpolateColor(originalColor, dimColor, fadeFactor);
          const fadedLabelColor = interpolateColor('#d9d9d9', dimLabelColor, fadeFactor);

          return {
            ...data,
            color: fadedColor,
            labelColor: fadedLabelColor,
            forceLabel: distance <= 2, // Only show labels for close nodes
          };
        }

        // No selection, no hover: return as is (border attrs already set on node)
        return data;
      },
      edgeReducer: selectedNodeId && distances
        ? (edge, data) => {
            const g = sigma.getGraph();
            const source = g.source(edge);
            const target = g.target(edge);

            const sourceDist = distances!.get(source);
            const targetDist = distances!.get(target);

            // Get original attributes
            const attrs = g.getEdgeAttributes(edge);
            const originalColor = attrs.color as string | undefined;
            const originalSize = attrs.size as number | undefined;

            if (sourceDist === undefined || targetDist === undefined) {
              // Edge to unreachable node: fully dimmed
              return { ...data, color: '#0a0a0a', size: 0.3 };
            }

            // Edge distance = max of endpoints
            const edgeDist = Math.max(sourceDist, targetDist);
            if (edgeDist === 0) {
              // Edges from selected node: full color
              return {
                ...data,
                color: originalColor ?? '#64748b',
                size: (originalSize ?? 1) * 1.5,
              };
            }

            // Fade edge based on distance
            const fadeFactor = Math.min(1, (edgeDist - 1) / (maxFadeDistance - 1));
            const baseColor = originalColor ?? '#64748b';
            const fadedColor = interpolateColor(baseColor, '#0a0a0a', fadeFactor);
            const fadedSize = Math.max(0.3, (originalSize ?? 1) * (1 - fadeFactor * 0.7));

            return {
              ...data,
              color: fadedColor,
              size: fadedSize,
            };
          }
        : undefined,
    });
  }, [selectedNodeId, hoveredNodeId, sigma, setSettings]);

  return null;
}

/** Zoom in/out/fit controls using Sigma's camera API. */
function ZoomButtons() {
  const sigma = useSigma();

  const handleZoomIn = useCallback(() => {
    const camera = sigma.getCamera();
    camera.animatedZoom({ duration: 300, factor: 1.5 });
  }, [sigma]);

  const handleZoomOut = useCallback(() => {
    const camera = sigma.getCamera();
    camera.animatedUnzoom({ duration: 300, factor: 1.5 });
  }, [sigma]);

  const handleZoomFit = useCallback(() => {
    const camera = sigma.getCamera();
    camera.animatedReset({ duration: 400 });
  }, [sigma]);

  return (
    <div className="absolute top-3 right-3 z-10 flex flex-col gap-1">
      <button
        onClick={handleZoomIn}
        className="p-1.5 rounded bg-card/80 border border-border text-muted-foreground hover:text-foreground hover:bg-card transition-colors"
        title="Zoom in"
      >
        <ZoomIn className="w-4 h-4" />
      </button>
      <button
        onClick={handleZoomOut}
        className="p-1.5 rounded bg-card/80 border border-border text-muted-foreground hover:text-foreground hover:bg-card transition-colors"
        title="Zoom out"
      >
        <ZoomOut className="w-4 h-4" />
      </button>
      <button
        onClick={handleZoomFit}
        className="p-1.5 rounded bg-card/80 border border-border text-muted-foreground hover:text-foreground hover:bg-card transition-colors"
        title="Fit to view"
      >
        <Maximize2 className="w-4 h-4" />
      </button>
    </div>
  );
}

export function GraphCanvas({ graph, onNodeClick, selectedNodeId }: GraphCanvasProps) {
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);

  const handleHoverChange = useCallback((nodeId: string | null) => {
    setHoveredNodeId(nodeId);
  }, []);

  const sigmaSettings = useMemo(
    () => ({
      allowInvalidContainer: true,
      renderLabels: true,
      labelFont: 'Sans-Serif',
      labelSize: 12,
      labelWeight: 'normal' as const,
      // Use 'labelColor' node attribute for per-node label colors, with fallback
      labelColor: { attribute: 'labelColor', color: '#d9d9d9' },
      edgeLabelFont: 'Sans-Serif',
      edgeLabelSize: 10,
      renderEdgeLabels: true,
      edgeLabelColor: { color: '#94a3b8' },
      defaultEdgeType: 'line' as const,
      defaultNodeType: 'border' as const,
      nodeProgramClasses: {
        border: NodeWithBorderProgram,
      },
      stagePadding: 60,
    }),
    [],
  );

  const isEmpty = graph.order === 0;

  return (
    <div
      className="absolute inset-0"
      style={{
        '--sigma-background-color': 'transparent',
      } as React.CSSProperties}
    >
      {/* SigmaContainer is always mounted so it has stable dimensions.
          EMPTY_GRAPH (MultiGraph) is a stable reference — Sigma won't recreate. */}
      <SigmaContainer
        graph={EMPTY_GRAPH}
        settings={sigmaSettings}
        style={{ width: '100%', height: '100%' }}
      >
        <GraphLoader graph={graph} />
        <GraphEventHandler onNodeClick={onNodeClick} onHoverChange={handleHoverChange} />
        <LayoutController graphOrder={graph.order} />
        <NodeHighlighter selectedNodeId={selectedNodeId} hoveredNodeId={hoveredNodeId} />
        <BreathingAnimation graphOrder={graph.order} />
        {!isEmpty && <ZoomButtons />}
      </SigmaContainer>

      {/* Empty state overlay */}
      {isEmpty && (
        <div className="absolute inset-0 flex flex-col items-center justify-center text-muted-foreground gap-2 pointer-events-none">
          <div className="text-sm">Search for something to explore the knowledge graph</div>
          <div className="text-xs">Try: &quot;Discord bot&quot;, &quot;GitHub Actions&quot;, &quot;deployment&quot;</div>
        </div>
      )}

      {/* Node count badge */}
      {!isEmpty && (
        <div className="absolute bottom-3 left-3 z-10 text-[10px] text-muted-foreground bg-card/80 px-2 py-1 rounded border border-border">
          {graph.order} nodes / {graph.size} edges
        </div>
      )}
    </div>
  );
}
