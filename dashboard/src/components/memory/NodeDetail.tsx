import { useState, useEffect } from 'react';
import { X, FileText, ChevronDown, ChevronRight } from 'lucide-react';
import type { GraphNode, FactData, PassageData } from './types';

interface NodeDetailProps {
  node: GraphNode | null;
  facts: FactData[];
  onClose: () => void;
  onEntityClick: (entityId: number) => void;
  embedded?: boolean;  // When true, render inline without wrapper
}

export function NodeDetail({ node, facts, onClose, onEntityClick, embedded = false }: NodeDetailProps) {
  const [passages, setPassages] = useState<PassageData[]>([]);
  const [passagesLoading, setPassagesLoading] = useState(false);
  const [expandedPassage, setExpandedPassage] = useState<number | null>(null);

  // Fetch passages when node changes
  const entityId = node?.entityId ?? null;
  useEffect(() => {
    setPassages([]);
    setExpandedPassage(null);

    if (!entityId) return;

    const controller = new AbortController();
    setPassagesLoading(true);

    fetch(`/api/memory/entity/${entityId}/passages?limit=10`, {
      signal: controller.signal,
    })
      .then((r) => {
        if (!r.ok) return { passages: [] };
        return r.json();
      })
      .then((data) => {
        setPassages(data.passages || []);
      })
      .catch((err) => {
        if (err.name !== 'AbortError') {
          setPassages([]);
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setPassagesLoading(false);
        }
      });

    return () => controller.abort();
  }, [entityId]);

  // Empty state for embedded mode
  if (embedded && !node) {
    return (
      <div className="flex flex-col h-full items-center justify-center text-muted-foreground text-sm p-4">
        <p>Click a node in the graph to see details</p>
      </div>
    );
  }

  if (!node) return null;

  const content = (
    <>
      {/* Compact metadata grid */}
      <div className="px-4 py-2 border-b border-border">
        <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
          <span className="text-muted-foreground">Type</span>
          <span className="text-foreground capitalize">{node.group.replace('-', ' ')}</span>

          {node.score !== undefined && (
            <>
              <span className="text-muted-foreground">Score</span>
              <span className="text-foreground font-mono">{node.score.toFixed(6)}</span>
            </>
          )}

          {node.distance !== undefined && (
            <>
              <span className="text-muted-foreground">Distance</span>
              <span className="text-foreground">{node.distance.toFixed(4)}</span>
            </>
          )}

          {node.passageId && (
            <>
              <span className="text-muted-foreground">Passage</span>
              <span className="text-foreground">#{node.passageId}</span>
            </>
          )}

          {node.entityId && (
            <>
              <span className="text-muted-foreground">Entity</span>
              <span className="text-foreground">#{node.entityId}</span>
            </>
          )}

          {(node.sessionSeq !== undefined || node.sessionId) && (
            <>
              <span className="text-muted-foreground">Location</span>
              <span className="text-foreground font-mono">
                {node.sessionSeq !== undefined && node.turnIndex !== undefined ? (
                  <span className="font-bold text-primary">S{node.sessionSeq}-T{node.turnIndex}</span>
                ) : (
                  <>
                    {node.sessionId?.slice(0, 8)}...
                    {node.turnIndex !== undefined && ` / T${node.turnIndex}`}
                  </>
                )}
              </span>
            </>
          )}

          {node.source && (
            <>
              <span className="text-muted-foreground">Source</span>
              <span className="text-foreground">{node.source}</span>
            </>
          )}
        </div>
      </div>

      {/* Linked Entities */}
      {node.linkedEntities && node.linkedEntities.length > 0 && (
        <div className="px-4 py-2 border-b border-border">
          <div className="text-xs text-muted-foreground font-medium mb-1">
            Linked Entities ({node.linkedEntities.length})
          </div>
          <div className="flex flex-wrap gap-1 max-h-24 overflow-y-auto">
            {node.linkedEntities.map((entity) => (
              <button
                key={entity.entity_id}
                onClick={() => onEntityClick(entity.entity_id)}
                className="text-xs bg-primary/10 text-primary hover:bg-primary/20 rounded px-2 py-0.5 transition-colors"
                title={`Importance: ${entity.importance.toFixed(4)}`}
              >
                {entity.name}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Passage Content */}
      {node.text && (
        <div className="px-4 py-2">
          <div className="text-xs text-muted-foreground font-medium mb-1">
            Passage Content
          </div>
          <div className="text-xs text-foreground whitespace-pre-wrap break-words bg-secondary rounded p-2">
            {node.text}
          </div>
        </div>
      )}

      {/* Facts - only for entity nodes */}
      {facts.length > 0 && (
        <div className="px-4 py-2 border-b border-border">
          <div className="text-xs text-muted-foreground font-medium mb-1">
            Connections ({facts.length})
          </div>
          <div className="space-y-1 max-h-32 overflow-y-auto">
            {facts.map((fact) => (
              <div
                key={fact.id}
                className="text-xs bg-secondary rounded-md p-2"
              >
                <div className="flex items-center gap-1 flex-wrap">
                  <button
                    onClick={() => onEntityClick(fact.subject.id)}
                    className="text-primary hover:underline font-medium"
                  >
                    {fact.subject.name}
                  </button>
                  <span className="text-muted-foreground italic">{fact.predicate}</span>
                  <button
                    onClick={() => onEntityClick(fact.object.id)}
                    className="text-primary hover:underline font-medium"
                  >
                    {fact.object.name}
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Source Passages - for entity nodes */}
      {node.entityId && (
        <div className="px-4 py-2">
          <div className="flex items-center gap-1 text-xs text-muted-foreground font-medium mb-1">
            <FileText className="w-3 h-3" />
            Source Passages
            {passagesLoading && (
              <span className="ml-1 animate-pulse">loading...</span>
            )}
            {!passagesLoading && passages.length > 0 && (
              <span>({passages.length})</span>
            )}
          </div>

          {!passagesLoading && passages.length === 0 && (
            <div className="text-xs text-muted-foreground italic">
              No passages found
            </div>
          )}

          <div className="space-y-1">
            {passages.map((passage) => {
              const isExpanded = expandedPassage === passage.id;
              const preview = passage.text.slice(0, 80) + (passage.text.length > 80 ? '...' : '');

              return (
                <div
                  key={passage.id}
                  className="bg-secondary rounded-md overflow-hidden"
                >
                  <button
                    onClick={() => setExpandedPassage(isExpanded ? null : passage.id)}
                    aria-expanded={isExpanded}
                    aria-label={`Toggle passage from ${passage.session_id}`}
                    className="w-full text-left p-2 flex items-start gap-1 hover:bg-secondary/80 transition-colors"
                  >
                    {isExpanded ? (
                      <ChevronDown className="w-3 h-3 mt-0.5 shrink-0 text-muted-foreground" />
                    ) : (
                      <ChevronRight className="w-3 h-3 mt-0.5 shrink-0 text-muted-foreground" />
                    )}
                    <div className="min-w-0">
                      <div className="text-xs text-foreground line-clamp-2">
                        {isExpanded ? '' : preview}
                      </div>
                      <div className="text-[10px] text-muted-foreground mt-0.5">
                        {passage.session_seq !== undefined ? (
                          <span className="font-mono font-bold text-primary">S{passage.session_seq}-T{passage.turn_index}</span>
                        ) : (
                          <>{passage.session_id.slice(0, 8)}... / turn {passage.turn_index}</>
                        )}
                      </div>
                    </div>
                  </button>

                  {isExpanded && (
                    <div className="px-2 pb-2">
                      <div className="text-xs text-foreground whitespace-pre-wrap break-words max-h-40 overflow-y-auto bg-background rounded p-2 border border-border">
                        {passage.text}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </>
  );

  // Embedded mode: render inline without outer wrapper
  if (embedded) {
    return (
      <div className="flex flex-col h-full overflow-hidden">
        {/* Header for embedded mode */}
        <div className="flex items-center justify-between px-4 py-2 shrink-0 border-b border-border">
          <h3 className="text-sm font-semibold text-foreground truncate">{node.name}</h3>
          <button
            onClick={onClose}
            aria-label="Clear selection"
            className="text-muted-foreground hover:text-foreground transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
        <div className="flex-1 min-h-0 overflow-y-auto">
          {content}
        </div>
      </div>
    );
  }

  // Legacy mode: full panel with wrapper
  return (
    <div className="hidden md:flex w-80 h-full bg-card border-l border-border flex-col overflow-hidden shrink-0">
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <h3 className="text-sm font-semibold text-foreground truncate">{node.name}</h3>
        <button
          onClick={onClose}
          aria-label="Close detail panel"
          className="text-muted-foreground hover:text-foreground transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
      {content}
    </div>
  );
}
