import { useState, useEffect, useRef, useCallback } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';

interface Entity {
  id: number;
  name: string;
  created_at: string | null;
  passage_count: number;
}

interface Passage {
  id: number;
  session_id: string;
  session_seq: number;
  turn_index: number;
  text_preview: string;
  entity_count: number;
}

interface EntityDetail {
  entity_id: number;
  passages: Array<{
    passage_id: number;
    session_id: string;
    turn_index: number;
    text: string;
  }>;
}

interface PassageDetail {
  passage_id: number;
  session_id: string;
  turn_index: number;
  text?: string;
  entities: Array<{
    entity_id: number;
    name: string;
    importance: number;
  }>;
}

type BrowserMode = 'entities' | 'passages';

export function DataBrowser() {
  const [mode, setMode] = useState<BrowserMode>('entities');
  const [entities, setEntities] = useState<Entity[]>([]);
  const [passages, setPassages] = useState<Passage[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [selectedEntity, setSelectedEntity] = useState<EntityDetail | null>(null);
  const [selectedPassage, setSelectedPassage] = useState<PassageDetail | null>(null);
  const [expandedPassageInEntity, setExpandedPassageInEntity] = useState<number | null>(null);

  const parentRef = useRef<HTMLDivElement>(null);
  const LIMIT = 50;

  // Fetch entities or passages
  const fetchData = useCallback(async (offset: number, reset = false) => {
    if (loading) return;
    setLoading(true);

    try {
      const endpoint = mode === 'entities' ? '/api/memory/entities' : '/api/memory/passages';
      const res = await fetch(`${endpoint}?offset=${offset}&limit=${LIMIT}`);
      if (!res.ok) throw new Error('Failed to fetch');

      const data = await res.json();
      if (mode === 'entities') {
        setEntities(prev => reset ? data.entities : [...prev, ...data.entities]);
      } else {
        setPassages(prev => reset ? data.passages : [...prev, ...data.passages]);
      }
      setTotal(data.total);
      setHasMore(offset + data[mode].length < data.total);
    } catch (err) {
      console.error('Fetch error:', err);
    } finally {
      setLoading(false);
    }
  }, [mode, loading]);

  // Reset when mode changes
  useEffect(() => {
    setEntities([]);
    setPassages([]);
    setTotal(0);
    setHasMore(true);
    setSelectedEntity(null);
    setSelectedPassage(null);
    fetchData(0, true);
  }, [mode]);

  // Virtualizer for current list
  const items = mode === 'entities' ? entities : passages;
  const rowVirtualizer = useVirtualizer({
    count: hasMore ? items.length + 1 : items.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 56,
    overscan: 5,
  });

  // Load more when scrolling to bottom
  useEffect(() => {
    const [lastItem] = [...rowVirtualizer.getVirtualItems()].reverse();
    if (!lastItem) return;

    if (lastItem.index >= items.length - 1 && hasMore && !loading) {
      fetchData(items.length);
    }
  }, [rowVirtualizer.getVirtualItems(), hasMore, loading, items.length, fetchData]);

  // Fetch entity detail
  const handleEntityClick = async (entity: Entity) => {
    try {
      const res = await fetch(`/api/memory/entity/${entity.id}/passages?limit=10`);
      if (!res.ok) throw new Error('Failed to fetch');
      const data = await res.json();
      setSelectedEntity({
        entity_id: entity.id,
        passages: data.passages || [],
      });
      setSelectedPassage(null);
      setExpandedPassageInEntity(null);
    } catch (err) {
      console.error('Entity detail error:', err);
    }
  };

  // Fetch passage detail
  const handlePassageClick = async (passage: Passage) => {
    try {
      const res = await fetch(`/api/memory/passage/${passage.id}/entities`);
      if (!res.ok) throw new Error('Failed to fetch');
      const data = await res.json();
      setSelectedPassage({
        passage_id: passage.id,
        session_id: data.session_id,
        turn_index: data.turn_index,
        text: data.text,
        entities: data.entities || [],
      });
      setSelectedEntity(null);
      setExpandedPassageInEntity(null);
    } catch (err) {
      console.error('Passage detail error:', err);
    }
  };

  // Handle passage click in entity detail view
  const handlePassageInEntityClick = (passageId: number) => {
    setExpandedPassageInEntity(prev => prev === passageId ? null : passageId);
  };

  return (
    <div className="flex h-full gap-4 min-h-0 overflow-hidden">
      {/* List panel - 1 part */}
      <Card className="flex-1 flex flex-col min-h-0 overflow-hidden">
        <CardHeader className="pb-2 shrink-0">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm">
              {mode === 'entities' ? 'Entities' : 'Passages'} ({total.toLocaleString()})
            </CardTitle>
            <div className="flex gap-1">
              <Button
                variant={mode === 'entities' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setMode('entities')}
              >
                Entities
              </Button>
              <Button
                variant={mode === 'passages' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setMode('passages')}
              >
                Passages
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent className="flex-1 min-h-0 p-0 overflow-hidden">
          <div
            ref={parentRef}
            className="h-full overflow-auto scrollbar-thin"
          >
            <div
              style={{
                height: `${rowVirtualizer.getTotalSize()}px`,
                width: '100%',
                position: 'relative',
              }}
            >
              {rowVirtualizer.getVirtualItems().map((virtualRow) => {
                const isLoader = virtualRow.index >= items.length;
                const item = items[virtualRow.index];

                return (
                  <div
                    key={virtualRow.index}
                    style={{
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      width: '100%',
                      height: `${virtualRow.size}px`,
                      transform: `translateY(${virtualRow.start}px)`,
                    }}
                    className="px-3"
                  >
                    {isLoader ? (
                      <div className="h-full flex items-center justify-center text-muted-foreground text-sm">
                        {loading ? 'Loading...' : 'Load more'}
                      </div>
                    ) : mode === 'entities' ? (
                      <EntityRow
                        entity={item as Entity}
                        onClick={() => handleEntityClick(item as Entity)}
                        selected={selectedEntity?.entity_id === (item as Entity).id}
                      />
                    ) : (
                      <PassageRow
                        passage={item as Passage}
                        onClick={() => handlePassageClick(item as Passage)}
                        selected={selectedPassage?.passage_id === (item as Passage).id}
                      />
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Detail panel - 2 parts */}
      <Card className="flex-[2] flex flex-col min-h-0 overflow-hidden">
        <CardHeader className="pb-2 shrink-0">
          <CardTitle className="text-sm">
            {selectedEntity ? `Entity: ${entities.find(e => e.id === selectedEntity.entity_id)?.name || selectedEntity.entity_id}` :
             selectedPassage ? `Passage #${selectedPassage.passage_id}` :
             'Select an item'}
          </CardTitle>
        </CardHeader>
        <CardContent className="flex-1 min-h-0 overflow-auto scrollbar-thin">
          {selectedEntity && (
            <div className="space-y-2">
              <div className="text-xs text-muted-foreground mb-2">
                Connected Passages ({selectedEntity.passages.length})
              </div>
              {selectedEntity.passages.map((p) => (
                <div
                  key={p.passage_id}
                  className={`p-2 rounded text-xs cursor-pointer transition-colors ${
                    expandedPassageInEntity === p.passage_id
                      ? 'bg-muted'
                      : 'bg-muted/50 hover:bg-muted/80'
                  }`}
                  onClick={() => handlePassageInEntityClick(p.passage_id)}
                >
                  <div className="font-medium mb-1 flex items-center justify-between">
                    <span>S{p.session_id?.slice(0, 8)}... T{p.turn_index}</span>
                    <span className="text-muted-foreground">
                      {expandedPassageInEntity === p.passage_id ? '▼' : '▶'}
                    </span>
                  </div>
                  {expandedPassageInEntity === p.passage_id ? (
                    <div className="mt-2 p-2 bg-background rounded border text-xs whitespace-pre-wrap max-h-64 overflow-auto scrollbar-thin">
                      {p.text}
                    </div>
                  ) : (
                    <div className="text-muted-foreground line-clamp-2">
                      {p.text?.slice(0, 100)}...
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
          {selectedPassage && (
            <div className="space-y-3">
              <div className="text-xs text-muted-foreground">
                Session: {selectedPassage.session_id?.slice(0, 8)}... | Turn: {selectedPassage.turn_index}
              </div>

              {/* Connected Entities */}
              <div>
                <div className="text-xs text-muted-foreground mb-2">
                  Connected Entities ({selectedPassage.entities.length})
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {selectedPassage.entities.map((e) => (
                    <span
                      key={e.entity_id}
                      className="px-2 py-1 bg-muted/50 rounded text-xs"
                      title={`Importance: ${(e.importance * 100).toFixed(0)}%`}
                    >
                      {e.name}
                    </span>
                  ))}
                </div>
              </div>

              {/* Passage Content */}
              {selectedPassage.text && (
                <div>
                  <div className="text-xs text-muted-foreground mb-2">
                    Passage Content
                  </div>
                  <div className="p-3 bg-muted/30 rounded border text-xs whitespace-pre-wrap max-h-96 overflow-auto scrollbar-thin">
                    {selectedPassage.text}
                  </div>
                </div>
              )}
            </div>
          )}
          {!selectedEntity && !selectedPassage && (
            <div className="text-muted-foreground text-sm text-center py-4">
              Click an item to see details
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function EntityRow({
  entity,
  onClick,
  selected,
}: {
  entity: Entity;
  onClick: () => void;
  selected: boolean;
}) {
  return (
    <div
      onClick={onClick}
      className={`h-full flex items-center justify-between px-3 py-2 rounded cursor-pointer hover:bg-muted/50 ${
        selected ? 'bg-muted' : ''
      }`}
    >
      <div className="flex-1 min-w-0">
        <div className="font-medium text-sm truncate">{entity.name}</div>
        <div className="text-xs text-muted-foreground">
          ID: {entity.id}
        </div>
      </div>
      <div className="text-xs text-muted-foreground shrink-0 ml-2">
        {entity.passage_count} passages
      </div>
    </div>
  );
}

function PassageRow({
  passage,
  onClick,
  selected,
}: {
  passage: Passage;
  onClick: () => void;
  selected: boolean;
}) {
  return (
    <div
      onClick={onClick}
      className={`h-full flex items-center justify-between px-3 py-2 rounded cursor-pointer hover:bg-muted/50 ${
        selected ? 'bg-muted' : ''
      }`}
    >
      <div className="flex-1 min-w-0">
        <div className="font-medium text-sm">
          S{passage.session_seq}-T{passage.turn_index}
        </div>
        <div className="text-xs text-muted-foreground truncate">
          {passage.text_preview}
        </div>
      </div>
      <div className="text-xs text-muted-foreground shrink-0 ml-2">
        {passage.entity_count} entities
      </div>
    </div>
  );
}
