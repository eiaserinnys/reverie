import { useCallback, useEffect, useMemo, useRef } from 'react';
import { ScrollArea } from '../ui/scroll-area';
import type { RetrievalResult } from './types';

interface ResultListProps {
  results: RetrievalResult[];
  selectedPassageId: number | null;
  onSelect: (passageId: number) => void;
  fullWidth?: boolean;  // When true, fills parent container width
}

/**
 * Left panel showing search results as a scrollable list.
 * - Sorted by score (descending)
 * - Click to select and highlight in graph
 * - Arrow keys to navigate up/down
 */
export function ResultList({ results, selectedPassageId, onSelect, fullWidth = false }: ResultListProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const itemRefs = useRef<Map<number, HTMLButtonElement>>(new Map());

  // Sort by score descending
  const sortedResults = useMemo(() => {
    return [...results].sort((a, b) => b.score - a.score);
  }, [results]);

  // Find current selection index
  const selectedIndex = useMemo(() => {
    if (selectedPassageId === null) return -1;
    return sortedResults.findIndex(r => r.passage_id === selectedPassageId);
  }, [sortedResults, selectedPassageId]);

  // Keyboard navigation handler
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (sortedResults.length === 0) return;

    let newIndex = selectedIndex;

    if (e.key === 'ArrowDown' || e.key === 'j') {
      e.preventDefault();
      newIndex = selectedIndex < sortedResults.length - 1 ? selectedIndex + 1 : 0;
    } else if (e.key === 'ArrowUp' || e.key === 'k') {
      e.preventDefault();
      newIndex = selectedIndex > 0 ? selectedIndex - 1 : sortedResults.length - 1;
    } else {
      return;
    }

    const result = sortedResults[newIndex];
    if (result) {
      onSelect(result.passage_id);
      // Scroll into view
      const button = itemRefs.current.get(result.passage_id);
      button?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
  }, [sortedResults, selectedIndex, onSelect]);

  // Register keyboard listener
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    container.addEventListener('keydown', handleKeyDown);
    return () => container.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  // Store button ref
  const setItemRef = useCallback((passageId: number, el: HTMLButtonElement | null) => {
    if (el) {
      itemRefs.current.set(passageId, el);
    } else {
      itemRefs.current.delete(passageId);
    }
  }, []);

  const containerClass = fullWidth
    ? 'flex flex-col flex-1 overflow-hidden'
    : 'hidden md:flex w-64 bg-card border-r border-border flex-col shrink-0';

  if (results.length === 0) {
    return (
      <div className={containerClass}>
        <div className="px-3 py-2 border-b border-border">
          <h3 className="text-sm font-semibold text-foreground">Results</h3>
        </div>
        <div className="flex-1 flex items-center justify-center p-4">
          <p className="text-xs text-muted-foreground text-center">
            No results found
          </p>
        </div>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      tabIndex={0}
      className={`${containerClass} overflow-hidden focus:outline-none`}
    >
      <div className="px-3 py-2 border-b border-border shrink-0">
        <h3 className="text-sm font-semibold text-foreground">
          Results ({results.length})
          <span className="text-[10px] font-normal text-muted-foreground ml-2">
            ↑↓ to navigate
          </span>
        </h3>
      </div>
      <div className="flex-1 min-h-0 relative">
        <div className="absolute inset-0 overflow-hidden">
          <ScrollArea className="h-full">
            <div className="p-2 space-y-1">
          {sortedResults.map((result, index) => {
            const isSelected = result.passage_id === selectedPassageId;
            const preview = result.text.slice(0, 80) + (result.text.length > 80 ? '...' : '');

            // Use session_seq/turn_index if available for display
            const locationLabel = result.session_seq !== undefined && result.session_seq !== null
              ? `S${result.session_seq}-T${result.turn_index}`
              : `P${result.passage_id}`;

            return (
              <button
                key={result.passage_id}
                ref={(el) => setItemRef(result.passage_id, el)}
                onClick={() => onSelect(result.passage_id)}
                className={`
                  w-full text-left p-2 rounded-md transition-colors
                  ${isSelected
                    ? 'bg-primary/20 border border-primary/40'
                    : 'hover:bg-secondary border border-transparent'
                  }
                `}
              >
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-[10px] font-mono text-muted-foreground">
                    #{index + 1}
                  </span>
                  <span className="text-[10px] font-mono font-bold text-primary">
                    {locationLabel}
                  </span>
                  <span className="text-[10px] font-mono text-amber-500 ml-auto">
                    {result.score.toFixed(4)}
                  </span>
                </div>
                <p className="text-xs text-foreground line-clamp-2">
                  {preview}
                </p>
              </button>
            );
          })}
            </div>
          </ScrollArea>
        </div>
      </div>
    </div>
  );
}
