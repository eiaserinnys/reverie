import { useMemo, useState } from 'react';
import { Check, Copy, ChevronDown, ChevronRight, Loader2 } from 'lucide-react';
import { ScrollArea } from '../ui/scroll-area';
import type { EnrichedFact, EnrichProgress, EnrichResponse } from './types';

interface EnrichResultProps {
  data: EnrichResponse | null;
  isLoading?: boolean;
  progress?: EnrichProgress | null;
  onPassageClick?: (passageId: number) => void;
}

/**
 * Get color class for relevance score.
 */
function getScoreColor(score: number): string {
  if (score >= 9) return 'text-green-400';
  if (score >= 7) return 'text-blue-400';
  if (score >= 5) return 'text-yellow-400';
  return 'text-muted-foreground';
}

/**
 * Get color class for confidence score.
 */
function getConfidenceColor(confidence: number): string {
  if (confidence >= 8) return 'text-emerald-400';
  if (confidence >= 6) return 'text-cyan-400';
  if (confidence >= 4) return 'text-orange-400';
  return 'text-red-400';
}

/**
 * Single fact item with expandable reason.
 */
function FactItem({
  fact,
  onPassageClick,
}: {
  fact: EnrichedFact;
  onPassageClick?: (passageId: number) => void;
}) {
  const [showReason, setShowReason] = useState(false);
  const hasCitation = fact.passage_id != null;
  const hasReason = fact.reason && fact.reason.trim().length > 0;
  const confidence = fact.confidence ?? 5;

  const handleCitationClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (hasCitation && onPassageClick) {
      onPassageClick(fact.passage_id!);
    }
  };

  return (
    <div className="bg-secondary/30 rounded-md px-3 py-2.5 overflow-hidden">
      <div
        className={`flex gap-2 text-[15px] leading-relaxed min-w-0 ${hasReason ? 'cursor-pointer' : ''}`}
        onClick={() => hasReason && setShowReason(!showReason)}
      >
        <span className="font-mono shrink-0 flex gap-1">
          <span className={getScoreColor(fact.score)}>[{fact.score}]</span>
          <span className={getConfidenceColor(confidence)}>({confidence})</span>
        </span>
        <span
          className="text-foreground min-w-0 flex-1"
          style={{ overflowWrap: 'anywhere', wordBreak: 'break-word' }}
        >
          {fact.text}
          {hasCitation && (
            <button
              onClick={handleCitationClick}
              className="italic text-blue-400 hover:text-blue-300 hover:underline ml-1"
            >
              [{fact.passage_id}]
            </button>
          )}
        </span>
      </div>
      {showReason && hasReason && (
        <div className="mt-2 ml-8 text-xs text-muted-foreground italic border-l-2 border-muted pl-2 break-words">
          {fact.reason}
        </div>
      )}
    </div>
  );
}

/**
 * Display Enrich pipeline results with scored facts.
 * The enricher uses LLM (gpt-5-mini) to extract and score
 * atomic facts from retrieved passages.
 */
export function EnrichResult({ data, isLoading = false, progress, onPassageClick }: EnrichResultProps) {
  const [copied, setCopied] = useState(false);
  const [showPassages, setShowPassages] = useState(false);

  // Sort facts by score descending
  const sortedFacts = useMemo(() => {
    if (!data?.facts) return [];
    return [...data.facts].sort((a, b) => b.score - a.score);
  }, [data?.facts]);

  // Format facts as copyable text
  const factsAsText = useMemo(() => {
    return sortedFacts.map(f => {
      const conf = f.confidence ?? 5;
      const ref = f.passage_id != null ? ` [${f.passage_id}]` : '';
      return `[${f.score}](${conf}) ${f.text}${ref}`;
    }).join('\n');
  }, [sortedFacts]);

  const handleCopy = async () => {
    if (!factsAsText) return;
    try {
      await navigator.clipboard.writeText(factsAsText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  // Loading state with progress
  if (isLoading) {
    const percent = progress ? Math.round((progress.current / progress.total) * 100) : 0;
    const stageLabel = progress?.stage === 'stage1' ? 'Extracting' : progress?.stage === 'stage2' ? 'Re-ranking' : 'Preparing';

    return (
      <div className="flex flex-col h-full items-center justify-center text-muted-foreground p-4">
        <Loader2 className="w-6 h-6 animate-spin mb-3" />
        <span className="text-sm mb-3">{stageLabel}...</span>

        {progress && (
          <div className="w-full max-w-xs">
            {/* Progress bar */}
            <div className="h-2 bg-secondary rounded-full overflow-hidden mb-2">
              <div
                className="h-full bg-primary transition-all duration-300"
                style={{ width: `${percent}%` }}
              />
            </div>
            {/* Progress text */}
            <div className="flex justify-between text-xs">
              <span>{progress.stage === 'stage1' ? 'Stage 1' : 'Stage 2'}</span>
              <span>{progress.current}/{progress.total}</span>
            </div>
            {progress.message && (
              <div className="text-xs text-center mt-1 text-muted-foreground/70">
                {progress.message}
              </div>
            )}
          </div>
        )}
      </div>
    );
  }

  // No data state
  if (!data) {
    return (
      <div className="flex flex-col h-full items-center justify-center text-muted-foreground text-sm p-4">
        <p>Search to see enriched context</p>
        <p className="text-xs mt-1">The enricher will extract scored facts from passages</p>
      </div>
    );
  }

  // Empty result state
  if (!data.facts || data.facts.length === 0) {
    return (
      <div className="flex flex-col h-full p-4">
        <div className="text-sm text-muted-foreground">
          No relevant facts found for this query.
        </div>
        <div className="mt-2 text-xs text-muted-foreground">
          Retrieved {data.passage_count} passages, but no facts were extracted.
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full overflow-hidden min-w-0">
      {/* Header with stats */}
      <div className="px-4 py-2 border-b border-border shrink-0 min-w-0">
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground">
            {data.passage_count} passages → {sortedFacts.length} facts in {data.elapsed_ms.toFixed(0)}ms
          </span>
          <button
            onClick={handleCopy}
            className="flex items-center gap-1 px-2 py-1 text-xs rounded-md bg-secondary hover:bg-secondary/80 transition-colors"
          >
            {copied ? (
              <>
                <Check className="w-3 h-3 text-green-500" />
                <span className="text-green-500">Copied!</span>
              </>
            ) : (
              <>
                <Copy className="w-3 h-3" />
                <span>Copy</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 overflow-hidden flex flex-col min-h-0 min-w-0">
        {/* Facts list - fills available space */}
        <div className="flex-1 flex flex-col min-h-0 min-w-0 p-4">
          <div className="text-xs text-muted-foreground font-medium mb-2 shrink-0 flex items-center gap-2">
            <span>Enriched Facts</span>
            <span className="text-[10px] font-normal">
              [score](confidence) • click for reason
            </span>
          </div>
          <div className="flex-1 min-h-0 min-w-0 relative">
            <div className="absolute inset-0 overflow-hidden">
              <ScrollArea className="h-full w-full">
                <div className="space-y-2 pr-2">
                  {sortedFacts.map((fact, idx) => (
                    <FactItem key={idx} fact={fact} onPassageClick={onPassageClick} />
                  ))}
                </div>
              </ScrollArea>
            </div>
          </div>
        </div>

        {/* Source passages section - collapsible */}
        <div className="border-t border-border shrink-0">
          <button
            onClick={() => setShowPassages(!showPassages)}
            className="flex items-center gap-2 w-full px-4 py-2 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors"
          >
            {showPassages ? (
              <ChevronDown className="w-3 h-3" />
            ) : (
              <ChevronRight className="w-3 h-3" />
            )}
            Source Passages ({data.retrieve_results.length})
          </button>

          {showPassages && (
            <div className="px-4 pb-3 max-h-48 overflow-y-auto">
              <div className="space-y-2">
                {data.retrieve_results.map((result) => (
                  <div
                    key={result.passage_id}
                    className="text-xs bg-secondary/50 rounded p-2"
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-mono text-muted-foreground">
                        {result.session_seq !== undefined
                          ? `S${result.session_seq}-T${result.turn_index}`
                          : `#${result.passage_id}`}
                      </span>
                      <span className="text-muted-foreground">
                        {result.score.toFixed(4)}
                      </span>
                    </div>
                    <div className="text-foreground line-clamp-2">
                      {result.text}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
