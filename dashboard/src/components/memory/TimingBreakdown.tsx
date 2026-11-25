/**
 * TimingBreakdown component for visualizing pipeline stage timing.
 *
 * Displays a horizontal bar chart showing the relative time spent
 * in each pipeline stage, with slow stages highlighted.
 */

import { type StageTiming } from './types';

/** Display names for pipeline stages */
const STAGE_LABELS: Record<string, string> = {
  // Pipeline internal stages
  graph_build: 'Graph Build',
  embed_query: 'Embed Query',
  knn_search: 'KNN Search',
  ner_extract: 'NER Extract',
  ner_embed: 'NER Embed',
  semantic_prop: 'Semantic Prop',
  ppr: 'PPR',
  hybrid_merge: 'Hybrid Merge',
  load_passages: 'Load Passages',
  // API-level stages
  pipeline: 'Pipeline',
  get_seed_entities: 'Get Seeds',
  build_result_list: 'Build Results',
  fetch_result_entities: 'Fetch Entities',
  get_traversal_graph: 'Traversal Graph',
  enrich_graph_nodes: 'Enrich Graph',
};

/** Colors for each stage (Tailwind classes) */
const STAGE_COLORS: Record<string, string> = {
  // Pipeline internal stages
  graph_build: 'bg-red-600',
  embed_query: 'bg-blue-500',
  knn_search: 'bg-green-500',
  ner_extract: 'bg-purple-500',
  ner_embed: 'bg-violet-500',
  semantic_prop: 'bg-amber-500',
  ppr: 'bg-rose-500',
  hybrid_merge: 'bg-cyan-500',
  load_passages: 'bg-indigo-500',
  // API-level stages
  pipeline: 'bg-blue-600',
  get_seed_entities: 'bg-emerald-500',
  build_result_list: 'bg-lime-500',
  fetch_result_entities: 'bg-teal-500',
  get_traversal_graph: 'bg-orange-500',
  enrich_graph_nodes: 'bg-pink-500',
};

/** Get label for stage (handles dynamic PPR iteration stages) */
function getStageLabel(key: string): string {
  if (STAGE_LABELS[key]) return STAGE_LABELS[key];
  // Handle ppr_i{N}_db and ppr_i{N}_prop
  const pprMatch = key.match(/^ppr_i(\d+)_(db|prop)$/);
  if (pprMatch) {
    const [, iter, type] = pprMatch;
    return type === 'db' ? `I${iter} DB` : `I${iter} P`;
  }
  return key;
}

/** Get color for stage (handles dynamic PPR iteration stages) */
function getStageColor(key: string): string {
  if (STAGE_COLORS[key]) return STAGE_COLORS[key];
  // Handle ppr_i{N}_db and ppr_i{N}_prop - alternate colors
  const pprMatch = key.match(/^ppr_i(\d+)_(db|prop)$/);
  if (pprMatch) {
    const [, , type] = pprMatch;
    return type === 'db' ? 'bg-rose-400' : 'bg-rose-600';
  }
  return 'bg-gray-500';
}

/** Base stage order (PPR iterations inserted dynamically) */
const BASE_STAGE_ORDER = [
  'graph_build',
  'embed_query',
  'knn_search',
  'ner_extract',
  'ner_embed',
  'semantic_prop',
  'ppr',  // Server-side PPR stage (or Python PPR iteration stages inserted here)
  // PPR iteration stages go here (ppr_i0_db, ppr_i0_prop, ppr_i1_db, ...)
  'hybrid_merge',
  'load_passages',
];

/** Stage order for API-level stages */
const API_STAGE_ORDER = [
  'pipeline',
  'get_seed_entities',
  'build_result_list',
  'fetch_result_entities',
  'get_traversal_graph',
  'enrich_graph_nodes',
];

interface TimingBreakdownProps {
  timing: StageTiming;
  /** Whether to show the expanded view with all stages */
  expanded?: boolean;
}

export function TimingBreakdown({ timing, expanded = true }: TimingBreakdownProps) {
  const { stages, total_ms, graph_build_ms, cache_hit, api_stages, seed_selection_details, ppr_details } = timing;

  // Merge graph_build_ms into stages if present
  const allStages = { ...stages };
  if (graph_build_ms !== null && graph_build_ms !== undefined) {
    allStages['graph_build'] = graph_build_ms;
  }

  // Build dynamic stage order with PPR iterations inserted
  const pprIterStages = Object.keys(allStages)
    .filter(k => k.match(/^ppr_i\d+_(db|prop)$/))
    .sort((a, b) => {
      // Sort by iteration number, then db before prop
      const matchA = a.match(/^ppr_i(\d+)_(db|prop)$/);
      const matchB = b.match(/^ppr_i(\d+)_(db|prop)$/);
      if (!matchA || !matchB) return 0;
      const iterA = parseInt(matchA[1]);
      const iterB = parseInt(matchB[1]);
      if (iterA !== iterB) return iterA - iterB;
      return matchA[2] === 'db' ? -1 : 1;
    });

  // Insert PPR iteration stages after semantic_prop
  const STAGE_ORDER = [...BASE_STAGE_ORDER];
  const insertIdx = STAGE_ORDER.indexOf('hybrid_merge');
  STAGE_ORDER.splice(insertIdx, 0, ...pprIterStages);

  // Check if using server-side PPR (no per-iteration stages = server-side)
  const isServerSidePpr = ppr_details?.server_side === true;

  // Sort pipeline stages by order, filter out missing stages
  const sortedStages = STAGE_ORDER
    .filter((stage) => allStages[stage] !== undefined)
    .map((stage) => {
      // For server-side PPR, show ppr stage with iteration count
      if (stage === 'ppr' && isServerSidePpr) {
        return {
          key: stage,
          label: `PPR (SQL, ${ppr_details?.iterations ?? '?'} iters)`,
          ms: allStages[stage],
          percent: total_ms > 0 ? (allStages[stage] / total_ms) * 100 : 0,
          color: 'bg-rose-500',
        };
      }
      return {
        key: stage,
        label: getStageLabel(stage),
        ms: allStages[stage],
        percent: total_ms > 0 ? (allStages[stage] / total_ms) * 100 : 0,
        color: getStageColor(stage),
      };
    });

  // Sort API-level stages
  const sortedApiStages = api_stages
    ? API_STAGE_ORDER
        .filter((stage) => api_stages[stage] !== undefined)
        .map((stage) => ({
          key: stage,
          label: STAGE_LABELS[stage] || stage,
          ms: api_stages[stage],
          color: STAGE_COLORS[stage] || 'bg-gray-500',
        }))
    : [];

  // Calculate total API time for percentage
  const totalApiMs = sortedApiStages.reduce((sum, s) => sum + s.ms, 0);

  // Identify slow stages (>50% of total)
  const slowThreshold = 50;

  if (!expanded) {
    // Compact view: bar with legend row
    return (
      <div className="space-y-1">
        {/* Pipeline stages bar */}
        <div className="flex items-center gap-2">
          <div className="flex h-2 flex-1 overflow-hidden rounded-full bg-muted">
            {sortedStages.map((stage) => (
              <div
                key={stage.key}
                className={`${stage.color} transition-all`}
                style={{ width: `${stage.percent}%` }}
                title={`${stage.label}: ${stage.ms.toFixed(1)}ms (${stage.percent.toFixed(0)}%)`}
              />
            ))}
          </div>
          <span className="text-xs text-muted-foreground whitespace-nowrap">
            {total_ms.toFixed(0)}ms
          </span>
        </div>
        {/* Pipeline compact legend - aggregate PPR iterations */}
        <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-muted-foreground">
          {(() => {
            // Separate PPR iteration stages from regular stages
            const regularStages = sortedStages.filter(s => !s.key.match(/^ppr_i\d+_(db|prop)$/));
            const pprDbStages = sortedStages.filter(s => s.key.match(/^ppr_i\d+_db$/));
            const pprPropStages = sortedStages.filter(s => s.key.match(/^ppr_i\d+_prop$/));

            const pprDbTotal = pprDbStages.reduce((sum, s) => sum + s.ms, 0);
            const pprPropTotal = pprPropStages.reduce((sum, s) => sum + s.ms, 0);
            const pprIterCount = pprDbStages.length;

            // Find where to insert PPR summary (after semantic_prop)
            const semanticPropIdx = regularStages.findIndex(s => s.key === 'semantic_prop');
            const beforePpr = regularStages.slice(0, semanticPropIdx + 1);
            const afterPpr = regularStages.slice(semanticPropIdx + 1);

            return (
              <>
                {beforePpr.map((stage) => (
                  <span key={stage.key} className="flex items-center gap-1">
                    <span className={`inline-block h-1.5 w-1.5 rounded-sm ${stage.color}`} />
                    <span>{stage.label}</span>
                    <span className="font-mono">{stage.ms.toFixed(0)}ms</span>
                  </span>
                ))}
                {/* Python PPR: show DB/Prop breakdown */}
                {pprIterCount > 0 && (
                  <>
                    <span className="flex items-center gap-1">
                      <span className="inline-block h-1.5 w-1.5 rounded-sm bg-rose-400" />
                      <span>PPR DB</span>
                      <span className="font-mono">{pprDbTotal.toFixed(0)}ms</span>
                    </span>
                    <span className="flex items-center gap-1">
                      <span className="inline-block h-1.5 w-1.5 rounded-sm bg-rose-600" />
                      <span>PPR Prop</span>
                      <span className="font-mono">{pprPropTotal.toFixed(0)}ms</span>
                    </span>
                    <span className="text-muted-foreground/60">({pprIterCount} iters)</span>
                  </>
                )}
                {/* Server-side PPR: semantic_prop already shows as "PPR (SQL, N iters)" */}
                {afterPpr.map((stage) => (
                  <span key={stage.key} className="flex items-center gap-1">
                    <span className={`inline-block h-1.5 w-1.5 rounded-sm ${stage.color}`} />
                    <span>{stage.label}</span>
                    <span className="font-mono">{stage.ms.toFixed(0)}ms</span>
                  </span>
                ))}
              </>
            );
          })()}
          <span className="ml-auto">
            {cache_hit ? '✓ cache' : '○ miss'}
          </span>
        </div>
        {/* API stages bar (if available) */}
        {sortedApiStages.length > 0 && (
          <>
            <div className="flex items-center gap-2 pt-1 border-t border-muted">
              <div className="flex h-2 flex-1 overflow-hidden rounded-full bg-muted">
                {sortedApiStages.map((stage) => {
                  const percent = totalApiMs > 0 ? (stage.ms / totalApiMs) * 100 : 0;
                  return (
                    <div
                      key={stage.key}
                      className={`${stage.color} transition-all`}
                      style={{ width: `${percent}%` }}
                      title={`${stage.label}: ${stage.ms.toFixed(1)}ms (${percent.toFixed(0)}%)`}
                    />
                  );
                })}
              </div>
              <span className="text-xs text-muted-foreground whitespace-nowrap">
                {totalApiMs.toFixed(0)}ms
              </span>
            </div>
            {/* API compact legend */}
            <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-muted-foreground">
              {sortedApiStages.map((stage) => (
                <span key={stage.key} className="flex items-center gap-1">
                  <span className={`inline-block h-1.5 w-1.5 rounded-sm ${stage.color}`} />
                  <span>{stage.label}</span>
                  <span className="font-mono">{stage.ms.toFixed(0)}ms</span>
                </span>
              ))}
            </div>
          </>
        )}
        {/* Seed selection details */}
        {seed_selection_details && (
          <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-muted-foreground pt-1 border-t border-muted">
            <span className="text-amber-500 font-medium">
              Seed Selection ({seed_selection_details.mode === 'entity'
                ? `${seed_selection_details.num_query_entities} entities`
                : 'query mode'}):
            </span>
            {seed_selection_details.mode === 'entity' && Object.entries(seed_selection_details)
              .filter(([k]) => k.startsWith('entity_') && k.endsWith('_ms'))
              .map(([k, v]) => (
                <span key={k} className="font-mono">
                  {k.replace('entity_', 'E').replace('_ms', '')}: {typeof v === 'number' ? v.toFixed(0) : v}ms
                </span>
              ))}
            <span className="ml-auto">
              {seed_selection_details.per_entity_avg_ms !== undefined && (
                <>avg: {seed_selection_details.per_entity_avg_ms.toFixed(0)}ms | </>
              )}
              seeds: {seed_selection_details.num_seeds_found}
            </span>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-2 text-sm">
      {/* Cache status */}
      <div className="flex items-center justify-between text-xs">
        <span className="text-muted-foreground">
          {cache_hit ? (
            <span className="text-green-500">● Cache Hit</span>
          ) : (
            <span className="text-amber-500">● Cache Miss</span>
          )}
          {graph_build_ms !== null && (
            <span className="ml-2 text-muted-foreground">
              (Graph build: {graph_build_ms.toFixed(0)}ms)
            </span>
          )}
        </span>
        <span className="font-medium">
          Total: {total_ms.toFixed(1)}ms
        </span>
      </div>

      {/* Stacked bar chart */}
      <div className="flex h-4 overflow-hidden rounded-full bg-muted">
        {sortedStages.map((stage) => (
          <div
            key={stage.key}
            className={`${stage.color} relative transition-all hover:brightness-110`}
            style={{ width: `${stage.percent}%` }}
            title={`${stage.label}: ${stage.ms.toFixed(1)}ms (${stage.percent.toFixed(0)}%)`}
          />
        ))}
      </div>

      {/* Legend with times */}
      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
        {sortedStages.map((stage) => {
          const isSlow = stage.percent >= slowThreshold;
          return (
            <div
              key={stage.key}
              className={`flex items-center justify-between ${isSlow ? 'text-amber-500 font-medium' : 'text-muted-foreground'}`}
            >
              <div className="flex items-center gap-1">
                <div className={`h-2 w-2 rounded-sm ${stage.color}`} />
                <span>{stage.label}</span>
              </div>
              <span className="font-mono">
                {stage.ms.toFixed(1)}ms
                {isSlow && ' ⚠'}
              </span>
            </div>
          );
        })}
      </div>

      {/* API-level stages (if available) */}
      {sortedApiStages.length > 0 && (
        <>
          <div className="border-t border-muted pt-2 mt-2">
            <div className="flex items-center justify-between text-xs mb-2">
              <span className="text-muted-foreground font-medium">API Stages</span>
              <span className="font-medium">{totalApiMs.toFixed(1)}ms</span>
            </div>
            {/* API stages bar */}
            <div className="flex h-4 overflow-hidden rounded-full bg-muted">
              {sortedApiStages.map((stage) => {
                const percent = totalApiMs > 0 ? (stage.ms / totalApiMs) * 100 : 0;
                return (
                  <div
                    key={stage.key}
                    className={`${stage.color} relative transition-all hover:brightness-110`}
                    style={{ width: `${percent}%` }}
                    title={`${stage.label}: ${stage.ms.toFixed(1)}ms (${percent.toFixed(0)}%)`}
                  />
                );
              })}
            </div>
          </div>
          {/* API stages legend */}
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
            {sortedApiStages.map((stage) => {
              const percent = totalApiMs > 0 ? (stage.ms / totalApiMs) * 100 : 0;
              const isSlow = percent >= slowThreshold;
              return (
                <div
                  key={stage.key}
                  className={`flex items-center justify-between ${isSlow ? 'text-orange-500 font-medium' : 'text-muted-foreground'}`}
                >
                  <div className="flex items-center gap-1">
                    <div className={`h-2 w-2 rounded-sm ${stage.color}`} />
                    <span>{stage.label}</span>
                  </div>
                  <span className="font-mono">
                    {stage.ms.toFixed(1)}ms
                    {isSlow && ' ⚠'}
                  </span>
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
