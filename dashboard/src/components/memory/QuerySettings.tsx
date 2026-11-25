import { Settings } from 'lucide-react';

export interface QueryParams {
  pprMinScore: number;
  pprMaxIter: number;
  tau: number;
  damping: number;
  beta: number;
  topK: number;
  halfLifeDays: number | null;  // null = disabled
}

interface QuerySettingsProps {
  params: QueryParams;
  onParamsChange: (params: QueryParams) => void;
  visible: boolean;
  embedded?: boolean;  // When true, render inline without absolute positioning
}

export function QuerySettings({ params, onParamsChange, visible, embedded = false }: QuerySettingsProps) {
  if (!visible) return null;

  const updateParam = <K extends keyof QueryParams>(key: K, value: QueryParams[K]) => {
    onParamsChange({ ...params, [key]: value });
  };

  const content = (
    <div className="space-y-4">
      {/* PPR Min Score - controls node pruning during PPR iteration */}
      <div className="space-y-1">
        <div className="flex justify-between text-xs">
          <label className="text-muted-foreground">PPR Min Score</label>
          <span className="font-mono text-foreground">{params.pprMinScore.toFixed(7)}</span>
        </div>
        <input
          type="range"
          min="0.0000001"
          max="0.001"
          step="0.0000001"
          value={params.pprMinScore}
          onChange={(e) => updateParam('pprMinScore', parseFloat(e.target.value))}
          className="w-full h-1 accent-purple-500"
        />
        <p className="text-[10px] text-muted-foreground">
          Min score for PPR nodes. Lower = more results but slower.
        </p>
      </div>

      {/* PPR Max Iter - controls iteration count */}
      <div className="space-y-1">
        <div className="flex justify-between text-xs">
          <label className="text-muted-foreground">PPR Max Iter</label>
          <span className="font-mono text-foreground">{params.pprMaxIter}</span>
        </div>
        <input
          type="range"
          min="10"
          max="50"
          step="1"
          value={params.pprMaxIter}
          onChange={(e) => updateParam('pprMaxIter', parseInt(e.target.value))}
          className="w-full h-1 accent-purple-500"
        />
        <p className="text-[10px] text-muted-foreground">
          Max PPR iterations. Higher = better convergence but slower.
        </p>
      </div>

      {/* Tau - Entity activation threshold */}
      <div className="space-y-1">
        <div className="flex justify-between text-xs">
          <label className="text-muted-foreground">Tau (Entity Activation)</label>
          <span className="font-mono text-foreground">{params.tau.toFixed(3)}</span>
        </div>
        <input
          type="range"
          min="0.1"
          max="0.9"
          step="0.001"
          value={params.tau}
          onChange={(e) => updateParam('tau', parseFloat(e.target.value))}
          className="w-full h-1 accent-purple-500"
        />
        <p className="text-[10px] text-muted-foreground">
          Entity activation threshold. Higher = fewer seed entities.
        </p>
      </div>

      {/* Damping - PPR damping factor */}
      <div className="space-y-1">
        <div className="flex justify-between text-xs">
          <label className="text-muted-foreground">Damping (PPR)</label>
          <span className="font-mono text-foreground">{params.damping.toFixed(3)}</span>
        </div>
        <input
          type="range"
          min="0.5"
          max="0.95"
          step="0.001"
          value={params.damping}
          onChange={(e) => updateParam('damping', parseFloat(e.target.value))}
          className="w-full h-1 accent-purple-500"
        />
        <p className="text-[10px] text-muted-foreground">
          PPR damping factor. Higher = more spread to neighbors.
        </p>
      </div>

      {/* Beta - PPR/KNN hybrid weight */}
      <div className="space-y-1">
        <div className="flex justify-between text-xs">
          <label className="text-muted-foreground">Beta (PPR Weight)</label>
          <span className="font-mono text-foreground">{params.beta.toFixed(3)}</span>
        </div>
        <input
          type="range"
          min="0"
          max="1"
          step="0.001"
          value={params.beta}
          onChange={(e) => updateParam('beta', parseFloat(e.target.value))}
          className="w-full h-1 accent-purple-500"
        />
        <p className="text-[10px] text-muted-foreground">
          PPR vs KNN weight. 1.0 = pure PPR, 0.0 = pure KNN.
        </p>
      </div>

      {/* Top K - Number of results */}
      <div className="space-y-1">
        <div className="flex justify-between text-xs">
          <label className="text-muted-foreground">Top K (Results)</label>
          <span className="font-mono text-foreground">{params.topK}</span>
        </div>
        <input
          type="range"
          min="10"
          max="100"
          step="1"
          value={params.topK}
          onChange={(e) => updateParam('topK', parseInt(e.target.value))}
          className="w-full h-1 accent-purple-500"
        />
        <p className="text-[10px] text-muted-foreground">
          Number of passages to retrieve.
        </p>
      </div>

      {/* Time Decay Half-Life */}
      <div className="space-y-1">
        <div className="flex justify-between text-xs">
          <label className="text-muted-foreground">Time Decay (Half-Life)</label>
          <span className="font-mono text-foreground">
            {params.halfLifeDays === null ? 'Off' : `${params.halfLifeDays}d`}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={params.halfLifeDays !== null}
            onChange={(e) => updateParam('halfLifeDays', e.target.checked ? 365 : null)}
            className="accent-purple-500"
          />
          <input
            type="range"
            min="7"
            max="1000"
            step="1"
            value={params.halfLifeDays ?? 365}
            disabled={params.halfLifeDays === null}
            onChange={(e) => updateParam('halfLifeDays', parseInt(e.target.value))}
            className="flex-1 h-1 accent-purple-500 disabled:opacity-50"
          />
        </div>
        <p className="text-[10px] text-muted-foreground">
          Older passages get lower scores. 365d = score halves after 1 year.
        </p>
      </div>
    </div>
  );

  // Embedded mode: render inline
  if (embedded) {
    return (
      <div>
        <h3 className="text-sm font-medium text-foreground mb-3 flex items-center gap-2">
          <Settings className="w-4 h-4" />
          Query Settings
        </h3>
        {content}
      </div>
    );
  }

  // Legacy mode: absolute positioned overlay (not used in new layout)
  return null;
}
