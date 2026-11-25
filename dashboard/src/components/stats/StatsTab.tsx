/**
 * StatsTab - System statistics and monitoring tab
 *
 * Displays:
 * - DB statistics (entities, passages, edges, sessions)
 * - Indexer status (workers, queue, current task)
 * - Entity/Passage browser
 */
import { SystemStats } from './SystemStats';
import { DataBrowser } from './DataBrowser';

export function StatsTab() {
  return (
    <div className="flex flex-col flex-1 overflow-hidden min-w-0 min-h-0 p-4 gap-4">
      {/* System Statistics - compact */}
      <div className="shrink-0">
        <SystemStats refreshInterval={3000} />
      </div>

      {/* Entity/Passage Browser - takes remaining space */}
      <div className="flex-1 min-h-0 overflow-hidden">
        <DataBrowser />
      </div>
    </div>
  );
}
