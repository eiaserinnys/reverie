import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';

interface MemoryStatus {
  entities: number;
  passages: number;
  entity_passages: number;
  passage_adjacency: number;
  sessions: number;
  last_indexed: string;
  process_memory_mb: number;
}

interface IndexStatus {
  running: boolean;
  current_session?: string | null;
  current_turn?: number | null;
  queue_size?: number;
  workers?: number;
  active_sessions?: string[];
}

interface SystemStatsProps {
  refreshInterval?: number;
}

function StatCard({
  title,
  value,
  subtitle,
}: {
  title: string;
  value: string | number;
  subtitle?: string;
}) {
  return (
    <Card className="flex-1 min-w-[140px]">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">
          {typeof value === 'number' ? value.toLocaleString() : value}
        </div>
        {subtitle && (
          <p className="text-xs text-muted-foreground mt-1">{subtitle}</p>
        )}
      </CardContent>
    </Card>
  );
}

export function SystemStats({ refreshInterval = 3000 }: SystemStatsProps) {
  const [memoryStatus, setMemoryStatus] = useState<MemoryStatus | null>(null);
  const [indexStatus, setIndexStatus] = useState<IndexStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchStatus() {
      try {
        // Fetch memory status
        const memoryRes = await fetch('/api/memory/status');
        if (memoryRes.ok) {
          const data = await memoryRes.json();
          if (!data.error) {
            setMemoryStatus(data);
          }
        }

        // Try to fetch index status
        try {
          const indexRes = await fetch('/api/index/status');
          if (indexRes.ok) {
            const data = await indexRes.json();
            if (!data.error) {
              setIndexStatus(data);
            }
          }
        } catch {
          // Index API might not be enabled
        }

        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch status');
      }
    }

    // Initial fetch
    fetchStatus();

    // Set up interval
    const interval = setInterval(fetchStatus, refreshInterval);

    return () => clearInterval(interval);
  }, [refreshInterval]);

  if (error) {
    return (
      <div className="p-4 bg-destructive/10 text-destructive rounded-lg">
        Error: {error}
      </div>
    );
  }

  if (!memoryStatus) {
    return (
      <div className="p-4 text-muted-foreground">
        Loading...
      </div>
    );
  }

  // Format indexer status for display
  const getIndexerStatus = () => {
    if (!indexStatus?.running) {
      return { status: 'Idle', subtitle: undefined };
    }
    const parts: string[] = [];
    if (indexStatus.current_session) {
      parts.push(`Session: ${indexStatus.current_session.slice(0, 8)}...`);
    }
    if (indexStatus.current_turn !== undefined && indexStatus.current_turn !== null) {
      parts.push(`Turn: ${indexStatus.current_turn}`);
    }
    return {
      status: 'Indexing',
      subtitle: parts.length > 0 ? parts.join(' | ') : undefined,
    };
  };

  const indexerInfo = getIndexerStatus();

  return (
    <div className="space-y-4">
      {/* Row 1: DB Statistics - 5 cards */}
      <div className="grid grid-cols-5 gap-4">
        <StatCard title="Entities" value={memoryStatus.entities} />
        <StatCard title="Passages" value={memoryStatus.passages} />
        <StatCard title="Entity-Passage Edges" value={memoryStatus.entity_passages} />
        <StatCard title="Adjacency Edges" value={memoryStatus.passage_adjacency} />
        <StatCard title="Sessions" value={memoryStatus.sessions} />
      </div>

      {/* Row 2: Indexer Status - 3 cards */}
      <div className="grid grid-cols-3 gap-4">
        <Card className="flex-1">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Indexer Status
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-start gap-4">
              <div className={`text-2xl font-bold ${indexStatus?.running ? 'text-green-500' : 'text-yellow-600'}`}>
                {indexerInfo.status}
              </div>
              {indexStatus?.running && indexStatus?.active_sessions && indexStatus.active_sessions.length > 0 && (
                <div className="flex-1 text-xs text-muted-foreground">
                  <div className="font-medium mb-1">Processing:</div>
                  <ul className="space-y-0.5">
                    {indexStatus.active_sessions.map((session, idx) => (
                      <li key={idx} className="font-mono truncate">
                        {session.slice(0, 12)}...
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
            {indexerInfo.subtitle && (
              <p className="text-xs text-muted-foreground mt-1">{indexerInfo.subtitle}</p>
            )}
          </CardContent>
        </Card>

        <StatCard
          title="Queue Size"
          value={indexStatus?.queue_size ?? 0}
          subtitle="pending sessions"
        />

        <StatCard
          title="Workers"
          value={indexStatus?.workers ?? 0}
          subtitle="index workers"
        />
      </div>
    </div>
  );
}
