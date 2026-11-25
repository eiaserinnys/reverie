import { useState, useEffect, useCallback } from 'react';
import { Tabs, TabsList, TabsTrigger, TabsContent } from './components/ui/tabs';
import { MemoryExplorerTab } from './components/memory/MemoryExplorerTab';
import { StatsTab } from './components/stats';
import { SetupWizard } from './components/setup/SetupWizard';
import { getSetupStatus } from './lib/setup-api';

type AppState = 'loading' | 'setup' | 'dashboard' | 'error';

/**
 * Custom hook to manage setup status
 */
function useSetupStatus() {
  const [state, setState] = useState<AppState>('loading');
  const [error, setError] = useState<string | null>(null);

  const checkStatus = useCallback(async () => {
    setState('loading');
    setError(null);
    try {
      const status = await getSetupStatus();
      setState(status.setup_complete ? 'dashboard' : 'setup');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to connect to server');
      setState('error');
    }
  }, []);

  useEffect(() => {
    checkStatus();
  }, [checkStatus]);

  return { state, error, refresh: checkStatus };
}

/**
 * Loading state UI
 */
function LoadingState() {
  return (
    <div className="flex items-center justify-center h-dvh bg-background">
      <div className="flex flex-col items-center gap-4">
        <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
        <p className="text-sm text-muted-foreground">Loading...</p>
      </div>
    </div>
  );
}

/**
 * Error state UI with retry button
 */
function ErrorState({ error, onRetry }: { error: string; onRetry: () => void }) {
  return (
    <div className="flex items-center justify-center h-dvh bg-background">
      <div className="flex flex-col items-center gap-4 text-center max-w-md p-6">
        <div className="w-12 h-12 rounded-full bg-red-500/10 flex items-center justify-center">
          <svg className="w-6 h-6 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        </div>
        <h3 className="text-lg font-medium">Connection Error</h3>
        <p className="text-sm text-muted-foreground">{error}</p>
        <button
          onClick={onRetry}
          className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors"
        >
          Retry
        </button>
      </div>
    </div>
  );
}

/**
 * Main Dashboard UI
 */
function Dashboard() {
  return (
    <div className="flex flex-col h-dvh max-w-full overflow-hidden fixed inset-0">
      <Tabs defaultValue="query" className="flex flex-col flex-1 overflow-hidden">
        {/* Header with title and tabs */}
        <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-card shrink-0">
          {/* Left: Title */}
          <h1 className="text-lg font-semibold text-foreground">
            Reverie Dashboard
          </h1>

          {/* Right: Tab navigation */}
          <TabsList>
            <TabsTrigger value="query">Query</TabsTrigger>
            <TabsTrigger value="stats">Stats</TabsTrigger>
          </TabsList>
        </div>

        {/* Tab content */}
        <TabsContent value="query" className="flex flex-1 overflow-hidden min-h-0">
          <MemoryExplorerTab />
        </TabsContent>

        <TabsContent value="stats" className="flex flex-1 overflow-hidden min-h-0">
          <StatsTab />
        </TabsContent>
      </Tabs>
    </div>
  );
}

/**
 * App - Entry point with setup status routing
 *
 * Routes to:
 * - LoadingState while checking setup status
 * - SetupWizard if setup is not complete
 * - Dashboard if setup is complete
 * - ErrorState on connection failure
 */
function App() {
  const { state, error, refresh } = useSetupStatus();

  switch (state) {
    case 'loading':
      return <LoadingState />;
    case 'error':
      return <ErrorState error={error || 'Unknown error'} onRetry={refresh} />;
    case 'setup':
      return <SetupWizard onComplete={refresh} />;
    case 'dashboard':
      return <Dashboard />;
    default: {
      // Exhaustive check - TypeScript will error if we miss a case
      const _exhaustiveCheck: never = state;
      return _exhaustiveCheck;
    }
  }
}

export default App;
