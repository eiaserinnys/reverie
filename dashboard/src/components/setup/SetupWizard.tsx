/**
 * Setup Wizard Component
 *
 * 메인 마법사 컨테이너
 * - 현재 Phase에 맞는 UI 렌더링
 * - 네비게이션 (이전/다음)
 */

import { SetupProvider, SETUP_PHASES } from './SetupContext';
import { useSetup } from './useSetup';
import { PhaseIndicator } from './shared';
import {
  GreetingPhase,
  DatabasePhase,
  ApiKeysPhase,
  ProfilePhase,
  MigrationPhase,
  VerificationPhase,
} from './phases';

// Map phase IDs to components
const PHASE_COMPONENTS: Record<string, React.ComponentType> = {
  greeting: GreetingPhase,
  database: DatabasePhase,
  apikeys: ApiKeysPhase,
  profile: ProfilePhase,
  migration: MigrationPhase,
  verification: VerificationPhase,
};

function PhaseContent({ phaseId }: { phaseId: string }) {
  const PhaseComponent = PHASE_COMPONENTS[phaseId];

  if (!PhaseComponent) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
        <div className="text-4xl mb-4">❓</div>
        <h3 className="text-lg font-medium">Unknown Phase</h3>
        <p className="text-sm mt-1">Phase ID: {phaseId}</p>
      </div>
    );
  }

  return <PhaseComponent />;
}

function SetupWizardContent() {
  const {
    currentPhase,
    loading,
    error,
  } = useSetup();

  const currentPhaseInfo = SETUP_PHASES[currentPhase];

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="flex flex-col items-center gap-4">
          <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
          <p className="text-sm text-muted-foreground">Loading setup status...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="flex flex-col items-center gap-4 text-center max-w-md">
          <div className="w-12 h-12 rounded-full bg-red-500/10 flex items-center justify-center">
            <svg className="w-6 h-6 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          </div>
          <h3 className="text-lg font-medium">Failed to load setup</h3>
          <p className="text-sm text-muted-foreground">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full">
      {/* Sidebar with phase indicator */}
      <aside className="w-64 border-r border-border p-4 shrink-0">
        <h2 className="text-lg font-semibold mb-4">Setup Wizard</h2>
        <PhaseIndicator />
      </aside>

      {/* Main content area */}
      <main className="flex-1 flex flex-col min-w-0">
        {/* Phase header */}
        <header className="px-6 py-4 border-b border-border shrink-0">
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">
              Step {currentPhase + 1} of {SETUP_PHASES.length}
            </span>
          </div>
          <h1 className="text-xl font-semibold mt-1">{currentPhaseInfo.title}</h1>
          <p className="text-sm text-muted-foreground mt-1">
            {currentPhaseInfo.description}
          </p>
        </header>

        {/* Phase content */}
        <div className="flex-1 overflow-auto p-6">
          <PhaseContent phaseId={currentPhaseInfo.id} />
        </div>
      </main>
    </div>
  );
}

export interface SetupWizardProps {
  /** Callback when setup is complete */
  onComplete?: () => void;
}

export function SetupWizard({ onComplete }: SetupWizardProps) {
  return (
    <SetupProvider onComplete={onComplete}>
      <div className="h-dvh bg-background">
        <SetupWizardContent />
      </div>
    </SetupProvider>
  );
}
