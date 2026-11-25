/**
 * Phase Indicator Component
 *
 * 설정 마법사의 단계 진행 표시
 */

import { SETUP_PHASES } from '../SetupContext';
import { useSetup } from '../useSetup';
import { cn } from '../../../lib/utils';

interface PhaseIndicatorProps {
  className?: string;
  compact?: boolean;
}

export function PhaseIndicator({ className, compact = false }: PhaseIndicatorProps) {
  const { currentPhase, goToPhase, status } = useSetup();

  const getPhaseStatus = (index: number): 'complete' | 'current' | 'upcoming' => {
    if (index < currentPhase) return 'complete';
    if (index === currentPhase) return 'current';
    return 'upcoming';
  };

  const getPhaseComplete = (phaseId: string): boolean => {
    if (!status) return false;
    const phase = status.phases[phaseId as keyof typeof status.phases];
    return phase?.status === 'complete';
  };

  if (compact) {
    return (
      <div className={cn('flex items-center gap-2', className)}>
        {SETUP_PHASES.map((phase, index) => {
          const phaseStatus = getPhaseStatus(index);
          const isComplete = getPhaseComplete(phase.id);

          return (
            <button
              key={phase.id}
              onClick={() => goToPhase(index)}
              className={cn(
                'w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium transition-colors',
                phaseStatus === 'current' && 'bg-primary text-primary-foreground',
                phaseStatus === 'complete' && 'bg-green-500 text-white',
                phaseStatus === 'upcoming' && 'bg-muted text-muted-foreground',
                isComplete && phaseStatus !== 'current' && 'bg-green-500 text-white'
              )}
              title={phase.title}
            >
              {isComplete ? (
                <CheckIcon className="w-4 h-4" />
              ) : (
                index + 1
              )}
            </button>
          );
        })}
      </div>
    );
  }

  return (
    <nav className={cn('flex flex-col gap-1', className)}>
      {SETUP_PHASES.map((phase, index) => {
        const phaseStatus = getPhaseStatus(index);
        const isComplete = getPhaseComplete(phase.id);

        return (
          <button
            key={phase.id}
            onClick={() => goToPhase(index)}
            className={cn(
              'flex items-center gap-3 px-3 py-2 rounded-lg text-left transition-colors',
              phaseStatus === 'current' && 'bg-primary/10 text-primary',
              phaseStatus === 'complete' && 'text-muted-foreground hover:bg-muted/50',
              phaseStatus === 'upcoming' && 'text-muted-foreground/50 hover:bg-muted/30'
            )}
          >
            <span
              className={cn(
                'w-7 h-7 rounded-full flex items-center justify-center text-sm font-medium shrink-0',
                phaseStatus === 'current' && 'bg-primary text-primary-foreground',
                phaseStatus === 'complete' && 'bg-green-500 text-white',
                phaseStatus === 'upcoming' && 'bg-muted text-muted-foreground',
                isComplete && phaseStatus !== 'current' && 'bg-green-500 text-white'
              )}
            >
              {isComplete ? (
                <CheckIcon className="w-4 h-4" />
              ) : (
                index + 1
              )}
            </span>
            <div className="flex flex-col min-w-0">
              <span className={cn(
                'text-sm font-medium truncate',
                phaseStatus === 'current' && 'text-foreground',
                phaseStatus !== 'current' && 'text-muted-foreground'
              )}>
                {phase.title}
              </span>
              <span className="text-xs text-muted-foreground truncate">
                {phase.description}
              </span>
            </div>
          </button>
        );
      })}
    </nav>
  );
}

// Simple check icon
function CheckIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={3}
    >
      <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
    </svg>
  );
}
