/**
 * Setup Wizard Context
 *
 * 설정 마법사의 전역 상태 관리
 */

import React, { createContext, useState, useCallback, useEffect, useMemo } from 'react';
import type { SetupStatus, CheckResult } from '../../lib/setup-api';
import { getSetupStatus } from '../../lib/setup-api';

// Setup Phases
export const SETUP_PHASES = [
  { id: 'greeting', title: 'Welcome', description: 'Welcome to Reverie setup' },
  { id: 'database', title: 'Database', description: 'Choose and configure your database' },
  { id: 'apikeys', title: 'API Keys', description: 'Set up OpenAI API key' },
  { id: 'profile', title: 'Profile', description: 'Configure profile settings' },
  { id: 'migration', title: 'Migration', description: 'Run database migrations' },
  { id: 'verification', title: 'Verification', description: 'Final system check' },
] as const;

export type PhaseId = (typeof SETUP_PHASES)[number]['id'];

export interface SetupContextType {
  // Current state
  currentPhase: number;
  status: SetupStatus | null;
  loading: boolean;
  error: string | null;

  // Phase navigation
  goToPhase: (phase: number) => void;
  nextPhase: () => void;
  prevPhase: () => void;
  canGoNext: boolean;
  canGoPrev: boolean;

  // Status management
  refreshStatus: () => Promise<void>;
  updatePhaseStatus: (phaseId: PhaseId, update: Partial<SetupStatus['phases'][PhaseId]>) => void;

  // Check items
  getCheckResult: (phaseId: PhaseId, checkName: string) => CheckResult | undefined;
  setCheckResult: (phaseId: PhaseId, checkName: string, result: CheckResult) => void;

  // Completion callback
  finishSetup: () => void;
}

export interface SetupProviderProps {
  children: React.ReactNode;
  onComplete?: () => void;
}

const SetupContext = createContext<SetupContextType | undefined>(undefined);

export function SetupProvider({ children, onComplete }: SetupProviderProps) {
  const [currentPhase, setCurrentPhase] = useState(0);
  const [status, setStatus] = useState<SetupStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load initial status
  const refreshStatus = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getSetupStatus();
      // 백엔드에서 greeting을 제공하지 않을 수 있으므로 기본값 추가
      const statusWithGreeting: SetupStatus = {
        ...data,
        phases: {
          ...data.phases,
          greeting: data.phases.greeting || { status: 'pending' },
        },
      };
      setStatus(statusWithGreeting);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load setup status');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshStatus();
  }, [refreshStatus]);

  // Phase navigation
  const goToPhase = useCallback((phase: number) => {
    if (phase >= 0 && phase < SETUP_PHASES.length) {
      setCurrentPhase(phase);
    }
  }, []);

  const nextPhase = useCallback(() => {
    if (currentPhase < SETUP_PHASES.length - 1) {
      setCurrentPhase(prev => prev + 1);
    }
  }, [currentPhase]);

  const prevPhase = useCallback(() => {
    if (currentPhase > 0) {
      setCurrentPhase(prev => prev - 1);
    }
  }, [currentPhase]);

  // Update phase status locally
  const updatePhaseStatus = useCallback(<T extends PhaseId>(
    phaseId: T,
    update: Partial<SetupStatus['phases'][T]>
  ) => {
    setStatus(prev => {
      if (!prev) return prev;
      return {
        ...prev,
        phases: {
          ...prev.phases,
          [phaseId]: {
            ...prev.phases[phaseId],
            ...update,
          },
        },
        last_updated: new Date().toISOString(),
      };
    });
  }, []);

  // Get check result for a specific phase and check name
  const getCheckResult = useCallback((phaseId: PhaseId, checkName: string): CheckResult | undefined => {
    if (!status) return undefined;
    const phase = status.phases[phaseId];
    if ('checks' in phase && phase.checks) {
      return (phase.checks as Record<string, CheckResult>)[checkName];
    }
    return undefined;
  }, [status]);

  // Set check result for a specific phase and check name
  const setCheckResult = useCallback((phaseId: PhaseId, checkName: string, result: CheckResult) => {
    setStatus(prev => {
      if (!prev) return prev;
      const phase = prev.phases[phaseId];
      if ('checks' in phase) {
        return {
          ...prev,
          phases: {
            ...prev.phases,
            [phaseId]: {
              ...phase,
              checks: {
                ...(phase.checks || {}),
                [checkName]: result,
              },
            },
          },
          last_updated: new Date().toISOString(),
        };
      }
      return prev;
    });
  }, []);

  const canGoNext = currentPhase < SETUP_PHASES.length - 1;
  const canGoPrev = currentPhase > 0;

  // Finish setup and trigger onComplete callback
  const finishSetup = useCallback(() => {
    if (onComplete) {
      onComplete();
    }
  }, [onComplete]);

  const value = useMemo<SetupContextType>(() => ({
    currentPhase,
    status,
    loading,
    error,
    goToPhase,
    nextPhase,
    prevPhase,
    canGoNext,
    canGoPrev,
    refreshStatus,
    updatePhaseStatus,
    getCheckResult,
    setCheckResult,
    finishSetup,
  }), [currentPhase, status, loading, error, goToPhase, nextPhase, prevPhase, canGoNext, canGoPrev, refreshStatus, updatePhaseStatus, getCheckResult, setCheckResult, finishSetup]);

  return (
    <SetupContext.Provider value={value}>
      {children}
    </SetupContext.Provider>
  );
}

export { SetupContext };
