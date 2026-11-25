/**
 * Verification Phase Component
 *
 * Phase 6: Final Verification
 * - 인덱서 시작 가능 여부
 * - API 엔드포인트 응답 확인
 * - 전체 시스템 상태 확인
 */

import { useState, useCallback, useEffect } from 'react';
import { useSetup } from '../useSetup';
import { CheckItem, type CheckItemStatus } from '../shared';
import { verifySetup, completeSetup } from '../../../lib/setup-api';

interface VerificationCheck {
  name: string;
  description: string;
  status: CheckItemStatus;
  message?: string;
  error?: string;
}

export function VerificationPhase() {
  const { status, updatePhaseStatus, prevPhase, finishSetup } = useSetup();

  const [checks, setChecks] = useState<VerificationCheck[]>([
    { name: 'Database Connection', description: 'Verify database is accessible', status: 'pending' },
    { name: 'Schema Verification', description: 'Check required tables exist', status: 'pending' },
    { name: 'API Key', description: 'Verify OpenAI API key is configured', status: 'pending' },
  ]);

  const [isVerifying, setIsVerifying] = useState(false);
  const [isCompleting, setIsCompleting] = useState(false);
  const [allPassed, setAllPassed] = useState(false);
  const [setupCompleted, setSetupCompleted] = useState(false);

  // Check if already verified
  useEffect(() => {
    if (status?.phases.verification.status === 'complete') {
      setAllPassed(true);
      setChecks(prev => prev.map(c => ({ ...c, status: 'passed' as CheckItemStatus })));
    }
    if (status?.setup_complete) {
      setSetupCompleted(true);
    }
  }, [status]);

  const handleVerify = useCallback(async () => {
    setIsVerifying(true);
    setAllPassed(false);
    updatePhaseStatus('verification', { status: 'in_progress' });

    // Set all to checking
    setChecks(prev => prev.map(c => ({ ...c, status: 'checking' as CheckItemStatus })));

    try {
      const result = await verifySetup();

      const newChecks: VerificationCheck[] = [
        {
          name: 'Database Connection',
          description: 'Verify database is accessible',
          status: result.checks?.database?.success ? 'passed' : 'failed',
          message: result.checks?.database?.message,
          error: result.checks?.database?.success ? undefined : result.checks?.database?.message,
        },
        {
          name: 'Schema Verification',
          description: 'Check required tables exist',
          status: result.checks?.tables?.success ? 'passed' : 'failed',
          message: result.checks?.tables?.message,
          error: result.checks?.tables?.success ? undefined : result.checks?.tables?.message,
        },
        {
          name: 'API Key',
          description: 'Verify OpenAI API key is configured',
          status: result.checks?.api_key?.success ? 'passed' : 'failed',
          message: result.checks?.api_key?.message,
          error: result.checks?.api_key?.success ? undefined : result.checks?.api_key?.message,
        },
      ];

      setChecks(newChecks);
      setAllPassed(result.success);

      if (result.success) {
        updatePhaseStatus('verification', { status: 'complete' });
      } else {
        updatePhaseStatus('verification', { status: 'failed' });
      }
    } catch (err) {
      setChecks(prev => prev.map(c => ({
        ...c,
        status: 'failed' as CheckItemStatus,
        error: err instanceof Error ? err.message : 'Verification failed',
      })));
      updatePhaseStatus('verification', { status: 'failed' });
    } finally {
      setIsVerifying(false);
    }
  }, [updatePhaseStatus]);

  const handleCompleteSetup = useCallback(async () => {
    setIsCompleting(true);

    try {
      const result = await completeSetup();

      if (result.success) {
        setSetupCompleted(true);
        // Trigger App to refresh status and transition to Dashboard
        setTimeout(() => {
          finishSetup();
        }, 1500);
      }
    } catch (err) {
      console.error('Failed to complete setup:', err);
    } finally {
      setIsCompleting(false);
    }
  }, [finishSetup]);

  return (
    <div className="space-y-6">
      {/* Description */}
      <div className="bg-muted/30 rounded-lg p-4">
        <p className="text-sm text-muted-foreground">
          Final verification step. This confirms all components are properly
          configured and the system is ready to use.
        </p>
      </div>

      {/* Verification Checks */}
      <div className="space-y-3">
        {checks.map((check, index) => (
          <CheckItem
            key={index}
            name={check.name}
            description={check.description}
            status={check.status}
            message={check.message}
            error={check.error}
          />
        ))}
      </div>

      {/* Success Message */}
      {allPassed && !setupCompleted && (
        <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
          <h4 className="text-sm font-medium text-green-600 dark:text-green-400 mb-2">
            All Checks Passed!
          </h4>
          <p className="text-xs text-muted-foreground">
            Your Reverie instance is properly configured. Click "Complete Setup"
            to finish and start using the dashboard.
          </p>
        </div>
      )}

      {/* Setup Completed Message */}
      {setupCompleted && (
        <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-6 text-center">
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-green-500 flex items-center justify-center">
            <svg className="w-8 h-8 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-green-600 dark:text-green-400 mb-2">
            Setup Complete!
          </h3>
          <p className="text-sm text-muted-foreground">
            Redirecting to dashboard...
          </p>
        </div>
      )}

      {/* Actions */}
      {!setupCompleted && (
        <div className="flex gap-3">
          <button
            onClick={prevPhase}
            className="px-4 py-2 text-sm font-medium rounded-md border border-border hover:bg-muted transition-colors"
          >
            ← Back
          </button>

          <button
            onClick={handleVerify}
            disabled={isVerifying || isCompleting}
            className="px-4 py-2 text-sm font-medium rounded-md border border-border hover:bg-muted disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isVerifying ? 'Verifying...' : 'Run Verification'}
          </button>

          {allPassed && (
            <button
              onClick={handleCompleteSetup}
              disabled={isCompleting}
              className="px-4 py-2 text-sm font-medium rounded-md bg-green-600 text-white hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {isCompleting ? 'Completing...' : 'Complete Setup'}
            </button>
          )}
        </div>
      )}

      {/* Next Steps */}
      {allPassed && !setupCompleted && (
        <div className="bg-muted/30 rounded-lg p-4">
          <h4 className="text-sm font-medium mb-2">What's Next?</h4>
          <ul className="text-xs text-muted-foreground space-y-1">
            <li>• Start the indexer to process Claude session files</li>
            <li>• Use the search interface to query your conversations</li>
            <li>• Configure additional settings in the dashboard</li>
          </ul>
        </div>
      )}
    </div>
  );
}
