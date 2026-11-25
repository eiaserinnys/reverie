/**
 * Migration Phase Component
 *
 * Phase 5: Database Migration
 * - 마이그레이션 실행
 * - 스키마 검증
 */

import { useState, useCallback, useEffect } from 'react';
import { useSetup } from '../useSetup';
import { CheckItem, type CheckItemStatus } from '../shared';
import { runMigration } from '../../../lib/setup-api';

interface MigrationState {
  status: CheckItemStatus;
  message?: string;
  error?: string;
}

export function MigrationPhase() {
  const { status, updatePhaseStatus, nextPhase, prevPhase } = useSetup();

  const [migrationState, setMigrationState] = useState<MigrationState>({ status: 'pending' });
  const [isRunning, setIsRunning] = useState(false);

  // Check if already completed
  useEffect(() => {
    if (status?.phases.migration.status === 'complete') {
      setMigrationState({
        status: 'passed',
        message: 'Migrations completed',
      });
    }
  }, [status]);

  const handleRunMigration = useCallback(async () => {
    setIsRunning(true);
    setMigrationState({ status: 'checking' });
    updatePhaseStatus('migration', { status: 'in_progress' });

    try {
      const result = await runMigration();

      if (result.success) {
        setMigrationState({
          status: 'passed',
          message: result.message || 'Migrations completed successfully',
        });
        updatePhaseStatus('migration', { status: 'complete' });
      } else {
        setMigrationState({
          status: 'failed',
          error: result.message || 'Migration failed',
        });
        updatePhaseStatus('migration', { status: 'failed' });
      }
    } catch (err) {
      const error = err instanceof Error ? err.message : 'Migration failed';
      setMigrationState({
        status: 'failed',
        error,
      });
      updatePhaseStatus('migration', { status: 'failed' });
    } finally {
      setIsRunning(false);
    }
  }, [updatePhaseStatus]);

  const isPassed = migrationState.status === 'passed';
  const isFailed = migrationState.status === 'failed';

  return (
    <div className="space-y-6">
      {/* Description */}
      <div className="bg-muted/30 rounded-lg p-4">
        <p className="text-sm text-muted-foreground">
          Run database migrations to create the required tables and indexes.
          This step sets up the schema for storing embeddings and indexed content.
        </p>
      </div>

      {/* Migration Status */}
      <div className="space-y-3">
        <CheckItem
          name="Database Migrations"
          description="Create tables and indexes for Reverie"
          status={migrationState.status}
          message={migrationState.message}
          error={migrationState.error}
          onRetry={handleRunMigration}
        />
      </div>

      {/* Warning for first run */}
      {migrationState.status === 'pending' && (
        <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-4">
          <h4 className="text-sm font-medium text-amber-600 dark:text-amber-400 mb-2">
            Before You Proceed
          </h4>
          <ul className="text-xs text-muted-foreground space-y-1">
            <li>• Ensure your database connection is properly configured</li>
            <li>• This will create the following tables: <code className="bg-muted px-1 rounded">passages</code>, <code className="bg-muted px-1 rounded">files</code>, <code className="bg-muted px-1 rounded">sessions</code></li>
            <li>• Existing tables will not be modified</li>
          </ul>
        </div>
      )}

      {/* Success Message */}
      {isPassed && (
        <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
          <h4 className="text-sm font-medium text-green-600 dark:text-green-400 mb-2">
            Migrations Complete
          </h4>
          <p className="text-xs text-muted-foreground">
            Database schema is ready. You can proceed to the final verification step.
          </p>
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-3">
        <button
          onClick={prevPhase}
          className="px-4 py-2 text-sm font-medium rounded-md border border-border hover:bg-muted transition-colors"
        >
          ← Back
        </button>

        {!isPassed && (
          <button
            onClick={handleRunMigration}
            disabled={isRunning}
            className="px-4 py-2 text-sm font-medium rounded-md bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isRunning ? 'Running Migrations...' : isFailed ? 'Retry Migration' : 'Run Migrations'}
          </button>
        )}

        {isPassed && (
          <button
            onClick={nextPhase}
            className="px-4 py-2 text-sm font-medium rounded-md bg-green-600 text-white hover:bg-green-700 transition-colors"
          >
            Continue →
          </button>
        )}
      </div>

      {/* Technical Details (collapsible) */}
      <details className="text-xs text-muted-foreground">
        <summary className="cursor-pointer font-medium hover:text-foreground">
          Technical Details
        </summary>
        <div className="mt-2 bg-muted/30 rounded-lg p-3 space-y-1">
          <p>Tables created:</p>
          <ul className="list-disc list-inside ml-2">
            <li><code>passages</code> - Indexed text passages with embeddings</li>
            <li><code>files</code> - Tracked session files</li>
            <li><code>sessions</code> - Claude session metadata</li>
          </ul>
          <p className="mt-2">Indexes:</p>
          <ul className="list-disc list-inside ml-2">
            <li>pgvector HNSW index for similarity search</li>
            <li>B-tree indexes for file path lookups</li>
          </ul>
        </div>
      </details>
    </div>
  );
}
