/**
 * Database Phase Component
 *
 * Phase 2 (이전 Prerequisites + Database 통합):
 * - 백엔드 선택 (PostgreSQL / SQLite)
 * - PostgreSQL: Docker 자동 설치 또는 수동 연결
 * - SQLite: sqlite-vec 설정
 */

import { useState, useCallback, useEffect } from 'react';
import { useSetup } from '../useSetup';
import { InputField, PasswordField, CheckItem, type CheckItemStatus } from '../shared';
import {
  checkPostgres,
  checkPgvector,
  saveDatabase,
  checkSqlite,
  installSqliteVec,
  selectSqliteBackend,
  selectPostgresBackend,
  checkDocker,
  getDockerStatus,
  installDocker,
  startPostgresWithProgress,
  type DatabaseConfigRequest,
  type DockerCheckResponse,
  type ProgressEvent,
} from '../../../lib/setup-api';

// 화면 상태 타입
type ViewState =
  | 'backend-select'      // 백엔드 선택 (PostgreSQL / SQLite)
  | 'postgres-method'     // PostgreSQL 설치 방식 선택 (Docker / Manual)
  | 'postgres-docker'     // Docker로 PostgreSQL 설치
  | 'postgres-manual'     // 수동 PostgreSQL 설정
  | 'sqlite-config';      // SQLite 설정

type DockerInstallState =
  | 'idle'
  | 'checking'
  | 'available'
  | 'installing'
  | 'installed'
  | 'not_available'
  | 'installing_docker'
  | 'needs_start';

interface FormData {
  host: string;
  port: string;
  user: string;
  password: string;
  database: string;
}

interface FormErrors {
  host?: string;
  port?: string;
  user?: string;
  password?: string;
  database?: string;
}

// 백엔드 선택 카드 컴포넌트
function BackendCard({
  title,
  description,
  subtitle,
  icon,
  selected,
  onSelect,
  disabled,
}: {
  title: string;
  description: string;
  subtitle?: string;
  icon: string;
  selected: boolean;
  onSelect: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      onClick={onSelect}
      disabled={disabled}
      className={`
        flex flex-col items-start p-4 rounded-lg border-2 transition-all text-left w-full
        ${selected
          ? 'border-primary bg-primary/5'
          : 'border-border hover:border-primary/50'
        }
        ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
      `}
    >
      <div className="text-2xl mb-2">{icon}</div>
      <h3 className="font-medium">{title}</h3>
      {subtitle && (
        <p className="text-xs text-primary/80 mt-0.5">{subtitle}</p>
      )}
      <p className="text-sm text-muted-foreground mt-1">{description}</p>
    </button>
  );
}

// SQLite 설정 화면
function SqliteConfig({
  onBack,
}: {
  onBack: () => void;
}) {
  const { updatePhaseStatus, nextPhase } = useSetup();
  const [sqlitePath, setSqlitePath] = useState('~/.reverie/reverie.db');
  const [checkStatus, setCheckStatus] = useState<CheckItemStatus>('pending');
  const [checkMessage, setCheckMessage] = useState<string>();
  const [checkError, setCheckError] = useState<string>();
  const [isChecking, setIsChecking] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isInstalling, setIsInstalling] = useState(false);

  // Check sqlite-vec on mount
  useEffect(() => {
    checkSqliteVec();
  }, []);

  const checkSqliteVec = async () => {
    setIsChecking(true);
    setCheckStatus('checking');
    setCheckMessage(undefined);
    setCheckError(undefined);

    try {
      const result = await checkSqlite();
      if (result.available) {
        setCheckStatus('passed');
        setCheckMessage(`sqlite-vec v${result.version} available`);
      } else {
        setCheckStatus('failed');
        setCheckError(result.error || 'sqlite-vec not available');
      }
    } catch (err) {
      setCheckStatus('failed');
      setCheckError(err instanceof Error ? err.message : 'Check failed');
    } finally {
      setIsChecking(false);
    }
  };

  const handleInstall = async () => {
    setIsInstalling(true);
    setCheckStatus('checking');
    setCheckMessage('Installing sqlite-vec...');
    setCheckError(undefined);

    try {
      const result = await installSqliteVec();
      if (result.success) {
        setCheckStatus('passed');
        setCheckMessage(result.message);
      } else {
        setCheckStatus('failed');
        setCheckError(result.message);
      }
    } catch (err) {
      setCheckStatus('failed');
      setCheckError(err instanceof Error ? err.message : 'Installation failed');
    } finally {
      setIsInstalling(false);
    }
  };

  const handleSaveAndContinue = async () => {
    // Check first if needed
    if (checkStatus === 'pending' || checkStatus === 'failed') {
      await checkSqliteVec();
      return;
    }

    if (checkStatus !== 'passed') {
      return;
    }

    setIsSaving(true);
    updatePhaseStatus('database', { status: 'in_progress' });

    try {
      const result = await selectSqliteBackend({ sqlite_path: sqlitePath });
      if (result.success) {
        updatePhaseStatus('database', {
          status: 'complete',
          config: {
            backend: 'sqlite',
            sqlite_path: result.sqlite_path,
          },
        });
        nextPhase();
      } else {
        updatePhaseStatus('database', { status: 'failed' });
        setCheckStatus('failed');
        setCheckError(result.error || 'Failed to save configuration');
      }
    } catch (err) {
      updatePhaseStatus('database', { status: 'failed' });
      setCheckStatus('failed');
      setCheckError(err instanceof Error ? err.message : 'Failed to save configuration');
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-muted/30 rounded-lg p-4">
        <p className="text-sm text-muted-foreground">
          SQLite is a lightweight database that stores everything in a single file.
          Perfect for getting started quickly without external dependencies.
        </p>
      </div>

      <InputField
        label="Database Path"
        value={sqlitePath}
        onChange={(e) => setSqlitePath(e.target.value)}
        placeholder="~/.reverie/reverie.db"
        description="Where to store the SQLite database file"
      />

      <CheckItem
        name="sqlite-vec Extension"
        description="Required for vector similarity search"
        status={checkStatus}
        message={checkMessage}
        error={checkError}
        onRetry={checkSqliteVec}
      />

      {/* Install button when sqlite-vec is not available */}
      {checkStatus === 'failed' && (
        <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
          <div className="flex items-start gap-3">
            <div className="flex-shrink-0 mt-0.5">
              <span className="text-xl">📦</span>
            </div>
            <div className="flex-1">
              <h4 className="text-sm font-medium text-blue-600 dark:text-blue-400">
                Install sqlite-vec
              </h4>
              <p className="text-xs text-muted-foreground mt-1">
                sqlite-vec is required for vector similarity search. Click below to install it automatically.
              </p>
              <button
                onClick={handleInstall}
                disabled={isInstalling}
                className="mt-3 px-4 py-2 text-sm font-medium rounded-md bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {isInstalling ? 'Installing...' : 'Install sqlite-vec'}
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="flex gap-3">
        <button
          onClick={onBack}
          className="px-4 py-2 text-sm font-medium rounded-md border border-border hover:bg-muted transition-colors"
        >
          ← Back
        </button>

        <button
          onClick={handleSaveAndContinue}
          disabled={isChecking || isSaving || isInstalling || checkStatus === 'failed'}
          className="px-4 py-2 text-sm font-medium rounded-md bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {isSaving ? 'Saving...' : 'Save & Continue →'}
        </button>
      </div>

      <div className="text-xs text-muted-foreground">
        <p>
          <strong>Note:</strong> SQLite is great for local development but may be slower
          for large datasets. Consider PostgreSQL for production use.
        </p>
      </div>
    </div>
  );
}

// PostgreSQL 수동 설정 화면
function PostgresManualConfig({
  onBack,
}: {
  onBack: () => void;
}) {
  const { status, updatePhaseStatus, nextPhase } = useSetup();

  // Get initial values from status (includes .env values)
  const config = status?.phases.database.config;
  const initialFormData: FormData = {
    host: config?.host || 'localhost',
    port: config?.port?.toString() || '5432',
    user: config?.user || '',
    password: config?.password || '',
    database: config?.database || 'reverie',
  };

  const [formData, setFormData] = useState<FormData>(initialFormData);
  const [errors, setErrors] = useState<FormErrors>({});
  const [testStatus, setTestStatus] = useState<CheckItemStatus>('pending');
  const [testMessage, setTestMessage] = useState<string>();
  const [testError, setTestError] = useState<string>();
  const [pgvectorStatus, setPgvectorStatus] = useState<CheckItemStatus>('pending');
  const [pgvectorMessage, setPgvectorMessage] = useState<string>();
  const [pgvectorError, setPgvectorError] = useState<string>();
  const [isTesting, setIsTesting] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  const handleChange = useCallback((field: keyof FormData, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    setErrors(prev => ({ ...prev, [field]: undefined }));
    setTestStatus('pending');
    setTestMessage(undefined);
    setTestError(undefined);
    setPgvectorStatus('pending');
    setPgvectorMessage(undefined);
    setPgvectorError(undefined);
  }, []);

  const validateForm = useCallback((): boolean => {
    const newErrors: FormErrors = {};

    if (!formData.host.trim()) {
      newErrors.host = 'Host is required';
    }

    const port = parseInt(formData.port, 10);
    if (!formData.port.trim() || isNaN(port) || port < 1 || port > 65535) {
      newErrors.port = 'Valid port number required (1-65535)';
    }

    if (!formData.user.trim()) {
      newErrors.user = 'Username is required';
    }

    if (!formData.database.trim()) {
      newErrors.database = 'Database name is required';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  }, [formData]);

  const getConfig = useCallback((): DatabaseConfigRequest => ({
    host: formData.host.trim(),
    port: parseInt(formData.port, 10),
    user: formData.user.trim(),
    password: formData.password,
    database: formData.database.trim(),
  }), [formData]);

  const handleTestConnection = useCallback(async (): Promise<boolean> => {
    if (!validateForm()) return false;

    setIsTesting(true);
    setTestStatus('checking');
    setTestMessage(undefined);
    setTestError(undefined);
    setPgvectorStatus('pending');
    setPgvectorMessage(undefined);
    setPgvectorError(undefined);

    try {
      const result = await checkPostgres(getConfig());

      if (result.success) {
        setTestStatus('passed');
        setTestMessage(result.version
          ? `Connected to ${result.version.split(' ').slice(0, 2).join(' ')}`
          : 'Connection successful');

        // Also check pgvector
        setPgvectorStatus('checking');
        const pgvectorResult = await checkPgvector();
        if (pgvectorResult.success) {
          setPgvectorStatus('passed');
          setPgvectorMessage(pgvectorResult.version
            ? `pgvector v${pgvectorResult.version}`
            : 'pgvector available');
          return true;
        } else {
          setPgvectorStatus('failed');
          setPgvectorError(pgvectorResult.error || 'pgvector not available');
          return false;
        }
      } else {
        setTestStatus('failed');
        setTestError(result.error || 'Connection failed');
        return false;
      }
    } catch (err) {
      setTestStatus('failed');
      setTestError(err instanceof Error ? err.message : 'Connection failed');
      return false;
    } finally {
      setIsTesting(false);
    }
  }, [validateForm, getConfig]);

  const handleSaveAndContinue = useCallback(async () => {
    if (!validateForm()) return;

    if (testStatus !== 'passed' || pgvectorStatus !== 'passed') {
      const testPassed = await handleTestConnection();
      if (!testPassed) return;
    }

    setIsSaving(true);
    updatePhaseStatus('database', { status: 'in_progress' });

    try {
      // First select postgres backend
      await selectPostgresBackend();

      // Then save the connection config
      const result = await saveDatabase(getConfig());

      if (result.success) {
        updatePhaseStatus('database', {
          status: 'complete',
          config: {
            backend: 'postgres',
            host: formData.host,
            port: parseInt(formData.port, 10),
            user: formData.user,
            database: formData.database,
          },
        });
        nextPhase();
      } else {
        updatePhaseStatus('database', { status: 'failed' });
        setTestStatus('failed');
        setTestError(result.error || 'Failed to save configuration');
      }
    } catch (err) {
      updatePhaseStatus('database', { status: 'failed' });
      setTestStatus('failed');
      setTestError(err instanceof Error ? err.message : 'Failed to save configuration');
    } finally {
      setIsSaving(false);
    }
  }, [validateForm, testStatus, pgvectorStatus, handleTestConnection, getConfig, updatePhaseStatus, nextPhase, formData]);

  const allPassed = testStatus === 'passed' && pgvectorStatus === 'passed';

  return (
    <div className="space-y-6">
      <div className="bg-muted/30 rounded-lg p-4">
        <p className="text-sm text-muted-foreground">
          Configure your PostgreSQL database connection. Make sure pgvector extension is installed.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <InputField
          label="Host"
          value={formData.host}
          onChange={(e) => handleChange('host', e.target.value)}
          placeholder="localhost"
          error={errors.host}
          required
        />

        <InputField
          label="Port"
          type="number"
          value={formData.port}
          onChange={(e) => handleChange('port', e.target.value)}
          placeholder="5432"
          error={errors.port}
          required
        />

        <InputField
          label="Username"
          value={formData.user}
          onChange={(e) => handleChange('user', e.target.value)}
          placeholder="postgres"
          error={errors.user}
          required
        />

        <PasswordField
          label="Password"
          value={formData.password}
          onChange={(e) => handleChange('password', e.target.value)}
          placeholder="••••••••"
          error={errors.password}
        />

        <InputField
          label="Database Name"
          value={formData.database}
          onChange={(e) => handleChange('database', e.target.value)}
          placeholder="reverie"
          error={errors.database}
          className="md:col-span-2"
          required
        />
      </div>

      {/* Connection check results */}
      <div className="space-y-3">
        {testStatus !== 'pending' && (
          <CheckItem
            name="PostgreSQL Connection"
            description="Testing database connection..."
            status={testStatus}
            message={testMessage}
            error={testError}
            onRetry={handleTestConnection}
          />
        )}

        {pgvectorStatus !== 'pending' && (
          <CheckItem
            name="pgvector Extension"
            description="Check if pgvector is available"
            status={pgvectorStatus}
            message={pgvectorMessage}
            error={pgvectorError}
          />
        )}
      </div>

      <div className="flex gap-3">
        <button
          onClick={onBack}
          className="px-4 py-2 text-sm font-medium rounded-md border border-border hover:bg-muted transition-colors"
        >
          ← Back
        </button>

        <button
          onClick={handleTestConnection}
          disabled={isTesting || isSaving}
          className="px-4 py-2 text-sm font-medium rounded-md border border-border hover:bg-muted disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {isTesting ? 'Testing...' : 'Test Connection'}
        </button>

        <button
          onClick={handleSaveAndContinue}
          disabled={isTesting || isSaving}
          className="px-4 py-2 text-sm font-medium rounded-md bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {isSaving ? 'Saving...' : allPassed ? 'Save & Continue →' : 'Verify & Continue →'}
        </button>
      </div>

      {/* Troubleshooting */}
      {(testStatus === 'failed' || pgvectorStatus === 'failed') && (
        <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-4">
          <h4 className="text-sm font-medium text-amber-600 dark:text-amber-400 mb-2">
            Troubleshooting
          </h4>
          <ul className="text-xs text-muted-foreground space-y-1">
            {testStatus === 'failed' && (
              <>
                <li>• Ensure PostgreSQL is running and accessible</li>
                <li>• Check the host, port, username, and password</li>
                <li>• Verify the database exists</li>
              </>
            )}
            {pgvectorStatus === 'failed' && (
              <>
                <li>• Install pgvector: <code className="bg-muted px-1 rounded">CREATE EXTENSION vector;</code></li>
                <li>• Or install the pgvector package for your PostgreSQL version</li>
              </>
            )}
          </ul>
        </div>
      )}

      <div className="text-xs text-muted-foreground">
        <p>
          <strong>Note:</strong> The password will be stored in your .env file.
          Make sure this file is not committed to version control.
        </p>
      </div>
    </div>
  );
}

// PostgreSQL Docker 자동 설치 화면
function PostgresDockerInstall({
  onBack,
}: {
  onBack: () => void;
}) {
  const { updatePhaseStatus, nextPhase, setCheckResult } = useSetup();

  const [dockerState, setDockerState] = useState<DockerInstallState>('idle');
  const [dockerInfo, setDockerInfo] = useState<DockerCheckResponse | null>(null);
  const [installProgress, setInstallProgress] = useState<string>('');
  const [installPercent, setInstallPercent] = useState<number>(0);

  // Verify connection and proceed to next phase
  const verifyAndProceed = useCallback(async () => {
    setInstallProgress('Verifying PostgreSQL connection...');

    const dockerConfig = {
      host: 'localhost',
      port: 5432,
      user: 'reverie',
      password: 'reverie',
      database: 'reverie',
    };

    try {
      const postgresResult = await checkPostgres(dockerConfig);

      if (postgresResult.success) {
        const msg = postgresResult.version
          ? `PostgreSQL ${postgresResult.version.split(' ')[1] || 'connected'} (Docker)`
          : 'PostgreSQL connected (Docker)';

        setCheckResult('database', 'postgres', {
          success: true,
          message: msg,
        });

        // Check pgvector
        const pgvectorResult = await checkPgvector();
        if (pgvectorResult.success) {
          const pgMsg = pgvectorResult.version
            ? `pgvector v${pgvectorResult.version}`
            : 'pgvector available';

          setCheckResult('database', 'pgvector', {
            success: true,
            message: pgMsg,
          });

          // Save the configuration
          await selectPostgresBackend();
          await saveDatabase(dockerConfig);

          updatePhaseStatus('database', {
            status: 'complete',
            config: {
              backend: 'postgres',
              host: dockerConfig.host,
              port: dockerConfig.port,
              user: dockerConfig.user,
              database: dockerConfig.database,
            },
          });

          setInstallProgress('');
          nextPhase();
        } else {
          setInstallProgress(`pgvector check failed: ${pgvectorResult.error}`);
        }
      } else {
        setInstallProgress(`Connection failed: ${postgresResult.error}`);
      }
    } catch (err) {
      setInstallProgress(`Error: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  }, [setCheckResult, updatePhaseStatus, nextPhase]);

  // Install PostgreSQL via Docker
  const installDockerPostgres = useCallback(() => {
    setDockerState('installing');
    setInstallProgress('Starting PostgreSQL installation...');
    setInstallPercent(0);

    startPostgresWithProgress(
      undefined,
      (event: ProgressEvent) => {
        setInstallProgress(event.message);
        setInstallPercent(event.progress);

        if (event.status === 'complete') {
          setDockerState('installed');
          setTimeout(() => {
            verifyAndProceed();
          }, 500);
        } else if (event.status === 'error') {
          setDockerState('available');
        }
      },
      (error: string) => {
        setInstallProgress(`Error: ${error}`);
        setDockerState('available');
      },
      () => {}
    );
  }, [verifyAndProceed]);

  // Check Docker availability
  const checkDockerAvailability = useCallback(async () => {
    setDockerState('checking');
    try {
      const result = await checkDocker();
      console.log('[Reverie] Docker check result:', result);
      setDockerInfo(result);

      if (result.installed && result.daemon_running && result.compose_available) {
        // Docker fully available - check if container is already running
        const containerStatus = await getDockerStatus();
        if (containerStatus.status === 'running') {
          setDockerState('installed');
          // Auto-verify and proceed
          verifyAndProceed();
        } else {
          setDockerState('available');
        }
      } else if (result.installed && !result.daemon_running) {
        setDockerState('needs_start');
      } else {
        setDockerState('not_available');
      }
    } catch {
      setDockerState('not_available');
    }
  }, [verifyAndProceed]);

  // Install Docker itself
  const installDockerApp = useCallback(() => {
    setDockerState('installing_docker');
    setInstallProgress('Starting Docker installation...');
    setInstallPercent(0);

    installDocker(
      (event: ProgressEvent) => {
        setInstallProgress(event.message);
        setInstallPercent(event.progress);

        if (event.status === 'complete') {
          // Docker Desktop needs time to start after installation
          // Show message and transition to needs_start state
          setInstallProgress('Docker installed! Please start Docker Desktop and click "Re-check Docker Status".');
          setDockerState('needs_start');
        } else if (event.status === 'error') {
          setDockerState('not_available');
        }
      },
      (error: string) => {
        setInstallProgress(`Error: ${error}`);
        setDockerState('not_available');
      },
      () => {}
    );
  }, []);

  // Check Docker availability on mount
  useEffect(() => {
    checkDockerAvailability();
  }, [checkDockerAvailability]);

  return (
    <div className="space-y-6">
      <div className="bg-muted/30 rounded-lg p-4">
        <p className="text-sm text-muted-foreground">
          We'll automatically install PostgreSQL with pgvector using Docker.
          This is the easiest way to get started.
        </p>
      </div>

      {/* Docker status */}
      <div className="space-y-4">
        {dockerState === 'checking' && (
          <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
            <div className="flex items-center gap-2 text-sm text-blue-600 dark:text-blue-400">
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              Checking Docker availability...
            </div>
          </div>
        )}

        {dockerState === 'available' && (
          <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
            <div className="flex items-start gap-3">
              <div className="flex-shrink-0 mt-0.5">
                <span className="text-xl">🐳</span>
              </div>
              <div className="flex-1">
                <h4 className="text-sm font-medium text-green-600 dark:text-green-400">
                  Docker is ready!
                </h4>
                <p className="text-xs text-muted-foreground mt-1">
                  Click below to install PostgreSQL with pgvector automatically.
                </p>
                {dockerInfo?.version && (
                  <p className="text-xs text-muted-foreground mt-1">
                    {dockerInfo.version}
                  </p>
                )}
                <button
                  onClick={installDockerPostgres}
                  className="mt-3 px-4 py-2 text-sm font-medium rounded-md bg-green-600 text-white hover:bg-green-700 transition-colors"
                >
                  Install PostgreSQL with Docker
                </button>
              </div>
            </div>
          </div>
        )}

        {dockerState === 'installing' && (
          <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
            <div className="space-y-3">
              <div className="flex items-center gap-2 text-sm text-blue-600 dark:text-blue-400">
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Installing PostgreSQL...
              </div>
              {installProgress && (
                <p className="text-xs text-muted-foreground">{installProgress}</p>
              )}
              {installPercent > 0 && (
                <div className="w-full bg-muted rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${installPercent}%` }}
                  />
                </div>
              )}
            </div>
          </div>
        )}

        {dockerState === 'installed' && (
          <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
            <div className="flex items-center gap-2 text-sm text-green-600 dark:text-green-400">
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
              PostgreSQL container is running!
            </div>
            {installProgress && (
              <p className="text-xs text-muted-foreground mt-2">{installProgress}</p>
            )}
            {installProgress && installProgress.toLowerCase().includes('failed') && (
              <button
                onClick={verifyAndProceed}
                className="mt-3 px-4 py-2 text-sm font-medium rounded-md bg-blue-600 text-white hover:bg-blue-700 transition-colors"
              >
                Retry Connection
              </button>
            )}
          </div>
        )}

        {dockerState === 'installing_docker' && (
          <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
            <div className="space-y-3">
              <div className="flex items-center gap-2 text-sm text-blue-600 dark:text-blue-400">
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Installing Docker...
              </div>
              {installProgress && (
                <p className="text-xs text-muted-foreground">{installProgress}</p>
              )}
              {installPercent > 0 && (
                <div className="w-full bg-muted rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${installPercent}%` }}
                  />
                </div>
              )}
            </div>
          </div>
        )}

        {dockerState === 'needs_start' && (
          <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-4">
            <div className="flex items-start gap-3">
              <div className="flex-shrink-0 mt-0.5">
                <svg className="h-5 w-5 text-amber-600 dark:text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <div className="flex-1">
                <h4 className="text-sm font-medium text-amber-600 dark:text-amber-400">
                  Docker Desktop Needs to Start
                </h4>
                <p className="text-xs text-muted-foreground mt-1">
                  Docker Desktop is installed but not running. Please start Docker Desktop and wait for it to initialize.
                </p>
                {dockerInfo?.version && (
                  <p className="text-xs text-muted-foreground mt-1">
                    Installed: {dockerInfo.version}
                  </p>
                )}
                <button
                  onClick={checkDockerAvailability}
                  className="mt-3 px-4 py-2 text-sm font-medium rounded-md bg-blue-600 text-white hover:bg-blue-700 transition-colors"
                >
                  Re-check Docker Status
                </button>
                <p className="text-xs text-muted-foreground mt-2">
                  <strong>Tip:</strong> On Windows, search for "Docker Desktop" in the Start menu and launch it.
                </p>
              </div>
            </div>
          </div>
        )}

        {dockerState === 'not_available' && (
          <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-4">
            <div className="flex items-start gap-3">
              <div className="flex-shrink-0 mt-0.5">
                <svg className="h-5 w-5 text-amber-600 dark:text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
              </div>
              <div className="flex-1">
                <h4 className="text-sm font-medium text-amber-600 dark:text-amber-400">
                  Docker Not Available
                </h4>
                <p className="text-xs text-muted-foreground mt-1">
                  Docker is required for automatic PostgreSQL installation.
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  <strong>Tip:</strong> If you've just installed Docker, try restarting your terminal (PowerShell/Command Prompt) to refresh the PATH.
                </p>
                <div className="flex flex-wrap gap-2 mt-3">
                  <button
                    onClick={installDockerApp}
                    className="px-4 py-2 text-sm font-medium rounded-md bg-blue-600 text-white hover:bg-blue-700 transition-colors"
                  >
                    Install Docker Automatically
                  </button>
                  <a
                    href="https://docs.docker.com/get-docker/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="px-4 py-2 text-sm font-medium rounded-md border border-muted-foreground/30 text-muted-foreground hover:bg-muted transition-colors"
                  >
                    Manual Install Guide →
                  </a>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="flex gap-3">
        <button
          onClick={onBack}
          disabled={dockerState === 'installing' || dockerState === 'installing_docker'}
          className="px-4 py-2 text-sm font-medium rounded-md border border-border hover:bg-muted disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          ← Back
        </button>
      </div>
    </div>
  );
}

// PostgreSQL 설치 방식 선택 화면
function PostgresMethodSelect({
  onBack,
  onSelectDocker,
  onSelectManual,
}: {
  onBack: () => void;
  onSelectDocker: () => void;
  onSelectManual: () => void;
}) {
  return (
    <div className="space-y-6">
      <div className="bg-muted/30 rounded-lg p-4">
        <p className="text-sm text-muted-foreground">
          Choose how you want to set up PostgreSQL. Docker is the easiest option
          if you don't have PostgreSQL installed yet.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <BackendCard
          title="Auto-install with Docker"
          subtitle="Recommended"
          description="Automatically set up PostgreSQL + pgvector with Docker. The easiest way to get started."
          icon="🐳"
          selected={false}
          onSelect={onSelectDocker}
        />

        <BackendCard
          title="Use Existing Installation"
          description="Connect to an existing PostgreSQL server. Requires pgvector extension."
          icon="💻"
          selected={false}
          onSelect={onSelectManual}
        />
      </div>

      <div className="flex gap-3">
        <button
          onClick={onBack}
          className="px-4 py-2 text-sm font-medium rounded-md border border-border hover:bg-muted transition-colors"
        >
          ← Back
        </button>
      </div>

      <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
        <h4 className="text-sm font-medium text-blue-400 mb-2">Which should I choose?</h4>
        <ul className="text-sm text-muted-foreground space-y-1">
          <li>• <strong>Docker</strong> - Recommended if you don't have PostgreSQL installed</li>
          <li>• <strong>Manual</strong> - Use your existing PostgreSQL server</li>
        </ul>
      </div>
    </div>
  );
}

// 백엔드 선택 화면 (첫 화면)
function BackendSelect({
  onSelectPostgres,
  onSelectSqlite,
}: {
  onSelectPostgres: () => void;
  onSelectSqlite: () => void;
}) {
  const { prevPhase } = useSetup();

  return (
    <div className="space-y-6">
      <div className="bg-muted/30 rounded-lg p-4">
        <p className="text-sm text-muted-foreground">
          Choose a database backend for storing embeddings and indexed content.
          You can change this later if needed.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <BackendCard
          title="PostgreSQL"
          subtitle="Recommended for production"
          description="Production-ready database with pgvector extension. Best for larger datasets and multiple users."
          icon="🐘"
          selected={false}
          onSelect={onSelectPostgres}
        />

        <BackendCard
          title="SQLite"
          subtitle="Easy setup, local development"
          description="Lightweight, zero-configuration database. Great for getting started quickly."
          icon="📦"
          selected={false}
          onSelect={onSelectSqlite}
        />
      </div>

      <div className="flex gap-3">
        <button
          onClick={prevPhase}
          className="px-4 py-2 text-sm font-medium rounded-md border border-border hover:bg-muted transition-colors"
        >
          ← Back
        </button>
      </div>

      <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
        <h4 className="text-sm font-medium text-blue-400 mb-2">Which should I choose?</h4>
        <ul className="text-sm text-muted-foreground space-y-1">
          <li>• <strong>SQLite</strong> - Best for local development, testing, or small datasets</li>
          <li>• <strong>PostgreSQL</strong> - Best for production, multiple users, or large datasets</li>
        </ul>
      </div>
    </div>
  );
}

// 메인 DatabasePhase 컴포넌트
export function DatabasePhase() {
  const { status } = useSetup();

  // Check if backend is already configured in .env
  const configuredBackend = status?.phases.database.config?.backend;

  // Always start from backend selection
  const getInitialView = (): ViewState => {
    return 'backend-select';
  };

  const [currentView, setCurrentView] = useState<ViewState>(getInitialView);

  // View navigation handlers
  const handleSelectPostgres = () => setCurrentView('postgres-method');
  const handleSelectSqlite = () => setCurrentView('sqlite-config');
  const handleSelectDocker = () => setCurrentView('postgres-docker');
  const handleSelectManual = () => setCurrentView('postgres-manual');
  const handleBackToBackendSelect = () => setCurrentView('backend-select');
  const handleBackToPostgresMethod = () => setCurrentView('postgres-method');

  // Render current view
  switch (currentView) {
    case 'backend-select':
      return (
        <BackendSelect
          onSelectPostgres={handleSelectPostgres}
          onSelectSqlite={handleSelectSqlite}
        />
      );

    case 'postgres-method':
      return (
        <PostgresMethodSelect
          onBack={handleBackToBackendSelect}
          onSelectDocker={handleSelectDocker}
          onSelectManual={handleSelectManual}
        />
      );

    case 'postgres-docker':
      return (
        <PostgresDockerInstall
          onBack={handleBackToPostgresMethod}
        />
      );

    case 'postgres-manual':
      return (
        <PostgresManualConfig
          onBack={configuredBackend ? handleBackToBackendSelect : handleBackToPostgresMethod}
        />
      );

    case 'sqlite-config':
      return (
        <SqliteConfig
          onBack={handleBackToBackendSelect}
        />
      );

    default:
      return (
        <BackendSelect
          onSelectPostgres={handleSelectPostgres}
          onSelectSqlite={handleSelectSqlite}
        />
      );
  }
}
