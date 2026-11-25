/**
 * Setup Wizard API Client
 *
 * Phase 1 백엔드의 /api/setup/* 엔드포인트와 통신
 */

import { fetchApi } from './api';

// Phase 상태
export type PhaseStatus = 'pending' | 'in_progress' | 'complete' | 'failed';

// 각 Phase의 체크 결과 (백엔드 형식)
export interface CheckResult {
  success: boolean;
  message?: string;
  checked_at?: string;
}

// 전체 설정 상태
export interface SetupStatus {
  setup_complete: boolean;
  phases: {
    // greeting은 프론트엔드 전용 phase (백엔드에서는 상태 추적 안함)
    greeting: {
      status: PhaseStatus;
    };
    // prerequisites는 더 이상 사용되지 않지만, 백엔드 호환성을 위해 optional로 유지
    prerequisites?: {
      status: PhaseStatus;
      checks: {
        python?: CheckResult;
        postgres?: CheckResult;
        pgvector?: CheckResult;
        docker?: CheckResult;
        docker_postgres?: CheckResult;
      };
    };
    database: {
      status: PhaseStatus;
      checks?: Record<string, CheckResult>;
      config?: {
        backend?: 'postgres' | 'sqlite';
        host?: string;
        port?: number;
        user?: string;
        password?: string;
        database?: string;
        sqlite_path?: string;
      };
    };
    apikeys: {
      status: PhaseStatus;
      validated: boolean;
      has_api_key?: boolean;
    };
    profile: {
      status: PhaseStatus;
      name?: string;
      session_paths?: string[];  // Multi-folder indexing
      session_path?: string;     // Deprecated, for backwards compatibility
    };
    migration: {
      status: PhaseStatus;
    };
    verification: {
      status: PhaseStatus;
    };
  };
  last_updated: string;
}

// API Response types
export interface CheckResponse {
  success: boolean;
  version?: string;
  message?: string;
  error?: string;
}

// Verification response with detailed checks
export interface VerifyResponse {
  success: boolean;
  checks: {
    database?: { success: boolean; message: string };
    tables?: { success: boolean; message: string };
    api_key?: { success: boolean; message: string };
  };
}

export interface DatabaseConfigRequest {
  host: string;
  port: number;
  user: string;
  password: string;
  database: string;
}

export interface ApiKeyRequest {
  openai_api_key: string;
  model?: string;
  indexing_ner_model?: string;
  query_ner_model?: string;
  preprocessor_model?: string;
  enrich_stage1_model?: string;
  enrich_stage2_model?: string;
}

export interface ProfileRequest {
  name: string;
  session_paths: string[];  // Multi-folder indexing support
  session_path?: string;    // Deprecated, for backwards compatibility
}

// Docker API types
export interface DockerCheckResponse {
  success: boolean;
  installed: boolean;
  daemon_running?: boolean;
  version?: string;
  compose_available: boolean;
  compose_version?: string;
  error?: string;
}

export interface DockerStartRequest {
  postgres_user?: string;
  postgres_password?: string;
  postgres_db?: string;
  postgres_port?: number;
}

export interface DockerStartResponse {
  success: boolean;
  message: string;
  connection_info?: {
    host: string;
    port: number;
    user: string;
    password: string;
    database: string;
  };
}

export interface DockerStatusResponse {
  success: boolean;
  status: 'running' | 'stopped' | 'not_found' | 'error';
  container_id?: string;
  health?: string;
  error?: string;
}

// SQLite API types
export interface SqliteCheckResponse {
  success: boolean;
  available: boolean;
  version?: string;
  error?: string;
}

export interface BackendSelectRequest {
  sqlite_path?: string;
}

export interface BackendSelectResponse {
  success: boolean;
  backend: 'postgres' | 'sqlite';
  sqlite_path?: string;
  error?: string;
}

// API Functions
export async function getSetupStatus(): Promise<SetupStatus> {
  return fetchApi<SetupStatus>('/api/setup/status');
}

export async function checkPython(): Promise<CheckResponse> {
  return fetchApi<CheckResponse>('/api/setup/check/python', {
    method: 'POST',
  });
}

export async function checkPostgres(config?: DatabaseConfigRequest): Promise<CheckResponse> {
  return fetchApi<CheckResponse>('/api/setup/check/postgres', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config || {}),
  });
}

export async function checkPgvector(): Promise<CheckResponse> {
  return fetchApi<CheckResponse>('/api/setup/check/pgvector', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({}),
  });
}

export async function saveDatabase(config: DatabaseConfigRequest): Promise<CheckResponse> {
  return fetchApi<CheckResponse>('/api/setup/database', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
}

export async function saveApiKey(data: ApiKeyRequest): Promise<CheckResponse> {
  return fetchApi<CheckResponse>('/api/setup/apikey', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
}

export async function saveProfile(data: ProfileRequest): Promise<CheckResponse> {
  // Map frontend field names to backend field names
  const body = {
    profile_name: data.name,
    sessions_paths: data.session_paths,
  };
  return fetchApi<CheckResponse>('/api/setup/profile', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

export async function runMigration(): Promise<CheckResponse> {
  return fetchApi<CheckResponse>('/api/setup/migrate', {
    method: 'POST',
  });
}

export async function verifySetup(): Promise<VerifyResponse> {
  return fetchApi<VerifyResponse>('/api/setup/verify');
}

export async function completeSetup(): Promise<CheckResponse> {
  return fetchApi<CheckResponse>('/api/setup/complete', {
    method: 'POST',
  });
}

// Docker API Functions
export async function checkDocker(): Promise<DockerCheckResponse> {
  return fetchApi<DockerCheckResponse>('/api/setup/check/docker');
}

export async function startDockerPostgres(
  config?: DockerStartRequest
): Promise<DockerStartResponse> {
  return fetchApi<DockerStartResponse>('/api/setup/docker/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: config ? JSON.stringify(config) : undefined,
  });
}

export async function stopDockerPostgres(): Promise<CheckResponse> {
  return fetchApi<CheckResponse>('/api/setup/docker/stop', {
    method: 'POST',
  });
}

export async function getDockerStatus(): Promise<DockerStatusResponse> {
  return fetchApi<DockerStatusResponse>('/api/setup/docker/status');
}

// SQLite API Functions
export async function checkSqlite(): Promise<SqliteCheckResponse> {
  return fetchApi<SqliteCheckResponse>('/api/setup/check/sqlite');
}

export interface SqliteInstallResponse {
  success: boolean;
  message: string;
  version?: string;
}

export async function installSqliteVec(): Promise<SqliteInstallResponse> {
  return fetchApi<SqliteInstallResponse>('/api/setup/sqlite/install', {
    method: 'POST',
  });
}

export async function selectSqliteBackend(
  config?: BackendSelectRequest
): Promise<BackendSelectResponse> {
  return fetchApi<BackendSelectResponse>('/api/setup/backend/sqlite', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: config ? JSON.stringify(config) : undefined,
  });
}

export async function selectPostgresBackend(): Promise<BackendSelectResponse> {
  return fetchApi<BackendSelectResponse>('/api/setup/backend/postgres', {
    method: 'POST',
  });
}

// SSE Progress Event
export interface ProgressEvent {
  status: 'starting' | 'in_progress' | 'complete' | 'error';
  message: string;
  progress: number;
  connection_info?: {
    host: string;
    port: number;
    user: string;
    password: string;
    database: string;
  };
  install_guide?: {
    title: string;
    url: string;
    steps: string[];
    notes: string[];
  };
}

// Install Docker with SSE progress streaming
export function installDocker(
  onProgress: (event: ProgressEvent) => void,
  onError: (error: string) => void,
  onComplete: () => void
): () => void {
  const baseUrl = import.meta.env.VITE_API_BASE_URL || '';
  const eventSource = new EventSource(`${baseUrl}/api/setup/docker/install`);

  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data) as ProgressEvent;
      onProgress(data);

      if (data.status === 'complete' || data.status === 'error') {
        eventSource.close();
        onComplete();
      }
    } catch (e) {
      console.error('Failed to parse SSE event:', e);
    }
  };

  eventSource.onerror = () => {
    eventSource.close();
    onError('Connection lost');
    onComplete();
  };

  // Return cleanup function
  return () => {
    eventSource.close();
  };
}

// Start PostgreSQL with SSE progress streaming
export function startPostgresWithProgress(
  config: DockerStartRequest | undefined,
  onProgress: (event: ProgressEvent) => void,
  onError: (error: string) => void,
  onComplete: () => void
): () => void {
  const baseUrl = import.meta.env.VITE_API_BASE_URL || '';

  // For SSE with POST, we need to use fetch with ReadableStream
  const controller = new AbortController();

  fetch(`${baseUrl}/api/setup/postgres/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: config ? JSON.stringify(config) : '{}',
    signal: controller.signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No response body');
      }

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Parse SSE events from buffer
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6)) as ProgressEvent;
              onProgress(data);

              if (data.status === 'complete' || data.status === 'error') {
                onComplete();
                return;
              }
            } catch (e) {
              console.error('Failed to parse SSE event:', e);
            }
          }
        }
      }

      onComplete();
    })
    .catch((err) => {
      if (err.name !== 'AbortError') {
        onError(err.message);
        onComplete();
      }
    });

  // Return cleanup function
  return () => {
    controller.abort();
  };
}

// Claude projects response
export interface ClaudeProject {
  folder_name: string;
  original_path: string;
  display_name: string;
  session_path: string;
  session_count: number;
}

export interface ClaudeProjectsResponse {
  success: boolean;
  config_path: string | null;
  projects: ClaudeProject[];
  error?: string;
}

/**
 * List available Claude Code projects
 */
export async function listClaudeProjects(): Promise<ClaudeProjectsResponse> {
  return fetchApi<ClaudeProjectsResponse>('/api/setup/claude-projects');
}
