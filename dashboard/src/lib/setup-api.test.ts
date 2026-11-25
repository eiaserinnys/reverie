import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as setupApi from './setup-api';
import * as api from './api';

// Mock fetchApi
vi.mock('./api', () => ({
  fetchApi: vi.fn(),
}));

describe('setup-api', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('getSetupStatus', () => {
    it('calls the correct endpoint', async () => {
      const mockStatus: setupApi.SetupStatus = {
        setup_complete: false,
        phases: {
          greeting: { status: 'pending' },
          database: { status: 'pending' },
          apikeys: { status: 'pending', validated: false },
          profile: { status: 'pending' },
          migration: { status: 'pending' },
          verification: { status: 'pending' },
        },
        last_updated: '2026-02-11T00:00:00Z',
      };

      (api.fetchApi as ReturnType<typeof vi.fn>).mockResolvedValue(mockStatus);

      const result = await setupApi.getSetupStatus();

      expect(api.fetchApi).toHaveBeenCalledWith('/api/setup/status');
      expect(result).toEqual(mockStatus);
    });
  });

  describe('checkPython', () => {
    it('calls the correct endpoint with POST', async () => {
      (api.fetchApi as ReturnType<typeof vi.fn>).mockResolvedValue({ success: true, version: '3.11.4' });

      const result = await setupApi.checkPython();

      expect(api.fetchApi).toHaveBeenCalledWith('/api/setup/check/python', {
        method: 'POST',
      });
      expect(result.success).toBe(true);
    });
  });

  describe('checkPostgres', () => {
    it('calls without config', async () => {
      (api.fetchApi as ReturnType<typeof vi.fn>).mockResolvedValue({ success: true });

      await setupApi.checkPostgres();

      expect(api.fetchApi).toHaveBeenCalledWith('/api/setup/check/postgres', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: '{}',
      });
    });

    it('calls with config', async () => {
      const config = {
        host: 'localhost',
        port: 5432,
        user: 'postgres',
        password: 'secret',
        database: 'reverie',
      };

      (api.fetchApi as ReturnType<typeof vi.fn>).mockResolvedValue({ success: true });

      await setupApi.checkPostgres(config);

      expect(api.fetchApi).toHaveBeenCalledWith('/api/setup/check/postgres', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
    });
  });

  describe('saveDatabase', () => {
    it('sends database config', async () => {
      const config = {
        host: 'localhost',
        port: 5432,
        user: 'postgres',
        password: 'secret',
        database: 'reverie',
      };

      (api.fetchApi as ReturnType<typeof vi.fn>).mockResolvedValue({ success: true });

      await setupApi.saveDatabase(config);

      expect(api.fetchApi).toHaveBeenCalledWith('/api/setup/database', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
    });
  });

  describe('saveApiKey', () => {
    it('sends API key data', async () => {
      const data = {
        openai_api_key: 'sk-test-key',
        model: 'gpt-4',
      };

      (api.fetchApi as ReturnType<typeof vi.fn>).mockResolvedValue({ success: true });

      await setupApi.saveApiKey(data);

      expect(api.fetchApi).toHaveBeenCalledWith('/api/setup/apikey', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
    });
  });

  describe('saveProfile', () => {
    it('sends profile data with session_paths', async () => {
      const data = {
        name: 'my-profile',
        session_paths: ['/path/to/sessions', '/another/path'],
      };

      (api.fetchApi as ReturnType<typeof vi.fn>).mockResolvedValue({ success: true });

      await setupApi.saveProfile(data);

      // saveProfile should map frontend fields to backend fields
      expect(api.fetchApi).toHaveBeenCalledWith('/api/setup/profile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          profile_name: 'my-profile',
          sessions_paths: ['/path/to/sessions', '/another/path'],
        }),
      });
    });
  });

  describe('runMigration', () => {
    it('calls migration endpoint', async () => {
      (api.fetchApi as ReturnType<typeof vi.fn>).mockResolvedValue({ success: true });

      await setupApi.runMigration();

      expect(api.fetchApi).toHaveBeenCalledWith('/api/setup/migrate', {
        method: 'POST',
      });
    });
  });

  describe('verifySetup', () => {
    it('calls verify endpoint', async () => {
      (api.fetchApi as ReturnType<typeof vi.fn>).mockResolvedValue({ success: true });

      await setupApi.verifySetup();

      expect(api.fetchApi).toHaveBeenCalledWith('/api/setup/verify');
    });
  });

  describe('completeSetup', () => {
    it('calls complete endpoint', async () => {
      (api.fetchApi as ReturnType<typeof vi.fn>).mockResolvedValue({ success: true });

      await setupApi.completeSetup();

      expect(api.fetchApi).toHaveBeenCalledWith('/api/setup/complete', {
        method: 'POST',
      });
    });
  });
});
