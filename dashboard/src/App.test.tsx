/**
 * App Component Tests
 *
 * 진입점 분기 로직 테스트:
 * - setup_complete: true → Dashboard 표시
 * - setup_complete: false → SetupWizard 표시
 * - 로딩 중 → 로딩 UI 표시
 * - 에러 → 에러 UI 표시
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import App from './App';
import * as setupApi from './lib/setup-api';

// Mock GraphCanvas to avoid WebGL dependencies
vi.mock('./components/memory/GraphCanvas', () => ({
  GraphCanvas: () => <div data-testid="graph-canvas-mock">Graph Canvas</div>,
}));

// Mock setup-api
vi.mock('./lib/setup-api', () => ({
  getSetupStatus: vi.fn(),
}));

describe('App', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('shows loading state initially', async () => {
    // Mock slow API response
    vi.mocked(setupApi.getSetupStatus).mockImplementation(
      () => new Promise(() => {}) // Never resolves
    );

    render(<App />);

    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });

  it('shows Dashboard when setup is complete', async () => {
    vi.mocked(setupApi.getSetupStatus).mockResolvedValueOnce({
      setup_complete: true,
      phases: {
        greeting: { status: 'complete' },
        database: { status: 'complete' },
        apikeys: { status: 'complete', validated: true },
        profile: { status: 'complete' },
        migration: { status: 'complete' },
        verification: { status: 'complete' },
      },
      last_updated: new Date().toISOString(),
    });

    render(<App />);

    // Dashboard should have the title "Reverie Dashboard"
    await waitFor(() => {
      expect(screen.getByText('Reverie Dashboard')).toBeInTheDocument();
    });

    // Should NOT show Setup Wizard
    expect(screen.queryByText('Setup Wizard')).not.toBeInTheDocument();
  });

  it('shows SetupWizard when setup is not complete', async () => {
    const mockStatus = {
      setup_complete: false,
      phases: {
        greeting: { status: 'pending' as const },
        database: { status: 'pending' as const },
        apikeys: { status: 'pending' as const, validated: false },
        profile: { status: 'pending' as const },
        migration: { status: 'pending' as const },
        verification: { status: 'pending' as const },
      },
      last_updated: new Date().toISOString(),
    };

    // Both App and SetupContext call getSetupStatus
    vi.mocked(setupApi.getSetupStatus).mockResolvedValue(mockStatus);

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('Setup Wizard')).toBeInTheDocument();
    });

    // Should NOT show Dashboard title
    expect(screen.queryByText('Reverie Dashboard')).not.toBeInTheDocument();
  });

  it('shows error state when API fails', async () => {
    vi.mocked(setupApi.getSetupStatus).mockRejectedValueOnce(
      new Error('Connection refused')
    );

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText(/connection refused/i)).toBeInTheDocument();
    });

    // Should show retry button
    expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument();
  });

  it('retries loading when retry button is clicked', async () => {
    // First call fails, second succeeds
    vi.mocked(setupApi.getSetupStatus)
      .mockRejectedValueOnce(new Error('Connection refused'))
      .mockResolvedValueOnce({
        setup_complete: true,
        phases: {
          greeting: { status: 'complete' },
          database: { status: 'complete' },
          apikeys: { status: 'complete', validated: true },
          profile: { status: 'complete' },
          migration: { status: 'complete' },
          verification: { status: 'complete' },
        },
        last_updated: new Date().toISOString(),
      });

    render(<App />);

    // Wait for error state
    await waitFor(() => {
      expect(screen.getByText(/connection refused/i)).toBeInTheDocument();
    });

    // Click retry
    fireEvent.click(screen.getByRole('button', { name: /retry/i }));

    // Should show dashboard after retry
    await waitFor(() => {
      expect(screen.getByText('Reverie Dashboard')).toBeInTheDocument();
    });

    expect(setupApi.getSetupStatus).toHaveBeenCalledTimes(2);
  });

  it('transitions from SetupWizard to Dashboard when setup completes', async () => {
    const mockIncomplete = {
      setup_complete: false,
      phases: {
        greeting: { status: 'pending' as const },
        database: { status: 'pending' as const },
        apikeys: { status: 'pending' as const, validated: false },
        profile: { status: 'pending' as const },
        migration: { status: 'pending' as const },
        verification: { status: 'pending' as const },
      },
      last_updated: new Date().toISOString(),
    };

    const mockComplete = {
      setup_complete: true,
      phases: {
        greeting: { status: 'complete' as const },
        database: { status: 'complete' as const },
        apikeys: { status: 'complete' as const, validated: true },
        profile: { status: 'complete' as const },
        migration: { status: 'complete' as const },
        verification: { status: 'complete' as const },
      },
      last_updated: new Date().toISOString(),
    };

    // App calls once, SetupContext calls once, then after completion both call again
    vi.mocked(setupApi.getSetupStatus)
      .mockResolvedValueOnce(mockIncomplete) // App initial check
      .mockResolvedValueOnce(mockIncomplete) // SetupContext initial check
      .mockResolvedValueOnce(mockComplete)   // App after onComplete
      .mockResolvedValue(mockComplete);      // Any subsequent calls

    render(<App />);

    // Should initially show SetupWizard
    await waitFor(() => {
      expect(screen.getByText('Setup Wizard')).toBeInTheDocument();
    });

    // Verify App and SetupContext both called getSetupStatus
    expect(setupApi.getSetupStatus).toHaveBeenCalledTimes(2);
  });
});
