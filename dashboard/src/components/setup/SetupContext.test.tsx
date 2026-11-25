import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { SetupProvider, SETUP_PHASES } from './SetupContext';
import { useSetup } from './useSetup';
import * as setupApi from '../../lib/setup-api';

// Mock the setup API
vi.mock('../../lib/setup-api', () => ({
  getSetupStatus: vi.fn(),
}));

// Test component that uses the context
function TestConsumer() {
  const {
    currentPhase,
    loading,
    error,
    status,
    canGoNext,
    canGoPrev,
    nextPhase,
    prevPhase,
    goToPhase,
  } = useSetup();

  return (
    <div>
      <div data-testid="loading">{loading ? 'loading' : 'loaded'}</div>
      <div data-testid="error">{error || 'no-error'}</div>
      <div data-testid="current-phase">{currentPhase}</div>
      <div data-testid="can-go-next">{canGoNext ? 'yes' : 'no'}</div>
      <div data-testid="can-go-prev">{canGoPrev ? 'yes' : 'no'}</div>
      <div data-testid="setup-complete">{status?.setup_complete ? 'complete' : 'incomplete'}</div>
      <button onClick={nextPhase}>Next</button>
      <button onClick={prevPhase}>Prev</button>
      <button onClick={() => goToPhase(3)}>Go to 3</button>
    </div>
  );
}

describe('SetupContext', () => {
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
    last_updated: new Date().toISOString(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
    (setupApi.getSetupStatus as ReturnType<typeof vi.fn>).mockResolvedValue(mockStatus);
  });

  it('loads setup status on mount', async () => {
    render(
      <SetupProvider>
        <TestConsumer />
      </SetupProvider>
    );

    // Initially loading
    expect(screen.getByTestId('loading')).toHaveTextContent('loading');

    // Wait for load
    await waitFor(() => {
      expect(screen.getByTestId('loading')).toHaveTextContent('loaded');
    });

    expect(setupApi.getSetupStatus).toHaveBeenCalledTimes(1);
  });

  it('handles API error gracefully', async () => {
    (setupApi.getSetupStatus as ReturnType<typeof vi.fn>).mockRejectedValue(new Error('Network error'));

    render(
      <SetupProvider>
        <TestConsumer />
      </SetupProvider>
    );

    await waitFor(() => {
      expect(screen.getByTestId('error')).toHaveTextContent('Network error');
    });
  });

  it('starts at phase 0', async () => {
    render(
      <SetupProvider>
        <TestConsumer />
      </SetupProvider>
    );

    await waitFor(() => {
      expect(screen.getByTestId('loading')).toHaveTextContent('loaded');
    });

    expect(screen.getByTestId('current-phase')).toHaveTextContent('0');
  });

  it('navigates to next phase', async () => {
    const user = userEvent.setup();

    render(
      <SetupProvider>
        <TestConsumer />
      </SetupProvider>
    );

    await waitFor(() => {
      expect(screen.getByTestId('loading')).toHaveTextContent('loaded');
    });

    await user.click(screen.getByText('Next'));
    expect(screen.getByTestId('current-phase')).toHaveTextContent('1');
  });

  it('navigates to previous phase', async () => {
    const user = userEvent.setup();

    render(
      <SetupProvider>
        <TestConsumer />
      </SetupProvider>
    );

    await waitFor(() => {
      expect(screen.getByTestId('loading')).toHaveTextContent('loaded');
    });

    // Go to phase 1 first
    await user.click(screen.getByText('Next'));
    expect(screen.getByTestId('current-phase')).toHaveTextContent('1');

    // Go back to phase 0
    await user.click(screen.getByText('Prev'));
    expect(screen.getByTestId('current-phase')).toHaveTextContent('0');
  });

  it('navigates to specific phase', async () => {
    const user = userEvent.setup();

    render(
      <SetupProvider>
        <TestConsumer />
      </SetupProvider>
    );

    await waitFor(() => {
      expect(screen.getByTestId('loading')).toHaveTextContent('loaded');
    });

    await user.click(screen.getByText('Go to 3'));
    expect(screen.getByTestId('current-phase')).toHaveTextContent('3');
  });

  it('reports canGoNext and canGoPrev correctly', async () => {
    const user = userEvent.setup();

    render(
      <SetupProvider>
        <TestConsumer />
      </SetupProvider>
    );

    await waitFor(() => {
      expect(screen.getByTestId('loading')).toHaveTextContent('loaded');
    });

    // At phase 0: can go next, cannot go prev
    expect(screen.getByTestId('can-go-next')).toHaveTextContent('yes');
    expect(screen.getByTestId('can-go-prev')).toHaveTextContent('no');

    // Go to last phase (6 phases = index 5)
    await user.click(screen.getByText('Go to 3'));
    await user.click(screen.getByText('Next'));
    await user.click(screen.getByText('Next'));

    // At last phase: cannot go next, can go prev
    expect(screen.getByTestId('current-phase')).toHaveTextContent(String(SETUP_PHASES.length - 1));
    expect(screen.getByTestId('can-go-next')).toHaveTextContent('no');
    expect(screen.getByTestId('can-go-prev')).toHaveTextContent('yes');
  });

  it('throws error when useSetup is used outside provider', () => {
    // Suppress console.error for this test
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    expect(() => {
      render(<TestConsumer />);
    }).toThrow('useSetup must be used within a SetupProvider');

    consoleSpy.mockRestore();
  });
});

describe('SETUP_PHASES', () => {
  it('has 6 phases', () => {
    expect(SETUP_PHASES).toHaveLength(6);
  });

  it('has correct phase ids', () => {
    const expectedIds = ['greeting', 'database', 'apikeys', 'profile', 'migration', 'verification'];
    expect(SETUP_PHASES.map(p => p.id)).toEqual(expectedIds);
  });
});
