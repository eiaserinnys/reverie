import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { SystemStats } from './SystemStats';

describe('SystemStats', () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    // Reset fetch before each test
  });

  afterEach(() => {
    vi.restoreAllMocks();
    globalThis.fetch = originalFetch;
  });

  it('shows loading state initially', () => {
    // Mock fetch to return pending promise
    vi.stubGlobal('fetch', vi.fn(() => new Promise(() => {})));

    render(<SystemStats />);
    expect(screen.getByText('Loading...')).toBeInTheDocument();
  });

  it('displays memory stats after fetch', async () => {
    const mockMemoryStatus = {
      entities: 100,
      passages: 50,
      entity_passages: 200,
      passage_adjacency: 30,
      sessions: 10,
      last_indexed: '2026-01-15T12:00:00Z',
      process_memory_mb: 128.5,
    };

    vi.stubGlobal('fetch', vi.fn((url: string) => {
      if (url === '/api/memory/status') {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(mockMemoryStatus),
        } as Response);
      }
      if (url === '/api/index/status') {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ running: false }),
        } as Response);
      }
      return Promise.resolve({
        ok: false,
        json: () => Promise.resolve({ error: 'Not found' }),
      } as Response);
    }));

    render(<SystemStats refreshInterval={60000} />);

    await waitFor(() => {
      expect(screen.getByText('100')).toBeInTheDocument(); // entities
    });

    expect(screen.getByText('50')).toBeInTheDocument(); // passages
    expect(screen.getByText('10')).toBeInTheDocument(); // sessions
    expect(screen.getByText('200')).toBeInTheDocument(); // entity_passages
    expect(screen.getByText('30')).toBeInTheDocument(); // passage_adjacency
  });

  it('shows error message on fetch failure', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new Error('Network error'))));

    render(<SystemStats refreshInterval={60000} />);

    await waitFor(() => {
      expect(screen.getByText(/Error: Network error/)).toBeInTheDocument();
    });
  });

  it('calls fetch on mount', async () => {
    const mockMemoryStatus = {
      entities: 100,
      passages: 50,
      entity_passages: 200,
      passage_adjacency: 30,
      sessions: 10,
      last_indexed: 'never',
      process_memory_mb: 128.5,
    };

    const fetchMock = vi.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockMemoryStatus),
      } as Response)
    );
    vi.stubGlobal('fetch', fetchMock);

    render(<SystemStats refreshInterval={60000} />);

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith('/api/memory/status');
    });
  });

  it('shows indexer idle when not running', async () => {
    const mockMemoryStatus = {
      entities: 100,
      passages: 50,
      entity_passages: 200,
      passage_adjacency: 30,
      sessions: 10,
      last_indexed: 'never',
      process_memory_mb: 128.5,
    };

    vi.stubGlobal('fetch', vi.fn((url: string) => {
      if (url === '/api/memory/status') {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(mockMemoryStatus),
        } as Response);
      }
      if (url === '/api/index/status') {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ running: false }),
        } as Response);
      }
      return Promise.resolve({
        ok: false,
        json: () => Promise.resolve({ error: 'Not found' }),
      } as Response);
    }));

    render(<SystemStats refreshInterval={60000} />);

    await waitFor(() => {
      expect(screen.getByText('Idle')).toBeInTheDocument();
    });
  });
});
