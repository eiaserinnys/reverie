import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { CheckItem } from './CheckItem';

describe('CheckItem', () => {
  it('renders pending state', () => {
    render(<CheckItem name="Python Check" status="pending" />);

    expect(screen.getByText('Python Check')).toBeInTheDocument();
  });

  it('renders checking state with spinner', () => {
    render(<CheckItem name="Python Check" status="checking" />);

    expect(screen.getByText('Python Check')).toBeInTheDocument();
    expect(screen.getByText('Checking...')).toBeInTheDocument();
  });

  it('renders passed state with message', () => {
    render(
      <CheckItem
        name="Python Check"
        status="passed"
        message="Python 3.11.4 installed"
      />
    );

    expect(screen.getByText('Python Check')).toBeInTheDocument();
    expect(screen.getByText('Python 3.11.4 installed')).toBeInTheDocument();
  });

  it('renders failed state with error and retry button', async () => {
    const user = userEvent.setup();
    const onRetry = vi.fn();

    render(
      <CheckItem
        name="Python Check"
        status="failed"
        error="Python not found"
        onRetry={onRetry}
      />
    );

    expect(screen.getByText('Python Check')).toBeInTheDocument();
    expect(screen.getByText('Python not found')).toBeInTheDocument();

    // Click retry
    await user.click(screen.getByText('Retry'));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it('renders description when provided', () => {
    render(
      <CheckItem
        name="Python Check"
        description="Checking Python version"
        status="pending"
      />
    );

    expect(screen.getByText('Python Check')).toBeInTheDocument();
    expect(screen.getByText('Checking Python version')).toBeInTheDocument();
  });

  it('does not show retry button when onRetry is not provided', () => {
    render(
      <CheckItem
        name="Python Check"
        status="failed"
        error="Python not found"
      />
    );

    expect(screen.queryByText('Retry')).not.toBeInTheDocument();
  });

  it('applies custom className', () => {
    const { container } = render(
      <CheckItem
        name="Python Check"
        status="pending"
        className="custom-class"
      />
    );

    expect(container.firstChild).toHaveClass('custom-class');
  });
});
