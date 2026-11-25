import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { SearchBar } from './SearchBar';

describe('SearchBar', () => {
  const defaultProps = {
    onSearch: vi.fn(),
    isLoading: false,
  };

  it('renders search input and button', () => {
    render(<SearchBar {...defaultProps} />);

    expect(screen.getByPlaceholderText(/Search memories/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Search/i })).toBeInTheDocument();
  });

  it('calls onSearch with query when form is submitted', () => {
    const onSearch = vi.fn();
    render(<SearchBar {...defaultProps} onSearch={onSearch} />);

    const input = screen.getByPlaceholderText(/Search memories/i);
    fireEvent.change(input, { target: { value: 'test query' } });

    const form = input.closest('form')!;
    fireEvent.submit(form);

    expect(onSearch).toHaveBeenCalledWith('test query');
  });

  it('does not call onSearch with empty query', () => {
    const onSearch = vi.fn();
    render(<SearchBar {...defaultProps} onSearch={onSearch} />);

    const form = screen.getByPlaceholderText(/Search memories/i).closest('form')!;
    fireEvent.submit(form);

    expect(onSearch).not.toHaveBeenCalled();
  });

  it('does not call onSearch when loading', () => {
    const onSearch = vi.fn();
    render(<SearchBar {...defaultProps} onSearch={onSearch} isLoading={true} />);

    const input = screen.getByPlaceholderText(/Search memories/i);
    fireEvent.change(input, { target: { value: 'test query' } });

    const form = input.closest('form')!;
    fireEvent.submit(form);

    expect(onSearch).not.toHaveBeenCalled();
  });

  it('disables search button when loading', () => {
    render(<SearchBar {...defaultProps} isLoading={true} />);

    expect(screen.getByRole('button', { name: /Searching/i })).toBeDisabled();
  });
});
