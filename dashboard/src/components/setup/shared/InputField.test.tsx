import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { InputField, PasswordField } from './InputField';

describe('InputField', () => {
  it('renders with label', () => {
    render(<InputField label="Email" />);

    expect(screen.getByLabelText('Email')).toBeInTheDocument();
  });

  it('renders with description', () => {
    render(<InputField label="Email" description="Your email address" />);

    expect(screen.getByText('Your email address')).toBeInTheDocument();
  });

  it('shows error message', () => {
    render(<InputField label="Email" error="Invalid email" />);

    expect(screen.getByText('Invalid email')).toBeInTheDocument();
  });

  it('shows success message', () => {
    render(<InputField label="Email" success="Email verified" />);

    expect(screen.getByText('Email verified')).toBeInTheDocument();
  });

  it('shows required indicator', () => {
    render(<InputField label="Email" required />);

    expect(screen.getByText('*')).toBeInTheDocument();
  });

  it('accepts input value', async () => {
    const user = userEvent.setup();

    render(<InputField label="Email" />);

    const input = screen.getByLabelText('Email');
    await user.type(input, 'test@example.com');

    expect(input).toHaveValue('test@example.com');
  });

  it('uses custom id when provided', () => {
    render(<InputField label="Email Address" id="custom-email-id" />);

    expect(screen.getByRole('textbox')).toHaveAttribute('id', 'custom-email-id');
  });

  it('generates id from label when not provided', () => {
    render(<InputField label="Email Address" />);

    expect(screen.getByRole('textbox')).toHaveAttribute('id', 'email-address');
  });

  it('applies custom className', () => {
    const { container } = render(<InputField label="Email" className="custom-class" />);

    expect(container.firstChild).toHaveClass('custom-class');
  });
});

describe('PasswordField', () => {
  it('renders as password by default', () => {
    render(<PasswordField label="Password" />);

    expect(screen.getByLabelText('Password')).toHaveAttribute('type', 'password');
  });

  it('toggles password visibility', async () => {
    const user = userEvent.setup();

    render(<PasswordField label="Password" />);

    const input = screen.getByLabelText('Password');
    expect(input).toHaveAttribute('type', 'password');

    // Find and click the toggle button
    const toggleButton = screen.getByRole('button');
    await user.click(toggleButton);

    expect(input).toHaveAttribute('type', 'text');

    // Toggle back
    await user.click(toggleButton);
    expect(input).toHaveAttribute('type', 'password');
  });

  it('hides toggle when showToggle is false', () => {
    render(<PasswordField label="Password" showToggle={false} />);

    expect(screen.queryByRole('button')).not.toBeInTheDocument();
  });
});
