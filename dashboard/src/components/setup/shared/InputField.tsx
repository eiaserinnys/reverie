/**
 * Input Field Component
 *
 * 입력 필드 with 레이블, 에러 표시, 검증
 */

import { forwardRef } from 'react';
import { cn } from '../../../lib/utils';

interface InputFieldProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label: string;
  description?: string;
  error?: string;
  success?: string;
}

export const InputField = forwardRef<HTMLInputElement, InputFieldProps>(
  ({ label, description, error, success, className, id, ...props }, ref) => {
    const inputId = id || label.toLowerCase().replace(/\s+/g, '-');

    return (
      <div className={cn('space-y-1.5', className)}>
        <label
          htmlFor={inputId}
          className="text-sm font-medium text-foreground"
        >
          {label}
          {props.required && <span className="text-red-500 ml-0.5">*</span>}
        </label>

        {description && (
          <p className="text-xs text-muted-foreground">{description}</p>
        )}

        <input
          id={inputId}
          ref={ref}
          className={cn(
            'w-full px-3 py-2 text-sm rounded-md border bg-background',
            'placeholder:text-muted-foreground',
            'focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2',
            'disabled:cursor-not-allowed disabled:opacity-50',
            error && 'border-red-500 focus:ring-red-500',
            success && 'border-green-500 focus:ring-green-500',
            !error && !success && 'border-input'
          )}
          {...props}
        />

        {error && (
          <p className="text-xs text-red-500 flex items-center gap-1">
            <ErrorIcon className="w-3 h-3" />
            {error}
          </p>
        )}

        {success && !error && (
          <p className="text-xs text-green-500 flex items-center gap-1">
            <CheckIcon className="w-3 h-3" />
            {success}
          </p>
        )}
      </div>
    );
  }
);

InputField.displayName = 'InputField';

// Password field with show/hide toggle
interface PasswordFieldProps extends Omit<InputFieldProps, 'type'> {
  showToggle?: boolean;
}

export const PasswordField = forwardRef<HTMLInputElement, PasswordFieldProps>(
  ({ showToggle = true, className, ...props }, ref) => {
    const [show, setShow] = React.useState(false);

    return (
      <div className="relative">
        <InputField
          ref={ref}
          type={show ? 'text' : 'password'}
          className={className}
          {...props}
        />
        {showToggle && (
          <button
            type="button"
            onClick={() => setShow(!show)}
            className="absolute right-3 top-[calc(50%+0.5rem)] -translate-y-1/2 text-muted-foreground hover:text-foreground"
            tabIndex={-1}
          >
            {show ? <EyeOffIcon className="w-4 h-4" /> : <EyeIcon className="w-4 h-4" />}
          </button>
        )}
      </div>
    );
  }
);

PasswordField.displayName = 'PasswordField';

// Missing React import
import React from 'react';

// Icons
function ErrorIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
    </svg>
  );
}

function CheckIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
    </svg>
  );
}

function EyeIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
      <path strokeLinecap="round" strokeLinejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
    </svg>
  );
}

function EyeOffIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
    </svg>
  );
}
