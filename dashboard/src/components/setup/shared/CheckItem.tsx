/**
 * Check Item Component
 *
 * 설정 체크 항목 표시 (pending/checking/passed/failed)
 */

import { cn } from '../../../lib/utils';

export type CheckItemStatus = 'pending' | 'checking' | 'passed' | 'failed';

interface CheckItemProps {
  name: string;
  description?: string;
  status: CheckItemStatus;
  message?: string;
  error?: string;
  onRetry?: () => void;
  className?: string;
}

export function CheckItem({
  name,
  description,
  status,
  message,
  error,
  onRetry,
  className,
}: CheckItemProps) {
  return (
    <div className={cn(
      'flex items-start gap-3 p-3 rounded-lg border',
      status === 'passed' && 'border-green-500/30 bg-green-500/5',
      status === 'failed' && 'border-red-500/30 bg-red-500/5',
      status === 'checking' && 'border-blue-500/30 bg-blue-500/5',
      status === 'pending' && 'border-border bg-muted/30',
      className
    )}>
      <StatusIcon status={status} />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="font-medium text-sm">{name}</span>
          {status === 'checking' && (
            <span className="text-xs text-blue-500">Checking...</span>
          )}
        </div>
        {description && (
          <p className="text-xs text-muted-foreground mt-0.5">{description}</p>
        )}
        {message && status === 'passed' && (
          <p className="text-xs text-green-600 dark:text-green-400 mt-1">{message}</p>
        )}
        {error && status === 'failed' && (
          <div className="mt-2">
            <p className="text-xs text-red-600 dark:text-red-400">{error}</p>
            {onRetry && (
              <button
                onClick={onRetry}
                className="mt-2 text-xs text-blue-500 hover:text-blue-600 font-medium"
              >
                Retry
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function StatusIcon({ status }: { status: CheckItemStatus }) {
  switch (status) {
    case 'passed':
      return (
        <div className="w-5 h-5 rounded-full bg-green-500 flex items-center justify-center shrink-0">
          <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
          </svg>
        </div>
      );
    case 'failed':
      return (
        <div className="w-5 h-5 rounded-full bg-red-500 flex items-center justify-center shrink-0">
          <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </div>
      );
    case 'checking':
      return (
        <div className="w-5 h-5 rounded-full border-2 border-blue-500 border-t-transparent animate-spin shrink-0" />
      );
    case 'pending':
    default:
      return (
        <div className="w-5 h-5 rounded-full border-2 border-muted-foreground/30 shrink-0" />
      );
  }
}
