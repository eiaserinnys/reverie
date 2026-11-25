// API utility functions

export async function fetchApi<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options);

  if (response.status === 401) {
    // Handle authentication error
    const pathname = window.location.pathname;
    const basePath = pathname.replace(/\/?$/, '').replace(/\/[^/]*\.[^/]*$/, '') || '';
    window.location.href = `${window.location.origin}${basePath}/auth/google`;
    throw new Error('Authentication required');
  }

  if (!response.ok) {
    const data = await response.json().catch(() => ({ error: 'Unknown error' }));
    throw new Error(data.error || `HTTP error ${response.status}`);
  }

  return response.json();
}

export function escapeHtml(text: string): string {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

export function formatDate(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diff = now.getTime() - date.getTime();

  if (diff < 60000) return 'just now';
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
  return date.toLocaleDateString('ko-KR');
}

export function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

export function resolvePath(baseDir: string, relativePath: string): string {
  // Handle absolute paths (starting with /)
  if (relativePath.startsWith('/')) {
    return relativePath.substring(1);
  }

  // Combine base directory with relative path
  const parts = (baseDir + relativePath).split('/').filter(p => p !== '');

  // Resolve .. and .
  const resolved: string[] = [];
  for (const part of parts) {
    if (part === '..') {
      resolved.pop();
    } else if (part !== '.') {
      resolved.push(part);
    }
  }

  return resolved.join('/');
}
