import { useState, useCallback, useRef, useEffect, type ReactNode, type MouseEvent } from 'react';

interface ResizablePanelProps {
  children: ReactNode;
  defaultWidth: number;  // percentage of container
  minWidth?: number;     // percentage
  maxWidth?: number;     // percentage
  side: 'left' | 'right';
  className?: string;
}

/**
 * A panel that can be resized by dragging its edge.
 * Width is specified as a percentage of the parent container.
 */
export function ResizablePanel({
  children,
  defaultWidth,
  minWidth = 10,
  maxWidth = 40,
  side,
  className = '',
}: ResizablePanelProps) {
  const [width, setWidth] = useState(defaultWidth);
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const startXRef = useRef(0);
  const startWidthRef = useRef(0);

  const handleMouseDown = useCallback((e: MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
    startXRef.current = e.clientX;
    startWidthRef.current = width;
  }, [width]);

  useEffect(() => {
    if (!isDragging) return;

    const handleMouseMove = (e: globalThis.MouseEvent) => {
      if (!containerRef.current) return;

      const parent = containerRef.current.parentElement;
      if (!parent) return;

      const parentWidth = parent.getBoundingClientRect().width;
      const deltaX = e.clientX - startXRef.current;
      const deltaPercent = (deltaX / parentWidth) * 100;

      // For left panel, dragging right increases width
      // For right panel, dragging left increases width
      const newWidth = side === 'left'
        ? startWidthRef.current + deltaPercent
        : startWidthRef.current - deltaPercent;

      const clampedWidth = Math.max(minWidth, Math.min(maxWidth, newWidth));
      setWidth(clampedWidth);
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, minWidth, maxWidth, side]);

  // Handle edge: left panel has handle on right, right panel has handle on left
  const handlePosition = side === 'left' ? 'right-0' : 'left-0';
  const borderClass = side === 'left' ? 'border-r' : 'border-l';

  return (
    <div
      ref={containerRef}
      className={`relative shrink-0 min-w-0 ${borderClass} border-border ${className}`}
      style={{ width: `${width}%` }}
    >
      {children}

      {/* Resize handle */}
      <div
        className={`absolute ${handlePosition} top-0 bottom-0 w-1 cursor-col-resize z-10 hover:bg-primary/30 transition-colors ${
          isDragging ? 'bg-primary/50' : ''
        }`}
        onMouseDown={handleMouseDown}
      />

      {/* Overlay to capture mouse events while dragging */}
      {isDragging && (
        <div className="fixed inset-0 cursor-col-resize z-50" />
      )}
    </div>
  );
}
