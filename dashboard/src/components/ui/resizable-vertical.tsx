import { useState, useCallback, useRef, useEffect, type ReactNode, type MouseEvent } from 'react';

interface ResizableVerticalProps {
  topContent: ReactNode;
  bottomContent: ReactNode;
  defaultTopHeight: number;  // percentage
  minTopHeight?: number;     // percentage
  maxTopHeight?: number;     // percentage
  className?: string;
}

/**
 * A vertical split layout with draggable divider.
 * Heights are specified as percentages.
 */
export function ResizableVertical({
  topContent,
  bottomContent,
  defaultTopHeight,
  minTopHeight = 20,
  maxTopHeight = 80,
  className = '',
}: ResizableVerticalProps) {
  const [topHeight, setTopHeight] = useState(defaultTopHeight);
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const startYRef = useRef(0);
  const startHeightRef = useRef(0);

  const handleMouseDown = useCallback((e: MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
    startYRef.current = e.clientY;
    startHeightRef.current = topHeight;
  }, [topHeight]);

  useEffect(() => {
    if (!isDragging) return;

    const handleMouseMove = (e: globalThis.MouseEvent) => {
      if (!containerRef.current) return;

      const containerHeight = containerRef.current.getBoundingClientRect().height;
      const deltaY = e.clientY - startYRef.current;
      const deltaPercent = (deltaY / containerHeight) * 100;

      const newHeight = startHeightRef.current + deltaPercent;
      const clampedHeight = Math.max(minTopHeight, Math.min(maxTopHeight, newHeight));
      setTopHeight(clampedHeight);
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
  }, [isDragging, minTopHeight, maxTopHeight]);

  return (
    <div
      ref={containerRef}
      className={`flex flex-col h-full overflow-hidden ${className}`}
    >
      {/* Top section */}
      <div
        className="shrink-0 overflow-hidden"
        style={{ height: `${topHeight}%` }}
      >
        {topContent}
      </div>

      {/* Resize handle */}
      <div
        className={`h-1 cursor-row-resize shrink-0 border-y border-border hover:bg-primary/30 transition-colors ${
          isDragging ? 'bg-primary/50' : 'bg-secondary/50'
        }`}
        onMouseDown={handleMouseDown}
      />

      {/* Bottom section */}
      <div className="flex-1 min-h-0 overflow-hidden">
        {bottomContent}
      </div>

      {/* Overlay to capture mouse events while dragging */}
      {isDragging && (
        <div className="fixed inset-0 cursor-row-resize z-50" />
      )}
    </div>
  );
}
