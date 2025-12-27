import React, { useEffect, useMemo, useRef, useState } from 'react';
import { api, VideoFileInfo } from '../api/client';

interface Props {
  value: string;
  onChange: (nextPath: string) => void;
}

function basename(path: string): string {
  const parts = path.split('/');
  return parts[parts.length - 1] || path;
}

/**
 * Dropdown for selecting a demo video shipped with the backend.
 *
 * Uses a custom popover so we can show thumbnails.
 */
export function VideoFileDropdown({ value, onChange }: Props) {
  const [open, setOpen] = useState(false);
  const [videos, setVideos] = useState<VideoFileInfo[]>([]);
  const [loaded, setLoaded] = useState(false);
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (loaded) return;
    api
      .listVideos()
      .then((v) => {
        setVideos(v);
        setLoaded(true);
      })
      .catch(() => {
        setVideos([]);
        setLoaded(true);
      });
  }, [loaded]);

  useEffect(() => {
    const onDocMouseDown = (e: MouseEvent) => {
      const el = containerRef.current;
      if (!el) return;
      if (!el.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onDocMouseDown);
    return () => document.removeEventListener('mousedown', onDocMouseDown);
  }, []);

  const selected = useMemo(() => {
    return videos.find((v) => v.path === value) || null;
  }, [videos, value]);

  return (
    <div className="relative w-full" ref={containerRef}>
      <button
        type="button"
        className="card px-3 py-2 w-full bg-slate-900/80 border border-slate-700 flex items-center justify-between"
        aria-label="Select demo video"
        onClick={() => setOpen((o) => !o)}
      >
        <span className="text-sm text-slate-200 truncate">
          {selected ? selected.name : value ? basename(value) : loaded ? 'Select a video…' : 'Loading…'}
        </span>
        <span className="text-xs text-slate-400">▾</span>
      </button>

      {open && (
        <div className="absolute z-20 mt-2 w-full card bg-slate-950 border border-slate-700 p-2 max-h-64 overflow-auto">
          {videos.length === 0 ? (
            <div className="px-2 py-2 text-sm text-slate-400">No demo videos found</div>
          ) : (
            videos.map((v) => (
              <button
                type="button"
                key={v.name}
                className="w-full flex items-center gap-3 px-2 py-2 rounded hover:bg-slate-900/70 text-left"
                onClick={() => {
                  onChange(v.path);
                  setOpen(false);
                }}
              >
                <img
                  className="w-16 h-10 object-cover rounded border border-slate-700"
                  src={`${api.base}${v.thumbnail_url}`}
                  alt={v.name}
                  loading="lazy"
                />
                <div className="flex-1 min-w-0">
                  <div className="text-sm text-slate-200 truncate">{v.name}</div>
                  <div className="text-xs text-slate-500 truncate">{v.path}</div>
                </div>
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
}
