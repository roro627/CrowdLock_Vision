import { useEffect, useRef, useState } from 'react';
import type { FramePayload } from '../types';
import clsx from 'clsx';

interface Props {
  frame: FramePayload | null;
  getFrameById?: (frameId: number) => FramePayload | undefined;
  showBoxes: boolean;
  showHead: boolean;
  showBody: boolean;
  showDensity: boolean;
  videoUrl: string;
  connection: string;
  backendError?: string | null;
}

function drawOverlay(canvas: HTMLCanvasElement, payload: FramePayload, options: Props) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);

  // Use frame_size from payload if available, otherwise fallback to 1280x720
  const rawFrameSize = payload.frame_size || [1280, 720];
  const frameW = rawFrameSize[0] > 0 ? rawFrameSize[0] : 1280;
  const frameH = rawFrameSize[1] > 0 ? rawFrameSize[1] : 720;
  // Match CSS object-cover behavior: keep aspect ratio, crop overflow.
  const scale = Math.max(width / frameW, height / frameH);
  const drawW = frameW * scale;
  const drawH = frameH * scale;
  const offsetX = (width - drawW) / 2;
  const offsetY = (height - drawH) / 2;

  payload.persons.forEach((p) => {
    const [x1, y1, x2, y2] = p.bbox;
    if (options.showBoxes) {
      ctx.strokeStyle = '#00aaff';
      ctx.lineWidth = 2;
      ctx.strokeRect(offsetX + x1 * scale, offsetY + y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale);
    }
    if (options.showHead) {
      ctx.fillStyle = '#1dd1a1';
      ctx.beginPath();
      ctx.arc(offsetX + p.head_center[0] * scale, offsetY + p.head_center[1] * scale, 5, 0, Math.PI * 2);
      ctx.fill();
    }
    if (options.showBody) {
      ctx.strokeStyle = '#ff9f43';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(offsetX + p.body_center[0] * scale, offsetY + p.body_center[1] * scale, 7, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.fillStyle = '#e2e8f0';
    ctx.font = '12px Inter';
    ctx.fillText(`ID ${p.id}`, offsetX + x1 * scale + 4, Math.max(12, offsetY + y1 * scale - 6));
  });

  if (options.showDensity && payload.density?.cells) {
    const [gx, gy] = payload.density.grid_size;
    const cellW = (frameW * scale) / gx;
    const cellH = (frameH * scale) / gy;
    const cells = payload.density.cells;
    const maxVal = Math.max(...cells.flat(), 1);
    cells.forEach((row, j) => {
      row.forEach((val, i) => {
        if (val <= 0) return;
        const alpha = Math.min(0.6, 0.1 + (val / maxVal) * 0.5);
        ctx.fillStyle = `rgba(255, 99, 72, ${alpha})`;
        ctx.fillRect(offsetX + i * cellW, offsetY + j * cellH, cellW, cellH);
      });
    });
    if (payload.density.max_cell) {
      const [mi, mj] = payload.density.max_cell;
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2;
      ctx.strokeRect(offsetX + mi * cellW, offsetY + mj * cellH, cellW, cellH);
    }
  }
}

function indexOfBytes(haystack: Uint8Array, needle: Uint8Array, fromIndex = 0): number {
  if (needle.length === 0) return fromIndex;
  outer: for (let i = fromIndex; i <= haystack.length - needle.length; i++) {
    for (let j = 0; j < needle.length; j++) {
      if (haystack[i + j] !== needle[j]) continue outer;
    }
    return i;
  }
  return -1;
}

function concatBytes(a: Uint8Array, b: Uint8Array): Uint8Array {
  if (a.length === 0) return b;
  if (b.length === 0) return a;
  const out = new Uint8Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

function drawCover(ctx: CanvasRenderingContext2D, bitmap: ImageBitmap, canvasW: number, canvasH: number) {
  const frameW = bitmap.width;
  const frameH = bitmap.height;
  const scale = Math.max(canvasW / frameW, canvasH / frameH);
  const drawW = frameW * scale;
  const drawH = frameH * scale;
  const offsetX = (canvasW - drawW) / 2;
  const offsetY = (canvasH - drawH) / 2;
  ctx.clearRect(0, 0, canvasW, canvasH);
  ctx.drawImage(bitmap, offsetX, offsetY, drawW, drawH);
}

export function VideoOverlay({ frame, getFrameById, showBoxes, showHead, showBody, showDensity, videoUrl, connection, backendError }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const videoCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [canvasSize, setCanvasSize] = useState<{ w: number; h: number }>({ w: 1280, h: 720 });
  const [alignedFrame, setAlignedFrame] = useState<FramePayload | null>(null);
  const pendingVideoFrameIdRef = useRef<number | null>(null);

  // Only resize canvases when the container size changes.
  // Resizing a canvas clears it; doing it on every metadata update makes the video look black at low FPS.
  useEffect(() => {
    const overlayCanvas = canvasRef.current;
    if (overlayCanvas) {
      overlayCanvas.width = canvasSize.w;
      overlayCanvas.height = canvasSize.h;
    }

    const videoCanvas = videoCanvasRef.current;
    if (videoCanvas) {
      videoCanvas.width = canvasSize.w;
      videoCanvas.height = canvasSize.h;
    }
  }, [canvasSize.w, canvasSize.h]);

  // Keep canvas resolution in sync with rendered video size to avoid overlay drift.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const update = (w: number, h: number) => {
      const nextW = Math.max(1, Math.round(w));
      const nextH = Math.max(1, Math.round(h));
      setCanvasSize((prev) => (prev.w === nextW && prev.h === nextH ? prev : { w: nextW, h: nextH }));
    };

    const rect = el.getBoundingClientRect();
    update(rect.width, rect.height);

    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const cr = entry.contentRect;
        update(cr.width, cr.height);
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const toDraw = alignedFrame || frame;
    if (!toDraw) return;
    drawOverlay(canvas, toDraw, {
      frame: toDraw,
      showBoxes,
      showHead,
      showBody,
      showDensity,
      videoUrl,
      connection,
      backendError,
    });
  }, [alignedFrame, frame, showBoxes, showHead, showBody, showDensity, videoUrl, connection, backendError]);

  // Draw MJPEG into a canvas so we can sync overlays by frame_id (X-Frame-Id).
  useEffect(() => {
    const vcanvas = videoCanvasRef.current;
    if (!vcanvas) return;
    const vctx = vcanvas.getContext('2d');
    if (!vctx) return;

    const ac = new AbortController();
    const encoder = new TextEncoder();
    const boundary = encoder.encode('--frame');
    const crlf = encoder.encode('\r\n');
    const headersSep = encoder.encode('\r\n\r\n');
    const textDecoder = new TextDecoder();

    let buffer = new Uint8Array(0);
    let decoding = false;
    let pending: { jpg: Uint8Array; frameId: number | null } | null = null;

    const kickDecode = async () => {
      if (decoding) return;
      if (!pending) return;
      decoding = true;
      const item = pending;
      pending = null;
      try {
        const blob = new Blob([item.jpg], { type: 'image/jpeg' });
        const bitmap = await createImageBitmap(blob);
        drawCover(vctx, bitmap, canvasSize.w, canvasSize.h);
        bitmap.close();

        if (item.frameId !== null && typeof getFrameById === 'function') {
          const found = getFrameById(item.frameId);
          if (found) {
            setAlignedFrame(found);
            pendingVideoFrameIdRef.current = null;
          } else {
            // Hold this frame_id until metadata arrives; this adds latency but keeps perfect alignment.
            pendingVideoFrameIdRef.current = item.frameId;
          }
        }
      } catch {
        // ignore
      } finally {
        decoding = false;
        if (pending) void kickDecode();
      }
    };

    const run = async () => {
      try {
        const res = await fetch(videoUrl, { signal: ac.signal, cache: 'no-store' });
        if (!res.ok || !res.body) return;
        const reader = res.body.getReader();

        while (true) {
          const { done, value } = await reader.read();
          if (done || ac.signal.aborted) break;
          if (value) buffer = concatBytes(buffer, value);

          while (true) {
            const b0 = indexOfBytes(buffer, boundary);
            if (b0 === -1) break;

            // Discard any junk before the boundary.
            if (b0 > 0) buffer = buffer.slice(b0);

            const afterBoundary = boundary.length;
            if (buffer.length < afterBoundary + 2) break;
            if (buffer[afterBoundary] !== crlf[0] || buffer[afterBoundary + 1] !== crlf[1]) {
              buffer = buffer.slice(1);
              continue;
            }

            const headersStart = afterBoundary + 2;
            const hEnd = indexOfBytes(buffer, headersSep, headersStart);
            if (hEnd === -1) break;
            const bodyStart = hEnd + headersSep.length;
            const headersRaw = textDecoder.decode(buffer.slice(headersStart, hEnd));
            let frameId: number | null = null;
            let contentLength: number | null = null;
            for (const line of headersRaw.split('\r\n')) {
              const idx = line.indexOf(':');
              if (idx === -1) continue;
              const k = line.slice(0, idx).trim().toLowerCase();
              const v = line.slice(idx + 1).trim();
              if (k === 'x-frame-id') {
                const n = Number.parseInt(v, 10);
                frameId = Number.isFinite(n) ? n : null;
              }
              if (k === 'content-length') {
                const n = Number.parseInt(v, 10);
                contentLength = Number.isFinite(n) && n > 0 ? n : null;
              }
            }

            // Prefer Content-Length (robust) to avoid false boundary matches inside JPEG bytes.
            if (contentLength !== null) {
              const end = bodyStart + contentLength;
              if (buffer.length < end) break;
              const jpg = buffer.slice(bodyStart, end);
              buffer = buffer.slice(end);

              // Optional CRLF after body.
              if (buffer.length >= 2 && buffer[0] === crlf[0] && buffer[1] === crlf[1]) {
                buffer = buffer.slice(2);
              }

              pending = { jpg, frameId };
              void kickDecode();
              continue;
            }

            // Fallback if Content-Length is missing: use boundary search.
            const nb = indexOfBytes(buffer, encoder.encode('\r\n--frame'), bodyStart);
            if (nb === -1) break;
            const jpg = buffer.slice(bodyStart, nb);
            buffer = buffer.slice(nb + 2);

            // If decoding can't keep up, keep only the most recent frame.
            pending = { jpg, frameId };
            void kickDecode();
          }
        }
      } catch {
        // ignore
      }
    };

    void run();
    return () => ac.abort();
  }, [videoUrl, canvasSize.w, canvasSize.h, getFrameById]);

  // When metadata arrives for a pending video frame, promote it to the aligned overlay.
  useEffect(() => {
    const pendingId = pendingVideoFrameIdRef.current;
    if (pendingId === null) return;
    if (typeof getFrameById !== 'function') return;
    const found = getFrameById(pendingId);
    if (found) {
      setAlignedFrame(found);
      pendingVideoFrameIdRef.current = null;
    }
  }, [frame, getFrameById]);

  return (
    <div
      ref={containerRef}
      className="relative w-full max-h-[70vh] aspect-video overflow-hidden rounded-2xl border border-slate-800 shadow-2xl"
    >
      <canvas ref={videoCanvasRef} className={clsx('w-full h-full block', 'bg-slate-950')} />
      <canvas ref={canvasRef} className="absolute inset-0 pointer-events-none" />
      <div className="absolute left-3 bottom-3 text-sm text-slate-200 bg-slate-900/60 px-3 py-1 rounded-full space-x-2 flex items-center">
        <span className="text-xs uppercase tracking-wide text-slate-400">{connection}</span>
        <span>•</span>
        <span>
          {(alignedFrame || frame)
            ? `${(alignedFrame || frame)!.fps.toFixed(1)} infer${typeof (alignedFrame || frame)!.stream_fps === 'number' ? ` / ${(alignedFrame || frame)!.stream_fps.toFixed(1)} stream` : ''} fps / ${(alignedFrame || frame)!.persons.length} people${typeof (alignedFrame || frame)!.latency_ms === 'number' ? ` / ${(alignedFrame || frame)!.latency_ms.toFixed(0)} ms` : ''}`
            : 'Buffering…'}
        </span>
        {backendError && <span className="text-red-400">({backendError})</span>}
      </div>
    </div>
  );
}

export default VideoOverlay;
