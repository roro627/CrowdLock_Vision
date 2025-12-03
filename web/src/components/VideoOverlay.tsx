import { useEffect, useRef } from 'react';
import type { FramePayload } from '../types';
import clsx from 'clsx';

interface Props {
  frame: FramePayload | null;
  showBoxes: boolean;
  showHead: boolean;
  showBody: boolean;
  showDensity: boolean;
  videoUrl: string;
}

function drawOverlay(canvas: HTMLCanvasElement, payload: FramePayload, options: Props) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);

  // Use frame_size from payload if available, otherwise fallback to 1280x720
  const [frameW, frameH] = payload.frame_size || [1280, 720];
  const scaleX = width / frameW;
  const scaleY = height / frameH;

  payload.persons.forEach((p) => {
    const [x1, y1, x2, y2] = p.bbox;
    if (options.showBoxes) {
      ctx.strokeStyle = '#00aaff';
      ctx.lineWidth = 2;
      ctx.strokeRect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);
    }
    if (options.showHead) {
      ctx.fillStyle = '#1dd1a1';
      ctx.beginPath();
      ctx.arc(p.head_center[0] * scaleX, p.head_center[1] * scaleY, 5, 0, Math.PI * 2);
      ctx.fill();
    }
    if (options.showBody) {
      ctx.strokeStyle = '#ff9f43';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(p.body_center[0] * scaleX, p.body_center[1] * scaleY, 7, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.fillStyle = '#e2e8f0';
    ctx.font = '12px Inter';
    ctx.fillText(`ID ${p.id}`, x1 * scaleX + 4, Math.max(12, y1 * scaleY - 6));
  });

  if (options.showDensity && payload.density?.cells) {
    const [gx, gy] = payload.density.grid_size;
    const cellW = width / gx;
    const cellH = height / gy;
    const cells = payload.density.cells;
    const maxVal = Math.max(...cells.flat(), 1);
    cells.forEach((row, j) => {
      row.forEach((val, i) => {
        if (val <= 0) return;
        const alpha = Math.min(0.6, 0.1 + (val / maxVal) * 0.5);
        ctx.fillStyle = `rgba(255, 99, 72, ${alpha})`;
        ctx.fillRect(i * cellW, j * cellH, cellW, cellH);
      });
    });
    if (payload.density.max_cell) {
      const [mi, mj] = payload.density.max_cell;
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2;
      ctx.strokeRect(mi * cellW, mj * cellH, cellW, cellH);
    }
  }
}

export function VideoOverlay({ frame, showBoxes, showHead, showBody, showDensity, videoUrl }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (frame && canvasRef.current) {
      drawOverlay(canvasRef.current, frame, {
        frame,
        showBoxes,
        showHead,
        showBody,
        showDensity,
        videoUrl
      });
    }
  }, [frame, showBoxes, showHead, showBody, showDensity, videoUrl]);

  return (
    <div className="relative w-full max-h-[70vh] aspect-video overflow-hidden rounded-2xl border border-slate-800 shadow-2xl">
      <img src={videoUrl} className={clsx('w-full h-full object-cover block', 'bg-slate-950')} alt="Video stream" />
      <canvas ref={canvasRef} className="absolute inset-0 pointer-events-none" width={1280} height={720} />
      <div className="absolute left-3 bottom-3 text-sm text-slate-200 bg-slate-900/60 px-3 py-1 rounded-full">
        {frame ? `${frame.fps.toFixed(1)} fps • ${frame.persons.length} people` : 'Waiting for stream'}
      </div>
    </div>
  );
}

export default VideoOverlay;
