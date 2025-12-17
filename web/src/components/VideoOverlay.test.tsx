import { describe, expect, it, vi } from 'vitest';

import type { FramePayload } from '../types';
import { drawOverlay } from './VideoOverlay';

function makePayload(partial?: Partial<FramePayload>): FramePayload {
  return {
    frame_id: 1,
    timestamp: 0,
    fps: 0,
    stream_fps: 0,
    frame_size: [1280, 720],
    persons: [],
    density: {
      grid_size: [8, 6],
      cells: Array.from({ length: 6 }, () => Array.from({ length: 8 }, () => 0)),
      max_cell: [0, 0],
    },
    ...partial,
  };
}

describe('drawOverlay (density band)', () => {
  it('draws a red band at the bottom when showDensity=true', () => {
    const canvas = document.createElement('canvas') as HTMLCanvasElement;
    canvas.width = 100;
    canvas.height = 80;

    const ctx: any = {
      clearRect: vi.fn(),
      fillRect: vi.fn(),
      strokeRect: vi.fn(),
      beginPath: vi.fn(),
      arc: vi.fn(),
      fill: vi.fn(),
      stroke: vi.fn(),
      fillText: vi.fn(),
      lineWidth: 0,
      strokeStyle: '',
      fillStyle: '',
      font: '',
    };

    (canvas as any).getContext = vi.fn().mockReturnValue(ctx);

    drawOverlay(
      canvas,
      makePayload({
        density: {
          grid_size: [8, 6],
          cells: Array.from({ length: 6 }, () => Array.from({ length: 8 }, () => 0)),
          max_cell: [4, 2],
          hotspot_bbox: [100, 200, 300, 400],
        },
      }),
      {
      frame: makePayload(),
      showBoxes: false,
      showHead: false,
      showBody: false,
      showDensity: true,
      videoUrl: 'http://example/stream/video',
      connection: 'connected',
      backendError: null,
      }
    );

    // Hotspot is rendered as a filled + stroked rectangle.
    expect(ctx.fillRect).toHaveBeenCalled();
    expect(ctx.strokeRect).toHaveBeenCalled();
  });

  it('does not draw the band when showDensity=false', () => {
    const canvas = document.createElement('canvas') as HTMLCanvasElement;
    canvas.width = 100;
    canvas.height = 80;

    const ctx: any = {
      clearRect: vi.fn(),
      fillRect: vi.fn(),
      strokeRect: vi.fn(),
    };

    (canvas as any).getContext = vi.fn().mockReturnValue(ctx);

    drawOverlay(canvas, makePayload(), {
      frame: makePayload(),
      showBoxes: false,
      showHead: false,
      showBody: false,
      showDensity: false,
      videoUrl: 'http://example/stream/video',
      connection: 'connected',
      backendError: null,
    });

    expect(ctx.fillRect).not.toHaveBeenCalled();
    expect(ctx.strokeRect).not.toHaveBeenCalled();
  });
});
