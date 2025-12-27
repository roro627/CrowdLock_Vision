import React from 'react';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import { SourceSelector } from './SourceSelector';

vi.mock('../api/client', () => {
  return {
    api: {
      base: 'http://localhost:8000',
      getConfig: vi.fn(async () => ({
        video_source: 'webcam',
        video_path: '',
        rtsp_url: '',
        model_name: 'yolo11l.pt',
        confidence: 0.35,
        grid_size: '10x10',
        smoothing: 0.2,
        inference_width: 640,
        jpeg_quality: 70,
        enable_backend_overlays: false,
      })),
      updateConfig: vi.fn(async (cfg: unknown) => cfg),
      getPresets: vi.fn(async () => ({ presets: [] })),
      applyPreset: vi.fn(async () => ({})),
      listVideos: vi.fn(async () => [
        {
          name: 'demo.mp4',
          path: 'testdata/videos/demo.mp4',
          thumbnail_url: '/media/videos/demo.mp4/thumbnail.jpg',
        },
      ]),
    },
  };
});

describe('SourceSelector file picker', () => {
  it('shows a demo-video dropdown and keeps a custom path input editable', async () => {
    const user = userEvent.setup();
    render(<SourceSelector onStatus={() => {}} />);

    const typeSelect = await screen.findByLabelText('Type');
    await user.selectOptions(typeSelect, 'file');

    const openBtn = await screen.findByRole('button', { name: /select demo video/i });
    await user.click(openBtn);

    const demoOption = await screen.findByRole('button', { name: /demo\.mp4/i });
    expect(within(demoOption).getByText('demo.mp4')).toBeInTheDocument();

    await user.click(demoOption);

    const customInput = screen.getByPlaceholderText('testdata/videos/clip.mp4');
    expect(customInput).toHaveValue('testdata/videos/demo.mp4');

    await user.clear(customInput);
    await user.type(customInput, 'testdata/videos/custom.mp4');
    expect(customInput).toHaveValue('testdata/videos/custom.mp4');
  });
});
