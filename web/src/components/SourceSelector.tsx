import React, { useEffect, useState } from 'react';
import { api, BackendConfig } from '../api/client';

interface Props {
  onStatus: (status: 'idle' | 'saving' | 'error' | 'saved') => void;
}

const defaultCfg: BackendConfig = {
  video_source: 'webcam',
  video_path: '',
  rtsp_url: '',
  model_name: 'yolov8n-pose.pt',
  device: null,
  confidence: 0.35,
  grid_size: '10x10',
  smoothing: 0.2,
  inference_width: 640,
  jpeg_quality: 70,
  enable_backend_overlays: false
};

export function SourceSelector({ onStatus }: Props) {
  const [cfg, setCfg] = useState<BackendConfig>(defaultCfg);

  useEffect(() => {
    api
      .getConfig()
      .then((data) => setCfg(data))
      .catch(() => onStatus('error'));
  }, [onStatus]);

  const handleChange = (key: keyof BackendConfig, value: string | number | boolean | null) => {
    setCfg((c) => ({ ...c, [key]: value } as BackendConfig));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    onStatus('saving');
    try {
      await api.updateConfig(cfg);
      onStatus('saved');
    } catch (err) {
      console.error(err);
      onStatus('error');
    }
  };

  return (
    <form className="card p-4 space-y-3" onSubmit={handleSubmit}>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-wide text-slate-400">Source</p>
          <p className="text-lg font-semibold">Video Input</p>
        </div>
        <button type="submit" className="button-primary">Apply</button>
      </div>
      <div className="grid grid-cols-1 gap-3">
        <label className="flex items-center gap-2">
          <span className="w-28 text-sm text-slate-300">Type</span>
          <select
            className="card px-3 py-2 w-full bg-slate-900/80 border border-slate-700"
            value={cfg.video_source}
            onChange={(e) => handleChange('video_source', e.target.value)}
          >
            <option value="webcam">Webcam</option>
            <option value="file">File</option>
            <option value="rtsp">RTSP</option>
          </select>
        </label>
        {cfg.video_source === 'file' && (
          <label className="flex items-center gap-2">
            <span className="w-28 text-sm text-slate-300">File Path</span>
            <input
              className="card px-3 py-2 w-full bg-slate-900/80 border border-slate-700"
              value={cfg.video_path || ''}
              onChange={(e) => handleChange('video_path', e.target.value)}
              placeholder="testdata/videos/clip.mp4"
            />
          </label>
        )}
        {cfg.video_source === 'rtsp' && (
          <label className="flex items-center gap-2">
            <span className="w-28 text-sm text-slate-300">RTSP URL</span>
            <input
              className="card px-3 py-2 w-full bg-slate-900/80 border border-slate-700"
              value={cfg.rtsp_url || ''}
              onChange={(e) => handleChange('rtsp_url', e.target.value)}
              placeholder="rtsp://user:pass@host:port/stream"
            />
          </label>
        )}
        <label className="flex items-center gap-2">
          <span className="w-28 text-sm text-slate-300">Model</span>
          <input
            className="card px-3 py-2 w-full bg-slate-900/80 border border-slate-700"
            value={cfg.model_name}
            onChange={(e) => handleChange('model_name', e.target.value)}
          />
        </label>
        <label className="flex items-center gap-2">
          <span className="w-28 text-sm text-slate-300">Device</span>
          <input
            className="card px-3 py-2 w-full bg-slate-900/80 border border-slate-700"
            value={cfg.device || ''}
            onChange={(e) => handleChange('device', e.target.value || null)}
            placeholder="cpu | cuda:0"
          />
        </label>
        <label className="flex items-center gap-2">
          <span className="w-28 text-sm text-slate-300">Confidence</span>
          <input
            type="number"
            min="0.01"
            max="1"
            step="0.01"
            className="card px-3 py-2 w-full bg-slate-900/80 border border-slate-700"
            value={cfg.confidence}
            onChange={(e) => handleChange('confidence', parseFloat(e.target.value))}
          />
        </label>
        <label className="flex items-center gap-2">
          <span className="w-28 text-sm text-slate-300">Grid</span>
          <input
            className="card px-3 py-2 w-full bg-slate-900/80 border border-slate-700"
            value={cfg.grid_size}
            onChange={(e) => handleChange('grid_size', e.target.value)}
            placeholder="10x10"
          />
        </label>
        <label className="flex items-center gap-2">
          <span className="w-28 text-sm text-slate-300">Smoothing</span>
          <input
            type="number"
            min="0"
            max="1"
            step="0.05"
            className="card px-3 py-2 w-full bg-slate-900/80 border border-slate-700"
            value={cfg.smoothing}
            onChange={(e) => handleChange('smoothing', parseFloat(e.target.value))}
          />
        </label>
        <label className="flex items-center gap-2">
          <span className="w-28 text-sm text-slate-300">Infer width</span>
          <input
            type="number"
            min="64"
            step="16"
            className="card px-3 py-2 w-full bg-slate-900/80 border border-slate-700"
            value={cfg.inference_width ?? 640}
            onChange={(e) => handleChange('inference_width', parseInt(e.target.value, 10))}
          />
        </label>
        <label className="flex items-center gap-2">
          <span className="w-28 text-sm text-slate-300">JPEG %</span>
          <input
            type="number"
            min="10"
            max="100"
            step="5"
            className="card px-3 py-2 w-full bg-slate-900/80 border border-slate-700"
            value={cfg.jpeg_quality ?? 70}
            onChange={(e) => handleChange('jpeg_quality', parseInt(e.target.value, 10))}
          />
        </label>
        <label className="flex items-center gap-2">
          <span className="w-28 text-sm text-slate-300">Server overlay</span>
          <input
            type="checkbox"
            className="h-4 w-4"
            checked={Boolean(cfg.enable_backend_overlays)}
            onChange={(e) => handleChange('enable_backend_overlays', e.target.checked)}
          />
        </label>
      </div>
    </form>
  );
}

export default SourceSelector;
