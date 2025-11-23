const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

export interface BackendConfig {
  video_source: 'webcam' | 'file' | 'rtsp';
  video_path?: string | null;
  rtsp_url?: string | null;
  model_name: string;
  device?: string | null;
  confidence: number;
  grid_size: string;
  smoothing: number;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init
  });
  if (!res.ok) {
    throw new Error(`Request failed: ${res.status}`);
  }
  return res.json();
}

export const api = {
  base: API_BASE,
  getConfig: () => request<BackendConfig>('/config'),
  updateConfig: (cfg: BackendConfig) => request<BackendConfig>('/config', {
    method: 'POST',
    body: JSON.stringify(cfg)
  })
};
