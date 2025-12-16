function defaultApiBaseFromLocation(): string {
  // Prefer same host as the web UI, with the backend port.
  const hostname = window.location.hostname || 'localhost';
  const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
  return `${protocol}//${hostname}:8000`;
}

function resolveApiBase(): string {
  const envBase = import.meta.env.VITE_API_BASE as string | undefined;
  if (!envBase) return defaultApiBaseFromLocation();

  try {
    const parsed = new URL(envBase);
    const badHost = parsed.hostname === 'localhost' || parsed.hostname === '0.0.0.0';
    const pageHost = window.location.hostname;
    if (badHost && pageHost && pageHost !== 'localhost') {
      parsed.hostname = pageHost;
      return parsed.toString().replace(/\/$/, '');
    }
  } catch {
    // If envBase isn't an absolute URL, fall back to it as-is.
  }

  return envBase.replace(/\/$/, '');
}

const API_BASE = resolveApiBase();

export interface BackendConfig {
  video_source: 'webcam' | 'file' | 'rtsp';
  video_path?: string | null;
  rtsp_url?: string | null;
  model_name: string;
  model_task?: 'detect' | 'pose' | 'auto' | null;
  confidence: number;
  grid_size: string;
  smoothing: number;
  inference_width?: number | null;
  inference_stride?: number;
  target_fps?: number | null;
  output_width?: number | null;
  jpeg_quality?: number | null;
  enable_backend_overlays?: boolean;
}

export interface PresetInfo {
  id: string;
  label: string;
  settings: Record<string, unknown>;
}

export interface PresetListResponse {
  presets: PresetInfo[];
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
  }),
  getPresets: () => request<PresetListResponse>('/config/presets'),
  applyPreset: (presetId: string) => request<BackendConfig>(`/config/presets/${presetId}`, {
    method: 'POST'
  })
};
