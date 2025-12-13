import { useEffect, useState } from 'react';
import type { StatsPayload } from '../types';

const POLL_MS = 2000;

export function useStats(apiBase: string) {
  const [stats, setStats] = useState<StatsPayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;

    const fetchStats = async () => {
      try {
        const res = await fetch(`${apiBase}/stats`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = (await res.json()) as StatsPayload;
        if (!cancelled) {
          setStats(data);
          setError(data.error ?? null);
        }
      } catch (e: any) {
        if (!cancelled) setError(e.message || 'Failed to fetch stats');
      } finally {
        if (!cancelled) timer = window.setTimeout(fetchStats, POLL_MS);
      }
    };

    fetchStats();

    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [apiBase]);

  return { stats, error };
}

