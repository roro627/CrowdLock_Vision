import { useEffect, useState } from 'react';
import type { StatsPayload } from '../types';

const POLL_MS = 2000;

/**
 * Poll the backend `/stats` endpoint on an interval.
 *
 * Returns the latest stats payload (or null) and a human-readable error string.
 */
export function useStats(apiBase: string): { stats: StatsPayload | null; error: string | null } {
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
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : 'Failed to fetch stats';
        if (!cancelled) setError(message);
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
