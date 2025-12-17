import { useCallback, useEffect, useRef, useState } from 'react';
import type { FramePayload } from '../types';

const WS_MIN_BACKOFF = 500;
const WS_MAX_BACKOFF = 8000;

/**
 * Normalize an HTTP(S) URL into a WS(S) URL.
 */
function normalizeUrl(raw: string): string {
  if (raw.startsWith('ws')) return raw;
  if (raw.startsWith('https://')) return raw.replace('https://', 'wss://');
  if (raw.startsWith('http://')) return raw.replace('http://', 'ws://');
  return raw;
}

/**
 * Subscribe to the backend metadata WebSocket and keep a bounded in-memory
 * lookup by `frame_id` for aligning overlays.
 */
export function useMetadataStream(url: string) {
  const [latest, setLatest] = useState<FramePayload | null>(null);
  const [status, setStatus] = useState<'connecting' | 'open' | 'closed'>('connecting');
  const framesByIdRef = useRef<Map<number, FramePayload>>(new Map());
  const retryRef = useRef(WS_MIN_BACKOFF);
  const wsRef = useRef<WebSocket | null>(null);
  const timeoutRef = useRef<number | null>(null);
  const pingIntervalRef = useRef<number | null>(null);

  // Estimate server_time ~= client_time + clockOffsetSec using an NTP-like exchange.
  const clockOffsetSecRef = useRef(0);
  const latencyMsEmaRef = useRef<number | null>(null);

  useEffect(() => {
    const wsUrl = normalizeUrl(url);
    let disposed = false;

    const scheduleReconnect = () => {
      if (disposed) return;
      setStatus('closed');
      if (timeoutRef.current) window.clearTimeout(timeoutRef.current);
      const delay = Math.min(retryRef.current, WS_MAX_BACKOFF);
      retryRef.current = Math.min(WS_MAX_BACKOFF, retryRef.current * 2);
      timeoutRef.current = window.setTimeout(connect, delay);
    };

    const connect = () => {
      if (disposed) return;
      setStatus('connecting');
      wsRef.current = new WebSocket(wsUrl);
      wsRef.current.onopen = () => {
        if (disposed) return;
        setStatus('open');
        retryRef.current = WS_MIN_BACKOFF;

        const sendPing = () => {
          const ws = wsRef.current;
          if (!ws || ws.readyState !== WebSocket.OPEN) return;
          try {
            ws.send(JSON.stringify({ type: 'ping', t: Date.now() / 1000 }));
          } catch {
            // Intentionally ignore send errors during teardown/reconnect.
          }
        };

        sendPing();
        if (pingIntervalRef.current) window.clearInterval(pingIntervalRef.current);
        pingIntervalRef.current = window.setInterval(sendPing, 2000);
      };
      wsRef.current.onclose = () => scheduleReconnect();
      wsRef.current.onerror = () => scheduleReconnect();
      wsRef.current.onmessage = (evt) => {
        if (disposed) return;
        try {
          const msg = JSON.parse(evt.data) as unknown;
          if (typeof msg !== 'object' || msg === null) return;

          const maybeAny = msg as Record<string, unknown>;
          if (maybeAny.type === 'pong') {
            const t1 = typeof maybeAny.t === 'number' ? maybeAny.t : null;
            const serverTime =
              typeof maybeAny.server_time === 'number' ? maybeAny.server_time : null;
            if (t1 !== null && serverTime !== null) {
              const t4 = Date.now() / 1000;
              const rtt = Math.max(0, t4 - t1);
              const offset = serverTime - (t1 + rtt / 2);
              // Slow EMA to keep stable even with jitter.
              clockOffsetSecRef.current = clockOffsetSecRef.current * 0.9 + offset * 0.1;
            }
            return;
          }

          const data = msg as FramePayload;
          // Compute latency in *server clock* seconds: (client_now + offset) - server_timestamp
          const nowServerSec = Date.now() / 1000 + clockOffsetSecRef.current;
          const rawMs = Math.max(0, (nowServerSec - data.timestamp) * 1000);
          const prev = latencyMsEmaRef.current;
          const ema = prev === null ? rawMs : prev * 0.85 + rawMs * 0.15;
          latencyMsEmaRef.current = ema;
          const payload: FramePayload = { ...data, latency_ms: ema };
          framesByIdRef.current.set(payload.frame_id, payload);
          // Keep memory bounded: store only the most recent N frames.
          while (framesByIdRef.current.size > 200) {
            const firstKey = framesByIdRef.current.keys().next().value as number | undefined;
            if (firstKey === undefined) break;
            framesByIdRef.current.delete(firstKey);
          }
          setLatest(payload);
        } catch (e) {
          // Ignore malformed messages to keep the stream resilient.
          void e;
        }
      };
    };

    connect();

    return () => {
      disposed = true;
      if (timeoutRef.current) window.clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
      if (pingIntervalRef.current) window.clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
      if (wsRef.current) {
        wsRef.current.onopen = null;
        wsRef.current.onclose = null;
        wsRef.current.onerror = null;
        wsRef.current.onmessage = null;
      }
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [url]);

  const getById = useCallback((frameId: number) => framesByIdRef.current.get(frameId), []);

  return { latest, status, getById };
}
