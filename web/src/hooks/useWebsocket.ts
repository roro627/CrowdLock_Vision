import { useEffect, useRef, useState } from 'react';
import type { FramePayload } from '../types';

const WS_MIN_BACKOFF = 500;
const WS_MAX_BACKOFF = 8000;

function normalizeUrl(raw: string): string {
  if (raw.startsWith('ws')) return raw;
  if (raw.startsWith('https://')) return raw.replace('https://', 'wss://');
  if (raw.startsWith('http://')) return raw.replace('http://', 'ws://');
  return raw;
}

export function useMetadataStream(url: string) {
  const [latest, setLatest] = useState<FramePayload | null>(null);
  const [status, setStatus] = useState<'connecting' | 'open' | 'closed'>('connecting');
  const retryRef = useRef(WS_MIN_BACKOFF);
  const wsRef = useRef<WebSocket | null>(null);
  const timeoutRef = useRef<number | null>(null);

  useEffect(() => {
    const wsUrl = normalizeUrl(url);

    const scheduleReconnect = () => {
      setStatus('closed');
      if (timeoutRef.current) window.clearTimeout(timeoutRef.current);
      const delay = Math.min(retryRef.current, WS_MAX_BACKOFF);
      retryRef.current = Math.min(WS_MAX_BACKOFF, retryRef.current * 2);
      timeoutRef.current = window.setTimeout(connect, delay);
    };

    const connect = () => {
      setStatus('connecting');
      wsRef.current = new WebSocket(wsUrl);
      wsRef.current.onopen = () => {
        setStatus('open');
        retryRef.current = WS_MIN_BACKOFF;
      };
      wsRef.current.onclose = scheduleReconnect;
      wsRef.current.onerror = scheduleReconnect;
      wsRef.current.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data) as FramePayload;
          setLatest(data);
        } catch (e) {
          console.error('Failed to parse message', e);
        }
      };
    };

    connect();

    return () => {
      if (timeoutRef.current) window.clearTimeout(timeoutRef.current);
      wsRef.current?.close();
    };
  }, [url]);

  return { latest, status };
}
