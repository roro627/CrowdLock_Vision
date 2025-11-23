import { useEffect, useState } from 'react';
import type { FramePayload } from '../types';

export function useMetadataStream(url: string) {
  const [latest, setLatest] = useState<FramePayload | null>(null);
  const [status, setStatus] = useState<'connecting' | 'open' | 'closed'>('connecting');

  useEffect(() => {
    const ws = new WebSocket(url);
    ws.onopen = () => setStatus('open');
    ws.onclose = () => setStatus('closed');
    ws.onerror = () => setStatus('closed');
    ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data) as FramePayload;
        setLatest(data);
      } catch (e) {
        console.error('Failed to parse message', e);
      }
    };
    return () => ws.close();
  }, [url]);

  return { latest, status };
}
