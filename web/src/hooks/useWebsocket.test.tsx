import { renderHook, act } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useMetadataStream } from './useWebsocket';

type VoidHandler = ((event?: unknown) => void) | null;

class MockWebSocket {
  static instances: MockWebSocket[] = [];

  onopen: VoidHandler = null;
  onclose: VoidHandler = null;
  onerror: VoidHandler = null;
  onmessage: ((event: { data: string }) => void) | null = null;

  constructor(public url: string) {
    MockWebSocket.instances.push(this);
  }

  close() {
    this.onclose?.();
  }
}

describe('useMetadataStream', () => {
  const originalWebSocket = globalThis.WebSocket;

  beforeEach(() => {
    MockWebSocket.instances = [];
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    globalThis.WebSocket = MockWebSocket as any;
    vi.useFakeTimers();
  });

  afterEach(() => {
    globalThis.WebSocket = originalWebSocket;
    vi.useRealTimers();
  });

  it('does not reconnect after unmount', () => {
    const { unmount } = renderHook(() => useMetadataStream('ws://example.test/stream'));
    expect(MockWebSocket.instances).toHaveLength(1);

    act(() => unmount());
    act(() => {
      vi.runAllTimers();
    });

    expect(MockWebSocket.instances).toHaveLength(1);
  });

  it('reconnects after a close', () => {
    const { result } = renderHook(() => useMetadataStream('ws://example.test/stream'));
    expect(MockWebSocket.instances).toHaveLength(1);

    act(() => {
      MockWebSocket.instances[0]?.onclose?.();
    });
    expect(result.current.status).toBe('closed');

    act(() => {
      vi.advanceTimersByTime(500);
    });

    expect(MockWebSocket.instances).toHaveLength(2);
    expect(result.current.status).toBe('connecting');
  });
});
