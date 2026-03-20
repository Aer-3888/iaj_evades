import React, { createContext, useContext, useEffect, useRef, useState, useCallback } from 'react';

type MessageType = 'Game' | 'Training' | 'Evaluation' | 'Log' | 'Status';

interface SocketMessage {
  type: MessageType;
  data: any;
}

interface SocketContextType {
  isConnected: boolean;
  isReconnecting: boolean;
  sendMessage: (msg: any) => void;
  subscribe: (type: MessageType, callback: (data: any) => void) => () => void;
}

const SocketContext = createContext<SocketContextType | null>(null);

export const useSocket = () => {
  const context = useContext(SocketContext);
  if (!context) throw new Error('useSocket must be used within a SocketProvider');
  return context;
};

function getWsUrl(): string {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  let host = window.location.host;
  // If we're on Vite's dev server (usually 5173), connect to the backend (usually 8080)
  if (host.includes(':5173')) {
    host = host.replace(':5173', ':8080');
  }
  return `${protocol}//${host}/ws`;
}

const MIN_RECONNECT_DELAY_MS = 2_000;
const MAX_RECONNECT_DELAY_MS = 10_000;

export const SocketProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isConnected, setIsConnected] = useState(false);
  const [isReconnecting, setIsReconnecting] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);
  const reconnectDelayRef = useRef(MIN_RECONNECT_DELAY_MS);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const unmountedRef = useRef(false);

  const subscribersRef = useRef<Record<string, Set<(data: any) => void>>>({
    Game: new Set(),
    Training: new Set(),
    Evaluation: new Set(),
    Log: new Set(),
    Status: new Set(),
  });

  const connect = useCallback(() => {
    if (unmountedRef.current) return;

    const wsUrl = getWsUrl();
    console.log('Connecting to WebSocket:', wsUrl);
    const s = new WebSocket(wsUrl);
    socketRef.current = s;

    s.onopen = () => {
      if (unmountedRef.current) { s.close(); return; }
      setIsConnected(true);
      setIsReconnecting(false);
      // Reset backoff on successful connection
      reconnectDelayRef.current = MIN_RECONNECT_DELAY_MS;
    };

    s.onclose = () => {
      if (unmountedRef.current) return;
      setIsConnected(false);
      // Schedule reconnect with exponential back-off
      const delay = reconnectDelayRef.current;
      reconnectDelayRef.current = Math.min(delay * 2, MAX_RECONNECT_DELAY_MS);
      console.warn(`WebSocket closed. Reconnecting in ${delay / 1000}s…`);
      setIsReconnecting(true);
      reconnectTimerRef.current = setTimeout(connect, delay);
    };

    s.onerror = () => {
      // onclose will fire after onerror, so let that handle reconnection.
    };

    s.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        const type = msg.type as MessageType;
        if (subscribersRef.current[type]) {
          subscribersRef.current[type].forEach(cb => cb(msg.data));
        }
      } catch (e) {
        console.error('Failed to parse socket message', e);
      }
    };
  }, []);

  useEffect(() => {
    unmountedRef.current = false;
    connect();
    return () => {
      unmountedRef.current = true;
      if (reconnectTimerRef.current !== null) {
        clearTimeout(reconnectTimerRef.current);
      }
      socketRef.current?.close();
    };
  }, [connect]);

  const sendMessage = (msg: any) => {
    const s = socketRef.current;
    if (s?.readyState === WebSocket.OPEN) {
      s.send(JSON.stringify(msg));
    }
  };

  const subscribe = (type: MessageType, callback: (data: any) => void) => {
    subscribersRef.current[type].add(callback);
    return () => {
      subscribersRef.current[type].delete(callback);
    };
  };

  return (
    <SocketContext.Provider value={{ isConnected, isReconnecting, sendMessage, subscribe }}>
      {children}
    </SocketContext.Provider>
  );
};
