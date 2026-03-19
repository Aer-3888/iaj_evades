import React, { createContext, useContext, useEffect, useRef, useState } from 'react';

type MessageType = 'Game' | 'Training' | 'Log' | 'Status';

interface SocketMessage {
  type: MessageType;
  data: any;
}

interface SocketContextType {
  isConnected: boolean;
  sendMessage: (msg: any) => void;
  subscribe: (type: MessageType, callback: (data: any) => void) => () => void;
}

const SocketContext = createContext<SocketContextType | null>(null);

export const useSocket = () => {
  const context = useContext(SocketContext);
  if (!context) throw new Error('useSocket must be used within a SocketProvider');
  return context;
};

export const SocketProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isConnected, setIsConnected] = useState(false);
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const subscribersRef = useRef<Record<string, Set<(data: any) => void>>>({
    Game: new Set(),
    Training: new Set(),
    Log: new Set(),
    Status: new Set(),
  });

  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    let host = window.location.host;
    
    // If we're on Vite's dev server (usually 5173), we want to connect to the backend (usually 8080)
    if (host.includes(':5173')) {
      host = host.replace(':5173', ':8080');
    }
    
    const wsUrl = `${protocol}//${host}/ws`;
    console.log('Connecting to WebSocket:', wsUrl);
    const s = new WebSocket(wsUrl);
    setSocket(s);

    s.onopen = () => setIsConnected(true);
    s.onclose = () => setIsConnected(false);
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

    return () => {
      s.close();
    };
  }, []);

  const sendMessage = (msg: any) => {
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(msg));
    }
  };

  const subscribe = (type: MessageType, callback: (data: any) => void) => {
    subscribersRef.current[type].add(callback);
    return () => {
      subscribersRef.current[type].delete(callback);
    };
  };

  return (
    <SocketContext.Provider value={{ isConnected, sendMessage, subscribe }}>
      {children}
    </SocketContext.Provider>
  );
};
