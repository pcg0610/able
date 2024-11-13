import { useEffect, useRef, useState } from 'react';
import { LogContainer, LogText } from '@features/deploy/ui/server/log-viewer.style';

const LogViewer = () => {
  const ws = useRef<WebSocket | null>(null);
  const logRef = useRef<HTMLDivElement | null>(null);
  const [logs, setLogs] = useState<string>('...');

  useEffect(() => {
    const connectWebSocket = () => {
      ws.current = new WebSocket('ws://localhost:8088/ws/logs');

      ws.current.onopen = () => {
        console.log('WebSocket Connected');
      };

      ws.current.onmessage = (event) => {
        setLogs(event.data);
      };

      ws.current.onclose = () => {
        console.log('WebSocket Disconnected');
        setTimeout(connectWebSocket, 5000);
      };

      ws.current.onerror = (error) => {
        console.error('WebSocket Error:', error);
      };
    };

    connectWebSocket();

    return () => {
      if (ws.current && ws.current.readyState === 1) {
        ws.current.close();
      }
    };
  }, []);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <LogContainer ref={logRef}>
      <LogText>{logs}</LogText>
    </LogContainer>
  );
};

export default LogViewer;
