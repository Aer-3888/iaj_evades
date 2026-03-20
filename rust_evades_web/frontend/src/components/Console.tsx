import { useEffect, useState, useRef } from 'react'
import { Terminal } from 'lucide-react'
import { useSocket } from '../contexts/SocketContext';

interface LogLine {
  stream: string;
  text: string;
}

export default function Console() {
  const [logs, setLogs] = useState<LogLine[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);
  const { subscribe } = useSocket();

  useEffect(() => {
    const unsubscribe = subscribe('Log', (data) => {
      setLogs(prev => [data, ...prev.slice(0, 199)]);
    });

    return () => unsubscribe();
  }, [subscribe]);

  const runCmd = async (cmd: string) => {
    try {
      await fetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command: cmd })
      });
    } catch (e) {
      console.error('Failed to run command', e);
    }
  };

  return (
    <div className="flex flex-col h-full bg-slate-950 border border-slate-800 rounded-xl overflow-hidden font-mono text-xs">
      <div className="bg-slate-900 px-4 py-2 border-b border-slate-800 flex items-center justify-between">
        <div className="flex items-center text-slate-400">
          <Terminal size={14} className="mr-2" /> SYSTEM CONSOLE
        </div>
        <div className="flex space-x-2">
           <button onClick={() => runCmd('cargo test')} className="px-2 py-0.5 bg-slate-800 hover:bg-slate-700 rounded text-[10px]">RUN TESTS</button>
           <button onClick={() => setLogs([])} className="px-2 py-0.5 bg-slate-800 hover:bg-slate-700 rounded text-[10px]">CLEAR</button>
        </div>
      </div>
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-1">
        {logs.map((log, i) => (
          <div key={i} className="flex">
            <span className={`w-16 shrink-0 ${log.stream === 'stderr' ? 'text-rose-500' : log.stream === 'system' ? 'text-blue-500' : 'text-slate-500'}`}>
              [{log.stream.toUpperCase()}]
            </span>
            <span className="text-slate-300 break-all">{log.text}</span>
          </div>
        ))}
        {logs.length === 0 && <div className="text-slate-700 italic">No output received...</div>}
      </div>
    </div>
  );
}
