import React, { useState, useEffect } from 'react';
import { Play, Square, CheckCircle, XCircle, Clock, Award, ShieldAlert } from 'lucide-react';
import StatCard from './StatCard';
import { useSocket } from '../contexts/SocketContext';

interface EvaluationSeedResult {
  seed: number;
  survival_time: number;
  total_return: number;
  evades: number;
  timed_out: boolean;
}

interface EvaluationSummary {
  average_survival_time: number;
  average_return: number;
  average_evades: number;
  min_survival_time: number;
  min_return: number;
  timeouts: number;
}

interface EvaluationProgress {
  current_seed_index: number;
  total_seeds: number;
  last_result?: EvaluationSeedResult;
  summary: EvaluationSummary;
}

const EvaluationPanel: React.FC = () => {
  const { subscribe } = useSocket();
  const [isRunning, setIsRunning] = useState(false);
  const [numSeeds, setNumSeeds] = useState(10);
  const [startSeed, setStartSeed] = useState(100);
  const [progress, setProgress] = useState<EvaluationProgress | null>(null);
  const [history, setHistory] = useState<EvaluationSeedResult[]>([]);

  useEffect(() => {
    const unsub = subscribe('Evaluation', (data: EvaluationProgress) => {
      setProgress(data);
      if (data.last_result) {
        setHistory(prev => [data.last_result!, ...prev].slice(0, 100));
      }
      if (data.current_seed_index === data.total_seeds) {
        setIsRunning(false);
      }
    });

    const checkStatus = async () => {
      try {
        const res = await fetch('/api/eval/status');
        const running = await res.json();
        setIsRunning(running);
      } catch (e) {
        console.error('Failed to check evaluation status', e);
      }
    };

    checkStatus();
    return () => unsub();
  }, [subscribe]);

  const startEval = async () => {
    try {
      setHistory([]);
      const res = await fetch('/api/eval/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ start_seed: startSeed, num_seeds: numSeeds }),
      });
      if (res.ok) {
        setIsRunning(true);
      } else {
        const err = await res.text();
        alert(`Failed to start evaluation: ${err}`);
      }
    } catch (e) {
      console.error('Failed to start evaluation', e);
    }
  };

  const stopEval = async () => {
    try {
      await fetch('/api/eval/stop', { method: 'POST' });
      setIsRunning(false);
    } catch (e) {
      console.error('Failed to stop evaluation', e);
    }
  };

  return (
    <div className="flex flex-col space-y-6 max-w-6xl mx-auto">
      {/* Header & Controls */}
      <div className="bg-slate-900 rounded-xl border border-slate-800 p-6 flex flex-col md:flex-row md:items-center justify-between gap-6 shadow-xl">
        <div className="flex flex-col space-y-1">
          <h2 className="text-xl font-bold text-slate-100">Model Evaluation</h2>
          <p className="text-sm text-slate-400">Benchmark your model across a fixed set of seeds.</p>
        </div>
        
        <div className="flex items-center gap-4 bg-slate-950 p-2 rounded-lg border border-slate-800/50">
          <div className="flex flex-col">
            <label className="text-[10px] uppercase font-bold text-slate-500 ml-1">Num Seeds</label>
            <input 
              type="number" 
              value={numSeeds}
              onChange={(e) => setNumSeeds(parseInt(e.target.value) || 1)}
              disabled={isRunning}
              className="bg-transparent text-sm font-mono text-blue-400 w-20 px-1 focus:outline-none"
            />
          </div>
          <div className="w-px h-8 bg-slate-800" />
          <div className="flex flex-col">
            <label className="text-[10px] uppercase font-bold text-slate-500 ml-1">Start Seed</label>
            <input 
              type="number" 
              value={startSeed}
              onChange={(e) => setStartSeed(parseInt(e.target.value) || 0)}
              disabled={isRunning}
              className="bg-transparent text-sm font-mono text-emerald-400 w-24 px-1 focus:outline-none"
            />
          </div>
          <button
            onClick={isRunning ? stopEval : startEval}
            className={`flex items-center space-x-2 px-6 py-2 rounded-lg font-bold transition-all shadow-lg active:scale-95 ${
              isRunning 
                ? 'bg-rose-600 hover:bg-rose-500 text-white' 
                : 'bg-blue-600 hover:bg-blue-500 text-white'
            }`}
          >
            {isRunning ? <Square size={18} fill="currentColor" /> : <Play size={18} fill="currentColor" />}
            <span>{isRunning ? 'STOP EVAL' : 'START EVAL'}</span>
          </button>
        </div>
      </div>

      {/* Progress Bar */}
      {isRunning && progress && (
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-6 shadow-lg overflow-hidden relative">
          <div className="flex justify-between items-center mb-2">
            <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">Evaluation Progress</span>
            <span className="text-xs font-mono text-blue-400">{progress.current_seed_index} / {progress.total_seeds}</span>
          </div>
          <div className="w-full bg-slate-800 h-3 rounded-full overflow-hidden">
            <div 
              className="bg-blue-500 h-full transition-all duration-300 shadow-[0_0_12px_rgba(59,130,246,0.5)]" 
              style={{ width: `${(progress.current_seed_index / progress.total_seeds) * 100}%` }}
            />
          </div>
        </div>
      )}

      {/* Stats Cards */}
      {progress && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard 
            label="Avg Survival" 
            value={`${progress.summary.average_survival_time.toFixed(2)}s`} 
            description={`Min: ${progress.summary.min_survival_time.toFixed(2)}s`}
            icon={<Clock className="text-blue-400" size={16} />} 
          />
          <StatCard 
            label="Avg Return" 
            value={progress.summary.average_return.toFixed(1)} 
            description={`Min: ${progress.summary.min_return.toFixed(1)}`}
            icon={<Award className="text-emerald-400" size={16} />} 
          />
          <StatCard 
            label="Avg Evades" 
            value={progress.summary.average_evades.toFixed(1)} 
            icon={<Activity className="text-purple-400" size={16} />} 
          />
          <StatCard 
            label="Success Rate" 
            value={`${((progress.summary.timeouts / progress.current_seed_index) * 100).toFixed(1)}%`}
            description={`${progress.summary.timeouts} / ${progress.current_seed_index}`}
            icon={<ShieldAlert className="text-rose-400" size={16} />} 
          />
        </div>
      )}

      {/* Results Table */}
      <div className="bg-slate-900 rounded-xl border border-slate-800 shadow-xl overflow-hidden min-h-[400px]">
        <div className="px-6 py-4 border-b border-slate-800 flex justify-between items-center">
          <h3 className="font-bold text-slate-300 text-sm uppercase tracking-widest">Seed Results</h3>
          <span className="text-[10px] text-slate-500 uppercase">Showing last 100 results</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="bg-slate-950/50 text-slate-500 text-[10px] uppercase font-bold tracking-tighter">
              <tr>
                <th className="px-6 py-3">Seed</th>
                <th className="px-6 py-3 text-center">Outcome</th>
                <th className="px-6 py-3 text-right">Survival</th>
                <th className="px-6 py-3 text-right">Return</th>
                <th className="px-6 py-3 text-right">Evades</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/50">
              {history.map((res, idx) => (
                <tr key={`${res.seed}-${idx}`} className="hover:bg-slate-800/30 transition group">
                  <td className="px-6 py-3 font-mono text-slate-400">{res.seed}</td>
                  <td className="px-6 py-3 text-center">
                    {res.timed_out ? (
                      <span className="inline-flex items-center space-x-1 text-emerald-500 bg-emerald-500/10 px-2 py-0.5 rounded text-[10px] font-bold">
                        <CheckCircle size={10} /> <span>SUCCESS</span>
                      </span>
                    ) : (
                      <span className="inline-flex items-center space-x-1 text-rose-500 bg-rose-500/10 px-2 py-0.5 rounded text-[10px] font-bold">
                        <XCircle size={10} /> <span>COLLISION</span>
                      </span>
                    )}
                  </td>
                  <td className="px-6 py-3 text-right font-mono text-slate-200">{res.survival_time.toFixed(2)}s</td>
                  <td className="px-6 py-3 text-right font-mono text-slate-200">{res.total_return.toFixed(1)}</td>
                  <td className="px-6 py-3 text-right font-mono text-slate-200">{res.evades}</td>
                </tr>
              ))}
              {history.length === 0 && (
                <tr>
                  <td colSpan={5} className="px-6 py-12 text-center text-slate-600 italic">
                    No results yet. Configure seeds and click "Start Eval" to begin.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default EvaluationPanel;

import { Activity } from 'lucide-react';
