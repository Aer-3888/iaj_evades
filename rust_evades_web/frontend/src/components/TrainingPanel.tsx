import { Play, Square, LineChart as ChartIcon, Zap, Target, Activity, Trophy } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import StatCard from './StatCard';
import { TrainingProgress } from '../App';
import { useToast } from '../contexts/ToastContext';

interface Props {
  history: TrainingProgress[];
  isRunning: boolean;
  setIsRunning: (val: boolean) => void;
}

export default function TrainingPanel({ history, isRunning, setIsRunning }: Props) {
  const { showToast } = useToast();
  const latest = history[history.length - 1] || {
    episode: 0,
    total_steps: 0,
    epsilon: 1,
    avg_survival: 0,
    min_survival: 0,
    avg_return: 0,
    avg_evades: 0,
    steps_per_second: 0,
    timeouts: 0,
    loss: 0,
    global_best_survival: 0
  };

  const handleStart = async () => {
    try {
      await fetch('/api/train/start', { method: 'POST' });
      setIsRunning(true);
      showToast('Training session started', 'success');
    } catch (e) {
      showToast('Failed to start training session', 'error');
    }
  };

  const handleStop = async () => {
    try {
      await fetch('/api/train/stop', { method: 'POST' });
      showToast('Training session stopped', 'info');
    } catch (e) {
      showToast('Failed to stop training session', 'error');
    }
  };

  const handlePromote = async () => {
    try {
      const res = await fetch('/api/train/promote', { method: 'POST' });
      const text = await res.text();
      showToast(text, 'success');
    } catch (e) {
      showToast('Failed to promote model', 'error');
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between bg-slate-900 border border-slate-800 p-6 rounded-xl shadow-lg">
        <div>
          <h2 className="text-xl font-bold text-emerald-400">DQN Training Control</h2>
          <p className="text-sm text-slate-400 mt-1">Configure and monitor reinforcement learning sessions.</p>
        </div>
        <div className="flex items-center space-x-4">
          {!isRunning && history.length > 0 && (
            <button 
              onClick={handlePromote}
              className="flex items-center px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded-lg text-xs font-bold transition"
              title="Promote this model to be used in the Simulation tab"
            >
              Promote to Simulation
            </button>
          )}
          <div className="flex items-center px-4 py-2 bg-slate-950 rounded-lg border border-slate-800">
            <div className={`w-2 h-2 rounded-full mr-3 ${isRunning ? 'bg-emerald-500 animate-pulse' : 'bg-slate-700'}`} />
            <span className="text-xs font-mono uppercase tracking-widest">
              {isRunning ? 'Training Active' : 'Idle'}
            </span>
          </div>
          {!isRunning ? (
            <button 
              onClick={handleStart}
              className="flex items-center px-6 py-2 bg-emerald-600 hover:bg-emerald-500 rounded-lg font-bold transition shadow-lg shadow-emerald-900/20 cursor-pointer"
            >
              <Play size={18} className="mr-2 fill-current" /> Start Session
            </button>
          ) : (
            <button 
              onClick={handleStop}
              className="flex items-center px-6 py-2 bg-rose-600 hover:bg-rose-500 rounded-lg font-bold transition cursor-pointer"
            >
              <Square size={18} className="mr-2 fill-current" /> Stop
            </button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
        <StatCard 
          label="Exploration" 
          value={`${(latest.epsilon * 100).toFixed(1)}%`} 
          description={latest.epsilon > 0.1 ? "Learning: Random moves" : "Expert: Greedy moves"}
          icon={<Zap size={14} />}
          color="text-amber-400"
        />
        <StatCard 
          label="Total Steps" 
          value={latest.total_steps.toLocaleString()} 
          description="Global progress"
          icon={<Activity size={14} />}
        />
        <StatCard 
          label="Best Survival" 
          value={`${latest.global_best_survival.toFixed(2)}s`} 
          description="All-time session record"
          icon={<Trophy size={14} />}
          color="text-emerald-400"
        />
        <StatCard 
          label="Avg Survival" 
          value={`${latest.avg_survival.toFixed(2)}s`} 
          description="Last evaluation avg"
          icon={<Zap size={14} />}
          color="text-blue-400"
        />
        <StatCard 
          label="Success" 
          value={`${latest.timeouts}`} 
          description="Eval seeds completed"
          icon={<Target size={14} />}
          color="text-rose-400"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl h-80">
          <h3 className="text-sm font-bold text-slate-500 uppercase mb-4 flex items-center">
            <ChartIcon size={16} className="mr-2" /> Survival Time (Avg)
          </h3>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="episode" stroke="#64748b" fontSize={12} />
              <YAxis stroke="#64748b" fontSize={12} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }}
                itemStyle={{ color: '#10b981' }}
              />
              <Line type="monotone" dataKey="avg_survival" stroke="#10b981" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="min_survival" stroke="#334155" strokeWidth={1} dot={false} strokeDasharray="5 5" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl h-80">
          <h3 className="text-sm font-bold text-slate-500 uppercase mb-4 flex items-center">
            <ChartIcon size={16} className="mr-2" /> Training Loss
          </h3>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="episode" stroke="#64748b" fontSize={12} />
              <YAxis stroke="#64748b" fontSize={12} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }}
                itemStyle={{ color: '#f43f5e' }}
              />
              <Line type="monotone" dataKey="loss" stroke="#f43f5e" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl h-80">
          <h3 className="text-sm font-bold text-slate-500 uppercase mb-4 flex items-center">
            <ChartIcon size={16} className="mr-2" /> Average Return
          </h3>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="episode" stroke="#64748b" fontSize={12} />
              <YAxis stroke="#64748b" fontSize={12} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }}
                itemStyle={{ color: '#3b82f6' }}
              />
              <Line type="monotone" dataKey="avg_return" stroke="#3b82f6" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl h-80">
          <h3 className="text-sm font-bold text-slate-500 uppercase mb-4 flex items-center">
            <Target size={16} className="mr-2" /> Performance (Success & Survival)
          </h3>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="episode" stroke="#64748b" fontSize={12} />
              <YAxis yAxisId="left" stroke="#f43f5e" fontSize={12} orientation="left" label={{ value: 'Wins', angle: -90, position: 'insideLeft', fill: '#f43f5e', fontSize: 10 }} />
              <YAxis yAxisId="right" stroke="#3b82f6" fontSize={12} orientation="right" label={{ value: 'Survival (s)', angle: 90, position: 'insideRight', fill: '#3b82f6', fontSize: 10 }} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }}
              />
              <Line yAxisId="left" type="monotone" dataKey="timeouts" name="Wins" stroke="#f43f5e" strokeWidth={2} dot={true} r={2} />
              <Line yAxisId="right" type="monotone" dataKey="avg_survival" name="Avg Survival" stroke="#3b82f6" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}
