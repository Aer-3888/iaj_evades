import { Play, Square, LineChart as ChartIcon, Zap, Target, Activity, Trophy, Settings2, FileJson, RefreshCw, X, Check, AlertTriangle } from 'lucide-react'
import { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import StatCard from './StatCard';
import { TrainingProgress } from '../App';
import { useToast } from '../contexts/ToastContext';

interface Props {
  history: TrainingProgress[];
  isRunning: boolean;
  setIsRunning: (val: boolean) => void;
}

interface TrainingConfig {
  model_type: 'dqn' | 'dqn2';
  map_design: 'Open' | 'Closed' | 'Arena';
  episodes: number;
  trainer_seed: number;
  learning_rate: number;
  epsilon_decay_steps: number;
  batch_size: number;
  replay_capacity: number;
  checkpoint_every: number;
  seed_focus_mode: "Original" | "BadSeeds";
  fixed_training_seeds: number[];
  random_seed_count_per_cycle: number;
  hidden_sizes: number[];
  warmup_steps: number;
  train_every: number;
  target_sync_interval: number;
  gamma: number;
  epsilon_start: number;
  epsilon_end: number;
  action_repeat: number;
  huber_delta: number;
  gradient_clip_norm: number;
}

const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
  model_type: 'dqn',
  map_design: 'Open',
  episodes: 100000,
  trainer_seed: 7,
  learning_rate: 0.0003,
  epsilon_decay_steps: 50000,
  batch_size: 128,
  replay_capacity: 50000,
  checkpoint_every: 100,
  seed_focus_mode: "BadSeeds",
  fixed_training_seeds: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
  random_seed_count_per_cycle: 2,
  hidden_sizes: [128, 128],
  warmup_steps: 1000,
  train_every: 2,
  target_sync_interval: 1000,
  gamma: 0.99,
  epsilon_start: 1.0,
  epsilon_end: 0.03,
  action_repeat: 2,
  huber_delta: 10.0,
  gradient_clip_norm: 1280.0,
};

interface ModelInfo {
  name: string;
  path: string;
  model_type: string;
}

function ModelTypeBadge({ type }: { type: string }) {
  const isDqn2 = type === 'dqn2';
  return (
    <span className={`ml-2 shrink-0 px-1.5 py-0.5 rounded text-[9px] font-bold uppercase tracking-wide ${
      isDqn2
        ? 'bg-purple-900/50 text-purple-300 border border-purple-700/50'
        : 'bg-emerald-900/40 text-emerald-400 border border-emerald-700/40'
    }`}>
      {isDqn2 ? 'DQN2' : 'DQN'}
    </span>
  );
}

export default function TrainingPanel({ history, isRunning, setIsRunning }: Props) {
  const { showToast } = useToast();
  const [showSettings, setShowSettings] = useState(false);
  const [config, setConfig] = useState<TrainingConfig>(DEFAULT_TRAINING_CONFIG);
  const [activeConfig, setActiveConfig] = useState<{ config: TrainingConfig, resume_model_path: string | null } | null>(null);
  const [resumeModelPath, setResumeModelPath] = useState<string | null>(null);
  const [resumeModelType, setResumeModelType] = useState<string | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);
  const [pendingCrossTypeStart, setPendingCrossTypeStart] = useState(false);

  const fetchActiveConfig = async () => {
    try {
      const res = await fetch('/api/train/active_config');
      const data = await res.json();
      setActiveConfig(data);
    } catch (e) {
      console.error('Failed to fetch active config', e);
    }
  };

  useEffect(() => {
    if (isRunning) {
      fetchActiveConfig();
    } else {
      setActiveConfig(null);
    }
  }, [isRunning]);

  const fetchModels = async () => {
    setLoadingModels(true);
    try {
      const res = await fetch('/api/models');
      const data = await res.json();
      setModels(data);
    } catch (e) {
      console.error('Failed to fetch models', e);
    } finally {
      setLoadingModels(false);
    }
  };

  useEffect(() => {
    if (showSettings) fetchModels();
  }, [showSettings]);

  const isCrossTypeResume =
    resumeModelPath !== null &&
    resumeModelType !== null &&
    resumeModelType !== config.model_type.toLowerCase();

  const doStart = async () => {
    try {
      const res = await fetch('/api/train/start', { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config, resume_model_path: resumeModelPath })
      });
      if (res.ok) {
        setIsRunning(true);
        showToast('Training session started', 'success');
      } else {
        const err = await res.text();
        showToast(`Failed to start: ${err}`, 'error');
      }
    } catch (e) {
      showToast('Failed to start training session', 'error');
    } finally {
      setPendingCrossTypeStart(false);
    }
  };

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
    global_best_survival: 0,
    mean_predicted_q: 0,
    mean_target_q: 0,
    mean_abs_td_error: 0,
    terminal_fraction: 0,
  };

  const handleStart = async () => {
    if (isCrossTypeResume) {
      setPendingCrossTypeStart(true);
      return;
    }
    await doStart();
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

  const selectResume = (model: ModelInfo) => {
    if (resumeModelPath === model.path) {
      setResumeModelPath(null);
      setResumeModelType(null);
    } else {
      setResumeModelPath(model.path);
      setResumeModelType(model.model_type);
    }
  };

  const configModelTypeLower = config.model_type.toLowerCase();

  const maxEpisode = latest.episode > 0 ? latest.episode : 1;

  const computeRolling = (field: keyof TrainingProgress): number[] => {
    if (history.length === 0) {
      return [];
    }

    const windowSize = Math.max(5, Math.min(100, Math.round(history.length * 0.05)));
    const halfWindow = Math.floor(windowSize / 2);

    return history.map((_, i) => {
      const start = Math.max(0, i - halfWindow);
      const end = Math.min(history.length - 1, i + halfWindow);
      let total = 0;

      for (let j = start; j <= end; j++) {
        total += history[j][field] as number;
      }

      return total / (end - start + 1);
    });
  };

  const rollingSurvival = computeRolling('avg_survival');
  const rollingReturn = computeRolling('avg_return');
  const rollingLoss = computeRolling('loss');
  const rollingTimeouts = computeRolling('timeouts');
  const rollingPredictedQ = computeRolling('mean_predicted_q');
  const rollingTargetQ = computeRolling('mean_target_q');
  const rollingTdError = computeRolling('mean_abs_td_error');
  const rollingTerminalFraction = computeRolling('terminal_fraction');

  const computedHistory = history.map((d, i) => ({
    ...d,
    lossLog: Math.log10(Math.max(1e-10, d.loss)),
    rolling_avg_survival: rollingSurvival[i],
    rolling_avg_return: rollingReturn[i],
    rolling_loss_log: Math.log10(Math.max(1e-10, rollingLoss[i])),
    rolling_timeouts: rollingTimeouts[i],
    rolling_mean_predicted_q: rollingPredictedQ[i],
    rolling_mean_target_q: rollingTargetQ[i],
    rolling_mean_abs_td_error: rollingTdError[i],
    rolling_terminal_fraction: rollingTerminalFraction[i],
  }));

  const xAxisProps = {
    dataKey: 'episode' as const,
    type: 'number' as const,
    domain: [0, maxEpisode] as [number, number],
    tickFormatter: (val: number) => {
      if (val >= 1000000) return `${(val / 1000000).toFixed(1)}M`;
      if (val >= 1000) return `${(val / 1000).toFixed(0)}K`;
      return String(val);
    },
    stroke: '#64748b' as const,
    fontSize: 12,
  };

  const tooltipLabelFormatter = (val: number) => `Episode: ${val.toLocaleString()}`;

  const lossValues = history.map(d => d.loss).filter(l => l > 0);
  const logMinLoss = lossValues.length > 0 ? Math.floor(Math.log10(Math.min(...lossValues))) : -4;
  const logMaxLoss = lossValues.length > 0 ? Math.ceil(Math.log10(Math.max(...lossValues))) : 0;
  const lossTicks: number[] = [];
  for (let i = logMinLoss; i <= logMaxLoss; i++) {
    lossTicks.push(i);
  }
  const lossYAxisProps = {
    ticks: lossTicks,
    domain: [logMinLoss, logMaxLoss] as [number, number],
    tickFormatter: (val: number) => {
      const v = Math.pow(10, val);
      return v >= 0.01 ? v.toPrecision(2) : v.toExponential(0);
    },
    stroke: '#64748b' as const,
    fontSize: 11,
  };

  return (
    <div className="space-y-6">
      {/* Cross-type resume warning dialog */}
      {pendingCrossTypeStart && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="bg-slate-900 border border-amber-700/60 rounded-xl shadow-2xl p-6 max-w-md w-full mx-4">
            <div className="flex items-start mb-4">
              <AlertTriangle className="text-amber-400 mr-3 mt-0.5 shrink-0" size={20} />
              <div>
                <h3 className="font-bold text-amber-400 text-base mb-1">Model type mismatch</h3>
                <p className="text-sm text-slate-300">
                  You're resuming a <span className="font-mono font-bold text-emerald-400">{(resumeModelType ?? '').toUpperCase()}</span> model but training as <span className="font-mono font-bold text-purple-400">{config.model_type.toUpperCase()}</span>.
                  The extra input weights will be initialised to <strong>zero</strong>. The model may need many episodes to recover.
                </p>
              </div>
            </div>
            <div className="flex justify-end gap-3">
              <button
                onClick={() => setPendingCrossTypeStart(false)}
                className="px-4 py-2 text-sm rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition"
              >
                Cancel
              </button>
              <button
                onClick={doStart}
                className="px-4 py-2 text-sm rounded-lg bg-amber-600 hover:bg-amber-500 font-bold text-white transition"
              >
                Proceed anyway
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="flex flex-col bg-slate-900 border border-slate-800 rounded-xl shadow-lg overflow-hidden">
        <div className="flex items-center justify-between p-6">
          <div>
            <h2 className="text-xl font-bold text-emerald-400">DQN Training Control</h2>
            <p className="text-sm text-slate-400 mt-1">Configure and monitor reinforcement learning sessions.</p>
          </div>
          <div className="flex items-center space-x-4">
            {!isRunning && (
              <button 
                onClick={() => setShowSettings(!showSettings)}
                className={`p-2 rounded-lg transition ${showSettings ? 'bg-emerald-600 text-white' : 'bg-slate-800 text-slate-400 hover:bg-slate-700'}`}
                title="Toggle Training Settings"
              >
                <Settings2 size={20} />
              </button>
            )}
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

        {isRunning && activeConfig && (
          <div className="border-t border-slate-800 p-4 bg-slate-950/50 text-xs text-slate-400">
             <div className="flex items-center mb-2">
               <Settings2 size={14} className="mr-2 text-emerald-500" />
               <span className="font-bold text-slate-300 uppercase tracking-widest">Active Session Settings</span>
             </div>
             <div className="grid grid-cols-2 md:grid-cols-4 gap-x-4 gap-y-2 font-mono mt-3">
               <div>Model: <span className="text-emerald-400">{activeConfig.config.model_type.toUpperCase()}</span></div>
               <div>Map: <span className="text-emerald-400">{activeConfig.config.map_design}</span></div>
               <div>Episodes: <span className="text-emerald-400">{activeConfig.config.episodes}</span></div>
               <div>LR: <span className="text-emerald-400">{activeConfig.config.learning_rate}</span></div>
               <div>Trainer Seed: <span className="text-emerald-400">{activeConfig.config.trainer_seed}</span></div>
               <div>Eps Decay: <span className="text-emerald-400">{activeConfig.config.epsilon_decay_steps}</span></div>
               <div>Batch: <span className="text-emerald-400">{activeConfig.config.batch_size}</span></div>
               <div>Replay: <span className="text-emerald-400">{activeConfig.config.replay_capacity}</span></div>
               <div>Action Rep: <span className="text-emerald-400">{activeConfig.config.action_repeat}</span></div>
               <div>Target Sync: <span className="text-emerald-400">{activeConfig.config.target_sync_interval}</span></div>
               <div>Huber Delta: <span className="text-emerald-400">{activeConfig.config.huber_delta}</span></div>
               <div>Grad Clip: <span className="text-emerald-400">{activeConfig.config.gradient_clip_norm}</span></div>
               <div>Seed Focus: <span className="text-emerald-400">{activeConfig.config.seed_focus_mode}</span></div>
               {activeConfig.resume_model_path && (
                 <div className="col-span-2 md:col-span-4 mt-1 border-t border-slate-800/50 pt-2">
                   Resume Model: <span className="text-blue-400">{activeConfig.resume_model_path.split('/').pop()}</span>
                 </div>
               )}
             </div>
          </div>
        )}

        {showSettings && !isRunning && (
          <div className="border-t border-slate-800 p-6 bg-slate-950/50 grid grid-cols-1 md:grid-cols-3 gap-6 animate-in slide-in-from-top-2">
            <div className="space-y-4">
               <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest">Core Parameters</h3>
               <div className="space-y-3">
                 {/* Model Type Selector */}
                  <div className="space-y-1">
                    <label className="text-xs font-medium text-slate-500">Model Type</label>
                   <div className="flex rounded overflow-hidden border border-slate-800">
                     <button
                       onClick={() => setConfig({...config, model_type: 'dqn'})}
                       className={`flex-1 py-1.5 text-xs font-bold transition ${config.model_type === 'dqn' ? 'bg-emerald-700 text-white' : 'bg-slate-900 text-slate-400 hover:bg-slate-800'}`}
                     >
                       DQN <span className="font-normal opacity-60 text-[10px]">(1 pass)</span>
                     </button>
                     <button
                       onClick={() => setConfig({...config, model_type: 'dqn2'})}
                       className={`flex-1 py-1.5 text-xs font-bold transition ${config.model_type === 'dqn2' ? 'bg-purple-700 text-white' : 'bg-slate-900 text-slate-400 hover:bg-slate-800'}`}
                     >
                       DQN2 <span className="font-normal opacity-60 text-[10px]">(2 passes)</span>
                     </button>
                   </div>
                   {config.model_type === 'dqn2' && (
                     <p className="text-[10px] text-purple-300/70 italic">
                       Near + far raycast (146 inputs vs. 74). Longer to train.
                     </p>
                   )}
                    {isCrossTypeResume && (
                      <p className="text-[10px] text-amber-400 flex items-center gap-1">
                        <AlertTriangle size={10} /> Resume model is {(resumeModelType ?? '').toUpperCase()} — weights will be zero-padded.
                      </p>
                    )}
                  </div>
                  <div className="space-y-1">
                    <label className="text-xs font-medium text-slate-500">Training Map</label>
                    <div className="flex rounded overflow-hidden border border-slate-800">
                      <button
                        onClick={() => setConfig({...config, map_design: 'Open'})}
                        className={`flex-1 py-1.5 text-xs font-bold transition ${config.map_design === 'Open' ? 'bg-emerald-700 text-white' : 'bg-slate-900 text-slate-400 hover:bg-slate-800'}`}
                      >
                        Open
                      </button>
                      <button
                        onClick={() => setConfig({...config, map_design: 'Arena'})}
                        className={`flex-1 py-1.5 text-xs font-bold transition ${config.map_design === 'Arena' ? 'bg-emerald-700 text-white' : 'bg-slate-900 text-slate-400 hover:bg-slate-800'}`}
                      >
                        Arena
                      </button>
                      <button
                        onClick={() => setConfig({...config, map_design: 'Closed'})}
                        className={`flex-1 py-1.5 text-xs font-bold transition ${config.map_design === 'Closed' ? 'bg-emerald-700 text-white' : 'bg-slate-900 text-slate-400 hover:bg-slate-800'}`}
                      >
                        Closed
                      </button>
                    </div>
                  </div>
                  <SettingInput 
                     label="Total Episodes" 
                     value={config.episodes} 
                    onChange={v => setConfig({...config, episodes: v})} 
                    min={100} max={1000000} step={1000}
                  />
                 <SettingInput 
                    label="Learning Rate" 
                    value={config.learning_rate} 
                    onChange={v => setConfig({...config, learning_rate: v})} 
                    min={0.00001} max={0.01} step={0.00001}
                    isFloat
                  />
                 <SettingInput 
                    label="Trainer Seed" 
                    value={config.trainer_seed} 
                    onChange={v => setConfig({...config, trainer_seed: v})} 
                    min={1} max={99999}
                  />
               </div>
            </div>
            <div className="space-y-4">
               <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest">DQN Logic</h3>
               <div className="space-y-3">
                 <SettingInput 
                    label="Epsilon Decay (steps)" 
                    value={config.epsilon_decay_steps} 
                    onChange={v => setConfig({...config, epsilon_decay_steps: v})} 
                    min={1000} max={500000} step={1000}
                  />
                 <SettingInput 
                    label="Batch Size" 
                    value={config.batch_size} 
                    onChange={v => setConfig({...config, batch_size: v})} 
                    min={32} max={1024} step={32}
                  />
                 <SettingInput 
                    label="Replay Capacity" 
                    value={config.replay_capacity} 
                    onChange={v => setConfig({...config, replay_capacity: v})} 
                    min={1000} max={500000} step={1000}
                  />
                 <SettingInput 
                    label="Huber Delta" 
                    value={config.huber_delta} 
                    onChange={v => setConfig({...config, huber_delta: v})} 
                    min={1} max={100} step={1}
                    isFloat
                  />
                 <SettingInput 
                    label="Gradient Clip Norm" 
                    value={config.gradient_clip_norm} 
                    onChange={v => setConfig({...config, gradient_clip_norm: v})} 
                    min={100} max={5000} step={10}
                  />
               </div>
            </div>
            <div className="space-y-4">
               <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest">Advanced</h3>
               <div className="space-y-3">
                  <div className="space-y-1">
                    <label className="text-xs font-medium text-slate-500">Seed Focus Mode</label>
                    <select 
                      value={config.seed_focus_mode}
                      onChange={e => setConfig({...config, seed_focus_mode: e.target.value as any})}
                      className="w-full bg-slate-900 border border-slate-800 rounded px-2 py-1 text-sm text-slate-300 focus:outline-none focus:border-emerald-500"
                    >
                      <option value="Original">Original (All seeds)</option>
                      <option value="BadSeeds">Bad Seeds (Focus on failures)</option>
                    </select>
                  </div>
                 <SettingInput 
                    label="Action Repeat" 
                    value={config.action_repeat} 
                    onChange={v => setConfig({...config, action_repeat: v})} 
                    min={1} max={4}
                  />
                 <SettingInput 
                    label="Target Sync (steps)" 
                    value={config.target_sync_interval} 
                    onChange={v => setConfig({...config, target_sync_interval: v})} 
                    min={100} max={10000} step={100}
                  />
               </div>
            </div>

            <div className="border-t border-slate-800 pt-6 md:col-span-3">
               <div className="flex justify-between items-center mb-4">
                  <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest">Resume from model (Optional)</h3>
                  <button 
                    onClick={fetchModels}
                    className="p-1 hover:bg-slate-800 rounded text-slate-400 transition"
                    title="Refresh model list"
                  >
                    <RefreshCw size={14} className={loadingModels ? 'animate-spin' : ''} />
                  </button>
               </div>
               
               <div className="bg-slate-950 rounded-lg border border-slate-800 overflow-hidden">
                  <div className="max-h-40 overflow-y-auto custom-scrollbar">
                    {models.length === 0 ? (
                      <div className="p-4 text-center text-slate-600 italic text-xs">
                        No JSON models found.
                      </div>
                    ) : (
                      <div className="divide-y divide-slate-800/50">
                        {models.map((model) => {
                          const isSelected = resumeModelPath === model.path;
                          const typeMismatch = isSelected && model.model_type !== configModelTypeLower;
                          return (
                            <button
                              key={model.path}
                              onClick={() => selectResume(model)}
                              className={`w-full flex items-center justify-between px-3 py-2 text-left transition hover:bg-slate-800/50 ${
                                isSelected ? (typeMismatch ? 'bg-amber-900/20' : 'bg-blue-900/20') : ''
                              }`}
                            >
                              <div className="flex items-center min-w-0 mr-4">
                                <FileJson size={14} className={`mr-2 shrink-0 ${isSelected ? (typeMismatch ? 'text-amber-400' : 'text-blue-400') : 'text-slate-500'}`} />
                                <span className={`text-[11px] truncate ${isSelected ? (typeMismatch ? 'text-amber-300 font-medium' : 'text-blue-300 font-medium') : 'text-slate-300'}`}>
                                  {model.name}
                                </span>
                                <ModelTypeBadge type={model.model_type} />
                              </div>
                              {isSelected && (
                                <div className={typeMismatch ? 'text-amber-400' : 'text-blue-400'}>
                                  {typeMismatch ? <AlertTriangle size={12} /> : <Check size={12} />}
                                </div>
                              )}
                            </button>
                          );
                        })}
                      </div>
                    )}
                  </div>
               </div>
               {resumeModelPath && (
                 <div className={`mt-2 flex items-center justify-between rounded-md px-2 py-1 border ${
                   isCrossTypeResume
                     ? 'bg-amber-950/20 border-amber-900/30'
                     : 'bg-blue-950/20 border-blue-900/30'
                 }`}>
                    <span className={`text-[10px] font-mono truncate mr-2 ${isCrossTypeResume ? 'text-amber-400' : 'text-blue-400'}`}>
                      {isCrossTypeResume ? '⚠ TYPE MISMATCH: ' : 'SELECTED: '}
                      {resumeModelPath.split('/').pop()}
                    </span>
                    <button onClick={() => { setResumeModelPath(null); setResumeModelType(null); }} className={isCrossTypeResume ? 'text-amber-400 hover:text-amber-300' : 'text-blue-400 hover:text-blue-300'}>
                      <X size={12}/>
                    </button>
                 </div>
               )}
            </div>
          </div>
        )}
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
            <LineChart data={computedHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis {...xAxisProps} />
              <YAxis stroke="#64748b" fontSize={12} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }}
                itemStyle={{ color: '#10b981' }}
                labelFormatter={tooltipLabelFormatter}
              />
              <Line type="monotone" dataKey="avg_survival" stroke="#10b981" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="min_survival" stroke="#334155" strokeWidth={1} dot={false} strokeDasharray="5 5" />
              <Line type="monotone" dataKey="rolling_avg_survival" stroke="#facc15" strokeWidth={1.5} dot={false} strokeDasharray="4 3" name="Rolling Avg" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl h-80">
          <h3 className="text-sm font-bold text-slate-500 uppercase mb-4 flex items-center">
            <ChartIcon size={16} className="mr-2" /> Training Loss
          </h3>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={computedHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis {...xAxisProps} />
              <YAxis {...lossYAxisProps} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }}
                itemStyle={{ color: '#f43f5e' }}
                labelFormatter={tooltipLabelFormatter}
                formatter={(val: number) => [Math.pow(10, val as number).toExponential(3), 'Loss']}
              />
              <Line type="monotone" dataKey="lossLog" name="Loss" stroke="#f43f5e" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="rolling_loss_log" stroke="#facc15" strokeWidth={1.5} dot={false} strokeDasharray="4 3" name="Rolling Avg" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl h-80">
          <h3 className="text-sm font-bold text-slate-500 uppercase mb-4 flex items-center">
            <ChartIcon size={16} className="mr-2" /> Average Return
          </h3>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={computedHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis {...xAxisProps} />
              <YAxis stroke="#64748b" fontSize={12} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }}
                itemStyle={{ color: '#3b82f6' }}
                labelFormatter={tooltipLabelFormatter}
              />
              <Line type="monotone" dataKey="avg_return" stroke="#3b82f6" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="rolling_avg_return" stroke="#facc15" strokeWidth={1.5} dot={false} strokeDasharray="4 3" name="Rolling Avg" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl h-80">
          <h3 className="text-sm font-bold text-slate-500 uppercase mb-4 flex items-center">
            <Target size={16} className="mr-2" /> Performance (Success & Survival)
          </h3>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={computedHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis {...xAxisProps} />
              <YAxis yAxisId="left" stroke="#f43f5e" fontSize={12} orientation="left" label={{ value: 'Wins', angle: -90, position: 'insideLeft', fill: '#f43f5e', fontSize: 10 }} />
              <YAxis yAxisId="right" stroke="#3b82f6" fontSize={12} orientation="right" label={{ value: 'Survival (s)', angle: 90, position: 'insideRight', fill: '#3b82f6', fontSize: 10 }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }}
                labelFormatter={tooltipLabelFormatter}
              />
              <Line yAxisId="left" type="monotone" dataKey="timeouts" name="Wins" stroke="#f43f5e" strokeWidth={2} dot={true} r={2} />
              <Line yAxisId="right" type="monotone" dataKey="avg_survival" name="Avg Survival" stroke="#3b82f6" strokeWidth={2} dot={false} />
              <Line yAxisId="left" type="monotone" dataKey="rolling_timeouts" stroke="#facc15" strokeWidth={1.5} dot={false} strokeDasharray="4 3" name="Rolling Wins" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl h-80">
          <h3 className="text-sm font-bold text-slate-500 uppercase mb-4 flex items-center">
            <ChartIcon size={16} className="mr-2" /> Q Values
          </h3>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={computedHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis {...xAxisProps} />
              <YAxis stroke="#64748b" fontSize={12} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }}
                labelFormatter={tooltipLabelFormatter}
              />
              <Line type="monotone" dataKey="mean_predicted_q" stroke="#22c55e" strokeWidth={2} dot={false} name="Mean Predicted Q" />
              <Line type="monotone" dataKey="mean_target_q" stroke="#38bdf8" strokeWidth={2} dot={false} name="Mean Target Q" />
              <Line type="monotone" dataKey="rolling_mean_predicted_q" stroke="#facc15" strokeWidth={1.5} dot={false} strokeDasharray="4 3" name="Predicted Rolling Avg" />
              <Line type="monotone" dataKey="rolling_mean_target_q" stroke="#fb7185" strokeWidth={1.5} dot={false} strokeDasharray="4 3" name="Target Rolling Avg" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl h-80">
          <h3 className="text-sm font-bold text-slate-500 uppercase mb-4 flex items-center">
            <ChartIcon size={16} className="mr-2" /> TD Error
          </h3>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={computedHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis {...xAxisProps} />
              <YAxis stroke="#64748b" fontSize={12} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }}
                labelFormatter={tooltipLabelFormatter}
              />
              <Line type="monotone" dataKey="mean_abs_td_error" stroke="#f97316" strokeWidth={2} dot={false} name="Mean Abs TD Error" />
              <Line type="monotone" dataKey="rolling_mean_abs_td_error" stroke="#facc15" strokeWidth={1.5} dot={false} strokeDasharray="4 3" name="Rolling Avg" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl h-80 lg:col-span-2">
          <h3 className="text-sm font-bold text-slate-500 uppercase mb-4 flex items-center">
            <ChartIcon size={16} className="mr-2" /> Sampled Batch Terminals
          </h3>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={computedHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis {...xAxisProps} />
              <YAxis stroke="#64748b" fontSize={12} domain={[0, 1]} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }}
                labelFormatter={tooltipLabelFormatter}
                formatter={(val: number) => [`${(val * 100).toFixed(1)}%`, 'Terminal Fraction']}
              />
              <Line type="monotone" dataKey="terminal_fraction" stroke="#a78bfa" strokeWidth={2} dot={false} name="Terminal Fraction" />
              <Line type="monotone" dataKey="rolling_terminal_fraction" stroke="#facc15" strokeWidth={1.5} dot={false} strokeDasharray="4 3" name="Rolling Avg" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}

function SettingInput({ label, value, onChange, min, max, step, isFloat }: { label: string, value: number, onChange: (v: number) => void, min: number, max: number, step?: number, isFloat?: boolean }) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between">
        <label className="text-xs font-medium text-slate-500">{label}</label>
        <span className="text-xs font-mono text-emerald-500">{isFloat ? value.toFixed(5) : value}</span>
      </div>
      <input 
        type="range" 
        min={min} max={max} step={step || 1}
        value={value}
        onChange={e => onChange(isFloat ? parseFloat(e.target.value) : parseInt(e.target.value))}
        className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
      />
    </div>
  )
}
