import { useState, useEffect } from 'react'
import Visualizer from './components/Visualizer'
import SettingsPanel from './components/SettingsPanel'
import TrainingPanel from './components/TrainingPanel'
import Console from './components/Console'
import ModelSelector from './components/ModelSelector'
import { Activity, Settings, Play, Pause, RotateCcw, Brain, User } from 'lucide-react'
import { SocketProvider, useSocket } from './contexts/SocketContext'

interface EngineStatus {
  running: boolean;
  ai_mode: boolean;
  has_model: boolean;
}

export interface TrainingProgress {
  episode: number;
  total_steps: number;
  epsilon: number;
  last_return: number;
  last_survival: number;
  last_evades: number;
  avg_survival: number;
  min_survival: number;
  avg_return: number;
  avg_evades: number;
  min_return: number;
  timeouts: number;
  loss: number;
  steps_per_second: number;
  global_best_survival: number;
}

function AppContent() {
  const [activeTab, setActiveTab] = useState<'game' | 'train' | 'settings'>('game')
  const { isConnected, sendMessage, subscribe } = useSocket()
  const [status, setStatus] = useState<EngineStatus>({ running: false, ai_mode: false, has_model: false })
  const [trainingHistory, setTrainingHistory] = useState<TrainingProgress[]>([])
  const [isTraining, setIsTraining] = useState(false)

  useEffect(() => {
    const unsubStatus = subscribe('Status', (data: EngineStatus) => {
      setStatus(data);
    });

    const unsubTraining = subscribe('Training', (data: TrainingProgress) => {
      setTrainingHistory(prev => [...prev.slice(-99), data]);
    });

    const checkTrainingStatus = async () => {
      try {
        const res = await fetch('/api/train/status');
        const running = await res.json();
        setIsTraining(running);
      } catch (e) {
        console.error('Failed to check training status', e);
      }
    };

    checkTrainingStatus();
    const interval = setInterval(checkTrainingStatus, 2000);

    return () => {
      unsubStatus();
      unsubTraining();
      clearInterval(interval);
    };
  }, [subscribe]);

  const togglePlay = () => {
    setStatus(prev => ({ ...prev, running: !prev.running }));
    sendMessage({ type: 'Control', data: { control: 'TogglePlay' } });
  };
  const resetGame = () => sendMessage({ type: 'Control', data: { control: 'Reset' } });
  const toggleAI = () => {
    setStatus(prev => ({ ...prev, ai_mode: !prev.ai_mode }));
    sendMessage({ type: 'Control', data: { control: 'ToggleAI' } });
  };

  return (
    <div className="flex h-screen w-full bg-slate-950 text-slate-100 overflow-hidden">
      {/* Sidebar */}
      <nav className="w-16 flex flex-col items-center py-4 bg-slate-900 border-r border-slate-800 space-y-8">
        <div className="text-blue-500 font-bold text-xl">E</div>
        <button 
          onClick={() => setActiveTab('game')}
          className={`p-2 rounded-lg transition ${activeTab === 'game' ? 'bg-blue-600' : 'hover:bg-slate-800'}`}
        >
          <Play size={24} />
        </button>
        <button 
          onClick={() => setActiveTab('train')}
          className={`p-2 rounded-lg transition ${activeTab === 'train' ? 'bg-blue-600' : 'hover:bg-slate-800'}`}
        >
          <Activity size={24} />
        </button>
        <button 
          onClick={() => setActiveTab('settings')}
          className={`p-2 rounded-lg transition ${activeTab === 'settings' ? 'bg-blue-600' : 'hover:bg-slate-800'}`}
        >
          <Settings size={24} />
        </button>
      </nav>

      {/* Main Content */}
      <main className="flex-1 flex flex-col relative min-w-0 overflow-hidden">
        <header className="h-12 border-b border-slate-800 flex items-center px-6 bg-slate-900 justify-between relative z-[60] pointer-events-auto">
          <h1 className="font-semibold text-slate-300 uppercase tracking-widest text-sm">
            Evades {activeTab === 'game' ? ':: Simulation' : `:: ${activeTab.toUpperCase()}`}
          </h1>
          {activeTab === 'game' && (
            <div className="flex items-center space-x-4">
              <button 
                onClick={resetGame}
                className="p-1.5 hover:bg-slate-800 rounded-md text-slate-400 transition flex items-center space-x-2 text-xs relative z-[70]"
                title="Reset Simulation"
              >
                <RotateCcw size={14} />
                <span>Reset</span>
              </button>
              <button 
                onClick={togglePlay}
                className={`px-3 py-1.5 rounded-md transition flex items-center space-x-2 text-xs font-bold relative z-[70] ${status.running ? 'bg-rose-600 hover:bg-rose-700' : 'bg-emerald-600 hover:bg-emerald-700'}`}
              >
                {status.running ? <Pause size={14} /> : <Play size={14} />}
                <span>{status.running ? 'PAUSE' : 'PLAY'}</span>
              </button>
            </div>
          )}
        </header>

        <div className="flex-1 overflow-y-auto p-6 relative z-10">
          {activeTab === 'game' && (
            <div className="flex flex-col space-y-6">
              <div className="bg-slate-900 rounded-xl border border-slate-800 p-2 shadow-2xl overflow-hidden self-center shrink-0">
                <Visualizer isRunning={status.running} isAiMode={status.ai_mode} />
              </div>
              <div className="grid grid-cols-3 gap-6 pb-12">
                 <div className="flex flex-col space-y-6">
                    <div className="bg-slate-900 p-4 rounded-lg border border-slate-800">
                        <div className="flex justify-between items-center mb-4">
                            <h3 className="text-xs font-bold text-slate-500 uppercase tracking-tighter">Control Mode</h3>
                            <button 
                                onClick={toggleAI}
                                className={`flex items-center space-x-2 px-2 py-1 rounded text-[10px] font-bold transition ${status.ai_mode ? 'bg-blue-600 text-white' : 'bg-slate-800 text-slate-400'}`}
                            >
                                {status.ai_mode ? <Brain size={12} /> : <User size={12} />}
                                <span>{status.ai_mode ? 'AI_DQN' : 'MANUAL'}</span>
                            </button>
                        </div>
                        <p className="text-[10px] text-slate-400 leading-relaxed">
                            {status.ai_mode 
                                ? "In AI Mode, the agent uses the pre-trained DQN model for decision making. Manual input is ignored." 
                                : "Manual control enabled. Use WASD or Arrow keys to navigate the arena."}
                        </p>
                    </div>
                    <div className="bg-slate-900 p-4 rounded-lg border border-slate-800">
                        <h3 className="text-xs font-bold text-slate-500 mb-2 uppercase tracking-tighter">System Status</h3>
                        <div className="space-y-3">
                            <div className={`${isConnected ? 'text-emerald-500' : 'text-rose-500'} text-[10px] flex items-center font-mono uppercase`}>
                                <span className={`mr-2 ${isConnected ? 'animate-pulse' : ''}`}>●</span> WS_{isConnected ? 'CONNECTED' : 'DISCONNECTED'}
                            </div>
                            <div className={`${status.running ? 'text-blue-500' : 'text-slate-500'} text-[10px] flex items-center font-mono uppercase`}>
                                <span className="mr-2">●</span> ENGINE_{status.running ? 'RUNNING_60HZ' : 'PAUSED'}
                            </div>
                            <div className={`${status.has_model ? 'text-purple-500' : 'text-rose-500'} text-[10px] flex items-center font-mono uppercase`}>
                                <span className="mr-2">●</span> MODEL_{status.has_model ? 'LOADED_BEST_JSON' : 'NOT_FOUND'}
                            </div>
                            <div className="text-slate-600 text-[9px] pt-2 border-t border-slate-800/50">
                                Models saved to:<br/>
                                <code className="text-slate-500">training_runs/web_run/</code>
                            </div>
                        </div>
                    </div>
                    <div className="bg-slate-900 p-4 rounded-lg border border-slate-800 shrink-0">
                        <ModelSelector />
                    </div>
                 </div>
                 <div className="col-span-2 min-h-[400px]">
                    <Console />
                 </div>
              </div>
            </div>
          )}
          
          {activeTab === 'settings' && <SettingsPanel />}
          {activeTab === 'train' && (
            <TrainingPanel 
              history={trainingHistory} 
              isRunning={isTraining} 
              setIsRunning={setIsTraining}
            />
          )}
        </div>
      </main>
    </div>
  )
}

export default function App() {
  return (
    <SocketProvider>
      <AppContent />
    </SocketProvider>
  )
}
