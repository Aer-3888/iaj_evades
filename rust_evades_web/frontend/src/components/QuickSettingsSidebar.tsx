import { useState, useEffect } from 'react'
import { RefreshCcw, Zap, Target, Settings2, Share2, Copy, RotateCcw as ReplayIcon, FileText } from 'lucide-react'
import { useSocket } from '../contexts/SocketContext'
import { useToast } from '../contexts/ToastContext'

interface GameConfig {
  map_design: "Open" | "Closed" | "Arena";
  player_speed: number;
  enemy_speed: number;
  player_radius: number;
  enemy_radius: number;
  max_episode_time: number;
  enemy_spawn_interval_min: number;
  enemy_spawn_interval_max: number;
  default_seed: number;
  render_fps: number;
  show_raycast: boolean;
  vision_only: boolean;
}

export default function QuickSettingsSidebar() {
  const [config, setConfig] = useState<GameConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [seedHistory, setSeedHistory] = useState<number[]>([]);
  const { showToast } = useToast();
  const { subscribe } = useSocket();

  useEffect(() => {
    const unsubscribe = subscribe('Game', (data: { base_seed: number }) => {
      if (data.base_seed !== undefined) {
        setSeedHistory(prev => {
          if (prev[0] === data.base_seed) return prev;
          return [data.base_seed, ...prev.filter(s => s !== data.base_seed)].slice(0, 50);
        });
      }
    });
    return () => unsubscribe();
  }, [subscribe]);

  const fetchConfig = async () => {
    try {
      const res = await fetch('/api/config');
      const data = await res.json();
      setConfig(data);
    } catch (e) {
      console.error('Failed to fetch config', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchConfig();
  }, []);

  const handleUpdate = async (newConfig: GameConfig) => {
    setConfig(newConfig);
    try {
      await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newConfig),
      });
    } catch (e) {
      showToast('Failed to update config', 'error');
    }
  };

  if (loading || !config) return null;

  return (
    <aside className="w-80 bg-slate-900 border-l border-slate-800 flex flex-col h-full overflow-y-auto">
      <div className="p-4 border-b border-slate-800 flex items-center justify-between bg-slate-900/50 sticky top-0 z-10 backdrop-blur-sm">
        <div className="flex items-center text-blue-400 font-bold text-sm uppercase tracking-wider">
          <Settings2 size={16} className="mr-2" /> Quick Config
        </div>
        <button 
          onClick={fetchConfig}
          className="p-1.5 hover:bg-slate-800 rounded-md text-slate-400 transition"
          title="Refresh from server"
        >
          <RefreshCcw size={14} />
        </button>
      </div>

      <div className="p-4 space-y-8">
        {/* Engine Settings */}
        <section className="space-y-6">
          <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Simulation Engine</label>
          <QuickSlider 
            label="Game Speed (FPS)" 
            value={config.render_fps} 
            min={10} max={240} step={1}
            onChange={(v) => handleUpdate({...config, render_fps: v})}
          />
          <div className="flex items-center justify-between pt-2">
            <label className="text-[11px] font-medium text-slate-400">Show Raycasting</label>
            <button
              onClick={() => handleUpdate({...config, show_raycast: !config.show_raycast})}
              className={`w-8 h-4 rounded-full transition-colors relative ${config.show_raycast ? 'bg-blue-600' : 'bg-slate-800'}`}
            >
              <div className={`absolute top-0.5 w-3 h-3 rounded-full bg-white transition-all ${config.show_raycast ? 'left-4.5' : 'left-0.5'}`} />
            </button>
          </div>
          <div className="flex items-center justify-between pt-1">
            <label className="text-[11px] font-medium text-slate-400">Vision Only Mode</label>
            <button
              onClick={() => handleUpdate({...config, vision_only: !config.vision_only})}
              className={`w-8 h-4 rounded-full transition-colors relative ${config.vision_only ? 'bg-blue-600' : 'bg-slate-800'}`}
            >
              <div className={`absolute top-0.5 w-3 h-3 rounded-full bg-white transition-all ${config.vision_only ? 'left-4.5' : 'left-0.5'}`} />
            </button>
          </div>
        </section>

        {/* Map Design */}
        <section className="space-y-3">
          <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Environment</label>
          <div className="grid grid-cols-3 bg-slate-950 p-1 rounded-lg border border-slate-800/50 gap-1">
            <button
              onClick={() => handleUpdate({...config, map_design: "Open"})}
              className={`flex-1 py-1.5 text-xs rounded-md transition font-medium ${config.map_design === "Open" ? "bg-blue-600 text-white shadow-sm" : "text-slate-500 hover:text-slate-300"}`}
            >
              Open
            </button>
            <button
              onClick={() => handleUpdate({...config, map_design: "Arena"})}
              className={`flex-1 py-1.5 text-xs rounded-md transition font-medium ${config.map_design === "Arena" ? "bg-blue-600 text-white shadow-sm" : "text-slate-500 hover:text-slate-300"}`}
            >
              Arena
            </button>
            <button
              onClick={() => handleUpdate({...config, map_design: "Closed"})}
              className={`flex-1 py-1.5 text-xs rounded-md transition font-medium ${config.map_design === "Closed" ? "bg-blue-600 text-white shadow-sm" : "text-slate-500 hover:text-slate-300"}`}
            >
              Closed
            </button>
          </div>
        </section>

        {/* Seed */}
        <section className="space-y-3">
          <div className="flex justify-between items-center">
            <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Simulation Seed</label>
            <span className="text-[10px] font-mono text-slate-600">0 = Random</span>
          </div>
          <div className="flex gap-2">
            <input 
              type="number"
              value={config.default_seed}
              onChange={(e) => handleUpdate({...config, default_seed: parseInt(e.target.value) || 0})}
              className="flex-1 bg-slate-950 border border-slate-800 rounded-lg px-2 py-1.5 text-xs font-mono text-blue-400 focus:outline-none focus:border-blue-500"
            />
            <button 
              onClick={() => handleUpdate({...config, default_seed: Math.floor(Math.random() * 1000)})}
              className="px-2 py-1.5 bg-slate-800 hover:bg-slate-700 rounded-lg text-slate-300 transition"
              title="Randomize"
            >
              <Share2 size={14} />
            </button>
          </div>
        </section>

        {/* Physics Sliders */}
        <section className="space-y-6">
          <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Entities Physics</label>
          
          <QuickSlider 
            label="Player Speed" 
            icon={<Zap size={12} />}
            value={config.player_speed} 
            min={100} max={600} step={10}
            onChange={(v) => handleUpdate({...config, player_speed: v})}
          />
          
          <QuickSlider 
            label="Enemy Speed" 
            icon={<Zap size={12} className="text-rose-400" />}
            value={config.enemy_speed} 
            min={100} max={600} step={10}
            onChange={(v) => handleUpdate({...config, enemy_speed: v})}
          />

          <QuickSlider 
            label="Enemy Radius" 
            icon={<Target size={12} className="text-rose-400" />}
            value={config.enemy_radius} 
            min={5} max={40} step={1}
            onChange={(v) => handleUpdate({...config, enemy_radius: v})}
          />
        </section>

        {/* Spawning */}
        <section className="space-y-6">
          <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Enemy Spawning</label>
          <QuickSlider 
            label="Spawn Int. (min)" 
            value={config.enemy_spawn_interval_min} 
            min={0.05} max={1.0} step={0.01}
            onChange={(v) => handleUpdate({...config, enemy_spawn_interval_min: v, enemy_spawn_interval_max: Math.max(v, config.enemy_spawn_interval_max)})}
          />
          <QuickSlider 
            label="Spawn Int. (max)" 
            value={config.enemy_spawn_interval_max} 
            min={0.05} max={1.0} step={0.01}
            onChange={(v) => handleUpdate({...config, enemy_spawn_interval_max: v, enemy_spawn_interval_min: Math.min(v, config.enemy_spawn_interval_min)})}
          />
        </section>

        {/* Seed History */}
        <section className="space-y-4 pt-4 border-t border-slate-800">
          <div className="flex justify-between items-center">
            <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Recent Seeds History</label>
            {seedHistory.length > 0 && (
              <button 
                onClick={() => {
                  navigator.clipboard.writeText(seedHistory.join('\n'));
                  showToast('All seeds copied to clipboard', 'success');
                }}
                className="flex items-center text-[10px] text-blue-400 hover:text-blue-300 transition font-bold"
              >
                <FileText size={12} className="mr-1" /> COPY ALL
              </button>
            )}
          </div>
          
          {seedHistory.length === 0 ? (
             <div className="text-[10px] text-slate-600 italic text-center py-2 bg-slate-950/30 rounded border border-dashed border-slate-800/50">
               No seeds recorded yet...
             </div>
          ) : (
            <div className="max-h-64 overflow-y-auto space-y-1.5 pr-1 custom-scrollbar">
              {seedHistory.map((seed, i) => (
                <div key={seed} className="group flex items-center justify-between bg-slate-950/50 hover:bg-slate-950 border border-slate-800/50 rounded-lg px-2 py-1.5 transition">
                  <span className={`text-[11px] font-mono ${i === 0 ? 'text-emerald-400 font-bold' : 'text-slate-400'}`}>
                    {seed}
                    {i === 0 && <span className="ml-2 text-[8px] opacity-50 uppercase tracking-tighter">(LIVE)</span>}
                  </span>
                  <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button 
                      onClick={() => {
                        navigator.clipboard.writeText(seed.toString());
                        showToast(`Seed ${seed} copied`, 'info');
                      }}
                      className="p-1 hover:bg-slate-800 rounded text-slate-400 hover:text-blue-400"
                      title="Copy seed"
                    >
                      <Copy size={12} />
                    </button>
                    <button 
                      onClick={() => handleUpdate({...config!, default_seed: seed})}
                      className="p-1 hover:bg-slate-800 rounded text-slate-400 hover:text-emerald-400"
                      title="Re-run this seed"
                    >
                      <ReplayIcon size={12} />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>

      <div className="mt-auto p-4 border-t border-slate-800 bg-slate-900/30">
        <div className="text-[10px] text-slate-500 text-center italic">
          Changes reset the simulation.
        </div>
      </div>
    </aside>
  )
}

function QuickSlider({ label, icon, value, min, max, step, onChange }: { 
  label: string, 
  icon?: React.ReactNode,
  value: number, 
  onChange: (v: number) => void, 
  min: number, 
  max: number, 
  step: number 
}) {
  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center">
        <div className="flex items-center text-[11px] font-medium text-slate-400">
          {icon && <span className="mr-1.5">{icon}</span>}
          {label}
        </div>
        <input 
          type="number"
          value={value}
          min={min}
          max={max}
          step={step}
          onChange={(e) => onChange(parseFloat(e.target.value) || min)}
          className="w-16 bg-slate-950 border border-slate-800 rounded px-1.5 py-0.5 text-[10px] font-mono text-blue-400 text-right focus:outline-none focus:border-blue-500"
        />
      </div>
      <input 
        type="range" 
        min={min} max={max} step={step} 
        value={value} 
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
      />
    </div>
  )
}
