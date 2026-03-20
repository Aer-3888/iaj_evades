import { useState, useEffect } from 'react'
import { Save, RefreshCcw } from 'lucide-react'
import ModelSelector from './ModelSelector'
import { useToast } from '../contexts/ToastContext'

interface GameConfig {
  map_design: "Open" | "Closed";
  player_speed: number;
  enemy_speed: number;
  player_radius: number;
  enemy_radius: number;
  max_episode_time: number;
  enemy_spawn_interval_min: number;
  enemy_spawn_interval_max: number;
  default_seed: number;
}

export default function SettingsPanel() {
  const [config, setConfig] = useState<GameConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const { showToast } = useToast();

  const fetchConfig = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/config');
      const data = await res.json();
      setConfig(data);
    } catch (e) {
      console.error('Failed to fetch config', e);
      showToast('Failed to load settings', 'error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchConfig();
  }, []);

  const handleSave = async () => {
    if (!config) return;
    try {
      await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
      showToast('Settings applied successfully!', 'success');
    } catch (e) {
      showToast('Failed to save settings', 'error');
    }
  };

  if (loading) return <div className="text-slate-500 font-mono animate-pulse">Loading configuration...</div>;
  if (!config) return <div className="text-rose-500">Failed to load config. Make sure the backend is running.</div>;

  return (
    <div className="space-y-8 pb-12">
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 md:p-8 shadow-lg">
        <div className="flex flex-col md:flex-row justify-between md:items-center mb-8 border-b border-slate-800 pb-6 gap-4">
          <div>
            <h2 className="text-xl font-bold text-blue-400">Game Physics & Parameters</h2>
            <p className="text-sm text-slate-400 mt-1">Configure the environment and entity rules.</p>
          </div>
          <div className="flex space-x-3">
            <button 
              onClick={fetchConfig}
              className="flex items-center px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-sm font-medium transition"
            >
              <RefreshCcw size={16} className="mr-2" /> Reset
            </button>
            <button 
              onClick={handleSave}
              className="flex items-center px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded-lg text-sm font-bold transition shadow-lg shadow-blue-900/20"
            >
              <Save size={16} className="mr-2" /> Apply Changes
            </button>
          </div>
        </div>

        <div className="space-y-12">
          {/* World Settings Section */}
          <section>
            <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-6">World Configuration</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-8">
              <div className="space-y-4">
                <label className="text-sm font-medium text-slate-400">Map Design</label>
                <div className="flex space-x-4">
                  <button
                    onClick={() => setConfig({...config, map_design: "Open"})}
                    className={`px-4 py-2 rounded-lg text-sm transition font-medium ${config.map_design === "Open" ? "bg-blue-600 text-white shadow-md shadow-blue-900/30" : "bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-200"}`}
                  >
                    Open (Infinite)
                  </button>
                  <button
                    onClick={() => setConfig({...config, map_design: "Closed"})}
                    className={`px-4 py-2 rounded-lg text-sm transition font-medium ${config.map_design === "Closed" ? "bg-blue-600 text-white shadow-md shadow-blue-900/30" : "bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-200"}`}
                  >
                    Closed (Corridor)
                  </button>
                </div>
              </div>
              <SettingItem 
                label="Max Episode Time (s)" 
                value={config.max_episode_time} 
                onChange={(v: number) => setConfig({...config, max_episode_time: v})}
                min={5} max={120} step={5}
              />
              <div className="space-y-4">
                <div className="flex justify-between items-end">
                  <label className="text-sm font-medium text-slate-400">Simulation Seed</label>
                  <span className="text-xs font-mono text-slate-500 italic">0 = Random</span>
                </div>
                <div className="flex space-x-2">
                   <input 
                      type="number"
                      value={config.default_seed}
                      onChange={(e) => setConfig({...config, default_seed: parseInt(e.target.value) || 0})}
                      className="bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm font-mono text-blue-400 w-full focus:outline-none focus:border-blue-500 transition"
                      placeholder="Enter seed (0 for random)"
                    />
                    <button 
                      onClick={() => setConfig({...config, default_seed: 0})}
                      className={`px-3 py-2 rounded-lg text-xs font-bold transition whitespace-nowrap ${config.default_seed === 0 ? "bg-blue-600 text-white" : "bg-slate-800 text-slate-400"}`}
                    >
                      Random
                    </button>
                </div>
              </div>
            </div>
          </section>

          {/* Entities Settings Section */}
          <section className="border-t border-slate-800/60 pt-8">
            <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-6">Entities Physics</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-8">
              <SettingItem 
                label="Player Speed" 
                value={config.player_speed} 
                onChange={(v: number) => setConfig({...config, player_speed: v})}
                min={100} max={1000} step={10}
              />
              <SettingItem 
                label="Enemy Speed" 
                value={config.enemy_speed} 
                onChange={(v: number) => setConfig({...config, enemy_speed: v})}
                min={100} max={1000} step={10}
              />
              <SettingItem 
                label="Player Radius" 
                value={config.player_radius} 
                onChange={(v: number) => setConfig({...config, player_radius: v})}
                min={5} max={50} step={1}
              />
              <SettingItem 
                label="Enemy Radius" 
                value={config.enemy_radius} 
                onChange={(v: number) => setConfig({...config, enemy_radius: v})}
                min={5} max={50} step={1}
              />
            </div>
          </section>

          {/* Spawn Settings Section */}
          <section className="border-t border-slate-800/60 pt-8">
            <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-6">Enemy Spawning</h3>
            <div className="space-y-6">
              <div className="flex flex-col space-y-2">
                <div className="flex justify-between items-end">
                  <label className="text-sm font-medium text-slate-400">Spawn Interval (min/max)</label>
                  <span className="text-sm font-mono text-blue-400 bg-blue-900/20 px-2 py-1 rounded">
                    {config.enemy_spawn_interval_min.toFixed(2)}s — {config.enemy_spawn_interval_max.toFixed(2)}s
                  </span>
                </div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-8 bg-slate-800/30 p-6 rounded-xl border border-slate-800/50">
                 <div className="space-y-3">
                   <div className="flex justify-between text-xs font-semibold text-slate-500">
                      <span>MINIMUM INTERVAL</span>
                      <span>{config.enemy_spawn_interval_min.toFixed(2)}s</span>
                   </div>
                   <input 
                      type="range" 
                      min={0.05} max={2.0} step={0.05} 
                      value={config.enemy_spawn_interval_min} 
                      onChange={(e) => {
                        const val = parseFloat(e.target.value);
                        setConfig({...config, enemy_spawn_interval_min: val, enemy_spawn_interval_max: Math.max(val, config.enemy_spawn_interval_max)});
                      }}
                      className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
                    />
                 </div>
                 <div className="space-y-3">
                   <div className="flex justify-between text-xs font-semibold text-slate-500">
                      <span>MAXIMUM INTERVAL</span>
                      <span>{config.enemy_spawn_interval_max.toFixed(2)}s</span>
                   </div>
                   <input 
                      type="range" 
                      min={0.05} max={2.0} step={0.05} 
                      value={config.enemy_spawn_interval_max} 
                      onChange={(e) => {
                        const val = parseFloat(e.target.value);
                        setConfig({...config, enemy_spawn_interval_max: val, enemy_spawn_interval_min: Math.min(val, config.enemy_spawn_interval_min)});
                      }}
                      className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
                    />
                 </div>
              </div>
            </div>
          </section>

          {/* Brain Model Selection Section */}
          <section className="border-t border-slate-800/60 pt-8">
            <ModelSelector />
          </section>
        </div>
      </div>
    </div>
  )
}

function SettingItem({ label, value, onChange, min, max, step }: { label: string, value: number, onChange: (v: number) => void, min: number, max: number, step: number }) {
  return (
    <div className="space-y-3">
      <div className="flex justify-between items-end">
        <label className="text-sm font-medium text-slate-400">{label}</label>
        <span className="text-sm font-mono text-blue-400 bg-blue-900/20 px-2 py-1 rounded">{value}</span>
      </div>
      <input 
        type="range" 
        min={min} max={max} step={step} 
        value={value} 
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
      />
    </div>
  )
}
