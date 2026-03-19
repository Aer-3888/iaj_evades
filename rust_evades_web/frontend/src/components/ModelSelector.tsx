import { useState, useEffect } from 'react'
import { FileJson, RefreshCw, Check } from 'lucide-react'

interface ModelInfo {
  name: string;
  path: string;
}

export default function ModelSelector() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [activePath, setActivePath] = useState<string | null>(null);

  const fetchModels = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/models');
      const data = await res.json();
      setModels(data);
    } catch (e) {
      console.error('Failed to fetch models', e);
    } finally {
      setLoading(false);
    }
  };

  const loadModel = async (path: string) => {
    setLoading(true);
    try {
      const res = await fetch('/api/models/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path }),
      });
      if (res.ok) {
        setActivePath(path);
        alert('Model loaded successfully!');
      } else {
        alert('Failed to load model');
      }
    } catch (e) {
      alert('Error loading model');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest">Brain Models (.json)</h3>
        <button 
          onClick={fetchModels}
          disabled={loading}
          className="p-1 hover:bg-slate-800 rounded text-slate-400 transition"
          title="Refresh model list"
        >
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
        <div className="max-h-60 overflow-y-auto custom-scrollbar">
          {models.length === 0 ? (
            <div className="p-8 text-center text-slate-600 italic text-sm">
              No JSON models found in training_runs/
            </div>
          ) : (
            <div className="divide-y divide-slate-800/50">
              {models.map((model) => (
                <button
                  key={model.path}
                  onClick={() => loadModel(model.path)}
                  disabled={loading}
                  className={`w-full flex items-center justify-between px-4 py-3 text-left transition hover:bg-slate-800/50 ${activePath === model.path ? 'bg-blue-900/20' : ''}`}
                >
                  <div className="flex items-center min-w-0 mr-4">
                    <FileJson size={16} className={`mr-3 shrink-0 ${activePath === model.path ? 'text-blue-400' : 'text-slate-500'}`} />
                    <span className={`text-sm truncate ${activePath === model.path ? 'text-blue-300 font-medium' : 'text-slate-300'}`}>
                      {model.name}
                    </span>
                  </div>
                  {activePath === model.path ? (
                    <div className="bg-blue-500/20 text-blue-400 p-1 rounded">
                      <Check size={12} />
                    </div>
                  ) : (
                    <span className="text-[10px] text-slate-600 font-mono">LOAD</span>
                  )}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
      <p className="text-[10px] text-slate-500 italic">
        Select a model to update the simulation brain in real-time.
      </p>
    </div>
  )
}
