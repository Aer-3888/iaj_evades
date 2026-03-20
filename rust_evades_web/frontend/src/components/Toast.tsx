import { X, CheckCircle, AlertCircle, Info } from 'lucide-react';
import { useToast } from '../contexts/ToastContext';

export default function ToastContainer() {
  const { toasts, removeToast } = useToast();

  return (
    <div className="fixed bottom-6 right-6 z-[100] flex flex-col gap-3 pointer-events-none">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className={`
            pointer-events-auto flex items-center gap-3 px-4 py-3 rounded-lg shadow-2xl border
            animate-in slide-in-from-right duration-300
            ${toast.type === 'success' ? 'bg-emerald-900/90 border-emerald-500 text-emerald-100' : ''}
            ${toast.type === 'error' ? 'bg-rose-900/90 border-rose-500 text-rose-100' : ''}
            ${toast.type === 'info' ? 'bg-slate-900/90 border-slate-500 text-slate-100' : ''}
          `}
        >
          {toast.type === 'success' && <CheckCircle size={18} className="text-emerald-400" />}
          {toast.type === 'error' && <AlertCircle size={18} className="text-rose-400" />}
          {toast.type === 'info' && <Info size={18} className="text-blue-400" />}
          
          <span className="text-sm font-medium pr-2">{toast.message}</span>
          
          <button
            onClick={() => removeToast(toast.id)}
            className="p-1 hover:bg-white/10 rounded-md transition-colors ml-auto"
          >
            <X size={14} />
          </button>
        </div>
      ))}
    </div>
  );
}
