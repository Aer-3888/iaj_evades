import React from 'react';

interface StatCardProps {
  label: string;
  value: string | number;
  icon?: React.ReactNode;
  description?: string;
  color?: string;
}

export default function StatCard({ label, value, icon, description, color = "text-emerald-400" }: StatCardProps) {
  return (
    <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl shadow-md">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider">{label}</h3>
        {icon && <div className="text-slate-600">{icon}</div>}
      </div>
      <div className={`text-2xl font-mono font-bold ${color}`}>
        {value}
      </div>
      {description && (
        <p className="text-[10px] text-slate-500 mt-1 uppercase tracking-tighter">
          {description}
        </p>
      )}
    </div>
  );
}
