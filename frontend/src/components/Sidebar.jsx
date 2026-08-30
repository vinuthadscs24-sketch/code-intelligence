import React from 'react';
import { Terminal, MessageSquare, FolderTree, GitCommit, AlertTriangle, Zap } from 'lucide-react';

export default function Sidebar({ activeTab, setActiveTab, activeRepo, onSwitchRepo }) {
  const navItems = [
    { id: 'overview', label: 'Overview', icon: Terminal },
    { id: 'ask', label: 'Ask Codebase', icon: MessageSquare },
    { id: 'explorer', label: 'Code Explorer', icon: FolderTree },
    { id: 'git', label: 'Git History', icon: GitCommit },
    { id: 'impact', label: 'Impact Analysis', icon: AlertTriangle },
  ];

  return (
    <aside className="w-64 bg-[#131B2E] border-r border-slate-800 flex flex-col justify-between p-4">
      <div className="flex flex-col gap-6">
        
        {/* Logo & Active Repo Badge */}
        <div className="flex flex-col gap-2 px-2 pt-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-7 h-7 rounded-lg bg-blue-600 flex items-center justify-center font-bold text-white">
                <Zap className="w-4 h-4" />
              </div>
              <span className="font-bold text-xs text-white">Code Intelligence</span>
            </div>
            {onSwitchRepo && (
              <button 
                onClick={onSwitchRepo} 
                className="text-[10px] text-slate-400 hover:text-white underline font-mono"
              >
                Switch Repo
              </button>
            )}
          </div>

          <div className="bg-[#0B0F17] p-2.5 rounded-lg border border-slate-800 flex flex-col gap-1">
            <span className="text-[10px] font-mono text-slate-500 uppercase">Active Target</span>
            <span className="text-xs font-mono text-blue-400 font-bold truncate">
              📦 {activeRepo?.repoName || activeRepo?.repo_url?.split('/').pop() || "spring-petclinic"}
            </span>
          </div>
        </div>

        {/* Navigation Items */}
        <nav className="flex flex-col gap-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const active = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`flex items-center gap-3 px-3 py-2 rounded-lg text-xs font-mono transition ${
                  active ? 'bg-blue-600 text-white font-semibold' : 'text-slate-400 hover:bg-slate-800/60 hover:text-slate-200'
                }`}
              >
                <Icon className="w-4 h-4" />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Engine Status Footer */}
      <div className="bg-[#0B0F17] p-3 rounded-lg border border-slate-800 text-[11px] font-mono text-slate-400 flex flex-col gap-1">
        <div className="flex justify-between">
          <span>Engine Status:</span>
          <span className="text-emerald-400">Online</span>
        </div>
      </div>
    </aside>
  );
}