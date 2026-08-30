import React from "react";
import { Cpu, HardDrive } from "lucide-react";

export default function Header({ activeRepo }) {
  return (
    <header className="h-14 border-b border-slate-800 bg-[#0F1623] px-6 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <span className="text-xs font-mono text-slate-400">Context:</span>
        <span className="text-xs font-mono bg-slate-800 px-2.5 py-1 rounded-md text-slate-200 border border-slate-700">
          {activeRepo?.repoName || activeRepo?.repo_url || "No Repository Selected"}
        </span>
      </div>

      <div className="flex items-center gap-4 text-xs font-mono text-slate-400">
        <div className="flex items-center gap-1.5">
          <Cpu className="w-3.5 h-3.5 text-blue-400" />
          <span>AST Indexed</span>
        </div>
        <div className="flex items-center gap-1.5">
          <HardDrive className="w-3.5 h-3.5 text-emerald-400" />
          <span>Vector Store Ready</span>
        </div>
      </div>
    </header>
  );
}
