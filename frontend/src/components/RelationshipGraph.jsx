import React from "react";

export default function RelationshipGraph({ callers = [], targetSymbol, callees = [] }) {
  if (!targetSymbol && !callers.length && !callees.length) return null;

  return (
    <div className="bg-[#131B2E] border border-slate-800 rounded-xl p-5 flex flex-col gap-3">
      <h3 className="text-xs font-mono uppercase tracking-wider text-slate-400">AST Relationship Graph</h3>
      <div className="grid grid-cols-3 gap-4 text-xs font-mono">
        <div className="bg-[#0B0F17] p-3 rounded-lg border border-slate-800">
          <span className="text-[10px] text-slate-500 uppercase block mb-2">Callers</span>
          {callers.map((c, i) => <div key={i} className="text-emerald-400 truncate">&lt;-- {c}</div>)}
        </div>
        <div className="bg-[#0B0F17] p-3 rounded-lg border border-blue-600/50 flex items-center justify-center font-bold text-blue-400 truncate">
          {targetSymbol || "Target Symbol"}
        </div>
        <div className="bg-[#0B0F17] p-3 rounded-lg border border-slate-800">
          <span className="text-[10px] text-slate-500 uppercase block mb-2">Callees</span>
          {callees.map((c, i) => <div key={i} className="text-purple-400 truncate">--&gt; {c}</div>)}
        </div>
      </div>
    </div>
  );
}
