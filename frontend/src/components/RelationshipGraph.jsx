import React from "react";

export default function RelationshipGraph({ callers = [], targetSymbol, callees = [] }) {
  if (!targetSymbol && !callers.length && !callees.length) return null;

  return (
    <div className="bg-[#131B2E] border border-slate-800 rounded-xl p-5 flex flex-col gap-3">
      <h3 className="text-xs font-mono uppercase tracking-wider text-slate-400">
        AST Relationship Graph
      </h3>
      <div className="grid grid-cols-3 gap-4 text-xs font-mono">
        {/* Callers Column */}
        <div className="bg-[#0B0F17] p-3 rounded-lg border border-slate-800 flex flex-col max-h-48 overflow-y-auto">
          <span className="text-[10px] text-slate-500 uppercase block mb-2 font-semibold">
            Callers ({callers.length})
          </span>
          {callers.length > 0 ? (
            callers.map((c, i) => (
              <div key={i} className="text-emerald-400 truncate py-0.5" title={c}>
                &lt;-- {c}
              </div>
            ))
          ) : (
            <span className="text-slate-600 italic text-[11px]">None</span>
          )}
        </div>

        {/* Target Symbol Column */}
        <div className="bg-[#0B0F17] p-3 rounded-lg border border-blue-600/50 flex flex-col items-center justify-center font-bold text-blue-400 text-center break-all">
          <span className="text-[10px] text-slate-500 uppercase block mb-1 font-normal">
            Target
          </span>
          {targetSymbol || "Target Symbol"}
        </div>

        {/* Callees Column */}
        <div className="bg-[#0B0F17] p-3 rounded-lg border border-slate-800 flex flex-col max-h-48 overflow-y-auto">
          <span className="text-[10px] text-slate-500 uppercase block mb-2 font-semibold">
            Callees ({callees.length})
          </span>
          {callees.length > 0 ? (
            callees.map((c, i) => (
              <div key={i} className="text-purple-400 truncate py-0.5" title={c}>
                --&gt; {c}
              </div>
            ))
          ) : (
            <span className="text-slate-600 italic text-[11px]">None</span>
          )}
        </div>
      </div>
    </div>
  );
}