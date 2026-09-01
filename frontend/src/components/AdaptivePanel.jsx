import React from 'react';
import { GitCommit, AlertTriangle, Network, Layers, FileCode } from 'lucide-react';

export default function AdaptivePanel({ response }) {
  if (!response) return null;

  const { panel_type, panel_data, evidence, retrieval_trace } = response;

  return (
    <div className="space-y-6 mt-4 font-mono">
      {/* 1. DEPENDENCY / EXECUTION GRAPH PANEL */}
      {(panel_type === "dependency_graph" || panel_type === "execution_graph") && (
        <div className="bg-[#131B2E] border border-slate-800 rounded-xl p-5">
          <div className="flex items-center gap-2 text-blue-400 font-bold text-xs mb-4">
            <Network className="w-4 h-4" />
            <span>CALL RELATIONSHIP GRAPH ({panel_data.target})</span>
          </div>
          <div className="flex flex-col gap-3">
            {panel_data.nodes?.map((node) => (
              <div 
                key={node.id} 
                className={`p-3 rounded-lg border text-xs flex justify-between items-center ${
                  node.type === 'target' 
                    ? 'bg-blue-950/80 border-blue-500 text-blue-200 font-bold' 
                    : 'bg-[#0B0F17] border-slate-800 text-slate-300'
                }`}
              >
                <span>{node.label}</span>
                <span className="text-[10px] uppercase px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                  {node.type}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 2. IMPACT SUMMARY PANEL */}
      {panel_type === "impact_summary" && (
        <div className="bg-[#131B2E] border border-slate-800 rounded-xl p-5">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2 text-amber-400 font-bold text-xs">
              <AlertTriangle className="w-4 h-4" />
              <span>IMPACT BLAST RADIUS</span>
            </div>
            <span className={`text-xs font-bold px-2.5 py-1 rounded ${
              panel_data.risk_level === 'HIGH' ? 'bg-red-950 text-red-400 border border-red-800' : 'bg-amber-950 text-amber-400'
            }`}>
              {panel_data.risk_level} RISK
            </span>
          </div>

          <div className="grid grid-cols-3 gap-3 mb-4 text-center">
            <div className="bg-[#0B0F17] p-3 rounded-lg border border-slate-800">
              <span className="text-xl font-bold text-white block">{panel_data.direct_dependents}</span>
              <span className="text-[10px] text-slate-500 uppercase">Direct Callers</span>
            </div>
            <div className="bg-[#0B0F17] p-3 rounded-lg border border-slate-800">
              <span className="text-xl font-bold text-white block">{panel_data.affected_classes}</span>
              <span className="text-[10px] text-slate-500 uppercase">Classes</span>
            </div>
            <div className="bg-[#0B0F17] p-3 rounded-lg border border-slate-800">
              <span className="text-xl font-bold text-white block">{panel_data.affected_files}</span>
              <span className="text-[10px] text-slate-500 uppercase">Affected Files</span>
            </div>
          </div>
        </div>
      )}

      {/* 3. GIT TIMELINE PANEL */}
      {panel_type === "git_timeline" && (
        <div className="bg-[#131B2E] border border-slate-800 rounded-xl p-5">
          <div className="flex items-center gap-2 text-emerald-400 font-bold text-xs mb-4">
            <GitCommit className="w-4 h-4" />
            <span>PROVENANCE TIMELINE</span>
          </div>
          <div className="space-y-3 border-l-2 border-slate-800 pl-4">
            {panel_data.commits?.map((commit, idx) => (
              <div key={idx} className="relative">
                <span className="absolute -left-[21px] top-1 w-2.5 h-2.5 rounded-full bg-emerald-500" />
                <p className="text-xs text-slate-200 font-bold">{commit.message}</p>
                <div className="text-[10px] text-slate-500 mt-0.5 flex gap-3">
                  <span>{commit.author}</span>
                  <span>{commit.date}</span>
                  <span className="text-blue-400">{commit.hash?.substring(0, 7)}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 4. EVIDENCE SNIPPETS */}
      {evidence && evidence.length > 0 && (
        <div className="bg-[#131B2E] border border-slate-800 rounded-xl p-5">
          <div className="flex items-center gap-2 text-slate-400 font-bold text-xs mb-3">
            <FileCode className="w-4 h-4 text-blue-400" />
            <span>RETRIEVED CODE EVIDENCE</span>
          </div>
          <div className="space-y-2">
            {evidence.map((item, i) => (
              <div key={i} className="bg-[#0B0F17] border border-slate-800 p-3 rounded-lg text-xs">
                <div className="text-blue-400 text-[11px] font-bold mb-1">
                  {item.file_path} {item.symbol_name && `-> ${item.symbol_name}`} (L{item.start_line}-L{item.end_line})
                </div>
                <pre className="text-[10px] text-slate-400 overflow-x-auto whitespace-pre-wrap">
                  {item.code_snippet}
                </pre>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 5. RETRIEVAL TRACE (INTERVIEW DEMO FEATURE) */}
      {retrieval_trace && (
        <div className="bg-[#131B2E] border border-blue-900/50 rounded-xl p-4 text-xs">
          <div className="flex items-center gap-2 text-blue-400 font-bold text-[11px] mb-2 uppercase tracking-wider">
            <Layers className="w-3.5 h-3.5" />
            <span>Retrieval Pipeline Execution Trace</span>
          </div>
          <div className="grid grid-cols-4 gap-2 text-center text-[10px]">
            <div className="bg-[#0B0F17] p-2 rounded border border-slate-800">
              <span className="text-slate-400 block">FAISS Candidates</span>
              <span className="text-blue-400 font-bold">{retrieval_trace.faiss_count}</span>
            </div>
            <div className="bg-[#0B0F17] p-2 rounded border border-slate-800">
              <span className="text-slate-400 block">BM25 Candidates</span>
              <span className="text-emerald-400 font-bold">{retrieval_trace.bm25_count}</span>
            </div>
            <div className="bg-[#0B0F17] p-2 rounded border border-slate-800">
              <span className="text-slate-400 block">Graph Expanded</span>
              <span className="text-amber-400 font-bold">{retrieval_trace.graph_expanded_count}</span>
            </div>
            <div className="bg-[#0B0F17] p-2 rounded border border-slate-800">
              <span className="text-slate-400 block">RRF Top K</span>
              <span className="text-purple-400 font-bold">{retrieval_trace.top_rrf_matches.length}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}