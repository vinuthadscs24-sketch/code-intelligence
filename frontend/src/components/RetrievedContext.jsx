import React from "react";

export default function RetrievedContext({ chunks = [] }) {
  if (!chunks.length) return null;

  return (
    <div className="flex flex-col gap-3">
      <h3 className="text-xs font-mono uppercase tracking-wider text-slate-400">Retrieved Chunks ({chunks.length})</h3>
      <div className="grid grid-cols-1 gap-3">
        {chunks.map((chunk, idx) => (
          <div key={idx} className="bg-[#131B2E] border border-slate-800 rounded-xl p-4 flex flex-col gap-2">
            <div className="flex justify-between text-[11px] font-mono text-blue-400">
              <span>?? {chunk.file_path || chunk.filename || `Chunk #${idx + 1}`}</span>
              <span className="text-slate-500">Score: {chunk.score ? chunk.score.toFixed(3) : "N/A"}</span>
            </div>
            <pre className="bg-[#0B0F17] p-3 rounded-lg text-[11px] font-mono text-slate-300 overflow-x-auto border border-slate-800/80">
              {chunk.code || chunk.content || chunk.text}
            </pre>
          </div>
        ))}
      </div>
    </div>
  );
}
