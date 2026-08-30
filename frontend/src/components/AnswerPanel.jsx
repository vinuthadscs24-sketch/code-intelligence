import React from "react";

export default function AnswerPanel({ answer }) {
  return (
    <div className="bg-[#131B2E] border border-slate-800 rounded-xl p-5">
      <h3 className="text-xs font-mono uppercase tracking-wider text-slate-400 mb-3">AI Engine Synthesis</h3>
      <div className="text-xs font-sans text-slate-300 leading-relaxed whitespace-pre-wrap">
        {answer || "No synthesis available."}
      </div>
    </div>
  );
}
