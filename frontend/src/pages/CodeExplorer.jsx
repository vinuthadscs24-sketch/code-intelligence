import React from "react";

export default function CodeExplorer() {
  return (
    <div className="p-6 max-w-5xl mx-auto text-slate-300 font-mono text-xs">
      <h1 className="text-xl font-bold text-white mb-2">Code Explorer</h1>
      <p className="text-slate-400">Select an indexed repository to inspect parsed classes, methods, and AST nodes.</p>
    </div>
  );
}
