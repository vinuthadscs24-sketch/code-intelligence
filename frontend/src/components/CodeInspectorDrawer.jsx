import React from 'react';
import { X, Code2, FileText, Hash, ExternalLink } from 'lucide-react';

export default function CodeInspectorDrawer({ isOpen, onClose, selectedNode }) {
  if (!isOpen || !selectedNode) return null;

  const fileName = selectedNode.file || selectedNode.file_path || "AppointmentService.java";
  const methodName = selectedNode.name || selectedNode.symbol_name || "bookAppointment()";
  const startLine = selectedNode.start_line || 42;
  const endLine = selectedNode.end_line || 68;
  const codeSnippet = selectedNode.code_snippet || selectedNode.content || `public void ${methodName} {\n    // Extracted AST symbol definition\n    System.out.println("Processing execution node...");\n}`;

  return (
    <div className="fixed inset-y-0 right-0 w-[480px] bg-[#131B2E] border-l border-slate-800 shadow-2xl z-50 flex flex-col transition-transform duration-300 ease-in-out">
      
      {/* Drawer Header */}
      <div className="p-4 border-b border-slate-800 flex items-center justify-between bg-[#1A243B]">
        <div className="flex items-center gap-2">
          <Code2 className="w-5 h-5 text-blue-400" />
          <h3 className="font-mono text-sm font-semibold text-slate-200">AST Node Inspector</h3>
        </div>
        <button 
          onClick={onClose}
          className="p-1 hover:bg-slate-700/60 rounded text-slate-400 hover:text-white transition"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      {/* Metadata Badges */}
      <div className="p-5 flex flex-col gap-4 border-b border-slate-800/60">
        <div>
          <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block mb-1">Symbol Name</span>
          <div className="text-base font-bold font-mono text-blue-400">{methodName}</div>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div className="bg-[#0B0F17] p-3 rounded-lg border border-slate-800/80">
            <div className="flex items-center gap-1.5 text-xs text-slate-400 mb-1">
              <FileText className="w-3.5 h-3.5 text-blue-400" /> File Path
            </div>
            <div className="text-xs font-mono text-slate-200 truncate" title={fileName}>{fileName}</div>
          </div>
          
          <div className="bg-[#0B0F17] p-3 rounded-lg border border-slate-800/80">
            <div className="flex items-center gap-1.5 text-xs text-slate-400 mb-1">
              <Hash className="w-3.5 h-3.5 text-blue-400" /> Lines
            </div>
            <div className="text-xs font-mono text-slate-200">L{startLine} - L{endLine}</div>
          </div>
        </div>
      </div>

      {/* Source Code Container */}
      <div className="flex-1 p-5 overflow-y-auto bg-[#0B0F17] flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Source Chunk</span>
          <span className="text-[10px] font-mono text-slate-500">Java AST</span>
        </div>
        <pre className="p-4 bg-[#131B2E] border border-slate-800 rounded-lg text-xs font-mono text-slate-300 overflow-x-auto leading-relaxed flex-1">
          <code>{codeSnippet}</code>
        </pre>
      </div>

    </div>
  );
}