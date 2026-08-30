import React, { useState, useEffect } from 'react';
import { ReactFlow, Background, Controls } from '@xyflow/react';
import { Play, Pause, RotateCcw, SkipForward } from 'lucide-react';

export default function ExecutionFlowGraph({ steps = [], onNodeClick }) {
  const [activeStep, setActiveStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  // Auto-advance step animation
  useEffect(() => {
    let timer;
    if (isPlaying && activeStep < steps.length - 1) {
      timer = setTimeout(() => {
        setActiveStep((prev) => prev + 1);
      }, 1200);
    } else if (activeStep >= steps.length - 1) {
      setIsPlaying(false);
    }
    return () => clearTimeout(timer);
  }, [isPlaying, activeStep, steps.length]);

  // Transform backend steps to React Flow nodes & edges dynamically
  const nodes = steps.map((step, index) => {
    const isActive = index <= activeStep;
    const isCurrent = index === activeStep;

    return {
      id: `step-${index}`,
      data: { 
        label: step.name || step.symbol_name || `Step ${index + 1}`,
        rawStep: step 
      },
      position: { x: 250, y: index * 110 + 40 },
      style: {
        background: isCurrent ? '#2563EB' : isActive ? '#1E293B' : '#0F172A',
        color: isCurrent ? '#FFFFFF' : isActive ? '#93C5FD' : '#475569',
        border: `2px solid ${isCurrent ? '#60A5FA' : isActive ? '#3B82F6' : '#1E293B'}`,
        boxShadow: isCurrent ? '0 0 20px rgba(59, 130, 246, 0.6)' : 'none',
        borderRadius: '8px',
        padding: '12px 20px',
        fontWeight: '600',
        fontSize: '13px',
        transition: 'all 0.4s ease',
        cursor: 'pointer'
      }
    };
  });

  const edges = steps.slice(0, -1).map((_, index) => {
    const isAnimated = index < activeStep;

    return {
      id: `edge-${index}`,
      source: `step-${index}`,
      target: `step-${index + 1}`,
      animated: isAnimated,
      style: {
        stroke: isAnimated ? '#60A5FA' : '#1E293B',
        strokeWidth: isAnimated ? 2.5 : 1.5,
        transition: 'all 0.4s ease'
      }
    };
  });

  return (
    <div className="relative w-full h-full bg-[#090D16]">
      {/* Playback Control Panel */}
      <div className="absolute top-4 right-4 z-10 flex items-center gap-2 bg-[#131B2E]/90 border border-slate-700 p-2 rounded-lg backdrop-blur-md">
        <button 
          onClick={() => setIsPlaying(!isPlaying)}
          className="p-1.5 bg-blue-600 hover:bg-blue-500 rounded text-white transition"
          title={isPlaying ? "Pause Flow" : "Play Flow"}
        >
          {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
        </button>
        <button 
          onClick={() => { setActiveStep(0); setIsPlaying(false); }}
          className="p-1.5 bg-slate-800 hover:bg-slate-700 rounded text-slate-300 transition"
          title="Reset"
        >
          <RotateCcw className="w-4 h-4" />
        </button>
        <button 
          onClick={() => setActiveStep((prev) => Math.min(prev + 1, steps.length - 1))}
          className="p-1.5 bg-slate-800 hover:bg-slate-700 rounded text-slate-300 transition"
          title="Next Step"
        >
          <SkipForward className="w-4 h-4" />
        </button>
        <span className="text-xs font-mono text-slate-400 ml-2">
          Step {activeStep + 1} / {steps.length || 1}
        </span>
      </div>

      <ReactFlow 
        nodes={nodes} 
        edges={edges}
        onNodeClick={(_, node) => onNodeClick && onNodeClick(node.data.rawStep)}
        fitView
      >
        <Background color="#1E293B" gap={18} />
        <Controls />
      </ReactFlow>
    </div>
  );
}