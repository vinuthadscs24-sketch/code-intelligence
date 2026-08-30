import React from 'react';
import { ReactFlow, Background, Controls } from '@xyflow/react';

export default function BlastRadiusGraph({ targetNode, impactedNodes = [], onNodeClick }) {
  const targetLabel = targetNode || 'Modified Target';

  // Position target at center, spread affected modules in a radial circle
  const nodes = [
    {
      id: 'target',
      data: { 
        label: `⚠️ Target: ${targetLabel}`,
        rawStep: { name: targetLabel, type: 'TARGET' }
      },
      position: { x: 300, y: 220 },
      style: {
        background: '#7F1D1D',
        color: '#FCA5A5',
        border: '2px solid #EF4444',
        boxShadow: '0 0 25px rgba(239, 68, 68, 0.5)',
        borderRadius: '8px',
        padding: '14px 20px',
        fontWeight: 'bold',
        fontSize: '13px',
        cursor: 'pointer'
      }
    },
    ...impactedNodes.map((item, index) => {
      const angle = (index / Math.max(impactedNodes.length, 1)) * 2 * Math.PI;
      const radius = 200;
      const itemName = item.name || item.file || `Affected Node ${index + 1}`;

      return {
        id: `impact-${index}`,
        data: { 
          label: itemName,
          rawStep: item
        },
        position: {
          x: 300 + radius * Math.cos(angle),
          y: 220 + radius * Math.sin(angle)
        },
        style: {
          background: '#1E293B',
          color: '#F87171',
          border: '1px solid #991B1B',
          borderRadius: '8px',
          padding: '10px 16px',
          fontSize: '12px',
          fontWeight: '600',
          cursor: 'pointer'
        }
      };
    })
  ];

  const edges = impactedNodes.map((_, index) => ({
    id: `edge-impact-${index}`,
    source: 'target',
    target: `impact-${index}`,
    animated: true,
    style: { stroke: '#EF4444', strokeWidth: 2, strokeDasharray: '5,5' }
  }));

  return (
    <div className="w-full h-full bg-[#090D16] relative">
      <div className="absolute top-4 left-4 z-10 bg-[#131B2E]/90 border border-red-900/50 p-2.5 rounded-lg backdrop-blur-md flex items-center gap-2">
        <span className="w-2.5 h-2.5 rounded-full bg-red-500 animate-pulse"></span>
        <span className="text-xs font-mono text-red-300 font-semibold">Blast Radius / Impact Analysis</span>
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