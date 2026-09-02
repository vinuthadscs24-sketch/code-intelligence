import { useMemo, useState } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  MarkerType,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

function CallGraph({ data }) {
  const [selectedNode, setSelectedNode] = useState(null);

  const nodes = useMemo(() => {
    if (!data?.nodes) {
      return [];
    }

    const callers = data.nodes.filter(
      (node) =>
        node.id !== data.target &&
        data.callers?.includes(node.id)
    );

    const target = data.nodes.find(
      (node) => node.id === data.target
    );

    const callees = data.nodes.filter(
      (node) =>
        node.id !== data.target &&
        !data.callers?.includes(node.id)
    );

    const result = [];

    callers.forEach((node, index) => {
      result.push({
        id: node.id,
        position: {
          x: 120 + index * 300,
          y: 40,
        },
        data: {
          label: (
            <div
              className="graph-node-content"
              onClick={() => setSelectedNode(node)}
            >
              <div className="graph-node-type">
                CALLER
              </div>

              <div className="graph-node-label">
                {node.label}
              </div>
            </div>
          ),
        },
        className: "call-graph-node caller-node",
      });
    });

    if (target) {
      result.push({
        id: target.id,
        position: {
          x: 320,
          y: 220,
        },
        data: {
          label: (
            <div
              className="graph-node-content"
              onClick={() => setSelectedNode(target)}
            >
              <div className="graph-node-type target-type">
                TARGET
              </div>

              <div className="graph-node-label target-label">
                {target.label}
              </div>
            </div>
          ),
        },
        className: "call-graph-node target-node",
      });
    }

    callees.forEach((node, index) => {
      result.push({
        id: node.id,
        position: {
          x: 40 + index * 190,
          y: 400,
        },
        data: {
          label: (
            <div
              className="graph-node-content"
              onClick={() => setSelectedNode(node)}
            >
              <div className="graph-node-type">
                CALLEE
              </div>

              <div className="graph-node-label">
                {node.label}
              </div>
            </div>
          ),
        },
        className: "call-graph-node callee-node",
      });
    });

    return result;
  }, [data]);

  const edges = useMemo(() => {
    if (!data?.edges) {
      return [];
    }

    return data.edges.map((edge, index) => ({
      id: `edge-${index}`,
      source: edge.source,
      target: edge.target,
      label: "CALLS",
      type: "smoothstep",
      markerEnd: {
        type: MarkerType.ArrowClosed,
      },
      className: "call-graph-edge",
    }));
  }, [data]);

  function handleNodeClick(_, node) {
    const originalNode = data?.nodes?.find(
      (item) => item.id === node.id
    );

    if (originalNode) {
      setSelectedNode(originalNode);
    }
  }

  return (
    <div className="call-graph-layout">

      {/* GRAPH */}

      <div className="call-graph">

        <div className="call-graph-header">

          <div>
            <div className="call-graph-title">
              CALL GRAPH
            </div>

            <div className="call-graph-subtitle">
              Dependency relationships for{" "}
              <strong>{data?.target}</strong>
            </div>
          </div>

          <div className="graph-stats">

            <div className="graph-stat">
              <span>
                {data?.callers?.length || 0}
              </span>
              callers
            </div>

            <div className="graph-stat">
              <span>
                {data?.callees?.length || 0}
              </span>
              callees
            </div>

            <div className="graph-stat">
              <span>
                {data?.edges?.length || 0}
              </span>
              relationships
            </div>

          </div>

        </div>

        <div className="call-graph-canvas">

          <ReactFlow
            nodes={nodes}
            edges={edges}
            fitView
            onNodeClick={handleNodeClick}
            nodesDraggable
            nodesConnectable={false}
            elementsSelectable
          >
            <Background />

            <Controls />

            <MiniMap />
          </ReactFlow>

        </div>

      </div>

      {/* INSPECTOR */}

      {selectedNode && (
        <aside className="node-inspector">

          <div className="inspector-header">

            <div>
              <div className="inspector-eyebrow">
                SYMBOL INSPECTOR
              </div>

              <div className="inspector-title">
                {selectedNode.label}
              </div>
            </div>

            <button
              className="inspector-close"
              onClick={() => setSelectedNode(null)}
            >
              ×
            </button>

          </div>

          <div className="inspector-section">

            <div className="inspector-label">
              TYPE
            </div>

            <div className="inspector-value">
              {selectedNode.type || "symbol"}
            </div>

          </div>

          <div className="inspector-section">

            <div className="inspector-label">
              ROLE
            </div>

            <div className="inspector-value">
              {selectedNode.id === data.target
                ? "Target"
                : data.callers?.includes(selectedNode.id)
                ? "Caller"
                : "Callee"}
            </div>

          </div>

          <div className="inspector-section">

            <div className="inspector-label">
              SYMBOL ID
            </div>

            <div className="inspector-code">
              {selectedNode.id}
            </div>

          </div>

          <div className="inspector-section">

            <div className="inspector-label">
              RELATIONSHIPS
            </div>

            <div className="relationship-list">

              {data.edges
                ?.filter(
                  (edge) =>
                    edge.source === selectedNode.id ||
                    edge.target === selectedNode.id
                )
                .map((edge, index) => (
                  <div
                    className="relationship-item"
                    key={index}
                  >

                    <span className="relationship-arrow">
                      {edge.source === selectedNode.id
                        ? "→"
                        : "←"}
                    </span>

                    <span>
                      {edge.source === selectedNode.id
                        ? edge.target
                        : edge.source}
                    </span>

                  </div>
                ))}

            </div>

          </div>

          <div className="inspector-section">

            <div className="inspector-label">
              SOURCE
            </div>

            <div className="source-placeholder">

              <div className="source-icon">
                ◇
              </div>

              <div>
                <div className="source-title">
                  Source inspection
                </div>

                <div className="source-text">
                  Connect this symbol to its
                  repository file to inspect the
                  implementation.
                </div>
              </div>

            </div>

          </div>

        </aside>
      )}

    </div>
  );
}

export default CallGraph;