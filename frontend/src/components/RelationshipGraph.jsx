import React, {
  memo,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";

import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  Handle,
  MarkerType,
  Position,
} from "@xyflow/react";

import { createPortal } from "react-dom";

import "@xyflow/react/dist/style.css";

/* =========================================================
   SYMBOL NODE
========================================================= */

const SymbolNode = memo(function SymbolNode({ data }) {
  return (
    <div
      className={`relationship-node ${data.nodeClass || ""}`}
    >
      {/* CALLER */}
      {data.role !== "CALLER" && (
        <Handle
          id="target"
          type="target"
          position={Position.Top}
          className="relationship-handle"
        />
      )}

      {/* TARGET / CALLER */}
      {data.role !== "CALLEE" && (
        <Handle
          id="source"
          type="source"
          position={Position.Bottom}
          className="relationship-handle"
        />
      )}

      <div className="relationship-node-role">
        {data.role}
      </div>

      <div className="relationship-node-label">
        {data.label}
      </div>
    </div>
  );
});

const nodeTypes = {
  symbolNode: SymbolNode,
};

/* =========================================================
   HELPERS
========================================================= */

function getName(value) {
  if (!value) return "Unknown";

  if (typeof value === "string") {
    return value;
  }

  if (typeof value === "object") {
    return (
      value.label ||
      value.name ||
      value.symbol ||
      value.id ||
      "Unknown"
    );
  }

  return String(value);
}

/* =========================================================
   RELATIONSHIP GRAPH
========================================================= */

export default function RelationshipGraph({
  callers = [],
  targetSymbol = "",
  callees = [],
}) {
  const [selectedNode, setSelectedNode] = useState(null);

  /* =======================================================
     CLOSE ONLY WITH ENTER
  ======================================================= */

  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key === "Enter") {
        setSelectedNode(null);
      }
    };

    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, []);

  /* =======================================================
     NORMALIZE DATA
  ======================================================= */

  const normalizedCallers = useMemo(() => {
    return callers.map(getName);
  }, [callers]);

  const normalizedCallees = useMemo(() => {
    return callees.map(getName);
  }, [callees]);

  const normalizedTarget = getName(targetSymbol);

  /* =======================================================
     GRAPH NODES
  ======================================================= */

  const nodes = useMemo(() => {
    const result = [];

    const targetId = `target-${normalizedTarget}`;

    const targetX = 500;

    const callerY = 50;
    const targetY = 230;
    const calleeY = 410;

    const horizontalGap = 220;

    /* -----------------------------------------------------
       CALLERS
    ----------------------------------------------------- */

    if (normalizedCallers.length > 0) {
      const totalWidth =
        (normalizedCallers.length - 1) *
        horizontalGap;

      normalizedCallers.forEach((caller, index) => {
        result.push({
          id: `caller-${caller}-${index}`,

          type: "symbolNode",

          position: {
            x:
              targetX -
              totalWidth / 2 +
              index * horizontalGap,

            y: callerY,
          },

          data: {
            label: caller,
            role: "CALLER",
            nodeClass: "caller-node",
          },

          draggable: true,
        });
      });
    }

    /* -----------------------------------------------------
       TARGET
    ----------------------------------------------------- */

    if (normalizedTarget) {
      result.push({
        id: targetId,

        type: "symbolNode",

        position: {
          x: targetX,
          y: targetY,
        },

        data: {
          label: normalizedTarget,
          role: "TARGET",
          nodeClass: "target-node",
        },

        draggable: true,
      });
    }

    /* -----------------------------------------------------
       CALLEES
    ----------------------------------------------------- */

    if (normalizedCallees.length > 0) {
      const totalWidth =
        (normalizedCallees.length - 1) *
        horizontalGap;

      normalizedCallees.forEach((callee, index) => {
        result.push({
          id: `callee-${callee}-${index}`,

          type: "symbolNode",

          position: {
            x:
              targetX -
              totalWidth / 2 +
              index * horizontalGap,

            y: calleeY,
          },

          data: {
            label: callee,
            role: "CALLEE",
            nodeClass: "callee-node",
          },

          draggable: true,
        });
      });
    }

    return result;
  }, [
    normalizedCallers,
    normalizedCallees,
    normalizedTarget,
  ]);

  /* =======================================================
     GRAPH EDGES
  ======================================================= */

  const edges = useMemo(() => {
    const result = [];

    const targetId = `target-${normalizedTarget}`;

    /* CALLER → TARGET */

    normalizedCallers.forEach((caller, index) => {
      result.push({
        id: `caller-edge-${index}`,

        source: `caller-${caller}-${index}`,

        target: targetId,

        sourceHandle: "source",

        targetHandle: "target",

        type: "smoothstep",

        animated: false,

        markerEnd: {
          type: MarkerType.ArrowClosed,
        },

        className: "relationship-edge",

        style: {
          strokeWidth: 2,
        },
      });
    });

    /* TARGET → CALLEE */

    normalizedCallees.forEach((callee, index) => {
      result.push({
        id: `callee-edge-${index}`,

        source: targetId,

        target: `callee-${callee}-${index}`,

        sourceHandle: "source",

        targetHandle: "target",

        type: "smoothstep",

        animated: false,

        markerEnd: {
          type: MarkerType.ArrowClosed,
        },

        className: "relationship-edge",

        style: {
          strokeWidth: 2,
        },
      });
    });

    return result;
  }, [
    normalizedCallers,
    normalizedCallees,
    normalizedTarget,
  ]);

  /* =======================================================
     NODE CLICK
  ======================================================= */

  const handleNodeClick = useCallback(
    (event, node) => {
      event.preventDefault();
      event.stopPropagation();

      setSelectedNode({
        id: node.id,
        label: node.data?.label || "Unknown",
        role: node.data?.role || "Symbol",
      });
    },
    []
  );

  /* =======================================================
     CLOSE INSPECTOR
  ======================================================= */

  const closeInspector = useCallback((event) => {
    if (event) {
      event.preventDefault();
      event.stopPropagation();
    }

    setSelectedNode(null);
  }, []);

  /* =======================================================
     EMPTY STATE
  ======================================================= */

  if (
    !normalizedTarget &&
    normalizedCallers.length === 0 &&
    normalizedCallees.length === 0
  ) {
    return null;
  }

  /* =======================================================
     SELECTED NODE RELATIONSHIPS
  ======================================================= */

  const selectedRelationships = selectedNode
    ? edges.filter(
        (edge) =>
          edge.source === selectedNode.id ||
          edge.target === selectedNode.id
      )
    : [];

  /* =======================================================
     INSPECTOR
     
     IMPORTANT:
     Render it outside the graph using a portal.
     This prevents ReactFlow / overflow / graph CSS
     from making it disappear.
  ======================================================= */

  const inspector =
    selectedNode && typeof document !== "undefined"
      ? createPortal(
          <div
            className="relationship-inspector-overlay"
            onClick={(event) => {
              event.stopPropagation();
            }}
          >
            <aside
              className="relationship-node-inspector"
              onClick={(event) => {
                event.stopPropagation();
              }}
            >
              {/* HEADER */}

              <div className="relationship-inspector-header">
                <div>
                  <div className="relationship-inspector-eyebrow">
                    SYMBOL INSPECTOR
                  </div>

                  <div className="relationship-inspector-title">
                    {selectedNode.label}
                  </div>
                </div>

                <button
                  type="button"
                  className="relationship-inspector-close"
                  onClick={closeInspector}
                  aria-label="Close inspector"
                >
                  ×
                </button>
              </div>

              {/* ROLE */}

              <div className="relationship-inspector-section">
                <div className="relationship-inspector-label">
                  ROLE
                </div>

                <div className="relationship-inspector-value">
                  {selectedNode.role}
                </div>
              </div>

              {/* SYMBOL ID */}

              <div className="relationship-inspector-section">
                <div className="relationship-inspector-label">
                  SYMBOL ID
                </div>

                <div className="relationship-inspector-code">
                  {selectedNode.id}
                </div>
              </div>

              {/* RELATIONSHIPS */}

              <div className="relationship-inspector-section">
                <div className="relationship-inspector-label">
                  RELATIONSHIPS
                </div>

                <div className="relationship-inspector-list">
                  {selectedRelationships.length > 0 ? (
                    selectedRelationships.map(
                      (edge, index) => {
                        const isSource =
                          edge.source ===
                          selectedNode.id;

                        const relatedNodeId = isSource
                          ? edge.target
                          : edge.source;

                        const relatedNode =
                          nodes.find(
                            (node) =>
                              node.id ===
                              relatedNodeId
                          );

                        return (
                          <div
                            key={`${edge.id}-${index}`}
                            className="relationship-inspector-item"
                          >
                            <span className="relationship-inspector-arrow">
                              {isSource ? "→" : "←"}
                            </span>

                            <span>
                              {relatedNode?.data
                                ?.label ||
                                relatedNodeId}
                            </span>
                          </div>
                        );
                      }
                    )
                  ) : (
                    <div className="relationship-inspector-empty">
                      No relationships found.
                    </div>
                  )}
                </div>
              </div>

              {/* SOURCE */}

              <div className="relationship-inspector-section">
                <div className="relationship-inspector-label">
                  SOURCE
                </div>

                <div className="relationship-source-placeholder">
                  <div className="relationship-source-icon">
                    ◇
                  </div>

                  <div>
                    <div className="relationship-source-title">
                      Source inspection
                    </div>

                    <div className="relationship-source-text">
                      Connect this symbol to its
                      repository file to inspect
                      the implementation.
                    </div>
                  </div>
                </div>
              </div>
            </aside>
          </div>,
          document.body
        )
      : null;

  /* =======================================================
     GRAPH
  ======================================================= */

  return (
    <>
      <div className="relationship-graph-wrapper">
        <div className="relationship-graph-card">

          {/* HEADER */}

          <div className="relationship-graph-header">
            <div>
              <div className="relationship-graph-title">
                AST RELATIONSHIP GRAPH
              </div>

              <div className="relationship-graph-subtitle">
                Dependency relationships for{" "}
                <strong>
                  {normalizedTarget ||
                    "Target Symbol"}
                </strong>
              </div>
            </div>

            {/* STATS */}

            <div className="relationship-graph-stats">
              <div className="relationship-stat">
                <span>
                  {normalizedCallers.length}
                </span>
                callers
              </div>

              <div className="relationship-stat">
                <span>
                  {normalizedCallees.length}
                </span>
                callees
              </div>

              <div className="relationship-stat">
                <span>
                  {edges.length}
                </span>
                relationships
              </div>
            </div>
          </div>

          {/* GRAPH CANVAS */}

          <div className="relationship-graph-canvas">
            <ReactFlow
              nodes={nodes}
              edges={edges}
              nodeTypes={nodeTypes}

              fitView

              fitViewOptions={{
                padding: 0.25,
                minZoom: 0.5,
                maxZoom: 1.5,
              }}

              onNodeClick={handleNodeClick}

              nodesDraggable={true}

              nodesConnectable={false}

              elementsSelectable={false}

              panOnDrag={true}

              zoomOnScroll={true}

              zoomOnPinch={true}

              zoomOnDoubleClick={false}
            >
              <Background />

              <Controls />

              <MiniMap
                pannable
                zoomable
              />
            </ReactFlow>
          </div>
        </div>
      </div>

      {inspector}
    </>
  );
}