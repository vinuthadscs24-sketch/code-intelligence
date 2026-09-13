import {
  memo,
  useMemo,
  useState,
  useCallback,
  useEffect,
} from "react";

import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  MarkerType,
  Handle,
  Position,
} from "@xyflow/react";

import "@xyflow/react/dist/style.css";

/* =========================================================
   CUSTOM SYMBOL NODE
========================================================= */

const SymbolNode = memo(function SymbolNode({ data }) {
  return (
    <div
      className={`graph-node-content ${data.nodeClass || ""}`}
    >
      {/* CALLER */}
      {data.role === "CALLER" && (
        <Handle
          id="source"
          type="source"
          position={Position.Bottom}
          className="graph-handle"
        />
      )}

      {/* TARGET */}
      {data.role === "TARGET" && (
        <>
          <Handle
            id="target"
            type="target"
            position={Position.Top}
            className="graph-handle"
          />

          <Handle
            id="source"
            type="source"
            position={Position.Bottom}
            className="graph-handle"
          />
        </>
      )}

      {/* CALLEE */}
      {data.role === "CALLEE" && (
        <Handle
          id="target"
          type="target"
          position={Position.Top}
          className="graph-handle"
        />
      )}

      {/* ROLE */}
      <div
        className={`graph-node-type ${
          data.role === "TARGET" ? "target-type" : ""
        }`}
      >
        {data.role}
      </div>

      {/* LABEL */}
      <div
        className={`graph-node-label ${
          data.role === "TARGET" ? "target-label" : ""
        }`}
      >
        {data.label}
      </div>
    </div>
  );
});

/* =========================================================
   NODE TYPES
========================================================= */

const nodeTypes = {
  symbolNode: SymbolNode,
};

/* =========================================================
   CALL GRAPH
========================================================= */

function CallGraph({ data }) {
  const [selectedNode, setSelectedNode] = useState(null);

  /* =======================================================
     ENTER KEY CLOSES INSPECTOR
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
     CREATE NODES
  ======================================================= */

  const nodes = useMemo(() => {
    if (!data?.nodes?.length) {
      return [];
    }

    /* -----------------------------------------------------
       TARGET
    ----------------------------------------------------- */

    const target = data.nodes.find(
      (node) => node.id === data.target
    );

    /* -----------------------------------------------------
       CALLERS
    ----------------------------------------------------- */

    const callers = data.nodes.filter(
      (node) =>
        node.id !== data.target &&
        data.callers?.includes(node.id)
    );

    /* -----------------------------------------------------
       CALLEES
       IMPORTANT:
       Use backend's explicit `callees` list.
    ----------------------------------------------------- */

    const callees = data.nodes.filter(
      (node) =>
        node.id !== data.target &&
        data.callees?.includes(node.id)
    );

    const result = [];

    /* =====================================================
       LAYOUT SETTINGS
    ===================================================== */

    const horizontalGap = 220;

    const centerX = 500;

    const callerY = 60;
    const targetY = 240;
    const calleeY = 440;

    /* =====================================================
       CALLERS
    ===================================================== */

    if (callers.length > 0) {
      const totalWidth =
        (callers.length - 1) *
        horizontalGap;

      callers.forEach((node, index) => {
        result.push({
          id: node.id,

          type: "symbolNode",

          position: {
            x:
              centerX -
              totalWidth / 2 +
              index * horizontalGap,

            y: callerY,
          },

          data: {
            label: node.label,
            role: "CALLER",
            nodeClass: "caller-node",
          },

          className:
            "call-graph-node caller-node",
        });
      });
    }

    /* =====================================================
       TARGET
    ===================================================== */

    if (target) {
      result.push({
        id: target.id,

        type: "symbolNode",

        position: {
          x: centerX,
          y: targetY,
        },

        data: {
          label: target.label,
          role: "TARGET",
          nodeClass: "target-node",
        },

        className:
          "call-graph-node target-node",
      });
    }

    /* =====================================================
       CALLEES
    ===================================================== */

    if (callees.length > 0) {
      const totalWidth =
        (callees.length - 1) *
        horizontalGap;

      callees.forEach((node, index) => {
        result.push({
          id: node.id,

          type: "symbolNode",

          position: {
            x:
              centerX -
              totalWidth / 2 +
              index * horizontalGap,

            y: calleeY,
          },

          data: {
            label: node.label,
            role: "CALLEE",
            nodeClass: "callee-node",
          },

          className:
            "call-graph-node callee-node",
        });
      });
    }

    return result;
  }, [
    data?.nodes,
    data?.target,
    data?.callers,
    data?.callees,
  ]);

  /* =======================================================
     CREATE EDGES
  ======================================================= */

  const edges = useMemo(() => {
    if (!data?.edges?.length) {
      return [];
    }

    return data.edges.map((edge, index) => {
      return {
        id:
          `edge-${edge.source}-${edge.target}-${index}`,

        source: edge.source,

        target: edge.target,

        sourceHandle: "source",

        targetHandle: "target",

        label: "CALLS",

        type: "smoothstep",

        animated: false,

        markerEnd: {
          type: MarkerType.ArrowClosed,
        },

        className:
          "call-graph-edge",

        style: {
          strokeWidth: 2,
        },

        labelStyle: {
          fontSize: 10,
          fontWeight: 600,
        },

        labelBgStyle: {
          fill: "white",
        },

        labelBgPadding: [5, 3],

        labelBgBorderRadius: 4,
      };
    });
  }, [data?.edges]);

  /* =======================================================
     NODE CLICK
  ======================================================= */

  const handleNodeClick = useCallback(
    (event, node) => {
      event.preventDefault();
      event.stopPropagation();

      const originalNode =
        data?.nodes?.find(
          (item) => item.id === node.id
        );

      if (!originalNode) {
        return;
      }

      setSelectedNode({
        ...originalNode,
      });
    },
    [data?.nodes]
  );

  /* =======================================================
     GET NODE ROLE
  ======================================================= */

  function getNodeRole(node) {
    if (!node) {
      return "Symbol";
    }

    if (node.id === data?.target) {
      return "Target";
    }

    if (
      data?.callers?.includes(node.id)
    ) {
      return "Caller";
    }

    if (
      data?.callees?.includes(node.id)
    ) {
      return "Callee";
    }

    return "Symbol";
  }

  /* =======================================================
     RENDER
  ======================================================= */

  return (
    <div className="call-graph-layout">

      {/* =================================================
          GRAPH
      ================================================= */}

      <div className="call-graph">

        {/* HEADER */}

        <div className="call-graph-header">

          <div>
            <div className="call-graph-title">
              CALL GRAPH
            </div>

            <div className="call-graph-subtitle">
              Dependency relationships
              for{" "}
              <strong>
                {data?.target}
              </strong>
            </div>
          </div>

          {/* STATS */}

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

        {/* =================================================
            CANVAS
        ================================================= */}

        <div className="call-graph-canvas">

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

      {/* =================================================
          INSPECTOR
      ================================================= */}

      {selectedNode && (
        <aside className="node-inspector">

          {/* HEADER */}

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
              type="button"
              className="inspector-close"
              onClick={(event) => {
                event.stopPropagation();

                setSelectedNode(null);
              }}
            >
              ×
            </button>

          </div>

          {/* TYPE */}

          <div className="inspector-section">

            <div className="inspector-label">
              TYPE
            </div>

            <div className="inspector-value">
              {selectedNode.type || "symbol"}
            </div>

          </div>

          {/* ROLE */}

          <div className="inspector-section">

            <div className="inspector-label">
              ROLE
            </div>

            <div className="inspector-value">
              {getNodeRole(selectedNode)}
            </div>

          </div>

          {/* SYMBOL ID */}

          <div className="inspector-section">

            <div className="inspector-label">
              SYMBOL ID
            </div>

            <div className="inspector-code">
              {selectedNode.id}
            </div>

          </div>

          {/* RELATIONSHIPS */}

          <div className="inspector-section">

            <div className="inspector-label">
              RELATIONSHIPS
            </div>

            <div className="relationship-list">

              {data?.edges
                ?.filter(
                  (edge) =>
                    edge.source === selectedNode.id ||
                    edge.target === selectedNode.id
                )
                .map((edge, index) => (
                  <div
                    className="relationship-item"
                    key={`${edge.source}-${edge.target}-${index}`}
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

          {/* SOURCE */}

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
                  Connect this symbol to
                  its repository file to
                  inspect the implementation.
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