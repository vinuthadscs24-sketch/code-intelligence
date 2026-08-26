from typing import List, Dict, Any, Set
from src.graph_builder import CodeKnowledgeGraph


class ImpactAnalyzer:
    """
    Analyzes the blast radius of a class or method change using
    the Code Knowledge Graph.

    The analyzer traverses upstream dependencies:

        Target Method
             ↑
          Callers
             ↑
          Callers
             ↑
        Higher-level callers

    This identifies components that may be affected when the
    target method changes.
    """

    def __init__(self, kg: CodeKnowledgeGraph):
        self.kg = kg

    def _find_matching_nodes(
        self,
        target_method: str
    ) -> List[str]:
        """
        Finds graph nodes matching the requested method.

        Supports:
            method
            method()
            Class.method
            Class.method()
            exact graph node IDs
            substring fallback
        """

        clean_target = target_method.strip()

        if not clean_target:
            return []

        graph = getattr(self.kg, "graph", None)

        if graph is None:
            return []

        matched_nodes: List[str] = []

        # ----------------------------------------------------------
        # Exact / structured matching
        # ----------------------------------------------------------

        for node in graph.nodes:

            node_string = str(node)

            if (
                node_string == clean_target
                or node_string == f"{clean_target}()"
                or node_string.endswith(
                    f".{clean_target}()"
                )
                or node_string.endswith(
                    f".{clean_target}"
                )
            ):
                matched_nodes.append(node)

        if matched_nodes:
            return matched_nodes

        # ----------------------------------------------------------
        # Case-insensitive structured matching
        # ----------------------------------------------------------

        clean_lower = clean_target.lower()

        for node in graph.nodes:

            node_string = str(node)
            node_lower = node_string.lower()

            if (
                node_lower == clean_lower
                or node_lower == f"{clean_lower}()"
                or node_lower.endswith(
                    f".{clean_lower}()"
                )
                or node_lower.endswith(
                    f".{clean_lower}"
                )
            ):
                matched_nodes.append(node)

        if matched_nodes:
            return matched_nodes

        # ----------------------------------------------------------
        # General substring fallback
        # ----------------------------------------------------------

        for node in graph.nodes:

            node_string = str(node)

            if clean_lower in node_string.lower():
                matched_nodes.append(node)

        return matched_nodes

    def _get_node_type(
        self,
        node: str
    ) -> str:
        """Returns the graph node type when available."""

        graph = getattr(self.kg, "graph", None)

        if graph is None:
            return "UNKNOWN"

        data = graph.nodes.get(node, {})

        return str(
            data.get(
                "type",
                "UNKNOWN"
            )
        )

    def _get_node_file(
        self,
        node: str
    ) -> str:
        """Returns the source file associated with a graph node."""

        graph = getattr(self.kg, "graph", None)

        if graph is None:
            return "unknown"

        data = graph.nodes.get(node, {})

        return str(
            data.get("file")
            or data.get("file_name")
            or data.get("file_path")
            or "unknown"
        )

    def _get_node_class(
        self,
        node: str
    ) -> str:
        """Returns the class associated with a graph node."""

        graph = getattr(self.kg, "graph", None)

        if graph is None:
            return "unknown"

        data = graph.nodes.get(node, {})

        return str(
            data.get("class_name")
            or data.get("class")
            or "unknown"
        )

    def _get_node_method(
        self,
        node: str
    ) -> str:
        """Returns the method associated with a graph node."""

        graph = getattr(self.kg, "graph", None)

        if graph is None:
            return "unknown"

        data = graph.nodes.get(node, {})

        return str(
            data.get("method_name")
            or data.get("method")
            or node
        )

    def _get_predecessors(
        self,
        node: str
    ) -> List[str]:
        """
        Gets upstream dependencies of a node.

        In a call graph:

            Caller → Target

        therefore predecessors of Target are callers.
        """

        graph = getattr(self.kg, "graph", None)

        if graph is None:
            return []

        try:
            return list(
                graph.predecessors(node)
            )
        except Exception:
            return []

    def analyze_blast_radius(
        self,
        target_method: str,
        depth: int = 2
    ) -> Dict[str, Any]:
        """
        Traverses upstream graph edges to identify components
        potentially affected by changing a target method.

        Example:

            Controller
                ↓
            Service
                ↓
            Repository

        If Service changes:

            depth=1 → Controller
            depth=2 → Controller's callers
        """

        if not target_method or not target_method.strip():
            return {
                "error": "Target method cannot be empty.",
                "target": target_method,
                "matched_nodes": [],
                "affected_components": [],
                "total_affected": 0,
                "depth_analyzed": 0,
            }

        if depth < 1:
            return {
                "error": "Depth must be at least 1.",
                "target": target_method,
                "matched_nodes": [],
                "affected_components": [],
                "total_affected": 0,
                "depth_analyzed": 0,
            }

        graph = getattr(
            self.kg,
            "graph",
            None
        )

        if graph is None:
            return {
                "error": "Knowledge Graph is not initialized.",
                "target": target_method,
                "matched_nodes": [],
                "affected_components": [],
                "total_affected": 0,
                "depth_analyzed": 0,
            }

        # ----------------------------------------------------------
        # Find target nodes
        # ----------------------------------------------------------

        matched_nodes = self._find_matching_nodes(
            target_method
        )

        if not matched_nodes:

            sample_methods = [
                str(node)
                for node, data in graph.nodes(
                    data=True
                )
                if data.get("type")
                in (
                    "METHOD",
                    "METHOD_CALL",
                )
            ][:10]

            return {
                "error": (
                    f"Method '{target_method}' "
                    "not found in Knowledge Graph."
                ),
                "available_method_samples": sample_methods,
                "target": target_method,
                "matched_nodes": [],
                "affected_components": [],
                "total_affected": 0,
                "depth_analyzed": 0,
            }

        # ----------------------------------------------------------
        # Breadth-first upstream traversal
        # ----------------------------------------------------------

        affected_nodes: Set[str] = set()

        visited_nodes: Set[str] = set(
            matched_nodes
        )

        current_level: Set[str] = set(
            matched_nodes
        )

        affected_by_depth: Dict[
            int,
            List[str]
        ] = {}

        for current_depth in range(
            1,
            depth + 1
        ):

            next_level: Set[str] = set()

            for node in current_level:

                predecessors = (
                    self._get_predecessors(node)
                )

                for predecessor in predecessors:

                    if predecessor in visited_nodes:
                        continue

                    visited_nodes.add(
                        predecessor
                    )

                    affected_nodes.add(
                        predecessor
                    )

                    next_level.add(
                        predecessor
                    )

            if not next_level:
                break

            affected_by_depth[
                current_depth
            ] = sorted(
                next_level,
                key=str
            )

            current_level = next_level

        # ----------------------------------------------------------
        # Build detailed component information
        # ----------------------------------------------------------

        affected_components = []

        for node in sorted(
            affected_nodes,
            key=str
        ):

            affected_components.append({
                "node": str(node),
                "type": self._get_node_type(node),
                "file": self._get_node_file(node),
                "class": self._get_node_class(node),
                "method": self._get_node_method(node),
            })

        # ----------------------------------------------------------
        # Separate direct and transitive impact
        # ----------------------------------------------------------

        direct_impact = affected_by_depth.get(
            1,
            []
        )

        transitive_impact = []

        for level, nodes in affected_by_depth.items():

            if level > 1:
                transitive_impact.extend(
                    nodes
                )

        return {
            "target": target_method,
            "matched_nodes": [
                str(node)
                for node in matched_nodes
            ],
            "depth_analyzed": depth,

            "total_affected": len(
                affected_nodes
            ),

            "direct_impact": [
                str(node)
                for node in direct_impact
            ],

            "transitive_impact": [
                str(node)
                for node in transitive_impact
            ],

            "affected_by_depth": {
                str(level): [
                    str(node)
                    for node in nodes
                ]
                for level, nodes
                in affected_by_depth.items()
            },

            "affected_components":
                affected_components,
        }

    def get_direct_impact(
        self,
        target_method: str
    ) -> Dict[str, Any]:
        """
        Returns only direct callers of a target method.
        """

        return self.analyze_blast_radius(
            target_method=target_method,
            depth=1
        )

    def get_full_blast_radius(
        self,
        target_method: str,
        max_depth: int = 5
    ) -> Dict[str, Any]:
        """
        Performs a deeper impact analysis.
        """

        return self.analyze_blast_radius(
            target_method=target_method,
            depth=max_depth
        )