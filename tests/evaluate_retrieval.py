import os
import time
import json
import logging
from typing import Dict, Any, List, Set, Tuple

# Set up logging format
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RetrievalEvaluator")


# Dynamic imports for local modules
try:
    from src.hybrid_retriever import HybridRetriever
except ImportError:
    HybridRetriever = None


# Define Golden Test Dataset (Ground Truth for Retrieval)
GOLDEN_TEST_SET = [
    {
        "query": "Find the method responsible for extracting line git blame and provenance.",
        "ground_truth_methods": ["get_line_blame", "get_method_provenance"],
        "ground_truth_files": ["git_intelligence.py"]
    },
    {
        "query": "How is reciprocal rank fusion combined across vector and graph search results?",
        "ground_truth_methods": ["search", "reciprocal_rank_fusion"],
        "ground_truth_files": ["hybrid_retriever.py"]
    },
    {
        "query": "Where is the LLM execution handled with model fallbacks?",
        "ground_truth_methods": ["_call_llm", "explain_why_changed", "answer_query"],
        "ground_truth_files": ["code_intelligence_engine.py", "engine.py"]
    },
    {
        "query": "Which function builds the AST call graph from source files?",
        "ground_truth_methods": ["build_graph", "parse_ast"],
        "ground_truth_files": ["graph_builder.py", "ast_parser.py"]
    }
]


class RetrievalEvaluator:
    """
    Evaluates Retrieval Quality across Vector-Only, Graph-Only, and Hybrid (RRF) modes.
    Computes Precision@K, Recall@K, F1@K, and MRR.
    """
    def __init__(self, vector_store=None, graph_db=None):
        self.vector_store = vector_store
        self.graph_db = graph_db
        
        if HybridRetriever and (vector_store or graph_db):
            self.hybrid_retriever = HybridRetriever(vector_store=vector_store, graph_db=graph_db)
        else:
            self.hybrid_retriever = None

    def _retrieve_vector_only(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if not self.vector_store or not hasattr(self.vector_store, "search"):
            return []
        try:
            raw_res = self.vector_store.search(query, top_k=top_k)
            return [r[0] if isinstance(r, tuple) else r for r in raw_res]
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def _retrieve_graph_only(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if not self.graph_db:
            return []
        try:
            # Simple term extraction for symbol matching against graph nodes
            query_terms = [t.lower() for t in query.replace("_", " ").split() if len(t) > 3]
            nodes = []
            
            if hasattr(self.graph_db, "get_all_nodes"):
                nodes = self.graph_db.get_all_nodes()
            elif hasattr(self.graph_db, "nodes"):
                nodes = list(self.graph_db.nodes(data=True))

            matched = []
            for node in nodes:
                node_name = node if isinstance(node, str) else str(node)
                if any(term in node_name.lower() for term in query_terms):
                    matched.append({
                        "method_name": node_name,
                        "file_name": getattr(node, "file_name", "unknown")
                    })
            return matched[:top_k]
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []

    def _retrieve_hybrid(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if self.hybrid_retriever:
            return self.hybrid_retriever.search(query, top_k=top_k)
        return []

    @staticmethod
    def calculate_metrics(
        retrieved_items: List[Dict[str, Any]], 
        truth_methods: List[str], 
        truth_files: List[str], 
        k: int
    ) -> Dict[str, float]:
        """
        Calculates Precision@K, Recall@K, F1@K, and Mean Reciprocal Rank (MRR).
        """
        top_k_items = retrieved_items[:k]
        if not top_k_items:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mrr": 0.0}

        relevant_found = 0
        first_rank = 0

        target_methods = {m.lower() for m in truth_methods}
        target_files = {f.lower() for f in truth_files}

        for rank, item in enumerate(top_k_items, start=1):
            item_method = str(item.get("method_name", "")).lower()
            item_file = str(item.get("file_name", "")).lower()

            # Item matches if method or file matches ground truth
            is_relevant = (item_method in target_methods) or any(tf in item_file for tf in target_files)

            if is_relevant:
                relevant_found += 1
                if first_rank == 0:
                    first_rank = rank

        precision = relevant_found / float(k)
        total_relevant = float(len(target_methods) + len(target_files))
        recall = relevant_found / total_relevant if total_relevant > 0 else 0.0
        
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        mrr = 1.0 / first_rank if first_rank > 0 else 0.0

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "mrr": round(mrr, 4)
        }

    def run_benchmark(self, k_list: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
        """
        Runs benchmarking across test suite for Vector, Graph, and Hybrid modes.
        """
        modes = ["Vector-Only", "Graph-Only", "Hybrid (RRF)"]
        results: Dict[str, Any] = {mode: {k: {"precision": [], "recall": [], "f1": [], "mrr": []} for k in k_list} for mode in modes}

        logger.info(f"Starting Benchmark across {len(GOLDEN_TEST_SET)} queries...")

        for item in GOLDEN_TEST_SET:
            query = item["query"]
            t_methods = item["ground_truth_methods"]
            t_files = item["ground_truth_files"]

            for k in k_list:
                # 1. Vector Only
                vec_retrieved = self._retrieve_vector_only(query, top_k=k)
                v_metrics = self.calculate_metrics(vec_retrieved, t_methods, t_files, k)
                for metric in v_metrics:
                    results["Vector-Only"][k][metric].append(v_metrics[metric])

                # 2. Graph Only
                graph_retrieved = self._retrieve_graph_only(query, top_k=k)
                g_metrics = self.calculate_metrics(graph_retrieved, t_methods, t_files, k)
                for metric in g_metrics:
                    results["Graph-Only"][k][metric].append(g_metrics[metric])

                # 3. Hybrid
                hyb_retrieved = self._retrieve_hybrid(query, top_k=k)
                h_metrics = self.calculate_metrics(hyb_retrieved, t_methods, t_files, k)
                for metric in h_metrics:
                    results["Hybrid (RRF)"][k][metric].append(h_metrics[metric])

        # Compute Averages
        summary = {}
        for mode in modes:
            summary[mode] = {}
            for k in k_list:
                summary[mode][f"K={k}"] = {
                    "MAP@K": round(sum(results[mode][k]["precision"]) / len(GOLDEN_TEST_SET), 4),
                    "MAR@K": round(sum(results[mode][k]["recall"]) / len(GOLDEN_TEST_SET), 4),
                    "F1@K": round(sum(results[mode][k]["f1"]) / len(GOLDEN_TEST_SET), 4),
                    "MRR": round(sum(results[mode][k]["mrr"]) / len(GOLDEN_TEST_SET), 4)
                }

        return summary


def print_formatted_results(benchmark_summary: Dict[str, Any]):
    """Prints a terminal evaluation report."""
    print("\n" + "=" * 80)
    print("                      CODE RETRIEVAL BENCHMARK REPORT                      ")
    print("=" * 80)

    for mode, k_data in benchmark_summary.items():
        print(f"\n--- Strategy: {mode} ---")
        print(f"{'Cutoff (K)':<12} | {'Precision@K':<12} | {'Recall@K':<12} | {'F1@K':<12} | {'MRR':<10}")
        print("-" * 70)
        for k_val, metrics in k_data.items():
            print(
                f"{k_val:<12} | "
                f"{metrics['MAP@K']:<12.4f} | "
                f"{metrics['MAR@K']:<12.4f} | "
                f"{metrics['F1@K']:<12.4f} | "
                f"{metrics['MRR']:<10.4f}"
            )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Mock fallback runner for standalone testing
    evaluator = RetrievalEvaluator()
    summary = evaluator.run_benchmark(k_list=[1, 3, 5])
    print_formatted_results(summary)