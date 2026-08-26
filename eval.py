import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Set

from src.parser import JavaASTParser
from src.chunker import CodeChunker
from src.vector_store import VectorStore
from src.graph_builder import CodeKnowledgeGraph
from src.hybrid_retriever import HybridRetriever


# =====================================================================
# Apache Commons Lang Benchmark Evaluation Dataset
# =====================================================================
BENCHMARK_DATASET = [
    {
        "id": 1,
        "query": "Where is function wrapping and FailableCallable conversion implemented?",
        "expected_symbols": {"Functions", "Failable", "asCallable", "FailableCallable"}
    },
    {
        "id": 2,
        "query": "Which classes handle date formatting and locale retrieval?",
        "expected_symbols": {"DatePrinter", "DateParser", "FastDateFormat", "getLocale"}
    },
    {
        "id": 3,
        "query": "Where is string manipulation or StringUtils located?",
        "expected_symbols": {"StringUtils", "substring", "isEmpty", "join"}
    },
    {
        "id": 4,
        "query": "How is reflection or FieldUtils implemented?",
        "expected_symbols": {"FieldUtils", "MethodUtils", "ClassUtils", "readField"}
    },
    {
        "id": 5,
        "query": "Where are array utilities and element lookup methods defined?",
        "expected_symbols": {"ArrayUtils", "add", "contains", "indexOf"}
    },
    {
        "id": 6,
        "query": "What classes handle system properties or SystemUtils?",
        "expected_symbols": {"SystemUtils", "IS_OS_WINDOWS", "getUserHome"}
    },
    {
        "id": 7,
        "query": "Where is object validation or Validate clause checking implemented?",
        "expected_symbols": {"Validate", "notNull", "isTrue", "notEmpty"}
    },
    {
        "id": 8,
        "query": "How is ExceptionUtils or stack trace printing structured?",
        "expected_symbols": {"ExceptionUtils", "getStackTrace", "getRootCause"}
    },
    {
        "id": 9,
        "query": "Where is RandomStringUtils or random string generation?",
        "expected_symbols": {"RandomStringUtils", "randomAlphanumeric", "randomAscii"}
    },
    {
        "id": 10,
        "query": "Where are tuple representations like Pair or Triple implemented?",
        "expected_symbols": {"Pair", "ImmutablePair", "Triple", "getLeft"}
    },
    {
        "id": 11,
        "query": "What classes handle number conversion and NumberUtils?",
        "expected_symbols": {"NumberUtils", "toInt", "createNumber", "isCreatable"}
    },
    {
        "id": 12,
        "query": "Where is Builder pattern or ToStringBuilder handled?",
        "expected_symbols": {"ToStringBuilder", "EqualsBuilder", "HashCodeBuilder", "append"}
    },
    {
        "id": 13,
        "query": "Which classes handle concurrent background initialization?",
        "expected_symbols": {"BackgroundInitializer", "CallableBackgroundInitializer", "initialize"}
    },
    {
        "id": 14,
        "query": "Where are char escaping utilities like StringEscapeUtils?",
        "expected_symbols": {"StringEscapeUtils", "escapeJava", "escapeHtml4"}
    },
    {
        "id": 15,
        "query": "Where is Range checking or numeric Range class defined?",
        "expected_symbols": {"Range", "between", "contains", "isAfter"}
    },
    {
        "id": 16,
        "query": "How are fast date parsing algorithms implemented in FastDateParser?",
        "expected_symbols": {"FastDateParser", "parse", "parsePattern"}
    },
    {
        "id": 17,
        "query": "Where is StopWatch or execution timing logic located?",
        "expected_symbols": {"StopWatch", "start", "stop", "getTime"}
    },
    {
        "id": 18,
        "query": "Where are functional consumer interface wrappers placed?",
        "expected_symbols": {"Functions", "asConsumer", "FailableConsumer"}
    },
    {
        "id": 19,
        "query": "Where is WordUtils for capitalization and word wrapping?",
        "expected_symbols": {"WordUtils", "capitalize", "wrap", "uncapitalize"}
    },
    {
        "id": 20,
        "query": "How is TypeUtils or generic type inspection implemented?",
        "expected_symbols": {"TypeUtils", "isAssignable", "getTypeArguments"}
    }
]


class CodebaseEvaluator:
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.vector_store = None
        self.graph_db = None
        self.retriever = None

    def setup(self):
        """Indexes codebase and initializes Vector Store and Knowledge Graph."""
        print(f"\n[Evaluator Setup] Indexing repository at '{self.repo_path}'...")
        parser = JavaASTParser()
        extracted_data = []

        for java_file in self.repo_path.rglob("*.java"):
            if "module-info.java" in java_file.name or "package-info.java" in java_file.name:
                continue
            try:
                res = parser.parse_file(str(java_file))
                if res:
                    if isinstance(res, dict):
                        extracted_data.append((Path(java_file), res))
                    elif isinstance(res, (list, tuple)):
                        tree = res[0]
                        source = res[1] if len(res) > 1 else ""
                        symbols = parser.extract_symbols_and_relations(tree, source)
                        extracted_data.append((Path(java_file), {
                            "tree": tree,
                            "source_code": source,
                            "symbols": symbols
                        }))
            except Exception:
                pass

        if not extracted_data:
            print("Warning: No Java files found.")
            return

        chunker = CodeChunker(parser=parser)
        chunks = chunker.create_chunks(extracted_data)

        self.graph_db = CodeKnowledgeGraph()
        self.graph_db.build_graph_from_chunks(chunks)

        self.vector_store = VectorStore()
        if not self.vector_store.load_index():
            self.vector_store.build_index(chunks)

        self.retriever = HybridRetriever(self.vector_store, self.graph_db)
        print(f"[Evaluator Setup] Ready: {len(chunks)} Chunks, {self.graph_db.get_summary()['total_nodes']} Graph Nodes.")

    def _extract_retrieved_symbols(self, chunk: Dict[str, Any]) -> Set[str]:
        """Extracts symbol names from a retrieved chunk dict."""
        symbols = set()
        if isinstance(chunk, dict):
            if chunk.get("class_name"):
                symbols.add(chunk["class_name"])
            if chunk.get("method_name"):
                symbols.add(chunk["method_name"])
            if chunk.get("file_name"):
                symbols.add(Path(chunk["file_name"]).stem)
            if chunk.get("id"):
                raw_id = str(chunk["id"])
                parts = raw_id.replace(".java", "").split("::")
                symbols.update(parts)
        return symbols

    def _evaluate_retrieved_set(self, retrieved_chunks: List[Dict[str, Any]], expected_symbols: Set[str], top_k: int) -> Dict[str, float]:
        """Calculates Recall@K, Precision@K, and MRR."""
        top_k_chunks = retrieved_chunks[:top_k]
        
        found_expected = set()
        first_hit_rank = 0

        for rank, chunk in enumerate(top_k_chunks, start=1):
            chunk_symbols = self._extract_retrieved_symbols(chunk)
            matched = chunk_symbols.intersection(expected_symbols)
            if matched:
                found_expected.update(matched)
                if first_hit_rank == 0:
                    first_hit_rank = rank

        recall = len(found_expected) / len(expected_symbols) if expected_symbols else 0.0
        # Formula Fix: Standard Precision@K calculation
        precision = len(found_expected) / top_k if top_k > 0 else 0.0
        mrr = 1.0 / first_hit_rank if first_hit_rank > 0 else 0.0

        return {
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "mrr": round(mrr, 4)
        }

    def run_benchmark(self, top_k: int = 5):
        """Runs evaluation queries across Vector, Graph, and Hybrid search strategies."""
        print(f"\n=======================================================================")
        print(f"       RUNNING CODEBASE RETRIEVAL BENCHMARK (Top-K = {top_k})")
        print(f"=======================================================================\n")

        results = {
            "vector": {"recall": [], "precision": [], "mrr": []},
            "graph": {"recall": [], "precision": [], "mrr": []},
            "hybrid": {"recall": [], "precision": [], "mrr": []}
        }

        print(f"{'ID':<4} | {'Query Snippet':<35} | {'Vector R@5':<10} | {'Graph R@5':<10} | {'Hybrid R@5':<10}")
        print("-" * 80)

        for item in BENCHMARK_DATASET:
            q_id = item["id"]
            query = item["query"]
            expected = item["expected_symbols"]

            # 1. Vector Search
            vec_chunks = []
            if self.vector_store and hasattr(self.vector_store, "search"):
                raw_vec = self.vector_store.search(query, top_k=top_k * 2)
                vec_chunks = [r[0] if isinstance(r, tuple) else r for r in raw_vec]
            vec_metrics = self._evaluate_retrieved_set(vec_chunks, expected, top_k)

            # 2. Graph Search (with vector store fallback)
            graph_chunks = []
            if self.graph_db and hasattr(self.graph_db, "search_graph"):
                graph_chunks = self.graph_db.search_graph(query, top_k=top_k * 2, vector_store=self.vector_store)
            graph_metrics = self._evaluate_retrieved_set(graph_chunks, expected, top_k)

            # 3. Hybrid Search
            hybrid_chunks = []
            if self.retriever:
                hybrid_chunks = self.retriever.search(query, top_k=top_k)
            else:
                hybrid_chunks = vec_chunks[:top_k]
            hybrid_metrics = self._evaluate_retrieved_set(hybrid_chunks, expected, top_k)

            # Record scores
            for key, metrics in [("vector", vec_metrics), ("graph", graph_metrics), ("hybrid", hybrid_metrics)]:
                results[key]["recall"].append(metrics["recall"])
                results[key]["precision"].append(metrics["precision"])
                results[key]["mrr"].append(metrics["mrr"])

            q_short = (query[:32] + "...") if len(query) > 35 else query
            print(f"{q_id:<4} | {q_short:<35} | {vec_metrics['recall']:<10} | {graph_metrics['recall']:<10} | {hybrid_metrics['recall']:<10}")

        print("\n" + "=" * 80)
        print("                        FINAL BENCHMARK SUMMARY")
        print("=" * 80)
        print(f"{'Strategy':<15} | {'Mean Recall@' + str(top_k):<15} | {'Mean Precision@' + str(top_k):<18} | {'Mean MRR':<10}")
        print("-" * 80)

        for strat in ["vector", "graph", "hybrid"]:
            avg_rec = sum(results[strat]["recall"]) / len(BENCHMARK_DATASET)
            avg_prec = sum(results[strat]["precision"]) / len(BENCHMARK_DATASET)
            avg_mrr = sum(results[strat]["mrr"]) / len(BENCHMARK_DATASET)
            print(f"{strat.capitalize():<15} | {avg_rec:<15.4f} | {avg_prec:<18.4f} | {avg_mrr:<10.4f}")

        print("=" * 80 + "\n")


if __name__ == "__main__":
    repo_input = sys.argv[1] if len(sys.argv) > 1 else "."
    evaluator = CodebaseEvaluator(repo_path=repo_input)
    evaluator.setup()
    evaluator.run_benchmark(top_k=5)