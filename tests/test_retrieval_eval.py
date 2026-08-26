import sys
import time
import argparse
from typing import List, Dict, Any
from pathlib import Path

# Add project root to path for relative imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Project imports
from src.vector_store import VectorStore
from src.graph_builder import CodeKnowledgeGraph  # Make sure this matches your graph class
from src.hybrid_retriever import HybridRetriever
from src.llm_engine import CodeIntelligenceEngine


DEFAULT_EVAL_BENCHMARK = [
    {
        "query": "Where are properties loaded and configured?",
        "expected_files": ["SystemProperties.java", "Configuration.java", "ConfigLoader.java"],
        "expected_methods": ["loadProperties", "getProperties", "init"]
    },
    {
        "query": "How is the HTTP client connection initialized?",
        "expected_files": ["HttpClient.java", "ConnectionManager.java", "NetworkClient.java"],
        "expected_methods": ["connect", "initClient", "sendRequest"]
    },
    {
        "query": "Where is user authentication token validated?",
        "expected_files": ["AuthManager.java", "TokenValidator.java", "SecurityFilter.java"],
        "expected_methods": ["validateToken", "authenticate", "verify"]
    }
]


class InteractiveRetrievalEvaluator:
    """
    Interactive evaluator and debugging tool for the Hybrid Retrieval Pipeline.
    """
    def __init__(self, repo_path: str = "./", chroma_path: str = "./chroma_db"):
        self.repo_path = repo_path
        self.chroma_path = chroma_path
        
        print("Initializing Retrieval Components...")
        self.vector_store = VectorStore(persist_directory=self.chroma_path)
        
        # Instantiate CodeKnowledgeGraph instead of undefined GraphDB
        self.graph_db = CodeKnowledgeGraph()
        
        # Load graph if database file exists
        graph_file = Path("codebase_graph.json")
        if graph_file.exists():
            print(f"Loading Graph DB from {graph_file}...")
            if hasattr(self.graph_db, "load_from_json"):
                self.graph_db.load_from_json(str(graph_file))

        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            graph_db=self.graph_db
        )
        self.llm_engine = CodeIntelligenceEngine(
            repo_path=self.repo_path,
            vector_store=self.vector_store,
            graph_db=self.graph_db
        )
        print("Pipeline components ready.\n")

    def evaluate_query(self, query: str, expected_files: List[str] = None, top_k: int = 5) -> Dict[str, Any]:
        """
        Executes search and calculates retrieval precision metrics.
        """
        start_time = time.time()
        results = self.retriever.search(query, top_k=top_k)
        latency_ms = (time.time() - start_time) * 1000

        retrieved_files = [r.get("file_name", r.get("file", "")) for r in results]
        
        hit_count = 0
        hit_targets = []
        if expected_files:
            expected_set = {f.lower() for f in expected_files}
            for rf in retrieved_files:
                rf_name = Path(rf).name.lower()
                if any(exp in rf_name for exp in expected_set):
                    hit_count += 1
                    hit_targets.append(rf)

        precision = (hit_count / len(results)) if results else 0.0

        return {
            "query": query,
            "latency_ms": round(latency_ms, 2),
            "top_k": top_k,
            "results_returned": len(results),
            "retrieved_files": retrieved_files,
            "hits": hit_count,
            "precision": round(precision, 2),
            "raw_results": results
        }

    def print_result_details(self, eval_res: Dict[str, Any], show_code: bool = False):
        """
        Prints clean formatted output of retrieval results.
        """
        print("\n" + "=" * 80)
        print(f" QUERY: {eval_res['query']}")
        print("=" * 80)
        print(f" Latency: {eval_res['latency_ms']} ms | Returned: {eval_res['results_returned']} chunks | Precision: {eval_res['precision']}")
        print("-" * 80)

        for i, chunk in enumerate(eval_res["raw_results"], 1):
            file_name = chunk.get("file_name", chunk.get("file", "Unknown"))
            class_name = chunk.get("class_name", "")
            method_name = chunk.get("method_name", "")
            score = chunk.get("combined_score", chunk.get("score", 0.0))

            callers = chunk.get("graph_callers", [])
            callees = chunk.get("graph_callees", [])
            graph_info = ""
            if callers or callees:
                graph_info = f" | Callers: {len(callers)} | Callees: {len(callees)}"

            print(f" [{i}] {file_name} -> {class_name}::{method_name} (Score: {score:.4f}){graph_info}")
            
            if show_code:
                code_snippet = chunk.get("code_content", chunk.get("text_representation", ""))[:200]
                print(f"     Code Preview:\n     {code_snippet.strip().replace('\n', '\n     ')}\n")
        print("=" * 80 + "\n")

    def run_benchmark_suite(self):
        """
        Runs automated evaluation benchmark suite.
        """
        print("\n Running Evaluation Benchmark Suite...")
        total_precision = 0.0
        total_latency = 0.0

        for bench in DEFAULT_EVAL_BENCHMARK:
            res = self.evaluate_query(
                query=bench["query"], 
                expected_files=bench["expected_files"], 
                top_k=5
            )
            self.print_result_details(res, show_code=False)
            total_precision += res["precision"]
            total_latency += res["latency_ms"]

        avg_precision = total_precision / len(DEFAULT_EVAL_BENCHMARK)
        avg_latency = total_latency / len(DEFAULT_EVAL_BENCHMARK)

        print("=" * 80)
        print(" BENCHMARK SUMMARY RESULTS")
        print("=" * 80)
        print(f" Average Precision @ Top-5: {avg_precision:.2f}")
        print(f" Average Latency:           {avg_latency:.2f} ms")
        print("=" * 80 + "\n")

    def start_interactive_repl(self):
        """
        Interactive Command Line REPL mode.
        """
        print("=" * 80)
        print(" INTERACTIVE RETRIEVAL EVALUATOR REPL")
        print(" Type your natural language queries below.")
        print(" Commands: ")
        print("   :code    - Toggle full code display")
        print("   :llm     - Generate full LLM answer for current query")
        print("   :bench   - Run benchmark evaluation suite")
        print("   :quit    - Exit interactive mode")
        print("=" * 80 + "\n")

        show_code = False
        last_query = None

        while True:
            try:
                user_input = input("eval-retrieval> ").strip()
                if not user_input:
                    continue

                if user_input.lower() in [":quit", ":exit", "exit", "quit"]:
                    print("Exiting interactive evaluator.")
                    break

                if user_input.lower() == ":code":
                    show_code = not show_code
                    print(f"Code preview display toggled to: {show_code}")
                    continue

                if user_input.lower() == ":bench":
                    self.run_benchmark_suite()
                    continue

                if user_input.lower() == ":llm":
                    if not last_query:
                        print("No previous query found. Please run a search query first.")
                        continue
                    print(f"\nGenerating LLM Answer for: '{last_query}'...")
                    answer_res = self.llm_engine.answer_query(last_query)
                    print("\n" + "=" * 80)
                    print(" LLM ANSWER")
                    print("=" * 80)
                    print(answer_res["answer"])
                    print("=" * 80 + "\n")
                    continue

                # Run query evaluation
                last_query = user_input
                res = self.evaluate_query(user_input, top_k=5)
                self.print_result_details(res, show_code=show_code)

            except KeyboardInterrupt:
                print("\nExiting interactive mode.")
                break
            except Exception as e:
                print(f"Error evaluating query: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Hybrid Retrieval Evaluator")
    parser.add_argument("--repo", type=str, default="./", help="Path to Git repository root")
    parser.add_argument("--chroma", type=str, default="./chroma_db", help="Path to ChromaDB storage")
    parser.add_argument("--bench", action="store_true", help="Run automated benchmark directly and exit")

    args = parser.parse_args()
    evaluator = InteractiveRetrievalEvaluator(repo_path=args.repo, chroma_path=args.chroma)

    if args.bench:
        evaluator.run_benchmark_suite()
    else:
        evaluator.start_interactive_repl()