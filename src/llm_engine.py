import os
import json
from typing import Dict, Any, Optional
from src.git_intelligence import GitIntelligence
from src.hybrid_retriever import HybridRetriever


class CodeIntelligenceEngine:
    """
    Phase 8: Synthesizes structural graph context, semantic vector search, 
    and Git provenance into evidence-backed LLM prompt answers.
    """
    def __init__(
        self, 
        repo_path: str, 
        vector_store=None, 
        graph_db=None, 
        model_name: str = "qwen2.5-coder:latest",
        provider: str = "auto"
    ):
        self.repo_path = os.path.abspath(repo_path)
        self.vector_store = vector_store
        self.graph_db = graph_db
        self.git_intel = GitIntelligence(self.repo_path)
        self.model_name = model_name
        self.provider = provider.lower()
        if self.vector_store and self.graph_db:
            self.retriever = HybridRetriever(self.vector_store, self.graph_db)
        else:
            self.retriever = None

    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Invokes the configured LLM client (OpenAI, Anthropic, or Ollama) with standard fallbacks.
        """
        sys_msg = system_prompt or "You are a specialized Codebase Intelligence AI."

        # 1. Attempt OpenAI Call
        if (self.provider in ["auto", "openai"]) and os.getenv("OPENAI_API_KEY"):
            try:
                import openai
                client = openai.OpenAI()
                response = client.chat.completions.create(
                    model=self.model_name if "gpt" in self.model_name else "gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if self.provider == "openai":
                    return f"Error executing OpenAI call: {str(e)}"

        # 2. Attempt Anthropic Call
        if (self.provider in ["auto", "anthropic"]) and os.getenv("ANTHROPIC_API_KEY"):
            try:
                import anthropic
                client = anthropic.Anthropic()
                response = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1024,
                    system=sys_msg,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text.strip()
            except Exception as e:
                if self.provider == "anthropic":
                    return f"Error executing Anthropic call: {str(e)}"

        # 3. Fallback to Local Ollama Call
        ollama_models = [self.model_name, "qwen2.5-coder:latest", "qwen2.5-coder", "llama3"]
        for target_model in ollama_models:
            try:
                import urllib.request
                req = urllib.request.Request(
                    "http://127.0.0.1:11434/api/generate",
                    data=json.dumps({
                        "model": target_model,
                        "prompt": f"{sys_msg}\n\n{prompt}",
                        "stream": False
                    }).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    res_data = json.loads(resp.read().decode("utf-8"))
                    answer = res_data.get("response", "").strip()
                    if answer:
                        return answer
            except Exception:
                continue

        return "[LLM Execution Skipped]: No API keys detected (OPENAI_API_KEY / ANTHROPIC_API_KEY) and local Ollama is offline. Prompt was built successfully."

    def explain_why_changed(self, relative_file_path: str, method_name: str, start_line: int, end_line: int) -> Dict[str, Any]:
        """
        Gathers Git provenance + AST method code + caller/callee context 
        and formats an LLM query answering: 'Why was this method changed?'
        """
        provenance = self.git_intel.get_method_provenance(
            relative_file_path=relative_file_path,
            method_name=method_name,
            start_line=start_line,
            end_line=end_line
        )

        primary_commit = provenance.get("primary_commit", {})

        callers = self.graph_db.find_callers(method_name) if self.graph_db and hasattr(self.graph_db, "find_callers") else []
        callees = self.graph_db.find_callees(method_name) if self.graph_db and hasattr(self.graph_db, "find_callees") else []

        context_payload = {
            "symbol": method_name,
            "file": relative_file_path,
            "line_range": [start_line, end_line],
            "structural_graph": {
                "callers": callers,
                "callees": callees
            },
            "git_evidence": {
                "commit_hash": primary_commit.get("commit_hash"),
                "author": primary_commit.get("author"),
                "date": primary_commit.get("date"),
                "commit_message": primary_commit.get("subject"),
                "patch_diff": (primary_commit.get("diff") or "")[:1000]
            }
        }

        prompt = f"""You are a specialized Codebase Intelligence AI.
Analyze the provided code method, its Git history, commit messages, and patch diffs to answer: "Why was this method changed?"

=== RETRIEVED CONTEXT & EVIDENCE ===
{json.dumps(context_payload, indent=2)}

=== INSTRUCTIONS ===
1. State the exact primary reason for the change based on the commit message and patch diff.
2. Cite the Commit Hash ({primary_commit.get('commit_hash')}) and Author ({primary_commit.get('author')}).
3. Explain the technical modifications made inside the line range [{start_line}-{end_line}].

Explanation:"""

        answer = self._call_llm(prompt)

        return {
            "method_name": method_name,
            "context_payload": context_payload,
            "prompt": prompt,
            "answer": answer
        }

    def answer_query(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Answers general natural language questions using hybrid retrieval (Vector + Graph).
        Defaults to top_k=10 for deeper contextual coverage.
        """
        retrieved_chunks = []
        if self.retriever:
            retrieved_chunks = self.retriever.search(query, top_k=top_k)
        elif self.vector_store and hasattr(self.vector_store, "search"):
            raw_res = self.vector_store.search(query, top_k=top_k)
            retrieved_chunks = [r[0] if isinstance(r, tuple) else r for r in raw_res]

        # Prioritize implementation chunks over simple SystemProperties if available
        impl_chunks = [c for c in retrieved_chunks if "SystemProperties" not in c.get('file_name', '')]
        final_chunks = impl_chunks if impl_chunks else retrieved_chunks

        # Build enriched context string containing graph metadata
        context_blocks = []
        for c in final_chunks[:top_k]:
            callers_str = f"Callers: {', '.join(c.get('graph_callers', []))}" if c.get('graph_callers') else ""
            callees_str = f"Callees: {', '.join(c.get('graph_callees', []))}" if c.get('graph_callees') else ""
            graph_meta = f" | {callers_str} {callees_str}".strip()

            block = (
                f"File: {c.get('file_name', 'Unknown')}\n"
                f"Class: {c.get('class_name', '')} | Method: {c.get('method_name', '')} {graph_meta}\n"
                f"Code:\n{c.get('code_content', c.get('text_representation', ''))}"
            )
            context_blocks.append(block)

        context_str = "\n---\n".join(context_blocks) if context_blocks else "No relevant code chunks retrieved."

        prompt = f"""You are an expert AI software architect analyzing a Java repository.
Answer the user query based strictly on the provided codebase context and structural graph relationships.

=== USER QUERY ===
{query}

=== RETRIEVED HYBRID CONTEXT ===
{context_str}

=== INSTRUCTIONS ===
- Provide a clear, technical answer explaining which files and methods are involved.
- Reference specific method signatures, caller/callee relationships, and class names directly.
- If context is insufficient, state what is known from the context.
"""

        answer = self._call_llm(prompt)
        return {
            "query": query,
            "retrieved_chunks": final_chunks,
            "prompt": prompt,
            "answer": answer
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 8 Context Builder & LLM Engine")
    parser.add_argument("--repo", type=str, default="./")
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)

    args = parser.parse_args()
    engine = CodeIntelligenceEngine(repo_path=args.repo)
    result = engine.explain_why_changed(args.file, args.method, args.start, args.end)

    print("\n" + "="*80)
    print(f" PHASE 8 PROMPT: Why was {args.method}() changed?")
    print("="*80)
    print(result["prompt"])
    print("="*80)
    print(" LLM ANSWER")
    print("="*80)
    print(result["answer"])
    print("="*80 + "\n")