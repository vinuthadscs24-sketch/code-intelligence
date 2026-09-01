from typing import List, Dict, Any, Optional
from src.git_intelligence import GitIntelligence


class CodeIntelligenceContextBuilder:  # Fixed space in class name
    """
    Combines hybrid retriever chunks (Vector + Graph) with Git line blame, 
    method commit history, and patch diffs into a unified LLM prompt context block.
    """
    def __init__(self, git_intel: Optional[GitIntelligence] = None, max_diff_chars: int = 1000):
        self.git_intel = git_intel
        self.max_diff_chars = max_diff_chars

    def build_context(
        self, 
        query: str, 
        retrieved_chunks: List[Dict[str, Any]], 
        include_git: bool = True
    ) -> str:
        """
        Formats retrieved code chunks and Git metadata into a structured prompt block.
        """
        context_blocks = []
        
        context_blocks.append("=== CODEBASE CONTEXT ===")
        context_blocks.append(f"USER QUERY: {query}\n")

        for idx, chunk in enumerate(retrieved_chunks, start=1):
            file_name = chunk.get("file_name", chunk.get("file_path", "unknown"))
            method_name = chunk.get("method_name", "N/A")
            class_name = chunk.get("class_name", "N/A")
            start_line = chunk.get("start_line", 1)
            end_line = chunk.get("end_line", start_line + 15)
            code_content = chunk.get("code_content", chunk.get("content", ""))
            
            callers = chunk.get("graph_callers", [])
            callees = chunk.get("graph_callees", [])
            sources = ", ".join(chunk.get("sources", ["hybrid"]))
            score = chunk.get("combined_score", chunk.get("rrf_score", 0.0))

            block = []
            block.append(f"--- Chunk [{idx}] ---")
            block.append(f"File: {file_name}")
            block.append(f"Location: Class={class_name} | Method={method_name} (Lines {start_line}-{end_line})")
            block.append(f"Retrieval Meta: Sources=[{sources}] | Combined Score={score}")
            
            if callers or callees:
                block.append(f"Knowledge Graph: Callers={callers} | Callees={callees}")

            # Extract Git metadata if enabled and GitIntelligence available
            if include_git and self.git_intel:
                git_meta = self.git_intel.get_method_provenance(
                    relative_file_path=file_name,
                    method_name=str(method_name),
                    start_line=start_line,
                    end_line=end_line,
                    max_commits=3
                )
                
                primary_commit = git_meta.get("primary_commit", {})
                block.append("\n[Git Intelligence]")
                block.append(f"Primary Commit: {primary_commit.get('commit_hash', 'N/A')} by {primary_commit.get('author', 'Unknown')} on {primary_commit.get('date', 'Unknown')}")
                block.append(f"Commit Subject: {primary_commit.get('subject', 'N/A')}")
                
                diff_text = primary_commit.get("diff", "").strip()
                if diff_text:
                    truncated_diff = diff_text[:self.max_diff_chars]
                    if len(diff_text) > self.max_diff_chars:
                        truncated_diff += "\n... [diff truncated]"
                    block.append(f"Recent Diff:\n```diff\n{truncated_diff}\n```")

            block.append(f"\n[Code Segment]\n```java\n{code_content}\n```\n")
            context_blocks.append("\n".join(block))

        context_blocks.append("=== END OF CONTEXT ===")
        return "\n".join(context_blocks)

    def format_llm_prompt(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Wraps formatted codebase context into a complete system instruction prompt.
        """
        context_str = self.build_context(query, retrieved_chunks, include_git=True)
        
        prompt = f"""You are an expert AI Codebase Assistant. Answer the user's technical query using ONLY the provided code snippets, dependency graph information, and Git commit history context.

{context_str}

INSTRUCTIONS:
1. Cite specific file names, method names, and line numbers when referencing code.
2. If answering "Why" a change occurred, analyze the provided Git commit messages and diffs.
3. If structural relationships matter, explain how callers and callees interact based on Knowledge Graph data.
4. If the context does not contain enough information to answer, state clearly what is missing.

User Question: {query}
Answer:"""
        return prompt
        