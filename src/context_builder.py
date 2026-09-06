from typing import List, Dict, Any, Optional

from src.git_intelligence import GitIntelligence


class CodeIntelligenceContextBuilder:
    """
    Combines hybrid retriever chunks with Git history,
    provenance, and diff information into LLM context.
    """

    def __init__(
        self,
        git_intel: Optional[GitIntelligence] = None,
        max_diff_chars: int = 1000,
    ):
        self.git_intel = git_intel
        self.max_diff_chars = max_diff_chars

    # =========================================================
    # WHY CHANGED / PROVENANCE
    # =========================================================

    def explain_why_changed(
        self,
        file_path: str,
        method_name: str,
        start_line: int,
        end_line: int,
    ) -> Dict[str, Any]:
        """
        Explain why a method or code region changed
        using Git history, commit metadata, and diffs.
        """

        if self.git_intel is None:
            return {
                "answer": "Git intelligence is not initialized.",
                "file": file_path,
                "method": method_name,
                "start_line": start_line,
                "end_line": end_line,
                "commits": [],
            }

        try:
            provenance = self.git_intel.get_method_provenance(
                relative_file_path=file_path,
                method_name=method_name,
                start_line=start_line,
                end_line=end_line,
                max_commits=5,
            )

            if not provenance:
                return {
                    "answer": (
                        f"No Git provenance information was found "
                        f"for {method_name} in {file_path}."
                    ),
                    "file": file_path,
                    "method": method_name,
                    "start_line": start_line,
                    "end_line": end_line,
                    "commits": [],
                }

            primary_commit = provenance.get(
                "primary_commit",
                {},
            )

            commits = provenance.get(
                "commits",
                [],
            )

            # Some GitIntelligence implementations may only
            # return primary_commit.
            if not commits and primary_commit:
                commits = [primary_commit]

            # If primary_commit is missing, use first commit.
            if not primary_commit and commits:
                primary_commit = commits[0]

            commit_hash = primary_commit.get(
                "commit_hash",
                "N/A",
            )

            author = primary_commit.get(
                "author",
                "Unknown",
            )

            date = primary_commit.get(
                "date",
                "Unknown",
            )

            subject = primary_commit.get(
                "subject",
                "No commit message available.",
            )

            diff = primary_commit.get(
                "diff",
                "",
            )

            answer = (
                f"{method_name} in {file_path} "
                f"was last associated with commit "
                f"{commit_hash}. "
                f"The change was made by {author} "
                f"on {date}. "
                f"Commit message: {subject}"
            )

            return {
                "answer": answer,
                "file": file_path,
                "method": method_name,
                "start_line": start_line,
                "end_line": end_line,
                "primary_commit": primary_commit,
                "commits": commits,
                "diff": diff,
            }

        except Exception as exc:
            return {
                "answer": (
                    f"Unable to determine why "
                    f"{method_name} changed: {exc}"
                ),
                "file": file_path,
                "method": method_name,
                "start_line": start_line,
                "end_line": end_line,
                "commits": [],
            }

    # =========================================================
    # BUILD CONTEXT
    # =========================================================

    def build_context(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        include_git: bool = True,
    ) -> str:
        """
        Formats retrieved code chunks, graph relationships,
        and Git metadata into a structured context block.
        """

        context_blocks = []

        context_blocks.append(
            "=== CODEBASE CONTEXT ==="
        )

        context_blocks.append(
            f"USER QUERY: {query}\n"
        )

        for idx, chunk in enumerate(
            retrieved_chunks,
            start=1,
        ):

            file_name = chunk.get(
                "file_name",
                chunk.get(
                    "file_path",
                    "unknown",
                ),
            )

            method_name = chunk.get(
                "method_name",
                "N/A",
            )

            class_name = chunk.get(
                "class_name",
                "N/A",
            )

            start_line = chunk.get(
                "start_line",
                1,
            )

            end_line = chunk.get(
                "end_line",
                start_line + 15,
            )

            code_content = chunk.get(
                "code_content",
                chunk.get(
                    "content",
                    "",
                ),
            )

            callers = chunk.get(
                "graph_callers",
                [],
            )

            callees = chunk.get(
                "graph_callees",
                [],
            )

            sources = chunk.get(
                "sources",
                ["hybrid"],
            )

            if isinstance(sources, list):
                sources = ", ".join(
                    str(source)
                    for source in sources
                )
            else:
                sources = str(sources)

            score = chunk.get(
                "combined_score",
                chunk.get(
                    "rrf_score",
                    0.0,
                ),
            )

            block = []

            block.append(
                f"--- Chunk [{idx}] ---"
            )

            block.append(
                f"File: {file_name}"
            )

            block.append(
                f"Location: Class={class_name} | "
                f"Method={method_name} "
                f"(Lines {start_line}-{end_line})"
            )

            block.append(
                f"Retrieval Meta: "
                f"Sources=[{sources}] | "
                f"Combined Score={score}"
            )

            # -------------------------------------------------
            # Knowledge Graph
            # -------------------------------------------------

            if callers or callees:

                block.append(
                    f"Knowledge Graph: "
                    f"Callers={callers} | "
                    f"Callees={callees}"
                )

            # -------------------------------------------------
            # Git Intelligence
            # -------------------------------------------------

            if include_git and self.git_intel:

                try:

                    git_meta = (
                        self.git_intel.get_method_provenance(
                            relative_file_path=file_name,
                            method_name=str(
                                method_name
                            ),
                            start_line=start_line,
                            end_line=end_line,
                            max_commits=3,
                        )
                    )

                    primary_commit = (
                        git_meta.get(
                            "primary_commit",
                            {},
                        )
                    )

                    block.append(
                        "\n[Git Intelligence]"
                    )

                    block.append(
                        "Primary Commit: "
                        f"{primary_commit.get('commit_hash', 'N/A')} "
                        "by "
                        f"{primary_commit.get('author', 'Unknown')} "
                        "on "
                        f"{primary_commit.get('date', 'Unknown')}"
                    )

                    block.append(
                        "Commit Subject: "
                        f"{primary_commit.get('subject', 'N/A')}"
                    )

                    diff_text = (
                        primary_commit.get(
                            "diff",
                            "",
                        )
                    )

                    if diff_text:

                        diff_text = (
                            diff_text.strip()
                        )

                        truncated_diff = (
                            diff_text[
                                :self.max_diff_chars
                            ]
                        )

                        if len(diff_text) > self.max_diff_chars:

                            truncated_diff += (
                                "\n... [diff truncated]"
                            )

                        block.append(
                            "Recent Diff:\n"
                            "```diff\n"
                            f"{truncated_diff}\n"
                            "```"
                        )

                except Exception as exc:

                    block.append(
                        "[Git Intelligence] "
                        f"Unavailable: {exc}"
                    )

            # -------------------------------------------------
            # Code
            # -------------------------------------------------

            block.append(
                "\n[Code Segment]\n"
                "```text\n"
                f"{code_content}\n"
                "```\n"
            )

            context_blocks.append(
                "\n".join(block)
            )

        context_blocks.append(
            "=== END OF CONTEXT ==="
        )

        return "\n".join(
            context_blocks
        )

    # =========================================================
    # LLM PROMPT
    # =========================================================

    def format_llm_prompt(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> str:
        """
        Wraps formatted codebase context into
        a complete LLM instruction prompt.
        """

        context_str = self.build_context(
            query,
            retrieved_chunks,
            include_git=True,
        )

        prompt = f"""
You are an expert AI Codebase Assistant.

Answer the user's technical query using ONLY the
provided code snippets, dependency graph information,
and Git commit history context.

{context_str}

INSTRUCTIONS:

1. Cite specific file names, method names,
   and line numbers when referencing code.

2. If answering "Why" a change occurred,
   analyze the provided Git commit messages
   and diffs.

3. If structural relationships matter,
   explain how callers and callees interact
   based on Knowledge Graph data.

4. If the context does not contain enough
   information to answer, state clearly
   what is missing.

User Question: {query}

Answer:
"""

        return prompt