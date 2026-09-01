import { apiClient } from "./client";

const BASE_URL = "http://localhost:8000";

export const queryApi = {
  // 1. Standard Non-Streaming Query
  askCodebase: async (repoId, queryText, topNFinal = 5) => {
    return apiClient("/v1/code-rag/query", {
      method: "POST",
      body: JSON.stringify({
        repo_id: repoId,
        query_text: queryText,
        top_n_final: topNFinal,
        vector_top_k: 10,
        bm25_top_k: 10,
        rewrite_query: null,
        rewrite_prompt: null,
      }),
    });
  },

  // 2. Adaptive Response Query (Returns visual panel data, evidence, & retrieval trace)
  askCodebaseAdaptive: async (repoId, queryText, debug = false) => {
    const url = `/v1/code-rag/query/adaptive?query=${encodeURIComponent(queryText)}&repo_id=${encodeURIComponent(repoId)}&debug=${debug}`;
    return apiClient(url, {
      method: "POST",
    });
  },

  // 3. Streamed Query
  askCodebaseStream: async function* (repoId, queryText, topNFinal = 5) {
    const response = await fetch(`${BASE_URL}/v1/code-rag/query/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        repo_id: repoId,
        query_text: queryText,
        top_n_final: topNFinal,
        vector_top_k: 10,
        bm25_top_k: 10,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      yield decoder.decode(value, { stream: true });
    }
  },
};