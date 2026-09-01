import { apiClient } from "./client";

export const queryApi = {
  askCodebase: async (
    repoId,
    queryText,
    topNFinal = 5
  ) => {
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
};