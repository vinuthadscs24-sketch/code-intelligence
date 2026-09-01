import { apiClient } from "./client";

export const repositoryApi = {
  indexRepo: async (repoUrl, repoId = "test-repo") => {
    return apiClient("/v1/code-rag/repository/setup", {
      method: "POST",
      body: JSON.stringify({
        repo_id: repoId,
        repo_url_or_path: repoUrl,
        access_token: null,
        force_reclone: false,
        force_reindex: false,
      }),
    });
  },

  getStatus: async (repoId) => {
    const path = "/v1/code-rag/repository/status/" + repoId;
    return apiClient(path);
  },
};
