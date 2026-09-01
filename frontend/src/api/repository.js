import { apiClient } from "./client";

export const repositoryApi = {
  indexRepo: async (repoUrlOrPath, repoId = null, forceReindex = true) => {
    // Automatically derive a clean repoId if none is provided
    const cleanInput = repoUrlOrPath.trim().replace(/\\/g, "/");
    const derivedRepoId =
      repoId ||
      cleanInput.split("/").filter(Boolean).pop().replace(".git", "") ||
      "indexed-repo";

    return apiClient("/v1/code-rag/repository/setup", {
      method: "POST",
      body: JSON.stringify({
        repo_id: derivedRepoId,
        repo_url_or_path: cleanInput,
        access_token: null,
        force_reclone: false,
        force_reindex: forceReindex,
      }),
    });
  },

  getStatus: async (repoId) => {
    const path = "/v1/code-rag/repository/status/" + encodeURIComponent(repoId);
    return apiClient(path);
  },
};