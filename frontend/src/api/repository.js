import { apiClient } from "./client";

export const repositoryApi = {
  indexRepo: async (repoUrl) => {
    return apiClient("/api/repository/index", {
      method: "POST",
      body: JSON.stringify({ repo_url: repoUrl }),
    });
  },
  getStatus: async (repoId) => {
    return apiClient(`/api/repository/status/${repoId}`);
  },
};
