import axios from 'axios';

const API_BASE = 'http://127.0.0.1:8000';

export const api = {
  // 1. Trigger repository setup & parsing
  setupRepo: async (repoUrl) => {
    const res = await axios.post(`${API_BASE}/v1/code-rag/repository/setup`, { repo_url: repoUrl });
    return res.data;
  },

  // 2. Fetch repository indexing status
  getRepoStatus: async (repoId) => {
    const res = await axios.get(`${API_BASE}/v1/code-rag/repository/status/${repoId}`);
    return res.data;
  },

  // 3. Query code retrieval & reasoning engine
  queryCode: async (query, repoId = "spring-petclinic") => {
    const res = await axios.post(`${API_BASE}/v1/code-rag/query`, {
      query,
      repo_id: repoId
    });
    return res.data;
  }
};