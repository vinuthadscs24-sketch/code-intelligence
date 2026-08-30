import { apiClient } from "./client";

export const queryApi = {
  askCodebase: async (question, topK = 5) => {
    return apiClient("/ask", {
      method: "POST",
      body: JSON.stringify({
        question: question,
        top_k: topK,
      }),
    });
  },
};