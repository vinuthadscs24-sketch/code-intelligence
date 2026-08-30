const BASE_URL = "http://localhost:8000";

export async function apiClient(endpoint, options = {}) {
  const config = {
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
    ...options,
  };

  try {
    const response = await fetch(`${BASE_URL}${endpoint}`, config);
    const data = await response.json();

    if (!response.ok) {
      // Extract FastAPI's HTTPException detail string
      throw new Error(data.detail || `Server error: ${response.status}`);
    }

    return data;
  } catch (error) {
    if (error.name === "TypeError" && error.message === "Failed to fetch") {
      throw new Error("Backend server is offline. Verify FastAPI on port 8000.");
    }
    throw error;
  }
}