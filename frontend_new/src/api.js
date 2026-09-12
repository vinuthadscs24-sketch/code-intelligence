const API_BASE_URL = "http://localhost:8000";

export async function askCodebase(query) {
  const response = await fetch(`${API_BASE_URL}/api/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      errorText || `Request failed: ${response.status}`
    );
  }

  return response.json();
}

export async function getDependencies(entityName) {
  const response = await fetch(
    `${API_BASE_URL}/dependencies/${encodeURIComponent(entityName)}`
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      errorText || `Request failed: ${response.status}`
    );
  }

  return response.json();
}

export async function getImpact(entityName, maxDepth = 3) {
  const response = await fetch(
    `${API_BASE_URL}/impact/${encodeURIComponent(
      entityName
    )}?max_depth=${maxDepth}`
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      errorText || `Request failed: ${response.status}`
    );
  }

  return response.json();
}