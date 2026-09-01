import React, { useState } from "react";
import { GitBranch, CheckCircle, Loader2, AlertCircle } from "lucide-react";
import { repositoryApi } from "../api/repository";

export default function RepositorySetup({ activeRepo, onRepoLoaded }) {
  const [url, setUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");
  const [error, setError] = useState(null);

  // Poll backend status until setup completes or explicitly fails
  const pollStatus = async (repoId) => {
    const maxRetries = 30; // 30 retries * 2s = 60s max wait
    for (let i = 0; i < maxRetries; i++) {
      await new Promise((resolve) => setTimeout(resolve, 2000));
      try {
        const statusData = await repositoryApi.getStatus(repoId);

        // Break early if completed or failed
        if (statusData.status === "completed" || statusData.status === "indexed") {
          return statusData;
        }
        if (statusData.status === "failed" || statusData.error) {
          throw new Error(statusData.error || "Repository setup failed on backend.");
        }

        setStatusMessage(`Indexing in progress... (${(i + 1) * 2}s)`);
      } catch (err) {
        // Stop polling immediately if backend reports an error
        if (err.message.includes("failed")) throw err;
        console.warn("Polling error:", err);
      }
    }
    throw new Error("Indexing timed out on server.");
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!url.trim()) return;

    setLoading(true);
    setError(null);
    setStatusMessage("Starting setup...");

    try {
      // Clean up Windows backslashes for path inputs
      const cleanInput = url.trim().replace(/\\/g, "/");

      // Extract a clean repoId from path or URL (e.g., 'farm-equipment-rental-app')
      const derivedRepoId = cleanInput.split("/").filter(Boolean).pop().replace(".git", "");

      // 1. Trigger Setup API
      const initialData = await repositoryApi.indexRepo(cleanInput, derivedRepoId);
      const repoId = initialData?.repo_id || derivedRepoId;

      // 2. Poll status until backend status switches
      const finalData = await pollStatus(repoId);

      // 3. Notify parent component to update active codebase state
      if (onRepoLoaded) {
        onRepoLoaded({
          repoId: repoId,
          repoName: repoId,
          ...finalData,
        });
      }
      setStatusMessage("");
    } catch (err) {
      setError(err.message || "Failed to index repository");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-[#131B2E] border border-slate-800 rounded-xl p-6 mb-6">
      <div className="flex items-center gap-3 mb-4">
        <GitBranch className="w-5 h-5 text-blue-400" />
        <h2 className="text-sm font-bold font-mono text-white">Target Repository Setup</h2>
      </div>

      <form onSubmit={handleSubmit} className="flex gap-3 mb-4">
        <input
          type="text"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="Enter Git URL or Local Path (e.g. C:\path\to\farm-equipment-rental-app)"
          className="flex-1 bg-[#0B0F17] border border-slate-800 rounded-lg px-4 py-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-blue-500"
        />
        <button
          type="submit"
          disabled={loading}
          className="bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white text-xs font-mono px-4 py-2 rounded-lg font-semibold flex items-center gap-2 transition-colors"
        >
          {loading && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
          {loading ? "Indexing..." : "Index Repository"}
        </button>
      </form>

      {/* Progress & Error Displays */}
      {statusMessage && loading && (
        <p className="text-xs font-mono text-blue-400 mb-2">{statusMessage}</p>
      )}

      {error && (
        <div className="flex items-center gap-2 text-xs font-mono text-red-400 mb-2">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>Error: {error}</span>
        </div>
      )}

      {activeRepo && !loading && !error && (
        <div className="flex items-center gap-2 text-xs font-mono text-emerald-400">
          <CheckCircle className="w-4 h-4 shrink-0" />
          <span>Active target set: {activeRepo.repoName || activeRepo.repoId}</span>
        </div>
      )}
    </div>
  );
}