import React, { useState } from "react";
import { GitBranch, CheckCircle, Loader2 } from "lucide-react";
import { repositoryApi } from "../api/repository";

export default function RepositorySetup({ activeRepo, onRepoLoaded }) {
  const [url, setUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!url) return;
    setLoading(true);
    setError(null);
    try {
      const data = await repositoryApi.indexRepo(url);
      if (onRepoLoaded) onRepoLoaded(data);
    } catch (err) {
      setError(err.message || "Failed to index repository");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-[#131B2E] border border-slate-800 rounded-xl p-6">
      <div className="flex items-center gap-3 mb-4">
        <GitBranch className="w-5 h-5 text-blue-400" />
        <h2 className="text-sm font-bold font-mono text-white">Target Repository Setup</h2>
      </div>

      <form onSubmit={handleSubmit} className="flex gap-3 mb-4">
        <input
          type="text"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="https://github.com/owner/repository"
          className="flex-1 bg-[#0B0F17] border border-slate-800 rounded-lg px-4 py-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-blue-500"
        />
        <button
          type="submit"
          disabled={loading}
          className="bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white text-xs font-mono px-4 py-2 rounded-lg font-semibold flex items-center gap-2"
        >
          {loading && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
          {loading ? "Indexing..." : "Index Repository"}
        </button>
      </form>

      {error && <p className="text-xs font-mono text-red-400">?? {error}</p>}
      {activeRepo && (
        <div className="flex items-center gap-2 text-xs font-mono text-emerald-400">
          <CheckCircle className="w-4 h-4" />
          <span>Active target set: {activeRepo.repoName || url}</span>
        </div>
      )}
    </div>
  );
}
