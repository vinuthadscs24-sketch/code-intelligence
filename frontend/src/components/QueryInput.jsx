import React, { useState } from "react";
import { Search, Loader2 } from "lucide-react";

export default function QueryInput({ onSubmit, loading }) {
  const [query, setQuery] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim() && !loading) {
      onSubmit(query);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="relative">
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="e.g. How does the PetController route incoming requests?"
        className="w-full bg-[#131B2E] border border-slate-800 rounded-xl py-3.5 pl-4 pr-12 text-xs font-mono text-slate-200 placeholder-slate-500 focus:outline-none focus:border-blue-500 shadow-inner"
      />
      <button
        type="submit"
        disabled={loading || !query.trim()}
        className="absolute right-2 top-2 p-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded-lg transition"
      >
        {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
      </button>
    </form>
  );
}
