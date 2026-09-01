import React, { useState } from "react";
import { Send, Loader2, Code2, Bug } from "lucide-react";
import RepositorySetup from "../components/RepositorySetup";
import AdaptivePanel from "../components/AdaptivePanel";
import { queryApi } from "../api/query";

export default function AskCodebase() {
  const [activeRepo, setActiveRepo] = useState({ repoId: "farm-equipment-rental-app" });
  const [queryText, setQueryText] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  
  // New Adaptive UI States
  const [debugMode, setDebugMode] = useState(true);
  const [adaptiveResponse, setAdaptiveResponse] = useState(null);

  const handleSendQuery = async (e) => {
    e.preventDefault();
    if (!queryText.trim() || !activeRepo?.repoId) return;

    setLoading(true);
    setAnswer("");
    setAdaptiveResponse(null);

    // 1. Primary Attempt: Query the Unified Adaptive Endpoint
    try {
      const data = await queryApi.askCodebaseAdaptive(
        activeRepo.repoId,
        queryText,
        debugMode
      );
      
      setAdaptiveResponse(data);
      setAnswer(data.answer_text || "Analysis complete.");
    } catch (adaptiveErr) {
      console.warn("Adaptive API failed, trying streaming fallback:", adaptiveErr);

      // 2. Fallback 1: Stream Response
      try {
        const stream = queryApi.askCodebaseStream(activeRepo.repoId, queryText);
        let streamContent = "";

        for await (const chunk of stream) {
          streamContent += chunk;
          setAnswer(streamContent);
        }

        if (!streamContent) {
          setAnswer("No response received from model stream.");
        }
      } catch (streamErr) {
        console.warn("Streaming failed, executing legacy non-streaming fallback:", streamErr);

        // 3. Fallback 2: Legacy Standard Response
        try {
          const res = await queryApi.askCodebase(activeRepo.repoId, queryText);
          setAnswer(res.answer || res.response || JSON.stringify(res));
        } catch (fallbackErr) {
          setAnswer(
            `Error querying ${activeRepo.repoId}: ${
              fallbackErr.message || "Failed to fetch response."
            }`
          );
        }
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0B0F17] text-slate-200 p-8 max-w-5xl mx-auto font-sans">
      {/* Header Bar */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <Code2 className="w-7 h-7 text-blue-500" />
          <h1 className="text-xl font-bold font-mono text-white">Ask Codebase</h1>
        </div>

        <div className="flex items-center gap-3">
          {/* Debug Mode Toggle */}
          <button
            onClick={() => setDebugMode(!debugMode)}
            className={`flex items-center gap-2 text-xs font-mono px-3.5 py-1.5 rounded-full border transition-all ${
              debugMode
                ? "bg-blue-950/80 border-blue-500 text-blue-300"
                : "bg-[#131B2E] border-slate-800 text-slate-500"
            }`}
          >
            <Bug className="w-3.5 h-3.5" />
            <span>Trace: {debugMode ? "ON" : "OFF"}</span>
          </button>

          {/* Active Target Badge */}
          <div className="bg-blue-950/60 border border-blue-800 text-blue-400 text-xs px-3.5 py-1.5 rounded-full font-mono flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-blue-400 animate-pulse" />
            <span>Context: <strong>{activeRepo.repoId}</strong></span>
          </div>
        </div>
      </div>

      {/* Indexing Section */}
      <RepositorySetup 
        activeRepo={activeRepo} 
        onRepoLoaded={(newRepoData) => setActiveRepo(newRepoData)} 
      />

      {/* Query Form */}
      <form onSubmit={handleSendQuery} className="flex gap-3 mb-6">
        <input
          type="text"
          value={queryText}
          onChange={(e) => setQueryText(e.target.value)}
          placeholder={`e.g. 'Who calls BookingService?' or 'What breaks if I change AuthController?'`}
          className="flex-1 bg-[#131B2E] border border-slate-800 rounded-xl px-5 py-3 text-sm font-mono text-slate-100 focus:outline-none focus:border-blue-500"
        />
        <button
          type="submit"
          disabled={loading}
          className="bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white px-6 py-3 rounded-xl font-mono text-sm font-semibold flex items-center gap-2 transition-colors"
        >
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
          Query
        </button>
      </form>

      {/* Output Display Container */}
      <div className="space-y-6">
        {/* AI Text Explanation Box */}
        <div className="bg-[#131B2E] border border-slate-800 rounded-xl p-6 font-mono text-xs leading-relaxed text-slate-300">
          <span className="text-[10px] uppercase font-bold text-blue-400 block mb-3 tracking-wider">
            AI Engine Synthesis
          </span>
          <div className="whitespace-pre-wrap">
            {answer || (loading ? "Generating analysis from engine..." : "Submit a query above to analyze dependencies, execution flow, git history, or impact.")}
          </div>
        </div>

        {/* Visual Adaptive Panel (Graphs, Metrics, Git Timelines, RRF Traces) */}
        {adaptiveResponse && <AdaptivePanel response={adaptiveResponse} />}
      </div>
    </div>
  );
}