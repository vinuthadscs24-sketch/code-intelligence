import React, { useState } from "react";
import { queryApi } from "../api/query";
import QueryInput from "../components/QueryInput";
import AnswerPanel from "../components/AnswerPanel";
import RetrievedContext from "../components/RetrievedContext";
import RelationshipGraph from "../components/RelationshipGraph";

export default function AskCodebase({ activeRepo }) {
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [error, setError] = useState(null);

  const handleQuery = async (queryText) => {
    setLoading(true);
    setError(null);
    setResponse(null);

    try {
      const repoId = activeRepo?.repoId || "test-repo";

      const data = await queryApi.askCodebase(
        repoId,
        queryText,
        5
      );

      setResponse(data);
    } catch (err) {
      setError(
        err.message || "Failed to query codebase backend"
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-6xl mx-auto flex flex-col gap-6">
      <div>
        <h1 className="text-xl font-bold font-mono text-white">
          Ask your codebase
        </h1>

        <p className="text-xs text-slate-400 mt-1">
          Ask questions about architecture, classes, methods,
          dependencies, and execution flows.
        </p>
      </div>

      <QueryInput
        onSubmit={handleQuery}
        loading={loading}
      />

      {error && (
        <div className="p-4 bg-red-950/40 border border-red-800/60 rounded-xl text-xs font-mono text-red-300">
          ERROR: {error}
        </div>
      )}

      {loading && (
        <div className="p-4 bg-[#131B2E] border border-slate-800 rounded-xl">
          <p className="text-xs font-mono text-blue-400">
            Analyzing codebase...
          </p>
        </div>
      )}

      {response && (
        <div className="flex flex-col gap-6">
          <AnswerPanel
            answer={
              response.answer ||
              response.response ||
              "No answer returned."
            }
          />

          <RelationshipGraph
            callers={response.callers || []}
            targetSymbol={response.target_symbol || ""}
            callees={response.callees || []}
          />

          <RetrievedContext
            chunks={
              response.chunks ||
              response.retrieved_context ||
              []
            }
          />
        </div>
      )}
    </div>
  );
}
