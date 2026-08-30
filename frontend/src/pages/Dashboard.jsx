import React from "react";
import RepositorySetup from "../components/RepositorySetup";

export default function Dashboard({ activeRepo, onRepoLoaded }) {
  return (
    <div className="p-6 max-w-5xl mx-auto flex flex-col gap-6">
      <h1 className="text-xl font-bold font-mono text-white">Repository Overview</h1>
      <RepositorySetup activeRepo={activeRepo} onRepoLoaded={onRepoLoaded} />
    </div>
  );
}
