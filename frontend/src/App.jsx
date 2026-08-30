import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import Dashboard from './pages/Dashboard';
import AskCodebase from './pages/AskCodebase';
import CodeExplorer from './pages/CodeExplorer';
import GitHistory from './pages/GitHistory';
import ImpactAnalysis from './pages/ImpactAnalysis';

export default function App() {
  const [activeTab, setActiveTab] = useState('ask'); // Default to Ask Codebase
  const [activeRepo, setActiveRepo] = useState(null);

  const renderContent = () => {
    switch (activeTab) {
      case 'overview':
        return <Dashboard activeRepo={activeRepo} onRepoLoaded={setActiveRepo} />;
      case 'ask':
        return <AskCodebase activeRepo={activeRepo} />;
      case 'explorer':
        return <CodeExplorer activeRepo={activeRepo} />;
      case 'git':
        return <GitHistory activeRepo={activeRepo} />;
      case 'impact':
        return <ImpactAnalysis activeRepo={activeRepo} />;
      default:
        return <AskCodebase activeRepo={activeRepo} />;
    }
  };

  return (
    <div className="flex h-screen w-screen bg-[#0B0F17] text-slate-200 overflow-hidden font-sans">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} activeRepo={activeRepo} />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header activeRepo={activeRepo} />
        <main className="flex-1 overflow-y-auto bg-[#0B0F17]">
          {renderContent()}
        </main>
      </div>
    </div>
  );
}