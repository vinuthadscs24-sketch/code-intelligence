import React, { useState } from 'react';
import { Github, Cpu, Database, Network, ArrowRight, CheckCircle2 } from 'lucide-react';

export default function RepoOnboarder({ onRepoLoaded }) {
  const [repoUrl, setRepoUrl] = useState('https://github.com/spring-projects/spring-petclinic');
  const [isProcessing, setIsProcessing] = useState(false);
  const [step, setStep] = useState(0);

  const steps = [
    { label: "Cloning Repository AST", icon: Github },
    { label: "Extracting Class & Method Symbols", icon: Network },
    { label: "Building Code Graph & Call Hierarchy", icon: Cpu },
    { label: "Indexing Hybrid FAISS + BM25 Embeddings", icon: Database }
  ];

  const handleStartOnboarding = (e) => {
    e.preventDefault();
    setIsProcessing(true);
    setStep(0);

    // Simulate real backend indexing pipeline progression
    const interval = setInterval(() => {
      setStep((prev) => {
        if (prev >= steps.length - 1) {
          clearInterval(interval);
          setTimeout(() => {
            onRepoLoaded({
              repoUrl,
              repoName: repoUrl.split('/').pop() || 'spring-petclinic',
              symbolsCount: 1420,
              filesCount: 84
            });
          }, 800);
          return prev;
        }
        return prev + 1;
      });
    }, 1200);
  };

  return (
    <div className="min-h-screen w-screen bg-[#0B0F17] flex items-center justify-center p-6 text-slate-200">
      <div className="max-w-xl w-full bg-[#131B2E] border border-slate-800 rounded-2xl p-8 shadow-2xl flex flex-col gap-6">
        
        {/* Header */}
        <div className="flex flex-col gap-2 text-center">
          <div className="mx-auto w-12 h-12 rounded-xl bg-blue-600/20 border border-blue-500/30 flex items-center justify-center text-blue-400 mb-2">
            <Github className="w-6 h-6" />
          </div>
          <h1 className="text-2xl font-bold text-white font-mono">Code Intelligence Workbench</h1>
          <p className="text-xs text-slate-400">
            Onboard a GitHub repository to build an AST Knowledge Graph & FAISS Vector Store before querying.
          </p>
        </div>

        {/* Form Input */}
        {!isProcessing ? (
          <form onSubmit={handleStartOnboarding} className="flex flex-col gap-4">
            <div className="flex flex-col gap-2">
              <label className="text-xs font-mono text-slate-400 uppercase tracking-wider">GitHub Repository URL</label>
              <input
                type="text"
                value={repoUrl}
                onChange={(e) => setRepoUrl(e.target.value)}
                placeholder="https://github.com/owner/repository"
                className="w-full bg-[#0B0F17] border border-slate-700 rounded-xl px-4 py-3 text-xs font-mono text-slate-200 focus:outline-none focus:border-blue-500 transition"
                required
              />
            </div>

            <button
              type="submit"
              className="w-full bg-blue-600 hover:bg-blue-500 text-white font-semibold py-3 rounded-xl text-xs flex items-center justify-center gap-2 transition shadow-lg shadow-blue-600/30"
            >
              <span>Onboard & Index Codebase</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </form>
        ) : (
          /* Indexing Status Visualizer */
          <div className="flex flex-col gap-4 py-4">
            <div className="text-xs font-mono text-blue-400 text-center font-semibold">
              Indexing target codebase... Please wait
            </div>

            <div className="flex flex-col gap-3">
              {steps.map((s, idx) => {
                const IconComponent = s.icon;
                const isDone = idx < step;
                const isCurrent = idx === step;

                return (
                  <div 
                    key={idx} 
                    className={`flex items-center justify-between p-3.5 rounded-xl border transition-all ${
                      isCurrent 
                        ? 'bg-blue-600/10 border-blue-500 text-blue-300' 
                        : isDone 
                        ? 'bg-[#0B0F17] border-slate-800 text-slate-300' 
                        : 'bg-[#0B0F17]/50 border-slate-900 text-slate-600'
                    }`}
                  >
                    <div className="flex items-center gap-3 text-xs font-mono">
                      <IconComponent className={`w-4 h-4 ${isCurrent ? 'animate-pulse text-blue-400' : isDone ? 'text-emerald-400' : ''}`} />
                      <span>{s.label}</span>
                    </div>
                    {isDone && <CheckCircle2 className="w-4 h-4 text-emerald-400" />}
                  </div>
                );
              })}
            </div>
          </div>
        )}

      </div>
    </div>
  );
}