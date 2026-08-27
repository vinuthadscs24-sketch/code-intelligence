import { useState } from "react";
import {
  Search,
  GitBranch,
  Network,
  Zap,
  FolderTree,
  History,
  ChevronRight,
  Sparkles,
  CircleDot,
  Code2,
} from "lucide-react";
import "./App.css";

const navigation = [
  { icon: FolderTree, label: "Repository" },
  { icon: Network, label: "Code Graph" },
  { icon: Zap, label: "Impact Analysis" },
  { icon: History, label: "Git Intelligence" },
];

function App() {
  const [query, setQuery] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();

    if (!query.trim()) return;

    console.log("Query:", query);
  };

  return (
    <div className="app">

      {/* Sidebar */}
      <aside className="sidebar">

        <div className="brand">
          <div className="brand-icon">
            <Code2 size={20} />
          </div>

          <div>
            <div className="brand-name">CodeAware</div>
            <div className="brand-subtitle">Code Intelligence</div>
          </div>
        </div>

        <div className="repository-card">
          <div className="repo-icon">
            <GitBranch size={16} />
          </div>

          <div className="repo-info">
            <span>Repository</span>
            <strong>code-intelligence</strong>
          </div>

          <ChevronRight size={15} />
        </div>

        <div className="nav-section">
          <div className="nav-title">EXPLORE</div>

          {navigation.map((item) => {
            const Icon = item.icon;

            return (
              <button className="nav-item" key={item.label}>
                <Icon size={17} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </div>

        <div className="sidebar-bottom">
          <div className="system-status">
            <span className="status-dot"></span>

            <div>
              <strong>System ready</strong>
              <span>RAG pipeline connected</span>
            </div>
          </div>
        </div>

      </aside>

      {/* Main */}
      <main className="main">

        {/* Header */}
        <header className="topbar">

          <div>
            <div className="breadcrumb">
              Repository <ChevronRight size={13} /> code-intelligence
            </div>

            <h1>Understand your codebase.</h1>
          </div>

          <div className="top-status">
            <CircleDot size={13} />
            Ready
          </div>

        </header>

        {/* Query */}
        <section className="query-section">

          <div className="query-label">
            <Sparkles size={16} />
            ASK YOUR CODEBASE
          </div>

          <h2>
            What do you want to understand?
          </h2>

          <p className="query-description">
            Ask questions about architecture, dependencies, code flow,
            impact, or implementation details.
          </p>

          <form onSubmit={handleSubmit} className="query-box">

            <Search size={20} />

            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g. How does HybridRetriever combine vector and BM25 search?"
            />

            <button type="submit">
              Ask
              <ChevronRight size={16} />
            </button>

          </form>

          <div className="suggestions">

            <button onClick={() => setQuery("How does the RAG pipeline work?")}>
              How does the RAG pipeline work?
            </button>

            <button
              onClick={() =>
                setQuery("What happens if I change HybridRetriever?")
              }
            >
              What happens if I change HybridRetriever?
            </button>

            <button
              onClick={() =>
                setQuery("Who calls the vector search component?")
              }
            >
              Who calls vector search?
            </button>

          </div>

        </section>

        {/* Workspace */}
        <section className="workspace">

          <div className="workspace-header">

            <div>
              <span className="section-eyebrow">CODE INTELLIGENCE</span>
              <h3>Visual understanding</h3>
            </div>

            <span className="coming-label">
              Waiting for query
            </span>

          </div>

          <div className="empty-workspace">

            <div className="graph-preview">

              <div className="graph-node node-one">
                RAGPipeline
              </div>

              <div className="graph-line line-one"></div>

              <div className="graph-node node-two">
                HybridRetriever
              </div>

              <div className="graph-line line-two"></div>

              <div className="graph-node node-three">
                Vector Store
              </div>

            </div>

            <div className="empty-content">

              <Network size={28} />

              <h3>
                Your code relationships will appear here
              </h3>

              <p>
                Ask a question to generate an interactive view of the
                relevant classes, functions, dependencies and execution flow.
              </p>

            </div>

          </div>

        </section>

        {/* Bottom cards */}
        <section className="feature-grid">

          <FeatureCard
            icon={<Network size={19} />}
            title="Code Graph"
            description="Explore relationships between classes, functions and modules."
          />

          <FeatureCard
            icon={<Zap size={19} />}
            title="Impact Analysis"
            description="Understand what could be affected before changing code."
          />

          <FeatureCard
            icon={<History size={19} />}
            title="Git Intelligence"
            description="Connect code behaviour with its history and changes."
          />

        </section>

      </main>
    </div>
  );
}

function FeatureCard({ icon, title, description }) {
  return (
    <div className="feature-card">

      <div className="feature-icon">
        {icon}
      </div>

      <div>
        <h4>{title}</h4>
        <p>{description}</p>
      </div>

      <ChevronRight size={16} className="feature-arrow" />

    </div>
  );
}

export default App;