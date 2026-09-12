import { useState } from "react";
import "./App.css";
import { askCodebase } from "./api";
import CallGraph from "./components/CallGraph";
import CodeExplorer from "./components/CodeExplorer";

const navigation = [
  { id: "ask", label: "Ask Codebase", icon: "⌕" },
  { id: "explorer", label: "Code Explorer", icon: "◇" },
  { id: "impact", label: "Impact Analysis", icon: "◎" },
  { id: "git", label: "Git History", icon: "↻" },
];

function App() {
  const [activeView, setActiveView] = useState("ask");
  const [query, setQuery] = useState("Who calls createBooking?");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleAsk() {
    const trimmedQuery = query.trim();

    if (!trimmedQuery || loading) {
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const data = await askCodebase(trimmedQuery);

      console.log("Backend response:", data);

      setResult(data);
    } catch (err) {
      console.error("Query error:", err);

      setError(
        "Unable to connect to the Code Intelligence backend. Make sure the FastAPI server is running on port 8000."
      );
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(event) {
    if (event.key === "Enter") {
      handleAsk();
    }
  }

  function useExample(example) {
    setQuery(example);
    setResult(null);
    setError("");
    setActiveView("ask");
  }

  function getFileName(path) {
    if (!path) {
      return "Unknown file";
    }

    return path.split("\\").pop().split("/").pop();
  }

  function getSymbolName(chunk) {
    if (chunk.class_name && chunk.method_name) {
      return `${chunk.class_name}.${chunk.method_name}`;
    }

    if (chunk.method_name) {
      return chunk.method_name;
    }

    if (chunk.class_name) {
      return chunk.class_name;
    }

    return "Code";
  }

  function renderRetrievedCode() {
    const chunks = result?.data?.retrieved_chunks;

    if (!chunks || chunks.length === 0) {
      return null;
    }

    return (
      <div className="retrieved-section">
        <div className="data-label">RELATED CODE</div>

        <div className="retrieved-count">
          {chunks.length} relevant code{" "}
          {chunks.length === 1 ? "chunk" : "chunks"}
        </div>

        <div className="retrieved-list">
          {chunks.map((chunk, index) => (
            <div
              className="code-card"
              key={chunk.chunk_id || index}
            >
              <div className="code-card-header">
                <div>
                  <div className="code-symbol">
                    {getSymbolName(chunk)}
                  </div>

                  <div className="code-location">
                    {getFileName(
                      chunk.file_name ||
                        chunk.file ||
                        chunk.file_path
                    )}
                    {" · "}
                    lines {chunk.start_line || "?"}–
                    {chunk.end_line || "?"}
                  </div>
                </div>

                <span className="code-rank">
                  #{chunk.final_rank || index + 1}
                </span>
              </div>

              <pre className="code-preview">
                {chunk.code_content ||
                  chunk.source_code ||
                  "No source code available."}
              </pre>

              <div className="code-card-footer">
                <span>
                  {chunk.chunk_type || "CODE"}
                </span>

                <span>
                  {chunk.sources?.join(" + ") ||
                    "retrieval"}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="app-shell">
      {/* TOP BAR */}
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">&lt;/&gt;</div>

          <div>
            <div className="brand-name">
              CODE INTELLIGENCE
            </div>

            <div className="brand-subtitle">
              AI-powered codebase analysis
            </div>
          </div>
        </div>

        <div className="repo-status">
          <div className="repo-info">
            <span className="status-dot"></span>

            <span className="repo-name">
              code-intelligence
            </span>

            <span className="branch">
              fix/code-indexing
            </span>
          </div>

          <div className="index-status">
            <span className="index-icon">◈</span>
            INDEXED
          </div>
        </div>

        <div className="top-actions">
          <button
            className="icon-button"
            title="Search"
          >
            /
          </button>

          <button
            className="icon-button"
            title="Settings"
          >
            ⚙
          </button>

          <div className="avatar">V</div>
        </div>
      </header>

      {/* MAIN LAYOUT */}
      <div className="main-layout">

        {/* SIDEBAR */}
        <aside className="sidebar">
          <div className="sidebar-section">
            <div className="section-label">
              INTELLIGENCE
            </div>

            {navigation.map((item) => (
              <button
                key={item.id}
                className={
                  activeView === item.id
                    ? "nav-item active"
                    : "nav-item"
                }
                onClick={() =>
                  setActiveView(item.id)
                }
              >
                <span className="nav-icon">
                  {item.icon}
                </span>

                <span>{item.label}</span>
              </button>
            ))}
          </div>

          <div className="sidebar-divider"></div>

          <div className="sidebar-section">
            <div className="section-label">
              REPOSITORY
            </div>

            <div className="tree-item root">
              <span>▾</span>
              <span>src</span>
            </div>

            <div className="tree-item">
              <span>›</span>
              <span>api</span>
            </div>

            <div className="tree-item">
              <span>›</span>
              <span>core</span>
            </div>

            <div className="tree-item">
              <span>›</span>
              <span>models</span>
            </div>

            <div className="tree-item">
              <span>›</span>
              <span>services</span>
            </div>
          </div>

          <div className="sidebar-bottom">
            <div className="pipeline-status">
              <span className="status-dot"></span>

              <div>
                <div className="pipeline-title">
                  Analysis ready
                </div>

                <div className="pipeline-meta">
                  2,481 symbols indexed
                </div>
              </div>
            </div>
          </div>
        </aside>

        {/* WORKSPACE */}
        <main className="workspace">

          <div className="workspace-header">
            <div>
              <div className="breadcrumb">
                WORKSPACE <span>/</span>{" "}
                {activeView
                  .toUpperCase()
                  .replace("-", " ")}
              </div>

              <h1>
                Understand your codebase.
              </h1>

              <p>
                Ask questions in natural language.
                Code Intelligence maps relationships,
                history, and impact automatically.
              </p>
            </div>
          </div>

          {/* ASK CODEBASE QUERY */}
          {activeView === "ask" && (
            <section className="query-section">
              <div className="query-label">
                <span className="query-command">
                  ⌘
                </span>

                ASK YOUR CODEBASE
              </div>

              <div className="query-box">
                <span className="query-prefix">
                  &gt;
                </span>

                <input
                  value={query}
                  onChange={(event) =>
                    setQuery(event.target.value)
                  }
                  onKeyDown={handleKeyDown}
                  placeholder="Who calls createBooking?"
                  aria-label="Ask your codebase"
                />

                <button
                  className="ask-button"
                  onClick={handleAsk}
                  disabled={loading}
                >
                  {loading
                    ? "ANALYZING"
                    : "ASK"}

                  <span>
                    {loading ? "..." : "↵"}
                  </span>
                </button>
              </div>

              <div className="query-hints">
                <button
                  onClick={() =>
                    useExample(
                      "Who calls createBooking?"
                    )
                  }
                >
                  Who calls createBooking?
                </button>

                <button
                  onClick={() =>
                    useExample(
                      "What breaks if I change UserService?"
                    )
                  }
                >
                  What breaks if I change UserService?
                </button>

                <button
                  onClick={() =>
                    useExample(
                      "Show recent booking changes"
                    )
                  }
                >
                  Show recent booking changes
                </button>
              </div>
            </section>
          )}

          {/* ANALYSIS WORKSPACE */}
          {activeView === "ask" && (
            <section className="analysis-workspace">
              <div className="workspace-grid"></div>

              {/* LOADING */}
              {loading && (
                <div className="analysis-state">
                  <div className="loading-indicator"></div>

                  <div className="state-title">
                    Analyzing your codebase
                  </div>

                  <div className="pipeline-steps">
                    <span>
                      QUERY CLASSIFIED
                    </span>

                    <span>
                      SYMBOLS RETRIEVED
                    </span>

                    <span>
                      DEPENDENCIES ANALYZED
                    </span>

                    <span>
                      GENERATING RESPONSE
                    </span>
                  </div>
                </div>
              )}

              {/* ERROR */}
              {!loading && error && (
                <div className="analysis-state error-state">
                  <div className="state-icon">
                    !
                  </div>

                  <div className="state-title">
                    Backend connection failed
                  </div>

                  <div className="state-description">
                    {error}
                  </div>
                </div>
              )}

              {/* EMPTY */}
              {!loading &&
                !error &&
                !result && (
                  <div className="empty-analysis">
                    <div className="analysis-symbol">
                      <div className="node node-a"></div>
                      <div className="node node-b"></div>
                      <div className="node node-c"></div>

                      <div className="connection connection-a"></div>
                      <div className="connection connection-b"></div>
                    </div>

                    <div className="empty-title">
                      Your code intelligence workspace
                    </div>

                    <div className="empty-description">
                      Ask a question above to generate
                      an adaptive analysis.
                      <br />
                      The visualization changes based
                      on what you ask.
                    </div>

                    <div className="analysis-types">
                      <span>CALL GRAPH</span>
                      <span>IMPACT</span>
                      <span>GIT HISTORY</span>
                      <span>RETRIEVAL</span>
                    </div>
                  </div>
                )}

              {/* RESULT */}
              {!loading &&
                !error &&
                result && (
                  <div className="result-panel">

                    <div className="result-header">
                      <div>
                        <div className="result-type">
                          {result.response_type ||
                            "ANALYSIS"}
                        </div>

                        <div className="result-query">
                          {result.query || query}
                        </div>
                      </div>

                      <div className="result-badge">
                        LIVE RESPONSE
                      </div>
                    </div>

                    {/* CALL GRAPH */}
                    {result.response_type ===
                      "call_graph" &&
                    result.data ? (
                      <>
                        <div className="result-answer">
                          {result.answer ||
                            "Call graph analysis completed."}
                        </div>

                        <CallGraph
                          data={result.data}
                        />
                      </>
                    ) : (
                      <>
                        {result.response_type ===
                        "answer" ? (
                          <div className="retrieval-summary">
                            Retrieved relevant code
                            from the indexed codebase.
                          </div>
                        ) : (
                          result.answer && (
                            <div className="result-answer">
                              {result.answer}
                            </div>
                          )
                        )}

                        {renderRetrievedCode()}

                        {result.data && (
                          <div className="result-data">
                            <details>
                              <summary>
                                Raw response
                              </summary>

                              <pre>
                                {JSON.stringify(
                                  result.data,
                                  null,
                                  2
                                )}
                              </pre>
                            </details>
                          </div>
                        )}
                      </>
                    )}
                  </div>
                )}
            </section>
          )}

          {/* CODE EXPLORER */}
          {activeView === "explorer" && (
            <CodeExplorer result={result} />
          )}

          {/* TEMPORARY IMPACT VIEW */}
          {activeView === "impact" && (
            <div className="explorer-empty">
              <div className="explorer-empty-icon">
                ◎
              </div>

              <div className="explorer-empty-title">
                Impact Analysis
              </div>

              <div className="explorer-empty-text">
                This section will use the real{" "}
                <strong>
                  /impact/{`{entity_name}`}
                </strong>{" "}
                backend endpoint.
              </div>
            </div>
          )}

          {/* TEMPORARY GIT VIEW */}
          {activeView === "git" && (
            <div className="explorer-empty">
              <div className="explorer-empty-icon">
                ↻
              </div>

              <div className="explorer-empty-title">
                Git History
              </div>

              <div className="explorer-empty-text">
                This section will use the real Git
                provenance backend endpoint.
              </div>
            </div>
          )}

        </main>
      </div>
    </div>
  );
}

export default App;