import { useState } from "react";

function getFileName(path) {
  if (!path) return "Unknown file";
  return path.split("\\").pop().split("/").pop();
}

function getSymbolName(chunk) {
  if (chunk.class_name && chunk.method_name) {
    return `${chunk.class_name}.${chunk.method_name}`;
  }

  if (chunk.method_name) return chunk.method_name;
  if (chunk.class_name) return chunk.class_name;

  return "Code";
}

function CodeExplorer({ result }) {
  const chunks = result?.data?.retrieved_chunks || [];

  const [selectedIndex, setSelectedIndex] = useState(0);

  if (chunks.length === 0) {
    return (
      <div className="explorer-empty">
        <div className="explorer-empty-icon">◇</div>

        <div className="explorer-empty-title">
          No code loaded
        </div>

        <div className="explorer-empty-text">
          Ask a question in <strong>Ask Codebase</strong> to
          retrieve relevant source code.
        </div>
      </div>
    );
  }

  const selectedChunk = chunks[selectedIndex];

  return (
    <div className="code-explorer">
      <div className="explorer-header">
        <div>
          <div className="explorer-eyebrow">
            CODE EXPLORER
          </div>

          <div className="explorer-title">
            Retrieved source
          </div>

          <div className="explorer-subtitle">
            {chunks.length} relevant code{" "}
            {chunks.length === 1 ? "chunk" : "chunks"}
          </div>
        </div>

        <div className="explorer-query">
          {result.query}
        </div>
      </div>

      <div className="explorer-body">
        <aside className="explorer-files">
          <div className="explorer-panel-title">
            RELATED CODE
          </div>

          {chunks.map((chunk, index) => {
            const filePath =
              chunk.file_name ||
              chunk.file ||
              chunk.file_path;

            return (
              <button
                key={chunk.chunk_id || index}
                className={
                  selectedIndex === index
                    ? "explorer-file active"
                    : "explorer-file"
                }
                onClick={() => setSelectedIndex(index)}
              >
                <div className="explorer-file-symbol">
                  {getSymbolName(chunk)}
                </div>

                <div className="explorer-file-name">
                  {getFileName(filePath)}
                </div>

                <div className="explorer-file-lines">
                  Lines {chunk.start_line || "?"}–
                  {chunk.end_line || "?"}
                </div>
              </button>
            );
          })}
        </aside>

        <section className="explorer-code">
          <div className="explorer-code-header">
            <div>
              <div className="explorer-code-file">
                {getFileName(
                  selectedChunk.file_name ||
                    selectedChunk.file ||
                    selectedChunk.file_path
                )}
              </div>

              <div className="explorer-code-symbol">
                {getSymbolName(selectedChunk)}
              </div>
            </div>

            <div className="explorer-code-meta">
              Lines {selectedChunk.start_line || "?"}–
              {selectedChunk.end_line || "?"}
            </div>
          </div>

          <pre className="explorer-source">
            {selectedChunk.code_content ||
              selectedChunk.source_code ||
              "No source code available."}
          </pre>
        </section>
      </div>
    </div>
  );
}

export default CodeExplorer;