import { useState } from "react";

function getFileName(path) {
  if (!path) return "Unknown file";

  return path
    .split("\\")
    .pop()
    .split("/")
    .pop();
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

  if (chunk.symbol_name) {
    return chunk.symbol_name;
  }

  if (chunk.name) {
    return chunk.name;
  }

  return "Code";
}

function getChunks(result) {
  if (!result) {
    return [];
  }

  /* Backend response:
     result.data.retrieved_chunks
  */
  if (
    Array.isArray(result?.data?.retrieved_chunks)
  ) {
    return result.data.retrieved_chunks;
  }

  /* Alternative backend response */
  if (
    Array.isArray(result?.data?.retrieved_context)
  ) {
    return result.data.retrieved_context;
  }

  /* Direct response */
  if (
    Array.isArray(result?.retrieved_chunks)
  ) {
    return result.retrieved_chunks;
  }

  /* Direct chunks */
  if (Array.isArray(result?.chunks)) {
    return result.chunks;
  }

  /* Direct retrieved context */
  if (
    Array.isArray(result?.retrieved_context)
  ) {
    return result.retrieved_context;
  }

  return [];
}

function CodeExplorer({ result }) {
  const chunks = getChunks(result);

  const [selectedIndex, setSelectedIndex] =
    useState(0);

  /* =====================================================
     EMPTY STATE
  ===================================================== */

  if (!result || chunks.length === 0) {
    return (
      <div className="explorer-empty">
        <div className="explorer-empty-icon">
          ◇
        </div>

        <div className="explorer-empty-title">
          No code loaded
        </div>

        <div className="explorer-empty-text">
          Ask a question in{" "}
          <strong>Ask Codebase</strong> to retrieve
          relevant source code.
        </div>
      </div>
    );
  }

  /* =====================================================
     SAFETY
  ===================================================== */

  const safeIndex =
    selectedIndex >= chunks.length
      ? 0
      : selectedIndex;

  const selectedChunk = chunks[safeIndex];

  /* =====================================================
     FILE PATH
  ===================================================== */

  const selectedFile =
    selectedChunk.file_name ||
    selectedChunk.file ||
    selectedChunk.file_path ||
    selectedChunk.path;

  /* =====================================================
     RENDER
  ===================================================== */

  return (
    <div className="code-explorer">

      {/* =================================================
          HEADER
      ================================================= */}

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
            {chunks.length === 1
              ? "chunk"
              : "chunks"}
          </div>

        </div>

        <div className="explorer-query">
          {result?.query || "Codebase query"}
        </div>

      </div>

      {/* =================================================
          BODY
      ================================================= */}

      <div className="explorer-body">

        {/* =================================================
            FILE LIST
        ================================================= */}

        <aside className="explorer-files">

          <div className="explorer-panel-title">
            RELATED CODE
          </div>

          {chunks.map((chunk, index) => {

            const filePath =
              chunk.file_name ||
              chunk.file ||
              chunk.file_path ||
              chunk.path;

            return (
              <button
                type="button"
                key={
                  chunk.chunk_id ||
                  chunk.id ||
                  index
                }
                className={
                  safeIndex === index
                    ? "explorer-file active"
                    : "explorer-file"
                }
                onClick={() =>
                  setSelectedIndex(index)
                }
              >

                <div className="explorer-file-symbol">
                  {getSymbolName(chunk)}
                </div>

                <div className="explorer-file-name">
                  {getFileName(filePath)}
                </div>

                <div className="explorer-file-lines">
                  Lines{" "}
                  {chunk.start_line ??
                    chunk.line_start ??
                    "?"}
                  –
                  {chunk.end_line ??
                    chunk.line_end ??
                    "?"}
                </div>

              </button>
            );
          })}

        </aside>

        {/* =================================================
            SOURCE CODE
        ================================================= */}

        <section className="explorer-code">

          {/* CODE HEADER */}

          <div className="explorer-code-header">

            <div>

              <div className="explorer-code-file">
                {getFileName(selectedFile)}
              </div>

              <div className="explorer-code-symbol">
                {getSymbolName(selectedChunk)}
              </div>

            </div>

            <div className="explorer-code-meta">
              Lines{" "}
              {selectedChunk.start_line ??
                selectedChunk.line_start ??
                "?"}
              –
              {selectedChunk.end_line ??
                selectedChunk.line_end ??
                "?"}
            </div>

          </div>

          {/* SOURCE */}

          <pre className="explorer-source">
            {selectedChunk.code_content ||
              selectedChunk.source_code ||
              selectedChunk.content ||
              selectedChunk.code ||
              "No source code available."}
          </pre>

        </section>

      </div>
    </div>
  );
}

export default CodeExplorer;