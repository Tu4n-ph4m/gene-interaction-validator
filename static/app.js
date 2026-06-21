const chatLog = document.getElementById("chat-log");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const geneFileInput = document.getElementById("gene-file");

const MAX_GENE_FILE_BYTES = 5 * 1024 * 1024; // 5MB is generous for a gene list

let history = []; // [{role: "user"|"assistant", content: string}]
let pendingFileGenes = null;
let pendingFileName = null;

function splitGenes(text) {
  return text
    .split(/[\s,]+/)
    .map((g) => g.trim())
    .filter(Boolean);
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

function scrollToBottom() {
  chatLog.scrollTop = chatLog.scrollHeight;
}

function appendMessage(role, html) {
  const div = document.createElement("div");
  div.className = `msg msg-${role}`;
  div.innerHTML = html;
  chatLog.appendChild(div);
  scrollToBottom();
  return div;
}

geneFileInput.addEventListener("change", () => {
  const file = geneFileInput.files[0];
  pendingFileGenes = null;
  pendingFileName = null;
  if (!file) return;

  if (file.size > MAX_GENE_FILE_BYTES) {
    appendMessage("assistant", `⚠️ "${escapeHtml(file.name)}" is too large (max 5 MB).`);
    geneFileInput.value = "";
    return;
  }

  const reader = new FileReader();
  reader.onload = () => {
    const genes = splitGenes(String(reader.result));
    if (!genes.length) {
      appendMessage("assistant", `⚠️ No gene symbols found in "${escapeHtml(file.name)}".`);
      geneFileInput.value = "";
      return;
    }
    pendingFileGenes = genes;
    pendingFileName = file.name;
    chatInput.placeholder = `${genes.length} genes loaded from ${file.name} -- add a tissue/species, or just press Send`;
  };
  reader.onerror = () => {
    appendMessage("assistant", `⚠️ Could not read "${escapeHtml(file.name)}".`);
  };
  reader.readAsText(file);
});

function renderResultsTable(results) {
  const wrapper = document.createElement("div");

  const toolbar = document.createElement("div");
  toolbar.className = "toolbar";
  toolbar.innerHTML = `<span></span><button class="download-btn">Download CSV</button>`;
  wrapper.appendChild(toolbar);

  const sorted = [...results].sort(
    (a, b) => (b.string_combined_score ?? -1) - (a.string_combined_score ?? -1)
  );

  const table = document.createElement("table");
  table.innerHTML = `
    <thead>
      <tr>
        <th>Gene 1</th><th>Gene 2</th><th>Verdict</th><th>StringDB Score</th>
        <th>Curated-Overlap Risk</th><th>BioGRID Evidence #</th>
        <th>StringDB</th><th>BioGRID</th>
      </tr>
    </thead>
    <tbody></tbody>
  `;
  const tbody = table.querySelector("tbody");

  for (const r of sorted) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${escapeHtml(r.gene1)}</td>
      <td>${escapeHtml(r.gene2)}</td>
      <td class="verdict-${r.verdict}">${escapeHtml(r.verdict)}</td>
      <td>${r.string_combined_score ?? ""}</td>
      <td class="risk-${r.string_curated_overlap_risk}">${escapeHtml(r.string_curated_overlap_risk)}</td>
      <td>${r.biogrid_evidence_count ?? ""}</td>
      <td>${r.string_source_url ? `<a href="${r.string_source_url}" target="_blank" rel="noopener">View ↗</a>` : ""}</td>
      <td>${r.biogrid_source_url ? `<a href="${r.biogrid_source_url}" target="_blank" rel="noopener">View ↗</a>` : ""}</td>
    `;
    tbody.appendChild(tr);
  }
  wrapper.appendChild(table);

  const caveatRows = results.filter((r) => (r.notes || "").includes("CAVEAT"));
  if (caveatRows.length) {
    const details = document.createElement("details");
    details.className = "caveats";
    details.innerHTML = `<summary>⚠️ ${caveatRows.length} pair(s) flagged with a tissue/independence caveat</summary>`;
    const ul = document.createElement("ul");
    for (const r of caveatRows) {
      const li = document.createElement("li");
      li.innerHTML = `<strong>${escapeHtml(r.gene1)} ↔ ${escapeHtml(r.gene2)}</strong>: ${escapeHtml(r.notes)}`;
      ul.appendChild(li);
    }
    details.appendChild(ul);
    wrapper.appendChild(details);
  }

  wrapper.querySelector(".download-btn").addEventListener("click", () => downloadCsv(results));
  return wrapper;
}

function downloadCsv(results) {
  if (!results.length) return;
  const columns = Object.keys(results[0]);
  const escapeCsv = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const lines = [columns.join(",")];
  for (const r of results) {
    lines.push(columns.map((c) => escapeCsv(r[c])).join(","));
  }
  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "gene_interaction_network.csv";
  a.click();
  URL.revokeObjectURL(url);
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  let message = chatInput.value.trim();
  if (pendingFileGenes) {
    const genePhrase = `Genes from ${pendingFileName}: ${pendingFileGenes.join(", ")}.`;
    message = message ? `${genePhrase} ${message}` : `Find interactions among these genes: ${pendingFileGenes.join(", ")}.`;
  }
  if (!message) return;

  appendMessage("user", escapeHtml(chatInput.value.trim() || `(attached ${pendingFileName})`));

  chatInput.value = "";
  chatInput.placeholder = "Type a message...";
  geneFileInput.value = "";
  pendingFileGenes = null;
  pendingFileName = null;

  sendBtn.disabled = true;
  chatInput.disabled = true;
  const thinkingMsg = appendMessage("assistant thinking", "Thinking...");

  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history }),
    });

    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${resp.status})`);
    }

    const data = await resp.json();
    thinkingMsg.remove();
    const bubble = appendMessage("assistant", escapeHtml(data.reply).replace(/\n/g, "<br>"));

    if (data.results) {
      if (data.results.invalid_genes && data.results.invalid_genes.length) {
        const warn = document.createElement("p");
        warn.className = "status-warn";
        warn.textContent = `Could not resolve: ${data.results.invalid_genes.join(", ")}`;
        bubble.appendChild(warn);
      }
      if (data.results.results && data.results.results.length) {
        bubble.appendChild(renderResultsTable(data.results.results));
      } else {
        const none = document.createElement("p");
        none.textContent = "No interactions found among the resolved genes.";
        bubble.appendChild(none);
      }
    }

    history.push({ role: "user", content: message });
    history.push({ role: "assistant", content: data.reply });
  } catch (err) {
    thinkingMsg.remove();
    appendMessage("assistant error", `⚠️ ${escapeHtml(err.message)}`);
  } finally {
    sendBtn.disabled = false;
    chatInput.disabled = false;
    chatInput.focus();
  }
});
