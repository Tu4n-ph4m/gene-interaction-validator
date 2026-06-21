const form = document.getElementById("network-form");
const submitBtn = document.getElementById("submit-btn");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const speciesSelect = document.getElementById("species");
const geneFileInput = document.getElementById("gene-file");
const fileInfoEl = document.getElementById("file-info");
const genesTextarea = document.getElementById("genes");

let lastResults = [];

const MAX_GENE_FILE_BYTES = 5 * 1024 * 1024; // 5MB is generous for a gene list

geneFileInput.addEventListener("change", () => {
  const file = geneFileInput.files[0];
  fileInfoEl.textContent = "";
  if (!file) return;

  if (file.size > MAX_GENE_FILE_BYTES) {
    fileInfoEl.textContent = `File too large (${(file.size / 1024 / 1024).toFixed(1)} MB, max 5 MB).`;
    geneFileInput.value = "";
    return;
  }

  const reader = new FileReader();
  reader.onload = () => {
    const genes = splitGenes(String(reader.result));
    if (!genes.length) {
      fileInfoEl.textContent = "No gene symbols found in that file.";
      return;
    }
    genesTextarea.value = genes.join(", ");
    fileInfoEl.textContent = `Loaded ${genes.length} gene(s) from ${file.name}.`;
  };
  reader.onerror = () => {
    fileInfoEl.textContent = `Could not read ${file.name}.`;
  };
  reader.readAsText(file);
});

async function loadSpeciesOptions() {
  try {
    const resp = await fetch("/api/species");
    if (!resp.ok) return;
    const data = await resp.json();
    speciesSelect.innerHTML = data.options
      .map((name) => `<option value="${name}">${name}</option>`)
      .join("");
  } catch {
    // Keep the static "Human" fallback already in the HTML.
  }
}
loadSpeciesOptions();

function splitGenes(text) {
  return text
    .split(/[\s,]+/)
    .map((g) => g.trim())
    .filter(Boolean);
}

function setStatus(html, cls) {
  statusEl.innerHTML = html ? `<div class="status-${cls}">${html}</div>` : "";
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

function renderResults(results, invalidGenes, totalGenes) {
  resultsEl.innerHTML = "";

  if (invalidGenes.length) {
    setStatus(
      `Could not resolve ${invalidGenes.length} gene(s): ${escapeHtml(invalidGenes.join(", "))}`,
      "warn"
    );
  }

  if (!results.length) {
    resultsEl.innerHTML = "<p>No interactions found among the resolved genes.</p>";
    return;
  }

  const resolvedCount = totalGenes - invalidGenes.length;
  const successMsg = `Found ${results.length} interacting pairs among ${resolvedCount} genes.`;
  if (!invalidGenes.length) setStatus(successMsg, "success");
  else statusEl.insertAdjacentHTML("beforeend", `<div class="status-success">${successMsg}</div>`);

  const sorted = [...results].sort(
    (a, b) => (b.string_combined_score ?? -1) - (a.string_combined_score ?? -1)
  );

  const toolbar = document.createElement("div");
  toolbar.className = "toolbar";
  toolbar.innerHTML = `<span></span><button id="download-btn">Download CSV</button>`;
  resultsEl.appendChild(toolbar);

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
  resultsEl.appendChild(table);

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
    resultsEl.appendChild(details);
  }

  document.getElementById("download-btn").addEventListener("click", () => downloadCsv(results));
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

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const genes = splitGenes(document.getElementById("genes").value);
  const tissue = document.getElementById("tissue").value.trim() || null;
  const species = speciesSelect.value || "human";

  if (genes.length < 2) {
    setStatus("Enter at least 2 gene symbols.", "error");
    return;
  }

  submitBtn.disabled = true;
  submitBtn.textContent = `Querying StringDB + BioGRID for ${genes.length} genes...`;
  setStatus("", "");
  resultsEl.innerHTML = "";

  try {
    const resp = await fetch("/api/network", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ genes, tissue, species }),
    });

    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${resp.status})`);
    }

    const data = await resp.json();
    lastResults = data.results;
    renderResults(data.results, data.invalid_genes, genes.length);
  } catch (err) {
    setStatus(escapeHtml(err.message), "error");
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Find interactions";
  }
});
