const API_BASE = '/api';
let validOptionsCache = null;

async function loadValidOptions() {
    if (!validOptionsCache) {
        validOptionsCache = await fetch(`${API_BASE}/valid_options`).then(r => r.json());
    }
    return validOptionsCache;
}


// --- Routing ---
window.addEventListener('hashchange', router);
window.addEventListener('load', router);

async function router() {
    const hash = window.location.hash || '#/';
    const app = document.getElementById('app');
    app.innerHTML = '<div class="container">Loading...</div>';

    if (hash === '#/') renderHome(app);
    else if (hash.startsWith('#/pair/')) {
        const parts = hash.split('/');
        renderPair(app, parts[2], parts[3]);
    }
    else if (hash === '#/disease') renderDisease(app);
    else renderHome(app);
}

// --- Views ---
async function renderHome(app) {
    const stats = await fetch(`${API_BASE}/stats`).then(r => r.json());
    const options = await loadValidOptions();

    app.innerHTML = `
        <div class="container">
            <div class="panel flex justify-between" style="align-items: center">
                <h2>EviDTI Prediction Database</h2>
                <div>
                     <a href="#/disease"><button>Disease Search</button></a>
                </div>
            </div>
            
            <div class="grid-2">
                <div class="panel flex-col" style="display:flex; flex-direction:column; gap:10px;">
                    <h3>Database Search</h3>
                    <label style="font-size: 0.85em; font-weight: bold; margin-bottom: -5px; color: var(--text-muted);">Drug</label>
                    <input list="drug-list" id="drugId" placeholder="Type to search (e.g. Drug_b755)">
                    <datalist id="drug-list">
                        ${options.drugs.map(d => `<option value="${d.id}">${d.label || d.id}</option>`).join('')}
                    </datalist>
                    <label style="font-size: 0.85em; font-weight: bold; margin-bottom: -5px; color: var(--text-muted);">Protein Target</label>
                    <input list="protein-list" id="proteinId" placeholder="Type to search (e.g. Mock Protein Q05603)">
                    <datalist id="protein-list">
                        ${options.proteins.map(p => `<option value="${p.id}">${p.label || p.id}</option>`).join('')}
                    </datalist>
                    <button onclick="viewPair()">View Prediction</button>
                    <p style="font-size:0.85em; color:var(--text-muted)">* </p>
                </div>
                
                <div class="panel">
                    <h3>NLQ Search</h3>
                    <input type="text" id="nlqInput" placeholder="Which drugs interact with Mock Protein Q05603?">
                    <button onclick="runNlq()" style="margin-top:10px">Ask</button>
                    <div id="nlqRes" style="margin-top: 1rem; font-size:0.9em;"></div>
                </div>
            </div>

            <div class="grid-4" style="margin-top:20px">
                <div class="stat-box"><h3>${stats.total_drugs || 0}</h3><p>Drugs</p></div>
                <div class="stat-box"><h3>50</h3><p>Proteins Cache</p></div>
                <div class="stat-box"><h3>${Object.values(stats.pairs || {}).reduce((a, b) => a + b, 0)}</h3><p>Predictions</p></div>
                <div class="stat-box"><h3>3</h3><p>Datasets</p></div>
            </div>
        </div>
    `;
    window.viewPair = () => {
        const d = document.getElementById('drugId').value;
        const p = document.getElementById('proteinId').value;
        if (d && p) window.location.hash = `#/pair/${d}/${p}`;
    }
    window.runNlq = async () => {
        const q = document.getElementById('nlqInput').value;
        const resDiv = document.getElementById('nlqRes');
        resDiv.innerHTML = "Searching...";
        const r = await fetch(`${API_BASE}/nlq`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: q, top_k: 5 })
        }).then(x => x.json());
        let html = `<div><i>${r.summary}</i> (${r.search_time_ms}ms)</div><br><p style="font-size:0.8em;color:var(--text-muted);">* Semantic Match based on local Document FAISS embeddings</p><ul>`;
        r.results.forEach(x => {
            html += `<li><a href="#/pair/${x.drug_id}/${x.protein_id}">${x.drug_name} x ${x.protein_id}</a> - ${x.decision} <span style="color:var(--brand-blue);">(Score: ${(x.relevance_score || 0).toFixed(2)})</span></li>`;
        });
        resDiv.innerHTML = html + "</ul>";
    }
}

async function renderPair(app, drugId, proteinId) {
    app.innerHTML = `<div class="container">Loading pair data...</div>`;
    try {
        const data = await fetch(`${API_BASE}/pair/${drugId}/${proteinId}`).then(r => r.json());

        let consensusHtml = data.predictions.map(p => `
            <tr>
                <td>${p.dataset}</td>
                <td>${(p.interaction_prob * 100).toFixed(1)}%</td>
                <td><span class="badge ${p.decision}">${p.decision}</span></td>
            </tr>
        `).join('');

        app.innerHTML = `
            <div class="container">
                <a href="#/">← Back to Search</a>
                <div class="panel" style="margin-top:20px">
                    <h2>Drug: ${data.drug_info.drug_name || 'Unknown'}</h2>
                    <p class="text-muted">ID: ${data.drug_info.drug_id} | SMILES: ${data.drug_info.smiles || ''}</p>
                    <hr style="border-color:var(--border); margin: 20px 0;">
                    <h2>Target: ${data.protein_info.protein_name || proteinId}</h2>
                    <p class="text-muted">${data.protein_info.gene_name || ''} | ${data.protein_info.organism || ''}</p>
                </div>
                
                <div class="grid-2">
                    <div class="panel">
                        <h3>Cross-Dataset Consensus</h3>
                        <table>
                            <thead><tr><th>Dataset</th><th>Probability</th><th>Decision</th></tr></thead>
                            <tbody>${consensusHtml}</tbody>
                        </table>
                    </div>
                </div>

                <div class="panel">
                    <h3>Per-Residue Light Attention Heatmap</h3>
                    <select id="datasetSelect" style="max-width: 200px; margin-bottom: 20px;" onchange="loadAttention('${drugId}', '${proteinId}')">
                        <option value="">Select Dataset...</option>
                        ${data.predictions.map(p => `<option value="${p.dataset}">${p.dataset}</option>`)}
                    </select>
                    <div id="heatmap" class="heatmap-container"></div>
                </div>
                
                <div style="margin-top:20px">
                    <button onclick="openScholar('${data.drug_info.drug_name}','${data.protein_info.protein_name}')">📄 Search Literature on Google Scholar</button>
                </div>
            </div>
        `;

        window.loadAttention = async (d, p) => {
            const ds = document.getElementById('datasetSelect').value;
            if (!ds) return;
            const res = await fetch(`${API_BASE}/pair/${d}/${p}/attention?dataset=${ds}`);
            if (!res.ok) {
                document.getElementById('heatmap').innerHTML = '<div style="padding:20px">Attention not available (maybe rejected).</div>';
                return;
            }
            const attn = await res.json();

            const trace = {
                x: attn.residue_indices,
                y: attn.attention_weights,
                type: 'bar',
                marker: {
                    color: attn.attention_weights,
                    colorscale: [
                        [0, '#1e293b'], [0.3, '#3b82f6'], [0.6, '#f59e0b'], [0.8, '#ef4444'], [1.0, '#dc2626']
                    ],
                },
                hovertemplate: 'Residue %{x}<br>Attention: %{y:.4f}<extra></extra>'
            };

            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#10233f' }, margin: { t: 30, b: 40, l: 50, r: 20 }
            };
            Plotly.newPlot('heatmap', [trace], layout);
        }

        window.openScholar = (d, p) => {
            const query = encodeURIComponent(`"${d}" "${p}" "drug-target interaction"`);
            window.open(`https://scholar.google.com/scholar?q=${query}`, '_blank');
        }

    } catch (e) {
        app.innerHTML = `<div class="container">Error loading pair: ${e}</div>`;
    }
}

async function renderDisease(app) {
    app.innerHTML = `
        <div class="container">
            <a href="#/">← Back</a>
            <div class="panel" style="margin-top:20px">
                <h2>Disease Search</h2>
                <div style="display:flex; gap:10px; max-width:400px">
                    <input type="text" id="disId" placeholder="Disease Query (e.g. cancer)" value="cancer">
                    <button onclick="searchDis()">Search</button>
                </div>
                <div id="disRes" style="margin-top:20px"></div>
                <div id="disCands" style="margin-top:20px"></div>
            </div>
        </div>
    `;

    window.searchDis = async (override_q) => {
        const q = override_q !== undefined ? override_q : document.getElementById('disId').value;
        const res = await fetch(`${API_BASE}/disease/search?q=${q}`).then(r => r.json());

        if (res.diseases.length === 0 && q !== '') {
            // fallback to showing all available diseases if the current query is sparse
            document.getElementById('disRes').innerHTML = "<p style='color:var(--badge-reject)'>No exact match. Loading all available cache candidates...</p>";
            setTimeout(() => window.searchDis(''), 800);
            return;
        }

        let html = `<ul>`;
        res.diseases.forEach(d => {
            html += `<li><a onclick="loadCands('${d.efo_id}')">${d.disease_name} (${d.efo_id})</a> - ${d.target_count} targets mapped</li>`;
        });
        document.getElementById('disRes').innerHTML = html + `</ul>`;
    }

    setTimeout(() => { window.searchDis(''); }, 100);

    window.loadCands = async (efo) => {
        document.getElementById('disCands').innerHTML = "Loading...";
        const res = await fetch(`${API_BASE}/disease/${efo}/drug_candidates`).then(r => r.json());
        let html = `<h3>Repurposing Candidates</h3><table>
            <thead><tr><th>Drug</th><th>Target</th><th>DTI Prob</th><th>Assoc Score</th></tr></thead><tbody>`;
        res.candidates.forEach(c => {
            html += `<tr>
                <td><a href="#/pair/${c.drug_id}/${c.target_protein}">${c.drug_name}</a></td>
                <td>${c.target_protein}</td>
                <td>${(c.interaction_prob * 100).toFixed(1)}%</td>
                <td>${c.association_score.toFixed(2)}</td>
            </tr>`;
        });
        document.getElementById('disCands').innerHTML = html + `</tbody></table>`;
    }
}
