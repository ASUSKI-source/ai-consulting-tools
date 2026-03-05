/**
 * AI Market Analyzer - Core Logic
 * Handles API communication, UI state, and data visualization.
 */

const API_BASE = window.location.origin;

// --- HISTORY & WATCHLIST STATE ---
let historyOffset = 0;
const HISTORY_PAGE_SIZE = 15;
let currentFilter = { ticker: '', asset_type: '' };
let currentAnalysisMeta = { ticker: '', asset_type: '', analysis_id: null };

// --- DOCUMENT Q&A STATE ---
let currentDocument = { collectionName: null, filename: null };
// --- LOADING STATE MANAGEMENT ---

let loadingInterval = null;

/**
 * Starts the phased loading experience.
 * Shows ticker-specific messages that cycle every 2 seconds.
 * Disables the active button to prevent double-clicks.
 * @param {string} ticker - The asset being analyzed (e.g. 'AAPL' or 'bitcoin')
 * @param {HTMLElement} button - The button to disable during loading
 * @param {string} originalText - The button's original label to restore later
 */
function startLoading(ticker, button, originalText) {
    // Hide results and errors, show loader
    document.getElementById('results').style.display = 'none';
    document.getElementById('results').style.opacity = '0';
    hideError();
    document.getElementById('loading').style.display = 'block';

    // Disable button and show in-progress label
    if (button) {
        button.disabled = true;
        button.textContent = 'Analyzing...';
        button._originalText = originalText;
    }

    const phases = [
        `Fetching data for ${ticker}...`,
        'Calculating indicators...',
        'Generating AI analysis...'
    ];

    let phaseIndex = 0;
    const loadingText = document.getElementById('loading-text');
    if (loadingText) loadingText.textContent = phases[0];

    // Clear any previous interval before starting a new one
    if (loadingInterval) clearInterval(loadingInterval);

    loadingInterval = setInterval(() => {
        phaseIndex++;
        if (loadingText && phaseIndex < phases.length) {
            loadingText.textContent = phases[phaseIndex];
        }
        // Stay on last message once reached — no looping back
    }, 2000);
}

/**
 * Stops the loading state and re-enables the button.
 * @param {HTMLElement} button - The button to re-enable
 */
function stopLoading(button) {
    clearInterval(loadingInterval);
    loadingInterval = null;
    document.getElementById('loading').style.display = 'none';

    if (button) {
        button.disabled = false;
        button.textContent = button._originalText || 'Analyze';
    }
}

/**
 * Fades the results section in smoothly after data arrives.
 */
function fadeInResults() {
    const results = document.getElementById('results');
    results.style.display = 'block';
    // Small timeout lets the browser register display:block before animating opacity
    setTimeout(() => {
        results.style.opacity = '1';
    }, 20);
}


// --- UI UTILITIES ---

/**
 * Displays an error message with an auto-hide timer.
 * Accepts either a plain string or a structured {user_message, suggestion} object.
 * @param {string|object} errorData
 */
function showError(errorData) {
    const errorDiv = document.getElementById('error-message');
    let html = '';

    if (typeof errorData === 'string') {
        html = `<span>${errorData}</span>`;
    } else {
        const msg = errorData.user_message || 'An error occurred.';
        const hint = errorData.suggestion || '';
        html = `<span><strong>${msg}</strong>${hint ? `<br><small>${hint}</small>` : ''}</span>`;
    }

    html += `<span class="close-btn" onclick="hideError()">×</span>`;
    errorDiv.innerHTML = html;
    errorDiv.style.display = 'flex';

    setTimeout(() => hideError(), 8000);
}

/**
 * Hides the error message div.
 */
function hideError() {
    document.getElementById('error-message').style.display = 'none';
}

/**
 * Shows a transient toast notification in the bottom-right corner.
 * @param {string} message
 * @param {number} duration
 */
function showToast(message, duration = 3000) {
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    document.body.appendChild(toast);

    // Trigger transition
    requestAnimationFrame(() => {
        toast.classList.add('show');
    });

    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => {
            toast.remove();
        }, 250);
    }, duration);
}


// --- FORMATTING FUNCTIONS ---

/**
 * Formats numbers as currency.
 * @param {number} price
 */
function formatPrice(price) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 6
    }).format(price);
}

/**
 * Formats percentages with + prefix and color-coding logic.
 * @param {number} pct
 */
function formatPct(pct) {
    if (pct === null || pct === undefined) return '0.00%';
    const prefix = pct > 0 ? '+' : '';
    const colorClass = pct > 0 ? 'positive' : pct < 0 ? 'negative' : '';
    return `<span class="${colorClass}">${prefix}${pct.toFixed(2)}%</span>`;
}

/**
 * Formats large market cap numbers (Billions/Millions).
 */
function formatMarketCap(cap) {
    if (cap >= 1e9) return `$${(cap / 1e9).toFixed(2)}B`;
    if (cap >= 1e6) return `$${(cap / 1e6).toFixed(2)}M`;
    return formatPrice(cap);
}

/**
 * Formats ISO timestamps into 'Mar 1 at 2:34 PM' style strings.
 * @param {string} iso
 */
function formatHistoryDate(iso) {
    if (!iso) return '-';
    const d = new Date(iso);
    if (isNaN(d.getTime())) return '-';
    const formatted = d.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: 'numeric',
        minute: '2-digit'
    });
    return formatted.replace(',', ' at');
}

/**
 * Escapes HTML in user-supplied strings so they can be safely injected.
 * @param {string} str
 */
function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/[&<>"']/g, (ch) => {
        switch (ch) {
            case '&':
                return '&amp;';
            case '<':
                return '&lt;';
            case '>':
                return '&gt;';
            case '"':
                return '&quot;';
            case "'":
                return '&#039;';
            default:
                return ch;
        }
    });
}

/**
 * Loads aggregate stats from the backend and renders the small stats bar.
 */
async function loadStats() {
    const el = document.getElementById('stats-bar');
    if (!el) return;

    try {
        const resp = await fetch(`${API_BASE}/stats`);
        const data = await resp.json();
        if (!resp.ok) {
            el.textContent = '';
            return;
        }

        const total = data.total_analyses || 0;
        const unique = data.unique_tickers || 0;
        const totalCost = Number(data.total_cost || 0).toFixed(4);

        el.textContent = `${total} analyses run  ·  ${unique} unique tickers  ·  Total AI cost: $${totalCost}`;
    } catch (err) {
        console.error('loadStats error:', err);
        el.textContent = '';
    }
}


// --- RESULTS DISPLAY LOGIC ---

/**
 * Populates the UI with Stock-specific data.
 */
function displayStockResults(data) {
    // Header & Price
    document.getElementById('display-name').textContent = data.company_name;
    document.getElementById('display-symbol').textContent = data.ticker;
    document.getElementById('current-price').textContent = formatPrice(data.current_price);

    // Toggle Section Visibility
    document.getElementById('stock-only-stats').style.display = 'grid';
    document.getElementById('crypto-only-stats').style.display = 'none';

    // Stock Stats
    document.getElementById('pe-ratio').textContent = data.pe_ratio ?? 'N/A';

    const yearLow = data.week_low_52;
    const yearHigh = data.week_high_52;
    if (typeof yearLow === 'number' && typeof yearHigh === 'number') {
        document.getElementById('year-range').textContent =
            `${formatPrice(yearLow)} - ${formatPrice(yearHigh)}`;
    } else {
        document.getElementById('year-range').textContent = '-';
    }

    // Technicals
    const rsiEl = document.getElementById('rsi-val');
    if (typeof data.rsi === 'number') {
        rsiEl.textContent = data.rsi.toFixed(2);
        rsiEl.className = data.rsi > 70 ? 'negative' : data.rsi < 30 ? 'positive' : '';
    } else {
        rsiEl.textContent = '-';
        rsiEl.className = '';
    }

    document.getElementById('sma5-val').textContent = formatPrice(data.sma_5);
    document.getElementById('sma20-val').textContent = formatPrice(data.sma_20);

    // AI Analysis & Cost
    document.getElementById('ai-response-text').innerText = data.analysis || 'No analysis returned.';
    document.getElementById('analysis-cost').textContent = data.estimated_cost.toFixed(6);

    // Track which asset is currently displayed for history/watchlist integration.
    currentAnalysisMeta = {
        ticker: data.ticker,
        asset_type: 'stock',
        analysis_id: data.analysis_id || null
    };

    fadeInResults();
}

/**
 * Populates the UI with Crypto-specific data.
 */
function displayCryptoResults(data) {
    document.getElementById('display-name').textContent = data.name;
    document.getElementById('display-symbol').textContent = data.symbol.toUpperCase();
    document.getElementById('current-price').textContent = formatPrice(data.current_price_usd);

    document.getElementById('stock-only-stats').style.display = 'none';
    document.getElementById('crypto-only-stats').style.display = 'grid';

    // Price Changes
    document.getElementById('change-24h').innerHTML = formatPct(data.price_change_24h_pct);
    document.getElementById('change-7d').innerHTML = formatPct(data.price_change_7d_pct);
    document.getElementById('change-30d').innerHTML = formatPct(data.price_change_30d_pct);

    document.getElementById('ai-response-text').innerHTML =
        typeof marked !== 'undefined' ? marked.parse(data.analysis || '') : (data.analysis || '');
    document.getElementById('analysis-cost').textContent = data.estimated_cost.toFixed(6);

    // Track which crypto is currently displayed for history/watchlist integration.
    currentAnalysisMeta = {
        ticker: data.symbol.toUpperCase(),
        asset_type: 'crypto',
        analysis_id: data.analysis_id || null
    };

    fadeInResults();
}


// --- HISTORY & WATCHLIST HELPERS ---

/**
 * Renders a single history row as an HTML string.
 * @param {object} analysis
 */
function renderHistoryItem(analysis) {
    const price = typeof analysis.current_price === 'number'
        ? formatPrice(analysis.current_price)
        : '-';

    const rsiVal = typeof analysis.rsi === 'number' ? analysis.rsi : null;
    const rsiClass = rsiVal !== null
        ? (rsiVal > 70 ? 'negative' : rsiVal < 30 ? 'positive' : '')
        : '';
    const rsiText = rsiVal !== null ? rsiVal.toFixed(2) : 'N/A';

    const m30 = typeof analysis.momentum_30d === 'number' ? analysis.momentum_30d : null;
    const momentumClass = m30 !== null
        ? (m30 > 0 ? 'positive' : m30 < 0 ? 'negative' : '')
        : '';
    const momentumText = m30 !== null ? m30.toFixed(2) : '0.00';

    const dateText = formatHistoryDate(analysis.created_at);

    const noteText = (analysis.notes || '').trim();
    const hasNote = noteText.length > 0;
    const safeNote = escapeHtml(noteText);

    const typeLabel = analysis.asset_type === 'comparison' ? 'COMPARE' : (analysis.asset_type || '').toUpperCase();
    const typeBadgeClass = analysis.asset_type === 'comparison' ? 'history-type-badge history-type-compare' : 'history-type-badge';

    return `
        <div class="history-item" data-id="${analysis.id}" data-asset-type="${analysis.asset_type}">
            <div class="history-main">
                <div class="history-ticker">${escapeHtml(analysis.ticker)}</div>
                <div class="history-company">${escapeHtml(analysis.company_name || '')}</div>
                ${typeLabel ? `<span class="${typeBadgeClass}">${typeLabel}</span>` : ''}
            </div>
            <div class="history-metrics">
                <span class="history-price">${price}</span>
                <span class="history-rsi ${rsiClass}">RSI: ${rsiText}</span>
                <span class="history-momentum ${momentumClass}">30d: ${momentumText}</span>
            </div>
            <div class="history-meta">
                <span class="history-date">${dateText}</span>
                <button class="btn-history-load" data-id="${analysis.id}">Load</button>
            </div>
            <div class="history-note-area">
                ${
                    hasNote
                        ? `<div class="history-note-display"><span class="history-note-text" data-id="${analysis.id}">${safeNote}</span></div>`
                        : `<button class="history-note-add" data-id="${analysis.id}">Add Note</button>`
                }
                <div class="history-note-edit" data-id="${analysis.id}" style="display:none;">
                    <input
                        type="text"
                        class="history-note-input"
                        placeholder="Type a note..."
                    />
                    <button class="history-note-save" data-id="${analysis.id}">Save</button>
                </div>
            </div>
        </div>
    `;
}

function renderDocumentCard(doc) {
    const uploadedAt = formatHistoryDate(doc.uploaded_at);
    return `
        <div class="document-card"
             data-collection="${escapeHtml(doc.collection_name)}"
             data-filename="${escapeHtml(doc.filename)}">
            <div class="document-main">
                <div class="document-filename">${escapeHtml(doc.filename)}</div>
                <div class="document-meta">
                    <span>${doc.chunks_indexed} chunks</span>
                    <span>${uploadedAt}</span>
                </div>
            </div>
            <div class="document-actions">
                <button class="btn-secondary doc-ask-btn">Ask Questions</button>
                <button class="btn-secondary doc-delete-btn danger">Delete</button>
            </div>
        </div>
    `;
}
/**
 * Opens the inline note editor for the given history item id.
 * @param {string|number} id
 */
function openHistoryNoteEditor(id) {
    const item = document.querySelector(`.history-item[data-id="${id}"]`);
    if (!item) return;

    const display = item.querySelector('.history-note-display');
    const addBtn = item.querySelector('.history-note-add');
    const edit = item.querySelector('.history-note-edit');
    const input = item.querySelector('.history-note-input');

    let existing = '';
    if (display) {
        const textEl = display.querySelector('.history-note-text');
        existing = textEl ? textEl.textContent : '';
        display.style.display = 'none';
    }
    if (addBtn) addBtn.style.display = 'none';

    if (edit && input) {
        edit.style.display = 'flex';
        input.value = existing;
        input.focus();
    }
}

/**
 * Saves a note for a given analysis id via the API.
 * @param {string|number} id
 * @param {string} text
 */
async function saveHistoryNote(id, text) {
    try {
        const resp = await fetch(`${API_BASE}/history/${id}/notes`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ notes: text })
        });
        const data = await resp.json();
        if (!resp.ok) {
            return showError(data.detail || 'Failed to save note.');
        }

        showToast('Note saved');
        // Reload history so the updated note appears consistently.
        loadHistory(true);
    } catch (err) {
        console.error('saveHistoryNote error:', err);
        showError('Unable to save note.');
    }
}

/**
 * Loads paginated analysis history from the backend and renders it.
 * @param {boolean} reset
 */
async function loadHistory(reset = true) {
    const listEl = document.getElementById('history-list');
    const loadMoreBtn = document.getElementById('history-load-more');

    if (!listEl) return;

    if (reset) {
        historyOffset = 0;
        listEl.innerHTML = '';
    }

    let url = `${API_BASE}/history?limit=${HISTORY_PAGE_SIZE}&offset=${historyOffset}`;

    if (currentFilter.ticker) {
        url += `&ticker=${encodeURIComponent(currentFilter.ticker)}`;
    }
    if (currentFilter.asset_type) {
        url += `&asset_type=${encodeURIComponent(currentFilter.asset_type)}`;
    }

    try {
        const resp = await fetch(url);
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            return showError(err.detail || 'Failed to load history.');
        }

        const items = await resp.json();

        if (reset && items.length === 0) {
            listEl.innerHTML = `<div class="history-empty">No analyses yet. Run your first analysis above.</div>`;
            loadMoreBtn.style.display = 'none';
            return;
        }

        const html = items.map(renderHistoryItem).join('');
        if (reset) {
            listEl.innerHTML = html;
        } else {
            listEl.insertAdjacentHTML('beforeend', html);
        }

        // Show load-more only when a full "page" was returned.
        if (items.length === HISTORY_PAGE_SIZE) {
            loadMoreBtn.style.display = 'block';
        } else {
            loadMoreBtn.style.display = 'none';
        }

        historyOffset += items.length;
    } catch (err) {
        console.error('loadHistory error:', err);
        showError('Unable to load history.');
    }
}

/**
 * Fetches a specific analysis by ID and displays it in the main results card.
 * @param {number|string} id
 */
async function loadAnalysisById(id) {
    try {
        const resp = await fetch(`${API_BASE}/history/${id}`);
        const data = await resp.json();
        if (!resp.ok) {
            return showError(data.detail || 'Failed to load saved analysis.');
        }

        if (data.asset_type === 'stock') {
            displayStockResults({
                ticker: data.ticker,
                company_name: data.company_name,
                current_price: data.current_price,
                market_cap: data.market_cap,
                pe_ratio: data.pe_ratio,
                week_high_52: data.week_high_52,
                week_low_52: data.week_low_52,
                rsi: data.rsi,
                sma_5: data.sma_5,
                sma_20: data.sma_20,
                momentum: {
                    change_5d: data.momentum_5d,
                    change_30d: data.momentum_30d
                },
                support_resistance: null,
                analysis: data.ai_analysis,
                estimated_cost: data.estimated_cost || 0,
                analysis_id: data.id
            });
        } else if (data.asset_type === 'crypto') {
            displayCryptoResults({
                name: data.company_name || data.ticker,
                symbol: data.ticker,
                current_price_usd: data.current_price,
                market_cap_usd: data.market_cap,
                volume_24h: 0,
                price_change_24h_pct: 0,
                price_change_7d_pct: 0,
                price_change_30d_pct: 0,
                ath: 0,
                ath_change_pct: 0,
                analysis: data.ai_analysis,
                estimated_cost: data.estimated_cost || 0,
                analysis_id: data.id
            });
        } else if (data.asset_type === 'comparison') {
            const names = (data.company_name || data.ticker || '').split(' vs ');
            displayCompareResults({
                asset1_name: (names[0] || 'Asset 1').trim(),
                asset2_name: (names[1] || 'Asset 2').trim(),
                analysis: data.ai_analysis || '',
                estimated_cost: data.estimated_cost || 0
            });
        }

        // When loading from history, scroll all the way to the top of the page.
        window.scrollTo({ top: 0, behavior: 'smooth' });

        showToast('Loaded analysis from history');
    } catch (err) {
        console.error('loadAnalysisById error:', err);
        showError('Unable to load saved analysis.');
    }
}

/**
 * Updates the current history filter based on the search box value.
 */
function filterHistory() {
    const input = document.getElementById('history-search-input');
    if (!input) return;
    currentFilter.ticker = input.value.trim();
    loadHistory(true);
}

/**
 * Sets the asset_type filter for the history list.
 * @param {'all'|'stock'|'crypto'} type
 */
function setTypeFilter(type) {
    const allBtn = document.getElementById('history-filter-all');
    const stockBtn = document.getElementById('history-filter-stocks');
    const cryptoBtn = document.getElementById('history-filter-crypto');

    if (type === 'all') {
        currentFilter.asset_type = '';
        allBtn.classList.add('active');
        stockBtn.classList.remove('active');
        cryptoBtn.classList.remove('active');
    } else if (type === 'stock') {
        currentFilter.asset_type = 'stock';
        stockBtn.classList.add('active');
        allBtn.classList.remove('active');
        cryptoBtn.classList.remove('active');
    } else if (type === 'crypto') {
        currentFilter.asset_type = 'crypto';
        cryptoBtn.classList.add('active');
        allBtn.classList.remove('active');
        stockBtn.classList.remove('active');
    }

    loadHistory(true);
}

/**
 * Adds the currently displayed asset to the watchlist.
 */
async function addToWatchlist() {
    const ticker = currentAnalysisMeta.ticker;
    const assetType = currentAnalysisMeta.asset_type;

    if (!ticker || !assetType) {
        return showError('Run an analysis first before adding to watchlist.');
    }

    try {
        const resp = await fetch(`${API_BASE}/watchlist/add`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker, asset_type: assetType })
        });
        const data = await resp.json();

        if (!resp.ok) {
            return showError(data.detail || 'Failed to add to watchlist.');
        }

        if (data.already_exists) {
            showToast('Already on watchlist');
        } else if (data.added) {
            showToast('Added to watchlist');
        }

        loadWatchlist();
    } catch (err) {
        console.error('addToWatchlist error:', err);
        showError('Unable to add to watchlist.');
    }
}

/**
 * Loads the current watchlist from the backend and renders chips.
 */
async function loadWatchlist() {
    const container = document.getElementById('watchlist-chips');
    if (!container) return;

    try {
        const resp = await fetch(`${API_BASE}/watchlist`);
        const data = await resp.json();
        if (!resp.ok) {
            return showError(data.detail || 'Failed to load watchlist.');
        }

        if (!Array.isArray(data) && Array.isArray(data.assets)) {
            // Backwards compatibility guard (should not happen with DB-backed API).
            container.innerHTML = '';
            return;
        }

        const chipsHtml = data
            .map(item => `
                <div class="watchlist-chip">
                    <span class="watchlist-chip-label">${item.ticker}</span>
                    <button class="watchlist-chip-remove" data-id="${item.id}">×</button>
                </div>
            `)
            .join('');

        container.innerHTML = chipsHtml || '<span class="history-empty">No watchlist items yet.</span>';
    } catch (err) {
        console.error('loadWatchlist error:', err);
        showError('Unable to load watchlist.');
    }
}

/**
 * Removes a watchlist entry by ID.
 * @param {number|string} id
 */
async function removeFromWatchlist(id) {
    try {
        const resp = await fetch(`${API_BASE}/watchlist/${id}`, {
            method: 'DELETE'
        });
        const data = await resp.json();
        if (!resp.ok) {
            return showError(data.detail || 'Failed to remove from watchlist.');
        }

        showToast('Removed from watchlist');
        loadWatchlist();
    } catch (err) {
        console.error('removeFromWatchlist error:', err);
        showError('Unable to remove from watchlist.');
    }
}

/**
 * Handles the display for the "Compare Two Assets" function.
 */
function displayCompareResults(data) {
    document.getElementById('display-name').textContent = `${data.asset1_name} vs ${data.asset2_name}`;
    document.getElementById('display-symbol').textContent = 'COMPARE';

    // Clear leftover data from any previous single-asset analysis
    document.getElementById('current-price').textContent = '';
    document.getElementById('pe-ratio').textContent = '';
    document.getElementById('year-range').textContent = '';
    document.getElementById('rsi-val').textContent = '';
    document.getElementById('sma5-val').textContent = '';
    document.getElementById('sma20-val').textContent = '';

    document.getElementById('stock-only-stats').style.display = 'none';
    document.getElementById('crypto-only-stats').style.display = 'none';

    document.getElementById('ai-response-text').innerHTML =
        typeof marked !== 'undefined'
            ? marked.parse(data.analysis || 'No comparison returned.')
            : (data.analysis || 'No comparison returned.');
    document.getElementById('analysis-cost').textContent =
        `This analysis cost: $${data.estimated_cost.toFixed(6)}`;

    fadeInResults();
    loadHistory(true);
}

async function uploadDocument(file) {
    if (!file) return;
    const statusEl = document.getElementById('doc-upload-status');
    if (statusEl) {
        statusEl.style.display = 'block';
        statusEl.textContent = `Uploading "${file.name}"...`;
    }
    const formData = new FormData();
    formData.append('file', file);
    try {
        const resp = await fetch(`${API_BASE}/documents/upload`, {
            method: 'POST',
            body: formData
        });
        const data = await resp.json();
        if (!resp.ok) {
            return showError(data.detail || data.message || 'Failed to upload document.');
        }
        showToast('Document indexed successfully');
        loadDocuments();
    } catch (err) {
        console.error('uploadDocument error:', err);
        showError('Unable to upload document.');
    } finally {
        if (statusEl) {
            statusEl.textContent = '';
            statusEl.style.display = 'none';
        }
    }
}
async function loadDocuments() {
    const container = document.getElementById('documents-list');
    if (!container) return;
    try {
        const resp = await fetch(`${API_BASE}/documents`);
        const data = await resp.json();
        if (!resp.ok) {
            return showError(data.detail || 'Failed to load documents.');
        }
        if (!Array.isArray(data) || data.length === 0) {
            container.innerHTML = '<div class="history-empty">No documents uploaded yet.</div>';
            return;
        }
        container.innerHTML = data.map(renderDocumentCard).join('');
    } catch (err) {
        console.error('loadDocuments error:', err);
        showError('Unable to load documents.');
    }
}
function selectDocument(collectionName, filename) {
    currentDocument = { collectionName, filename };
    const panel = document.getElementById('doc-qa-panel');
    const nameEl = document.getElementById('doc-active-name');
    const answerEl = document.getElementById('doc-answer-text');
    const costEl = document.getElementById('doc-cost');
    const confEl = document.getElementById('doc-confidence-badge');
    const sourcesContainer = document.getElementById('doc-sources-container');
    const sourcesList = document.getElementById('doc-sources-list');
    const historyList = document.getElementById('doc-history-list');
    if (panel) panel.style.display = 'block';
    if (nameEl) nameEl.textContent = filename || collectionName;
    if (answerEl) answerEl.textContent = '';
    if (costEl) costEl.textContent = '';
    if (confEl) {
        confEl.textContent = '';
        confEl.className = 'confidence-badge';
    }
    if (sourcesContainer) sourcesContainer.style.display = 'none';
    if (sourcesList) sourcesList.innerHTML = '';
    if (historyList) historyList.innerHTML = '';
    loadDocumentHistory(collectionName);
}
async function loadDocumentHistory(collectionName) {
    const container = document.getElementById('doc-history-list');
    if (!container) return;
    try {
        const resp = await fetch(`${API_BASE}/documents/${encodeURIComponent(collectionName)}/history`);
        const data = await resp.json();
        if (!resp.ok) {
            console.error('loadDocumentHistory error:', data.detail || data);
            return;
        }
        if (!Array.isArray(data) || data.length === 0) {
            container.innerHTML = '<div class="history-empty">No previous Q&A yet for this document.</div>';
            return;
        }
        container.innerHTML = data
            .map((item, idx) => `
                <div class="doc-history-item">
                    <div class="doc-history-question">
                        ${escapeHtml(item.question)}
                    </div>
                    <div class="doc-history-answer">
                        ${escapeHtml(item.answer || '')}
                    </div>
                    <div class="doc-history-meta">
                        ${formatHistoryDate(item.created_at)}
                    </div>
                </div>
            `)
            .join('');
    } catch (err) {
        console.error('loadDocumentHistory error:', err);
    }
}
async function askQuestion() {
    const input = document.getElementById('doc-question-input');
    if (!currentDocument.collectionName) {
        return showError('Select a document first.');
    }
    if (!input || !input.value.trim()) {
        return showError('Please enter a question.');
    }
    const question = input.value.trim();
    try {
        const resp = await fetch(`${API_BASE}/documents/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question,
                collection_name: currentDocument.collectionName,
                n_results: 4
            })
        });
        const data = await resp.json();
        if (!resp.ok) {
            return showError(data.detail || 'Failed to ask question.');
        }
        const answerEl = document.getElementById('doc-answer-text');
        const costEl = document.getElementById('doc-cost');
        const confEl = document.getElementById('doc-confidence-badge');
        const sourcesContainer = document.getElementById('doc-sources-container');
        const sourcesList = document.getElementById('doc-sources-list');
        if (answerEl) {
            answerEl.textContent = data.answer || 'No answer returned.';
        }
        if (costEl) {
            const cost = Number(data.estimated_cost || 0);
            costEl.textContent =
                `Estimated cost: $${cost.toFixed(6)} ` +
                `(in: ${data.input_tokens || 0}, out: ${data.output_tokens || 0})`;
        }
        if (confEl) {
            const ok = !!data.found_relevant_context;
            confEl.textContent = ok
                ? 'High confidence'
                : 'Low confidence (weak context)';
            confEl.className =
                'confidence-badge ' + (ok ? 'confidence-high' : 'confidence-low');
        }
        if (sourcesList) {
            const sources = Array.isArray(data.sources) ? data.sources : [];
            if (sources.length === 0) {
                sourcesList.innerHTML = '<div class="history-empty">No source passages returned.</div>';
            } else {
                sourcesList.innerHTML = sources
                    .map((src, idx) => `
                        <div class="doc-source">
                            <div class="doc-source-label">Passage ${idx + 1}</div>
                            <div class="doc-source-text">${escapeHtml(src)}</div>
                        </div>
                    `)
                    .join('');
            }
        }
        if (sourcesContainer) {
            sourcesContainer.style.display = 'none';
        }
        const toggleBtn = document.getElementById('doc-sources-toggle');
        if (toggleBtn) {
            toggleBtn.textContent = 'Show sources';
        }
        // Refresh history so the new Q&A appears there too
        loadDocumentHistory(currentDocument.collectionName);
    } catch (err) {
        console.error('askQuestion error:', err);
        showError('Unable to ask question.');
    }
}
async function deleteDocument(collectionName) {
    if (!collectionName) return;
    const confirmed = window.confirm('Delete this document and its collection?');
    if (!confirmed) return;
    try {
        const resp = await fetch(`${API_BASE}/documents/${encodeURIComponent(collectionName)}`, {
            method: 'DELETE'
        });
        const data = await resp.json();
        if (!resp.ok) {
            return showError(data.detail || 'Failed to delete document.');
        }
        showToast('Document deleted');
        if (currentDocument.collectionName === collectionName) {
            currentDocument = { collectionName: null, filename: null };
            const panel = document.getElementById('doc-qa-panel');
            if (panel) panel.style.display = 'none';
        }
        loadDocuments();
    } catch (err) {
        console.error('deleteDocument error:', err);
        showError('Unable to delete document.');
    }
}
function setupDragAndDrop() {
    const dropzone = document.getElementById('doc-upload-dropzone');
    const fileInput = document.getElementById('doc-file-input');
    if (!dropzone || !fileInput) return;
    dropzone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', () => {
        const file = fileInput.files && fileInput.files[0];
        if (file) uploadDocument(file);
        fileInput.value = '';
    });
    ['dragenter', 'dragover'].forEach(evtName => {
        dropzone.addEventListener(evtName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('drag-over');
        });
    });
    ['dragleave', 'drop'].forEach(evtName => {
        dropzone.addEventListener(evtName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('drag-over');
        });
    });
    dropzone.addEventListener('drop', (e) => {
        const files = e.dataTransfer && e.dataTransfer.files;
        if (files && files.length > 0) {
            uploadDocument(files[0]);
        }
    });
}


// --- API ACTIONS ---

async function analyzeStock() {
    const ticker = document.getElementById('stock-ticker').value.trim().toUpperCase();
    if (!ticker) return showError('Please enter a stock ticker.');

    const btn = document.getElementById('analyze-stock-btn');
    startLoading(ticker, btn, 'Analyze');

    try {
        const response = await fetch(`${API_BASE}/analyze/stock`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker })
        });
        const data = await response.json();
        stopLoading(btn);
        if (response.ok) {
            displayStockResults(data);
            loadStats();
        }
        else showError(data.detail || 'Error analyzing stock.');
    } catch (err) {
        stopLoading(btn);
        console.error('analyzeStock error:', err);
        showError('Connection failed. Is the server running?');
    }
}

async function analyzeCrypto() {
    const coinId = document.getElementById('crypto-id').value.trim().toLowerCase();
    if (!coinId) return showError('Please enter a CoinGecko ID.');

    const btn = document.getElementById('analyze-crypto-btn');
    startLoading(coinId, btn, 'Analyze');

    try {
        const response = await fetch(`${API_BASE}/analyze/crypto`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ coin_id: coinId })
        });
        const data = await response.json();
        stopLoading(btn);
        if (response.ok) {
            displayCryptoResults(data);
            loadStats();
        }
        else showError(data.detail || 'Error analyzing crypto.');
    } catch (err) {
        stopLoading(btn);
        console.error('analyzeCrypto error:', err);
        showError('Connection failed. Is the server running?');
    }
}

async function compareAssets() {
    const a1 = document.getElementById('asset1-input').value.trim();
    const t1 = document.getElementById('asset1-type').value;
    const a2 = document.getElementById('asset2-input').value.trim();
    const t2 = document.getElementById('asset2-type').value;

    if (!a1 || !a2) return showError('Both assets are required for comparison.');

    const btn = document.getElementById('compare-btn');
    startLoading(`${a1} vs ${a2}`, btn, 'Compare');

    try {
        const response = await fetch(`${API_BASE}/analyze/compare`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ asset1: a1, asset1_type: t1, asset2: a2, asset2_type: t2 })
        });
        const data = await response.json();
        stopLoading(btn);
        if (response.ok) displayCompareResults(data);
        else showError(data.detail || 'Comparison failed.');
    } catch (err) {
        stopLoading(btn);
        console.error('compareAssets error:', err);
        showError('Connection failed. Is the server running?');
    }
}


// --- INITIALIZATION & EVENT LISTENERS ---

document.addEventListener('DOMContentLoaded', () => {
    // Tab Switching Logic
    const stockTab = document.getElementById('tab-stocks');
    const cryptoTab = document.getElementById('tab-crypto');
    const stockSearch = document.getElementById('stock-search');
    const cryptoSearch = document.getElementById('crypto-search');

    stockTab.addEventListener('click', () => {
        stockTab.classList.add('active');
        cryptoTab.classList.remove('active');
        stockSearch.style.display = 'flex';
        cryptoSearch.style.display = 'none';
    });

    cryptoTab.addEventListener('click', () => {
        cryptoTab.classList.add('active');
        stockTab.classList.remove('active');
        cryptoSearch.style.display = 'flex';
        stockSearch.style.display = 'none';
    });

    // Button Click Listeners
    document.getElementById('analyze-stock-btn').onclick = analyzeStock;
    document.getElementById('analyze-crypto-btn').onclick = analyzeCrypto;
    document.getElementById('compare-btn').onclick = compareAssets;

    // Enter Key Support
    const inputs = document.querySelectorAll('input[type="text"]');
    inputs.forEach(input => {
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                if (input.id === 'stock-ticker') analyzeStock();
                else if (input.id === 'crypto-id') analyzeCrypto();
                else if (input.id.startsWith('asset')) compareAssets();
            }
        });
    });

    // History toggle and controls
    const historyToggleBtn = document.getElementById('history-toggle-btn');
    const historyContent = document.getElementById('history-content');
    const historyRefreshBtn = document.getElementById('history-refresh-btn');
    const historyLoadMoreBtn = document.getElementById('history-load-more');
    const historySearchInput = document.getElementById('history-search-input');
    const historyList = document.getElementById('history-list');

    if (historyToggleBtn && historyContent) {
        historyToggleBtn.addEventListener('click', () => {
            const isHidden = historyContent.style.display === 'none' || historyContent.style.display === '';
            historyContent.style.display = isHidden ? 'block' : 'none';
            if (isHidden) {
                loadHistory(true);
            }
        });
    }

    if (historyRefreshBtn) {
        historyRefreshBtn.addEventListener('click', () => loadHistory(true));
    }

    if (historyLoadMoreBtn) {
        historyLoadMoreBtn.addEventListener('click', () => loadHistory(false));
    }

    // Debounced search for history filtering
    let historySearchTimeout = null;
    if (historySearchInput) {
        historySearchInput.addEventListener('input', () => {
            clearTimeout(historySearchTimeout);
            historySearchTimeout = setTimeout(() => filterHistory(), 400);
        });
    }

    // Type filter buttons
    const filterAllBtn = document.getElementById('history-filter-all');
    const filterStockBtn = document.getElementById('history-filter-stocks');
    const filterCryptoBtn = document.getElementById('history-filter-crypto');

    if (filterAllBtn) filterAllBtn.addEventListener('click', () => setTypeFilter('all'));
    if (filterStockBtn) filterStockBtn.addEventListener('click', () => setTypeFilter('stock'));
    if (filterCryptoBtn) filterCryptoBtn.addEventListener('click', () => setTypeFilter('crypto'));

    // Delegate click for history "Load" buttons
    if (historyList) {
        historyList.addEventListener('click', (e) => {
            if (e.target.classList.contains('btn-history-load')) {
                const id = e.target.getAttribute('data-id');
                if (id) loadAnalysisById(id);
            } else if (
                e.target.classList.contains('history-note-add') ||
                e.target.classList.contains('history-note-text')
            ) {
                const id = e.target.getAttribute('data-id')
                    || e.target.closest('.history-item')?.getAttribute('data-id');
                if (id) openHistoryNoteEditor(id);
            } else if (e.target.classList.contains('history-note-save')) {
                const id = e.target.getAttribute('data-id');
                const editContainer = e.target.closest('.history-note-edit');
                const input = editContainer
                    ? editContainer.querySelector('.history-note-input')
                    : null;
                if (id && input) {
                    saveHistoryNote(id, input.value.trim());
                }
            }
        });

        historyList.addEventListener('keydown', (e) => {
            if (
                e.target.classList.contains('history-note-input') &&
                e.key === 'Enter'
            ) {
                const editContainer = e.target.closest('.history-note-edit');
                const id = editContainer
                    ? editContainer.getAttribute('data-id')
                    : null;
                if (id) {
                    e.preventDefault();
                    saveHistoryNote(id, e.target.value.trim());
                }
            }
        });
    }

    // Watchlist actions
    const addWatchlistBtn = document.getElementById('add-to-watchlist-btn');
    if (addWatchlistBtn) {
        addWatchlistBtn.addEventListener('click', addToWatchlist);
    }

    const watchlistChips = document.getElementById('watchlist-chips');
    if (watchlistChips) {
        watchlistChips.addEventListener('click', (e) => {
            if (e.target.classList.contains('watchlist-chip-remove')) {
                const id = e.target.getAttribute('data-id');
                if (id) removeFromWatchlist(id);
            }
        });
    }

    // Document Q&A setup
    setupDragAndDrop();
    loadDocuments();
    const documentsList = document.getElementById('documents-list');
    if (documentsList) {
        documentsList.addEventListener('click', (e) => {
            const card = e.target.closest('.document-card');
            if (!card) return;
            const collectionName = card.getAttribute('data-collection');
            const filename = card.getAttribute('data-filename');
            if (e.target.classList.contains('doc-ask-btn')) {
                selectDocument(collectionName, filename);
            } else if (e.target.classList.contains('doc-delete-btn')) {
                deleteDocument(collectionName);
            }
        });
    }
    const askBtn = document.getElementById('doc-ask-btn');
    if (askBtn) {
        askBtn.addEventListener('click', askQuestion);
    }
    const sourcesToggle = document.getElementById('doc-sources-toggle');
    if (sourcesToggle) {
        sourcesToggle.addEventListener('click', () => {
            const container = document.getElementById('doc-sources-container');
            if (!container) return;
            const isHidden = container.style.display === 'none' || container.style.display === '';
            container.style.display = isHidden ? 'block' : 'none';
            sourcesToggle.textContent = isHidden ? 'Hide sources' : 'Show sources';
        });
    }

    // Auto-load stats, history and watchlist on page load so data is visible immediately.
    loadStats();
    loadHistory(true);
    loadWatchlist();
});