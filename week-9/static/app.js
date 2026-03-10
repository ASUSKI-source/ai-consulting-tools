/**
 * AI Market Analyzer - Core Logic
 * Handles API communication, UI state, and data visualization.
 */

const API_BASE = window.location.origin;

/**
 * Returns the Authorization header object if a token exists.
 * Passes it to every fetch() call to protected endpoints.
 */
function authHeaders() {
    const token = localStorage.getItem('token');
    return token
        ? { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` }
        : { 'Content-Type': 'application/json' };
}

/**
 * If response is 401, clear auth state, show message, and reload to login. Returns true if handled.
 */
function handle401(resp) {
    if (resp.status === 401) {
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        showError('Your session has expired. Please sign in again.');
        setTimeout(() => location.reload(), 2000);
        return true;
    }
    return false;
}

/**
 * Clear auth state and redirect to login page.
 */
async function logout() {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    window.location.href = '/login';
}

// --- HISTORY & WATCHLIST STATE ---
let historyOffset = 0;
const HISTORY_PAGE_SIZE = 15;
let currentFilter = { ticker: '', asset_type: '' };
let currentAnalysisMeta = { ticker: '', asset_type: '', analysis_id: null };

// --- DOCUMENT Q&A STATE ---
let currentDocument = { collectionName: null, filename: null };
let allDocuments = [];
let collectionsList = [];
let currentCollectionFilter = '';
// --- LOADING STATE MANAGEMENT ---

let loadingInterval = null;

// --- Financial Research Agent Chat State ---
const conversationHistory = [];
const activeActivityItems = {};

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

    const user = JSON.parse(localStorage.getItem('user') || '{}');
    const userName = user.full_name || user.email?.split('@')[0] || '';
    const forLabel = userName ? ` for ${userName}` : '';

    const phases = [
        `Fetching data for ${ticker}...`,
        `Calculating indicators${forLabel}...`,
        `Generating AI analysis${forLabel}...`
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

// --- Document Q&A phased loading (variable wait; phases give feedback) ---
const docQaPhases = [
    { id: 'phase-search', text: 'Searching document...', duration: 2000 },
    { id: 'phase-budget', text: 'Selecting best passages...', duration: 1500 },
    { id: 'phase-claude', text: 'Generating answer with Claude...', duration: 0 }
];
let loadingPhaseTimeoutId = null;

function showLoadingPhases() {
    document.getElementById('loading').style.display = 'block';
    let currentPhase = 0;
    const el = document.getElementById('loading-text');
    if (!el) return;

    function advancePhase() {
        if (currentPhase < docQaPhases.length) {
            el.textContent = docQaPhases[currentPhase].text;
            if (docQaPhases[currentPhase].duration > 0) {
                loadingPhaseTimeoutId = setTimeout(advancePhase, docQaPhases[currentPhase].duration);
            }
            currentPhase++;
        }
    }
    advancePhase();
}

function hideLoadingPhases() {
    if (loadingPhaseTimeoutId !== null) {
        clearTimeout(loadingPhaseTimeoutId);
        loadingPhaseTimeoutId = null;
    }
    document.getElementById('loading').style.display = 'none';
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

/** Category-specific icons for the error box. */
const ERROR_ICONS = {
    not_found: '🔍',
    rate_limit: '⏱',
    file_error: '📄',
    unknown: '⚠️'
};

/**
 * Displays an error message with an auto-hide timer.
 * Accepts either a plain string (backwards compatible) or a dict with
 * { user_message, suggestion, category }.
 * If dict: line 1 (larger) = user_message, line 2 (smaller, italic) = suggestion.
 * Shows a category-specific icon before the message when category is present.
 * @param {string|object} errorData
 */
function showError(errorData) {
    const errorDiv = document.getElementById('error-message');
    let html = '';

    if (typeof errorData === 'string') {
        html = `<span class="error-message-text">${escapeHtml(errorData)}</span>`;
    } else {
        const msg = errorData.user_message || 'An error occurred.';
        const hint = errorData.suggestion || '';
        const category = (errorData.category || 'unknown').toLowerCase();
        const icon = ERROR_ICONS[category] || ERROR_ICONS.unknown;
        html = '<span class="error-message-content">';
        html += `<span class="error-message-icon" aria-hidden="true">${icon}</span>`;
        html += '<span class="error-message-body">';
        html += `<span class="error-message-text">${escapeHtml(msg)}</span>`;
        if (hint) {
            html += `<br><span class="error-message-suggestion">${escapeHtml(hint)}</span>`;
        }
        html += '</span></span>';
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
        const resp = await fetch(`${API_BASE}/stats`, { headers: authHeaders() });
        const data = await resp.json();
        if (handle401(resp)) return;
        if (!resp.ok) {
            el.textContent = '';
            return;
        }

        const total = data.total_analyses || 0;
        const unique = data.unique_tickers || 0;
        const totalCost = Number(data.total_cost || 0).toFixed(4);

        el.textContent = `${total} analyses run  ·  ${unique} unique tickers  ·  Total AI cost: $${totalCost}`;
    } catch (err) {
        console.error('Error in loadStats:', err);
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
    const coll = doc.collection_group || doc.collection_name || '';
    return `
        <div class="document-card"
             data-collection="${escapeHtml(coll)}"
             data-filename="${escapeHtml(doc.filename)}">
            <div class="document-main">
                <div class="document-filename">${escapeHtml(doc.filename)}</div>
                <div class="document-meta">
                    <span>${doc.chunks_indexed} chunks</span>
                    <span>${uploadedAt}</span>
                </div>
            </div>
            <div class="document-actions">
                <button type="button" class="btn-secondary doc-preview-chunks-btn">Preview Chunks</button>
                <button class="btn-secondary doc-ask-btn">Ask Questions</button>
                <button class="btn-secondary doc-delete-btn danger">Delete</button>
            </div>
            <div class="doc-chunk-preview" style="display: none;" data-offset="0">
                <div class="doc-chunk-preview-list"></div>
                <button type="button" class="btn-secondary small doc-chunk-preview-more" style="display: none;">Show More</button>
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
            headers: authHeaders(),
            body: JSON.stringify({ notes: text })
        });
        const data = await resp.json().catch(() => ({}));
        if (handle401(resp)) return;
        if (!resp.ok) {
            showError(data.detail || data.user_message || 'An error occurred.');
            return;
        }

        showToast('Note saved');
        // Reload history so the updated note appears consistently.
        loadHistory(true);
    } catch (err) {
        console.error('Error in saveHistoryNote:', err);
        showError('Connection failed. Is the server running?');
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
        const resp = await fetch(url, { headers: authHeaders() });
        const data = await resp.json().catch(() => ({}));
        if (handle401(resp)) return;
        if (!resp.ok) {
            showError(data.detail || data.user_message || 'An error occurred.');
            return;
        }

        const items = data;

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
        console.error('Error in loadHistory:', err);
        showError('Connection failed. Is the server running?');
    }
}

/**
 * Fetches a specific analysis by ID and displays it in the main results card.
 * @param {number|string} id
 */
async function loadAnalysisById(id) {
    try {
        const resp = await fetch(`${API_BASE}/history/${id}`, { headers: authHeaders() });
        const data = await resp.json().catch(() => ({}));
        if (handle401(resp)) return;
        if (!resp.ok) {
            showError(data.detail || data.user_message || 'An error occurred.');
            return;
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
        console.error('Error in loadAnalysisById:', err);
        showError('Connection failed. Is the server running?');
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
            headers: authHeaders(),
            body: JSON.stringify({ ticker, asset_type: assetType })
        });
        const data = await resp.json().catch(() => ({}));
        if (handle401(resp)) return;
        if (!resp.ok) {
            showError(data.detail || data.user_message || 'An error occurred.');
            return;
        }

        if (data.already_exists) {
            showToast('Already on watchlist');
        } else if (data.added) {
            showToast('Added to watchlist');
        }

        loadWatchlist();
    } catch (err) {
        console.error('Error in addToWatchlist:', err);
        showError('Connection failed. Is the server running?');
    }
}

/**
 * Loads the current watchlist from the backend and renders chips.
 */
async function loadWatchlist() {
    const container = document.getElementById('watchlist-chips');
    if (!container) return;

    try {
        const resp = await fetch(`${API_BASE}/watchlist`, { headers: authHeaders() });
        const data = await resp.json().catch(() => ({}));
        if (handle401(resp)) return;
        if (!resp.ok) {
            showError(data.detail || data.user_message || 'An error occurred.');
            return;
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
        console.error('Error in loadWatchlist:', err);
        showError('Connection failed. Is the server running?');
    }
}

/**
 * Removes a watchlist entry by ID.
 * @param {number|string} id
 */
async function removeFromWatchlist(id) {
    try {
        const resp = await fetch(`${API_BASE}/watchlist/${id}`, {
            method: 'DELETE',
            headers: authHeaders()
        });
        const data = await resp.json().catch(() => ({}));
        if (handle401(resp)) return;
        if (!resp.ok) {
            showError(data.detail || data.user_message || 'An error occurred.');
            return;
        }

        showToast('Removed from watchlist');
        loadWatchlist();
    } catch (err) {
        console.error('Error in removeFromWatchlist:', err);
        showError('Connection failed. Is the server running?');
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

async function loadCollections() {
    try {
        const resp = await fetch(`${API_BASE}/collections`, { headers: authHeaders() });
        const data = await resp.json().catch(() => ({}));
        if (handle401(resp)) return;
        if (!resp.ok) {
            showError(data.detail || data.user_message || 'An error occurred.');
            return;
        }
        collectionsList = Array.isArray(data) ? data : [];
        const tabsList = document.getElementById('doc-collection-tabs-list');
        if (tabsList) {
            tabsList.innerHTML = collectionsList
                .map((c) => `<button type="button" class="tab-btn doc-collection-tab" data-collection="${escapeHtml(c.name)}">${escapeHtml(c.name)}</button>`)
                .join('');
        }
        const uploadSelect = document.getElementById('doc-upload-collection');
        if (uploadSelect) {
            const current = uploadSelect.value;
            if (collectionsList.length === 0) {
                uploadSelect.innerHTML = '<option value="default">default</option>';
            } else {
                uploadSelect.innerHTML = collectionsList
                    .map((c) => `<option value="${escapeHtml(c.name)}">${escapeHtml(c.name)}</option>`)
                    .join('');
            }
            if (collectionsList.length > 0 && !collectionsList.some((c) => c.name === current)) {
                uploadSelect.value = collectionsList[0].name;
            } else if (current) {
                uploadSelect.value = current;
            }
        }
    } catch (err) {
        console.error('Error in loadCollections:', err);
        showError('Connection failed. Is the server running?');
    }
}
function selectCollection(name) {
    currentCollectionFilter = name || '';
    document.querySelectorAll('.doc-collection-tab').forEach((btn) => {
        const data = btn.getAttribute('data-collection') || '';
        btn.classList.toggle('active', data === currentCollectionFilter);
    });
    renderDocumentsList();
}
async function createCollection(name, description) {
    if (!name || !name.trim()) return showError('Collection name is required.');
    try {
        const resp = await fetch(`${API_BASE}/collections`, {
            method: 'POST',
            headers: authHeaders(),
            body: JSON.stringify({ name: name.trim(), description: description ? description.trim() : null })
        });
        const data = await resp.json().catch(() => ({}));
        if (handle401(resp)) return;
        if (!resp.ok) {
            showError(data.detail || data.user_message || 'An error occurred.');
            return;
        }
        showToast('Collection created');
        document.getElementById('doc-create-collection-form').style.display = 'none';
        document.getElementById('doc-new-collection-name').value = '';
        document.getElementById('doc-new-collection-desc').value = '';
        loadCollections();
        loadDocuments();
    } catch (err) {
        console.error('Error in createCollection:', err);
        showError('Connection failed. Is the server running?');
    }
}
function renderDocumentsList() {
    const container = document.getElementById('documents-list');
    if (!container) return;
    const filtered = currentCollectionFilter
        ? allDocuments.filter((d) => (d.collection_group || d.collection_name) === currentCollectionFilter)
        : allDocuments;
    if (filtered.length === 0) {
        container.innerHTML = '<div class="history-empty">No documents in this view.</div>';
        return;
    }
    container.innerHTML = filtered.map(renderDocumentCard).join('');
}
async function uploadDocument(file) {
    if (!file) return;
    const statusEl = document.getElementById('doc-upload-status');
    const uploadSelect = document.getElementById('doc-upload-collection');
    const collectionGroup = (uploadSelect && uploadSelect.value) || 'default';
    if (statusEl) {
        statusEl.style.display = 'block';
        statusEl.textContent = `Uploading "${file.name}"...`;
    }
    const formData = new FormData();
    formData.append('file', file);
    formData.append('collection_group', collectionGroup);
    try {
        const token = localStorage.getItem('token');
        const headers = token ? { 'Authorization': `Bearer ${token}` } : {};
        const resp = await fetch(`${API_BASE}/documents/upload`, {
            method: 'POST',
            headers,
            body: formData
        });
        const data = await resp.json().catch(() => ({}));
        if (handle401(resp)) return;
        if (!resp.ok) {
            showError(data.detail || data.user_message || 'An error occurred.');
            return;
        }
        showToast('Document indexed successfully');
        loadDocuments();
        loadCollections();
    } catch (err) {
        console.error('Error in uploadDocument:', err);
        showError('Connection failed. Is the server running?');
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
        const resp = await fetch(`${API_BASE}/documents`, { headers: authHeaders() });
        const data = await resp.json().catch(() => ({}));
        if (handle401(resp)) return;
        if (!resp.ok) {
            showError(data.detail || data.user_message || 'An error occurred.');
            return;
        }
        allDocuments = Array.isArray(data) ? data : [];
        if (allDocuments.length === 0) {
            container.innerHTML = '<div class="history-empty">No documents uploaded yet.</div>';
            return;
        }
        renderDocumentsList();
    } catch (err) {
        console.error('Error in loadDocuments:', err);
        showError('Connection failed. Is the server running?');
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
    const attributionsEl = document.getElementById('doc-source-attributions');
    if (attributionsEl) attributionsEl.innerHTML = '';
    loadDocumentHistory(collectionName);
}
async function loadDocumentHistory(collectionName) {
    const container = document.getElementById('doc-history-list');
    if (!container) return;
    try {
        const resp = await fetch(`${API_BASE}/documents/${encodeURIComponent(collectionName)}/history`, { headers: authHeaders() });
        const data = await resp.json().catch(() => ({}));
        if (handle401(resp)) return;
        if (!resp.ok) {
            showError(data.detail || data.user_message || 'An error occurred.');
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
        console.error('Error in loadDocumentHistory:', err);
        showError('Connection failed. Is the server running?');
    }
}
/** Confidence bar fill colors by API color string. */
const CONFIDENCE_COLORS = {
    green: '#22C55E',
    yellow: '#EAB308',
    orange: '#F97316',
    red: '#EF4444',
};

/** Hint text per confidence color. */
const CONFIDENCE_HINTS = {
    green: 'Strong match — the answer is well-supported by the document.',
    yellow: 'Moderate match — verify the answer against the source passages.',
    orange: 'Weak match — the document may not contain a clear answer.',
    red: 'Very weak match — treat this answer with caution.',
};

/**
 * Updates the confidence bar above the answer (Document Q&A).
 * @param {{ score: number, label: string, color: string }} confidence
 */
function updateConfidence(confidence) {
    const container = document.getElementById('confidence-container');
    const labelEl = document.getElementById('confidence-label-text');
    const scoreEl = document.getElementById('confidence-score-text');
    const fillEl = document.getElementById('confidence-bar-fill');
    const hintEl = document.getElementById('confidence-hint');
    if (!container || !labelEl || !scoreEl || !fillEl || !hintEl) return;

    container.style.display = 'block';
    labelEl.textContent = confidence.label || 'Confidence';
    scoreEl.textContent = (confidence.score != null ? confidence.score : 0) + '/100';
    fillEl.style.width = (confidence.score != null ? Math.max(0, Math.min(100, confidence.score)) : 0) + '%';
    const colorKey = (confidence.color || 'red').toLowerCase();
    fillEl.style.backgroundColor = CONFIDENCE_COLORS[colorKey] || CONFIDENCE_COLORS.red;
    hintEl.textContent = CONFIDENCE_HINTS[colorKey] || CONFIDENCE_HINTS.red;
}

function renderSourceAttributions(sources) {
    const container = document.getElementById('doc-source-attributions');
    if (!container) return;
    if (!Array.isArray(sources) || sources.length === 0) {
        container.innerHTML = '';
        return;
    }
    const seen = new Set();
    const items = [];
    for (const src of sources) {
        const file = (src && src.source_file) || '';
        const coll = (src && src.collection_group) || currentDocument.collectionName || '';
        const key = `${coll}:${file}`;
        if (!file || seen.has(key)) continue;
        seen.add(key);
        const label = `${escapeHtml(file)} (${escapeHtml(coll)})`;
        items.push(
            `<a href="#" class="doc-source-attribution-link" data-collection="${escapeHtml(coll)}" data-filename="${escapeHtml(file)}">${label}</a>`
        );
    }
    container.innerHTML = items.length
        ? '<span class="doc-sources-label">Sources: </span>' + items.join(', ')
        : '';
}
async function askQuestion() {
    const input = document.getElementById('doc-question-input');
    const scopeEl = document.querySelector('input[name="doc-search-scope"]:checked');
    const scope = (scopeEl && scopeEl.value) || 'document';
    if (scope !== 'all' && !currentDocument.collectionName) {
        return showError('Select a document first (or use "All documents" search).');
    }
    if (!input || !input.value.trim()) {
        return showError('Please enter a question.');
    }
    const question = input.value.trim();
    const answerEl = document.getElementById('doc-answer-text');
    const attributionsEl = document.getElementById('doc-source-attributions');
    if (answerEl) answerEl.textContent = '';
    if (attributionsEl) attributionsEl.innerHTML = '';
    const confidenceContainer = document.getElementById('confidence-container');
    if (confidenceContainer) confidenceContainer.style.display = 'none';
    showLoadingPhases();
    try {
        let resp;
        if (scope === 'all') {
            resp = await fetch(`${API_BASE}/documents/ask-all`, {
                method: 'POST',
                headers: authHeaders(),
                body: JSON.stringify({ question })
            });
        } else {
            const body = {
                question,
                collection_name: currentDocument.collectionName,
                n_results: 10
            };
            if (scope === 'document' && currentDocument.filename) {
                body.source_file = currentDocument.filename;
            }
            resp = await fetch(`${API_BASE}/documents/ask`, {
                method: 'POST',
                headers: authHeaders(),
                body: JSON.stringify(body)
            });
        }
        const data = await resp.json().catch(() => ({}));
        hideLoadingPhases();
        if (handle401(resp)) return;
        if (!resp.ok) {
            showError(data.detail || data.user_message || 'An error occurred.');
            return;
        }
        const costEl = document.getElementById('doc-cost');
        const confEl = document.getElementById('doc-confidence-badge');
        const sourcesContainer = document.getElementById('doc-sources-container');
        const sourcesList = document.getElementById('doc-sources-list');
        if (answerEl) {
            answerEl.textContent = data.answer || 'No answer returned.';
        }
        if (data.confidence) {
            updateConfidence(data.confidence);
        } else {
            const container = document.getElementById('confidence-container');
            if (container) container.style.display = 'none';
        }
        const sources = Array.isArray(data.sources) ? data.sources : [];
        const sourcesText = Array.isArray(data.sources_text) ? data.sources_text : [];
        renderSourceAttributions(sources);
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
        const sourcesDetails = document.getElementById('doc-sources-details');
        if (sourcesList) {
            if (sources.length === 0 && sourcesText.length === 0) {
                sourcesList.innerHTML = '<div class="history-empty">No source passages returned.</div>';
                if (sourcesDetails) sourcesDetails.removeAttribute('open');
            } else {
                const listHtml = (sourcesText.length ? sourcesText : sources.map(s => (s && s.text) || '')).map((text, idx) => {
                    const src = sources[idx] || {};
                    const fname = (src.source_file || '').toString();
                    const distance = src.distance != null ? Number(src.distance).toFixed(3) : '—';
                    const preview = (text || '').replace(/\s+/g, ' ').trim().slice(0, 200);
                    const label = fname ? `Passage ${idx + 1} · ${escapeHtml(fname)} · distance ${distance}` : `Passage ${idx + 1} · distance ${distance}`;
                    return `
                        <div class="doc-source">
                            <div class="doc-source-label">${label}</div>
                            <blockquote class="doc-source-blockquote">${escapeHtml(preview)}${(text || '').length > 200 ? '…' : ''}</blockquote>
                        </div>
                    `;
                }).join('');
                sourcesList.innerHTML = listHtml;
                if (sourcesDetails) sourcesDetails.removeAttribute('open');
            }
        }
        if (scope !== 'all' && currentDocument.collectionName) {
            loadDocumentHistory(currentDocument.collectionName);
        }
    } catch (err) {
        hideLoadingPhases();
        console.error('Error in askQuestion:', err);
        showError('Connection failed. Is the server running?');
    }
}
const CHUNK_PREVIEW_PAGE_SIZE = 5;

/**
 * Load chunks for a document and display in the card's preview panel.
 * @param {string} collectionName
 * @param {string} filename
 * @param {HTMLElement} cardEl - .document-card
 * @param {number} offset - pagination offset
 * @param {boolean} append - if true, append to list; else replace
 */
async function loadChunkPreview(collectionName, filename, cardEl, offset, append) {
    const panel = cardEl.querySelector('.doc-chunk-preview');
    const listEl = cardEl.querySelector('.doc-chunk-preview-list');
    const moreBtn = cardEl.querySelector('.doc-chunk-preview-more');
    if (!panel || !listEl) return;

    try {
        const url = `${API_BASE}/documents/${encodeURIComponent(collectionName)}/chunks?source_file=${encodeURIComponent(filename)}&limit=${CHUNK_PREVIEW_PAGE_SIZE}&offset=${offset}`;
        const resp = await fetch(url, { headers: authHeaders() });
        const data = await resp.json().catch(() => ({}));
        if (handle401(resp)) return;
        if (!resp.ok) {
            showError(data.detail || data.user_message || 'An error occurred.');
            return;
        }
        const chunks = data.chunks || [];
        const totalChunks = data.total_chunks || 0;
        const nextOffset = offset + chunks.length;

        if (!append) {
            listEl.innerHTML = '';
        }
        chunks.forEach((chunk) => {
            const preview = (chunk.text || '').replace(/\s+/g, ' ').trim().slice(0, 150);
            const wc = chunk.word_count != null ? chunk.word_count : (chunk.text || '').split(/\s+/).length;
            const item = document.createElement('div');
            item.className = 'doc-chunk-preview-item';
            item.innerHTML = `
                <div class="doc-chunk-preview-meta">Chunk ${chunk.index + 1} · ${wc} words · ${escapeHtml(chunk.source_file || '')}</div>
                <div class="doc-chunk-preview-text">${escapeHtml(preview)}${(chunk.text || '').length > 150 ? '…' : ''}</div>
            `;
            listEl.appendChild(item);
        });

        panel.dataset.offset = String(nextOffset);
        panel.style.display = 'block';
        if (moreBtn) {
            moreBtn.style.display = nextOffset < totalChunks ? 'block' : 'none';
        }
    } catch (err) {
        console.error('Error in loadChunkPreview:', err);
        showError('Connection failed. Is the server running?');
    }
}

async function deleteDocument(collectionName) {
    if (!collectionName) return;
    const confirmed = window.confirm('Delete this document and its collection?');
    if (!confirmed) return;
    try {
        const resp = await fetch(`${API_BASE}/documents/${encodeURIComponent(collectionName)}`, {
            method: 'DELETE',
            headers: authHeaders()
        });
        const data = await resp.json().catch(() => ({}));
        if (handle401(resp)) return;
        if (!resp.ok) {
            showError(data.detail || data.user_message || 'An error occurred.');
            return;
        }
        showToast('Document deleted');
        if (currentDocument.collectionName === collectionName) {
            currentDocument = { collectionName: null, filename: null };
            const panel = document.getElementById('doc-qa-panel');
            if (panel) panel.style.display = 'none';
        }
        loadDocuments();
    } catch (err) {
        console.error('Error in deleteDocument:', err);
        showError('Connection failed. Is the server running?');
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
            headers: authHeaders(),
            body: JSON.stringify({ ticker })
        });
        const data = await response.json().catch(() => ({}));
        stopLoading(btn);
        if (handle401(response)) return;
        if (!response.ok) {
            showError(data.detail || data.user_message || 'An error occurred.');
            return;
        }
        displayStockResults(data);
        loadStats();
    } catch (err) {
        stopLoading(btn);
        console.error('Error in analyzeStock:', err);
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
            headers: authHeaders(),
            body: JSON.stringify({ coin_id: coinId })
        });
        const data = await response.json().catch(() => ({}));
        stopLoading(btn);
        if (handle401(response)) return;
        if (!response.ok) {
            showError(data.detail || data.user_message || 'An error occurred.');
            return;
        }
        displayCryptoResults(data);
        loadStats();
    } catch (err) {
        stopLoading(btn);
        console.error('Error in analyzeCrypto:', err);
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
            headers: authHeaders(),
            body: JSON.stringify({ asset1: a1, asset1_type: t1, asset2: a2, asset2_type: t2 })
        });
        const data = await response.json().catch(() => ({}));
        stopLoading(btn);
        if (handle401(response)) return;
        if (!response.ok) {
            showError(data.detail || data.user_message || 'An error occurred.');
            return;
        }
        displayCompareResults(data);
    } catch (err) {
        stopLoading(btn);
        console.error('Error in compareAssets:', err);
        showError('Connection failed. Is the server running?');
    }
}


// --- INITIALIZATION & EVENT LISTENERS ---

document.addEventListener('DOMContentLoaded', async () => {
    const user = JSON.parse(localStorage.getItem('user') || '{}');
    const greeting = document.getElementById('user-greeting');
    if (greeting) {
        greeting.textContent = 'Hello, ' + (user.full_name || user.email?.split('@')[0] || 'User');
    }
    const userBar = document.getElementById('user-bar');
    if (userBar) userBar.style.display = 'flex';

    // Wire logout button
    const logoutBtn = document.getElementById('logout-btn');
    if (logoutBtn) logoutBtn.addEventListener('click', logout);

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
    loadCollections().then(() => loadDocuments());
    const docTabsContainer = document.querySelector('.doc-collection-tabs');
    if (docTabsContainer) {
        docTabsContainer.addEventListener('click', (e) => {
            const tab = e.target.closest('.doc-collection-tab');
            if (tab) {
                selectCollection(tab.getAttribute('data-collection') || '');
                return;
            }
            if (e.target.id === 'doc-collection-add-btn') {
                document.getElementById('doc-create-collection-form').style.display = 'block';
            }
        });
    }
    const createSubmit = document.getElementById('doc-create-collection-submit');
    if (createSubmit) {
        createSubmit.addEventListener('click', () => {
            const name = document.getElementById('doc-new-collection-name').value;
            const desc = document.getElementById('doc-new-collection-desc').value;
            createCollection(name, desc);
        });
    }
    const createCancel = document.getElementById('doc-create-collection-cancel');
    if (createCancel) {
        createCancel.addEventListener('click', () => {
            document.getElementById('doc-create-collection-form').style.display = 'none';
        });
    }
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
            } else if (e.target.classList.contains('doc-preview-chunks-btn')) {
                const panel = card.querySelector('.doc-chunk-preview');
                if (panel && panel.style.display === 'none') {
                    loadChunkPreview(collectionName, filename, card, 0, false);
                } else if (panel) {
                    panel.style.display = 'none';
                }
            } else if (e.target.classList.contains('doc-chunk-preview-more')) {
                const panel = card.querySelector('.doc-chunk-preview');
                const offset = parseInt(panel.dataset.offset || '0', 10);
                loadChunkPreview(collectionName, filename, card, offset, true);
            }
        });
    }
    const docQaPanel = document.getElementById('doc-qa-panel');
    if (docQaPanel) {
        docQaPanel.addEventListener('click', (e) => {
            const link = e.target.closest('.doc-source-attribution-link');
            if (!link) return;
            e.preventDefault();
            selectDocument(link.getAttribute('data-collection'), link.getAttribute('data-filename'));
        });
    }
    const askBtn = document.getElementById('doc-ask-btn');
    if (askBtn) {
        askBtn.addEventListener('click', askQuestion);
    }
    // Agent chat wiring
    const chatSendBtn = document.getElementById('chat-send-btn');
    const chatInput = document.getElementById('chat-input');
    const chatClearBtn = document.getElementById('chat-clear-btn');
    if (chatSendBtn) {
        chatSendBtn.addEventListener('click', sendMessage);
    }
    if (chatInput) {
        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                sendMessage();
            }
        });
    }
    if (chatClearBtn) {
        chatClearBtn.addEventListener('click', clearConversation);
    }
    // Auto-load stats, history and watchlist on page load so data is visible immediately.
    loadStats();
    loadHistory(true);
    loadWatchlist();
});

/**
 * Sends a user question to the Financial Research Agent backend and renders
 * the conversational reply in the chat panel.
 */
async function sendMessage() {
    const inputEl = document.getElementById('chat-input');
    const messagesEl = document.getElementById('chat-messages');
    const statusEl = document.getElementById('agent-status');
    const sendBtn = document.getElementById('chat-send-btn');

    if (!inputEl || !messagesEl || !statusEl || !sendBtn) return;

    const userText = inputEl.value.trim();
    if (!userText) return;

    // Clear input and append user bubble
    inputEl.value = '';
    const userBubble = document.createElement('div');
    userBubble.className = 'chat-message message-user';
    userBubble.textContent = userText;
    messagesEl.appendChild(userBubble);
    messagesEl.scrollTop = messagesEl.scrollHeight;

    // Disable send button and show initial status while streaming.
    statusEl.textContent = 'Agent is researching...';
    sendBtn.disabled = true;

    // Use the streaming endpoint for the agent reply.
    sendMessageStreaming(userText);
}

/**
 * Clears the current chat transcript and resets the in-memory conversation history.
 */
function clearConversation() {
    const messagesEl = document.getElementById('chat-messages');
    const statusEl = document.getElementById('agent-status');
    if (messagesEl) {
        messagesEl.innerHTML = '';
    }
    if (statusEl) {
        statusEl.textContent = '';
    }
    conversationHistory.length = 0;
}

/**
 * Clears all items from the agent activity feed panel.
 */
function clearActivityFeed() {
    const feed = document.getElementById('activity-feed');
    if (feed) {
        feed.innerHTML = '';
    }
    // Reset any stored references to active activity items.
    Object.keys(activeActivityItems).forEach((key) => {
        delete activeActivityItems[key];
    });
}

/**
 * Adds a new activity item row to the agent activity feed and scrolls to bottom.
 * @param {string} toolName
 * @param {'pending'|'complete'} status
 * @param {string} detail
 * @returns {HTMLElement|null}
 */
function addActivityItem(toolName, status, detail = '') {
    const feed = document.getElementById('activity-feed');
    if (!feed) return null;

    const container = document.createElement('div');
    container.className = `activity-item ${status === 'complete' ? 'complete' : 'pending'}`;
    container.dataset.toolName = toolName;
    container.dataset.detail = detail || '';

    const iconSpan = document.createElement('span');
    iconSpan.className = 'activity-item-icon';

    const labelWrap = document.createElement('div');
    labelWrap.className = 'activity-item-main';

    const labelSpan = document.createElement('div');
    labelSpan.className = 'activity-item-label';
    labelSpan.textContent = toolName;

    const detailSpan = document.createElement('div');
    detailSpan.className = 'activity-item-detail';
    detailSpan.textContent = detail || '';

    const timeSpan = document.createElement('span');
    timeSpan.className = 'activity-item-timestamp';
    const now = new Date();
    timeSpan.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    // Choose icon based on tool name / status
    let icon = '⚡';
    if (toolName === 'search_documents') icon = '🔍';
    else if (toolName === 'get_stock_data') icon = '📈';
    else if (toolName === 'get_crypto_data') icon = '💱';
    else if (toolName === 'compare_assets') icon = '⚖️';
    else if (toolName === 'done') icon = '✓';
    iconSpan.textContent = icon;

    labelWrap.appendChild(labelSpan);
    if (detail) {
        labelWrap.appendChild(detailSpan);
    }

    container.appendChild(iconSpan);
    container.appendChild(labelWrap);
    container.appendChild(timeSpan);

    feed.appendChild(container);
    feed.scrollTop = feed.scrollHeight;

    return container;
}

/**
 * Creates and appends an empty agent chat bubble, returning the element so
 * the caller can update its contents as new text arrives.
 * @param {string} text
 * @returns {HTMLElement}
 */
function appendAgentBubble(text) {
    const messagesEl = document.getElementById('chat-messages');
    const bubble = document.createElement('div');
    bubble.className = 'chat-message message-agent';
    bubble.textContent = text || '';
    if (messagesEl) {
        messagesEl.appendChild(bubble);
        scrollChatToBottom();
    }
    return bubble;
}

/**
 * Appends a tools badge under the given agent bubble to summarize which tools
 * were invoked during the agent's reasoning.
 * @param {HTMLElement} bubbleEl
 * @param {Array<any>} toolsUsed
 */
function appendToolsBadge(bubbleEl, toolsUsed) {
    if (!bubbleEl || !Array.isArray(toolsUsed) || toolsUsed.length === 0) return;
    const toolNames = toolsUsed
        .map((t) => (t && (t.tool || t.name)))
        .filter(Boolean);
    if (toolNames.length === 0) return;
    const badge = document.createElement('div');
    badge.className = 'tools-badge';
    badge.textContent = `Tools used: ${toolNames.join(', ')}`;
    bubbleEl.appendChild(document.createElement('br'));
    bubbleEl.appendChild(badge);
}

/**
 * Appends a small cost / token usage line just below a given agent message.
 * @param {HTMLElement} bubbleEl
 * @param {number} totalTokens
 */
function appendMessageMeta(bubbleEl, totalTokens) {
    if (!bubbleEl || typeof totalTokens !== 'number') return;
    const estimatedCost = (totalTokens * 0.000003).toFixed(4); // $3 per 1M tokens
    const meta = document.createElement('div');
    meta.className = 'message-meta';
    meta.textContent = `${totalTokens} tokens · ~$${estimatedCost}`;
    if (bubbleEl.parentNode) {
        bubbleEl.parentNode.insertBefore(meta, bubbleEl.nextSibling);
    }
}

/**
 * Updates the small status line beneath the chat input to indicate what the
 * agent is currently doing (e.g., calling tools, streaming answer, errors).
 * @param {string} text
 */
function updateAgentStatus(text) {
    const statusEl = document.getElementById('agent-status');
    if (statusEl) {
        statusEl.textContent = text || '';
    }
}

/**
 * Scrolls the chat messages container to the bottom so the latest messages
 * and streamed chunks remain in view.
 */
function scrollChatToBottom() {
    const messagesEl = document.getElementById('chat-messages');
    if (messagesEl) {
        messagesEl.scrollTop = messagesEl.scrollHeight;
    }
}

/**
 * Re-enables the chat send button after a streaming session completes or
 * encounters an error.
 */
function enableSendButton() {
    const sendBtn = document.getElementById('chat-send-btn');
    if (sendBtn) {
        sendBtn.disabled = false;
    }
}

/**
 * Streaming version of the Financial Research Agent chat. Uses an EventSource
 * connected to the /agent/stream SSE endpoint and incrementally appends text
 * deltas to a single agent bubble as they arrive.
 * @param {string} userText
 */
function sendMessageStreaming(userText) {
    const encodedMsg = encodeURIComponent(userText);
    // Include serialized conversation history so the backend can maintain context.
    const historyParam = encodeURIComponent(JSON.stringify(conversationHistory || []));
    const es = new EventSource(
        `${API_BASE}/agent/stream?message=${encodedMsg}&conversation_history=${historyParam}`
    );
    let agentBubble = null;
    let fullText = '';

    // New run: clear prior activity entries.
    clearActivityFeed();

    es.onmessage = function (event) {
        let data;
        try {
            data = JSON.parse(event.data);
        } catch {
            return;
        }

        if (data.type === 'tool_call') {
            // Show which tool is being called and log to activity feed
            updateAgentStatus(`Calling ${data.tool}...`);
            const detail = (data.input && (data.input.ticker || data.input.query)) || '';
            const item = addActivityItem(data.tool, 'pending', detail);
            if (item) {
                activeActivityItems[data.tool] = item;
            }
        } else if (data.type === 'tool_result') {
            // Mark corresponding activity item complete when tool result arrives
            const item = activeActivityItems[data.tool];
            if (item && item.classList.contains('pending')) {
                item.classList.remove('pending');
                item.classList.add('complete');
            }
            // Update label/detail to include duration, e.g. "get_stock_data AAPL  →  243ms"
            if (item) {
                const labelEl = item.querySelector('.activity-item-label');
                const detailEl = item.querySelector('.activity-item-detail');
                const baseLabel = item.dataset.toolName || data.tool;
                const baseDetail = item.dataset.detail || (detailEl ? detailEl.textContent : '');
                const durationMs = typeof data.duration_ms === 'number' ? data.duration_ms : null;

                if (labelEl) {
                    labelEl.textContent = baseLabel + (baseDetail ? `  ${baseDetail}` : '');
                }

                if (durationMs !== null) {
                    const durationText = `→  ${durationMs}ms`;
                    if (detailEl) {
                        detailEl.textContent = durationText;
                    } else {
                        const newDetail = document.createElement('div');
                        newDetail.className = 'activity-item-detail';
                        newDetail.textContent = durationText;
                        const main = item.querySelector('.activity-item-main') || item;
                        main.appendChild(newDetail);
                    }
                }
            }
        } else if (data.type === 'text_delta') {
            // Create bubble on first chunk
            if (!agentBubble) {
                agentBubble = appendAgentBubble('');
                updateAgentStatus('');
            }
            fullText += data.text;
            agentBubble.textContent = fullText;
            scrollChatToBottom();
        } else if (data.type === 'done') {
            // Finalize — add tools badge, update history
            appendToolsBadge(agentBubble, data.tools_used || []);
            appendMessageMeta(agentBubble, data.total_tokens || 0);
            if (Array.isArray(data.conversation_history)) {
                conversationHistory.length = 0;
                data.conversation_history.forEach((m) => conversationHistory.push(m));
            }
            addActivityItem('done', 'complete', `${data.total_tokens || 0} tokens`);
            es.close();
            enableSendButton();
        } else if (data.type === 'error') {
            updateAgentStatus('Error: ' + (data.message || 'Unknown error'));
            es.close();
            enableSendButton();
        }
    };

    es.onerror = function () {
        updateAgentStatus('Connection lost. Please try again.');
        es.close();
        enableSendButton();
    };
}