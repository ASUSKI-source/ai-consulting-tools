/**
 * AI Market Analyzer - Core Logic
 * Handles API communication, UI state, and data visualization.
 */

const API_BASE = window.location.origin;

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
    document.getElementById('pe-ratio').textContent = data.pe_ratio || 'N/A';
    document.getElementById('year-range').textContent =
        `${formatPrice(data.week_low_52)} - ${formatPrice(data.week_high_52)}`;

    // Technicals
    const rsiEl = document.getElementById('rsi-val');
    rsiEl.textContent = data.rsi.toFixed(2);
    rsiEl.className = data.rsi > 70 ? 'negative' : data.rsi < 30 ? 'positive' : '';

    document.getElementById('sma5-val').textContent = formatPrice(data.sma_5);
    document.getElementById('sma20-val').textContent = formatPrice(data.sma_20);

    // AI Analysis & Cost
    document.getElementById('ai-response-text').innerText = data.analysis || 'No analysis returned.';
    document.getElementById('analysis-cost').textContent = data.estimated_cost.toFixed(6);

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

    fadeInResults();
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
        if (response.ok) displayStockResults(data);
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
        if (response.ok) displayCryptoResults(data);
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
});