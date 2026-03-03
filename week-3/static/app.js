/**
 * AI Market Analyzer - Core Logic
 * Handles API communication, UI state, and data visualization.
 */

const API_BASE = window.location.origin;

// --- UI UTILITIES ---

/**
 * Toggles the visibility of the loading spinner.
 * @param {boolean} visible 
 */
function showLoading(visible) {
    const loader = document.getElementById('loading');
    loader.style.display = visible ? 'block' : 'none';
    if (visible) {
        document.getElementById('results').style.display = 'none';
        hideError();
    }
}

/**
 * Displays an error message with an auto-hide timer.
 * @param {string} message 
 */
function showError(message) {
    const errorDiv = document.getElementById('error-message');
    errorDiv.innerHTML = `<span>${message}</span><span class="close-btn" onclick="hideError()">×</span>`;
    errorDiv.style.display = 'flex';
    
    // Auto-hide after 8 seconds
    setTimeout(() => {
        hideError();
    }, 8000);
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
    showLoading(false);
    document.getElementById('results').style.display = 'block';
    
    // Header & Price
    document.getElementById('display-name').textContent = data.company_name;
    document.getElementById('display-symbol').textContent = data.ticker;
    document.getElementById('current-price').textContent = formatPrice(data.current_price);
    
    // Toggle Section Visibility
    document.getElementById('stock-only-stats').style.display = 'grid';
    document.getElementById('crypto-only-stats').style.display = 'none';

    // Stock Stats
    document.getElementById('pe-ratio').textContent = data.pe_ratio || 'N/A';
    document.getElementById('year-range').textContent = `${formatPrice(data.week_low_52)} - ${formatPrice(data.week_high_52)}`;

    // Technicals
    const rsiEl = document.getElementById('rsi-val');
    rsiEl.textContent = data.rsi.toFixed(2);
    rsiEl.className = data.rsi > 70 ? 'negative' : data.rsi < 30 ? 'positive' : '';

    document.getElementById('sma5-val').textContent = formatPrice(data.sma_5);
    document.getElementById('sma20-val').textContent = formatPrice(data.sma_20);

    // AI Analysis & Cost
    document.getElementById('ai-response-text').innerText = data.analysis || 'No analysis returned.';
    document.getElementById('analysis-cost').textContent = data.estimated_cost.toFixed(6);
}

/**
 * Populates the UI with Crypto-specific data, including advanced run metrics.
 */
function displayCryptoResults(data) {
    showLoading(false);
    document.getElementById('results').style.display = 'block';

    document.getElementById('display-name').textContent = data.name;
    document.getElementById('display-symbol').textContent = data.symbol.toUpperCase();
    document.getElementById('current-price').textContent = formatPrice(data.current_price_usd);

    document.getElementById('stock-only-stats').style.display = 'none';
    document.getElementById('crypto-only-stats').style.display = 'grid';

    // Price Changes
    document.getElementById('change-24h').innerHTML = formatPct(data.price_change_24h_pct);
    document.getElementById('change-7d').innerHTML = formatPct(data.price_change_7d_pct);
    document.getElementById('change-30d').innerHTML = formatPct(data.price_change_30d_pct);

    // Advanced Metrics (Velocity/Funding/Depth)
    //document.getElementById('slap-freq-val').textContent = data.slap_frequency || 'Normal';
    //document.getElementById('funding-rate-val').textContent = `${(data.funding_rate * 100).toFixed(4)}%`;
    //document.getElementById('str-ratio-val').textContent = data.short_reserve_ratio || '0.00';
    //document.getElementById('liq-gap-dist').textContent = data.liquidity_gap || 'Calculating...';

    document.getElementById('ai-response-text').innerHTML = 
    typeof marked !== 'undefined' ? marked.parse(data.analysis || '') : (data.analysis || '');
    document.getElementById('analysis-cost').textContent = data.estimated_cost.toFixed(6);
}

/**
 * Handles the display for the "Compare Two Assets" function.
 */
function displayCompareResults(data) {
    showLoading(false);
    document.getElementById('results').style.display = 'block';
    
    document.getElementById('display-name').textContent = `${data.asset1_name} vs ${data.asset2_name}`;
    document.getElementById('display-symbol').textContent = "COMPARE";
    
    // Clear leftover data from previous single-stock analysis
    document.getElementById('current-price').textContent = '';
    document.getElementById('pe-ratio').textContent = '';
    document.getElementById('year-range').textContent = '';
    document.getElementById('rsi-val').textContent = '';
    document.getElementById('sma5-val').textContent = '';
    document.getElementById('sma20-val').textContent = '';
    
    // Hide standard grids for comparison view
    document.getElementById('stock-only-stats').style.display = 'none';
    document.getElementById('crypto-only-stats').style.display = 'none';
    document.getElementById('price-container') && 
        (document.getElementById('price-container').style.display = 'none');
    
    // Use data.analysis — not data.comparison_analysis
    document.getElementById('ai-response-text').innerHTML = 
    typeof marked !== 'undefined' ? marked.parse(data.analysis || 'No comparison returned.') : (data.analysis || 'No comparison returned.');
    document.getElementById('analysis-cost').textContent = `This analysis cost: $${data.estimated_cost.toFixed(6)}`;
}


// --- API ACTIONS ---

async function analyzeStock() {
    const ticker = document.getElementById('stock-ticker').value.trim().toUpperCase();
    if (!ticker) return showError("Please enter a stock ticker.");

    showLoading(true);
    try {
        const response = await fetch(`${API_BASE}/analyze/stock`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker })
        });
        const data = await response.json();
        if (response.ok) displayStockResults(data);
        else showError(data.detail || "Error analyzing stock.");
    } catch (err) {
        showError("Connection failed. Ensure the bridge is active.");
    }
}

async function analyzeCrypto() {
    const coinId = document.getElementById('crypto-id').value.trim().toLowerCase();
    if (!coinId) return showError("Please enter a CoinGecko ID.");

    showLoading(true);
    try {
        const response = await fetch(`${API_BASE}/analyze/crypto`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ coin_id: coinId })
        });
        const data = await response.json();
        if (response.ok) displayCryptoResults(data);
        else showError(data.detail || "Error analyzing crypto.");
    } catch (err) {
        showError("Connection failed. Hard refresh the bridge.");
    }
}

async function compareAssets() {
    const a1 = document.getElementById('asset1-input').value.trim();
    const t1 = document.getElementById('asset1-type').value;
    const a2 = document.getElementById('asset2-input').value.trim();
    const t2 = document.getElementById('asset2-type').value;

    if (!a1 || !a2) return showError("Both assets are required for comparison.");

    showLoading(true);
    try {
        const response = await fetch(`${API_BASE}/analyze/compare`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ asset1: a1, asset1_type: t1, asset2: a2, asset2_type: t2 })
        });
        const data = await response.json();
        console.log("Compare response status:", response.status);
        console.log("Compare response data:", JSON.stringify(data));
        if (response.ok) displayCompareResults(data);
        else showError(data.detail || "Comparison failed.");
    } catch (err) {
        console.error("Compare error:", err);
        showError("Error: " + err.message);
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