/**
 * CogniSense CX - Storage & App Utilities
 * Manages complaint history in localStorage
 */

const Store = (() => {
  const KEY = 'cognisense_complaints';

  function getAll() {
    try {
      return JSON.parse(localStorage.getItem(KEY)) || [];
    } catch { return []; }
  }

  function save(result) {
    const complaints = getAll();
    result.id = Date.now() + '_' + Math.random().toString(36).substr(2, 5);
    complaints.unshift(result);
    // Keep max 500 entries
    if (complaints.length > 500) complaints.splice(500);
    localStorage.setItem(KEY, JSON.stringify(complaints));
    return result;
  }

  function clear() {
    localStorage.removeItem(KEY);
  }

  function getStats() {
    const all = getAll();
    const sentimentCounts = { Positive: 0, Negative: 0, Neutral: 0 };
    const categoryCounts = {};
    const priorityCounts = { High: 0, Medium: 0, Low: 0 };
    const dailyTrend = {};

    all.forEach(c => {
      // Sentiment
      if (c.sentiment) sentimentCounts[c.sentiment.label] = (sentimentCounts[c.sentiment.label] || 0) + 1;
      // Category
      const cat = c.classification?.category || 'General Inquiry';
      categoryCounts[cat] = (categoryCounts[cat] || 0) + 1;
      // Priority
      if (c.priority) priorityCounts[c.priority.level] = (priorityCounts[c.priority.level] || 0) + 1;
      // Daily trend
      const day = c.timestamp ? c.timestamp.substring(0, 10) : 'unknown';
      dailyTrend[day] = (dailyTrend[day] || 0) + 1;
    });

    return { total: all.length, sentimentCounts, categoryCounts, priorityCounts, dailyTrend };
  }

  function getHighPriority(limit = 10) {
    return getAll()
      .filter(c => c.priority?.level === 'High')
      .slice(0, limit);
  }

  return { getAll, save, clear, getStats, getHighPriority };
})();

// ─── Shared UI Utilities ─────────────────────────────────────────────────────
function formatTimeAgo(isoString) {
  const diff = Date.now() - new Date(isoString).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

function truncate(text, len = 80) {
  if (!text) return '';
  return text.length > len ? text.substring(0, len) + '…' : text;
}

function animateCounter(el, target, duration = 1200) {
  let start = 0;
  const step = Math.ceil(target / (duration / 16));
  const timer = setInterval(() => {
    start += step;
    if (start >= target) { el.textContent = target; clearInterval(timer); }
    else { el.textContent = start; }
  }, 16);
}

function ensurePageBanner() {
  let el = document.getElementById('pageBanner');
  if (!el) {
    el = document.createElement('div');
    el.id = 'pageBanner';
    el.style.cssText = 'display:none;margin-bottom:0.9rem;padding:0.7rem 0.95rem;border-radius:10px;font-size:0.84rem;font-weight:600;border:1px solid transparent;';
    const mount = document.querySelector('.app-main') || document.body;
    mount.insertBefore(el, mount.firstChild);
  }
  return el;
}

function showBanner(message, type = 'info') {
  const colors = {
    info: { bg: 'rgba(79,158,255,0.12)', border: 'rgba(79,158,255,0.35)', text: '#9ecbff' },
    error: { bg: 'rgba(248,113,113,0.14)', border: 'rgba(248,113,113,0.35)', text: '#fca5a5' },
    success: { bg: 'rgba(16,217,138,0.12)', border: 'rgba(16,217,138,0.35)', text: '#86efac' },
  };
  const el = ensurePageBanner();
  const c = colors[type] || colors.info;
  el.style.display = 'block';
  el.style.background = c.bg;
  el.style.borderColor = c.border;
  el.style.color = c.text;
  el.textContent = message;
}

function clearBanner() {
  const el = document.getElementById('pageBanner');
  if (el) el.style.display = 'none';
}

function withLoading(buttonEl, loadingText, action) {
  const original = buttonEl ? buttonEl.textContent : '';
  if (buttonEl) {
    buttonEl.disabled = true;
    buttonEl.textContent = loadingText;
  }
  try {
    return action();
  } finally {
    if (buttonEl) {
      buttonEl.disabled = false;
      buttonEl.textContent = original;
    }
  }
}

// ─── Theme Toggle ────────────────────────────────────────────────────────────
function initTheme() {
  const saved = localStorage.getItem('cx_theme') || 'dark';
  document.documentElement.setAttribute('data-theme', saved);
}

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme');
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('cx_theme', next);

  const btn = document.getElementById('themeToggle');
  if (btn) btn.textContent = next === 'dark' ? '☀️' : '🌙';
}

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  const btn = document.getElementById('themeToggle');
  if (btn) {
    const current = document.documentElement.getAttribute('data-theme');
    btn.textContent = current === 'dark' ? '☀️' : '🌙';
    btn.addEventListener('click', toggleTheme);
  }
});
