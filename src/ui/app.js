const API = 'https://book-recommender-api.azurewebsites.net';

// ── DOM REFS ──────────────────────────────────────────────
const $ = id => document.getElementById(id);

const pip         = $('pip');
const statusText  = $('statusText');
const searchInput = $('searchInput');
const clearBtn    = $('clearBtn');
const dropdown    = $('dropdown');
const mainEl      = $('main');
const errBanner   = $('errBanner');
const seedStrip   = $('seedStrip');
const controlsBar = $('controlsBar');
const tabsBar     = $('tabsBar');
const panelsWrap  = $('panelsWrap');
const topKRange   = $('topKRange');
const topKInput   = $('topKInput');
const refreshBtn  = $('refreshBtn');

// Modal
const modalBackdrop  = $('modalBackdrop');
const modal          = $('modal');
const modalRank      = $('modalRank');
const modalTitle     = $('modalTitle');
const modalAuthor    = $('modalAuthor');
const modalStats     = $('modalStats');
const modalDescSec   = $('modalDescSection');
const modalDesc      = $('modalDesc');
const modalTagsSec   = $('modalTagsSection');
const modalTags      = $('modalTags');
const modalScoreBar  = $('modalScoreBar');
const modalScoreVal  = $('modalScoreVal');

// ── STATE ────────────────────────────────────────────────
let currentIsbn  = null;
let debounceTimer = null;

// ── HEALTH CHECK ─────────────────────────────────────────
(async () => {
  try {
    const r = await fetch(`${API}/health`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json();
    pip.className       = 'status-pip ok';
    statusText.textContent = `api online · ${d.strategy_count} strategies`;
  } catch (e) {
    pip.className       = 'status-pip err';
    statusText.textContent = 'api unreachable';
    showError(
      `Cannot reach the API (${e.message}). ` +
      `Serve this folder with a local server: python -m http.server 8080`
    );
  }
})();

// ── SEARCH ───────────────────────────────────────────────
searchInput.addEventListener('input', () => {
  const q = searchInput.value.trim();
  clearBtn.classList.toggle('visible', q.length > 0);
  clearTimeout(debounceTimer);
  if (q.length < 2) { closeDropdown(); return; }
  debounceTimer = setTimeout(() => runSearch(q), 300);
});

searchInput.addEventListener('keydown', e => {
  if (e.key === 'Escape') { closeDropdown(); searchInput.blur(); }
});

clearBtn.addEventListener('click', () => {
  searchInput.value = '';
  clearBtn.classList.remove('visible');
  closeDropdown();
  searchInput.focus();
});

document.addEventListener('click', e => {
  if (!e.target.closest('.search-container')) closeDropdown();
});

function closeDropdown() {
  dropdown.className = 'search-dropdown';
  dropdown.innerHTML = '';
}

async function runSearch(q) {
  dropdown.className = 'search-dropdown open';
  dropdown.innerHTML = '<div class="dropdown-msg">Searching…</div>';
  try {
    const r = await fetch(`${API}/books/search?q=${encodeURIComponent(q)}&max_results=8`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    renderDropdown(await r.json());
  } catch (e) {
    dropdown.innerHTML = `<div class="dropdown-msg" style="color:var(--red)">Error: ${e.message}</div>`;
  }
}

function renderDropdown(books) {
  if (!books.length) {
    dropdown.innerHTML = '<div class="dropdown-msg">No results found.</div>';
    return;
  }
  dropdown.innerHTML = books.map(b => `
    <div class="dropdown-item"
      data-isbn="${attr(b.isbn)}"
      data-title="${attr(b.title)}"
      data-author="${attr(b.author)}"
      data-count="${b.rating_count}"
      data-rating="${b.bayesian_rating.toFixed(2)}">
      <div>
        <div class="di-title">${tc(b.title)}</div>
        <div class="di-author">${tc(b.author)}</div>
      </div>
      <div class="di-meta">★ ${b.bayesian_rating.toFixed(1)}<br>${fmt(b.rating_count)}</div>
    </div>
  `).join('');
  dropdown.querySelectorAll('.dropdown-item').forEach(el =>
    el.addEventListener('click', () => pickBook(el.dataset))
  );
}

// ── PICK BOOK ────────────────────────────────────────────
function pickBook(d) {
  searchInput.value = tc(d.title);
  clearBtn.classList.add('visible');
  closeDropdown();
  errBanner.classList.remove('visible');

  $('seedTitle').textContent  = tc(d.title);
  $('seedAuthor').textContent = tc(d.author);
  $('seedStats').innerHTML    = `<b>★ ${parseFloat(d.rating).toFixed(2)}</b><br>${fmt(d.count)} ratings`;

  currentIsbn = d.isbn;
  seedStrip.classList.add('visible');
  controlsBar.classList.add('visible');
  mainEl.classList.add('visible');

  loadRecs(d.isbn);
}

// ── TOP-K CONTROLS ───────────────────────────────────────
topKRange.addEventListener('input', () => {
  topKInput.value = topKRange.value;
});

topKInput.addEventListener('input', () => {
  let v = parseInt(topKInput.value, 10);
  if (isNaN(v)) return;
  v = Math.min(50, Math.max(1, v));
  topKRange.value = v;
});

refreshBtn.addEventListener('click', () => {
  if (!currentIsbn) return;
  loadRecs(currentIsbn);
});

// ── LOAD RECOMMENDATIONS ─────────────────────────────────
async function loadRecs(isbn) {
  const topK = Math.min(50, Math.max(1, parseInt(topKInput.value, 10) || 10));
  topKInput.value = topK;
  topKRange.value = topK;

  tabsBar.innerHTML    = '';
  panelsWrap.innerHTML = `
    <div class="load-state">
      <div class="spinner"></div>
      <span>Loading recommendations…</span>
    </div>`;

  refreshBtn.disabled = true;
  setTimeout(() => mainEl.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);

  try {
    const r = await fetch(`${API}/recommend`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ isbn, top_k: topK }),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${r.status}`);
    }
    renderStrategies((await r.json()).strategies);
  } catch (e) {
    panelsWrap.innerHTML =
      `<div class="empty-state" style="color:var(--red)">Failed: ${e.message}</div>`;
  } finally {
    refreshBtn.disabled = false;
  }
}

// ── RENDER STRATEGIES ────────────────────────────────────
function renderStrategies(strategies) {
  tabsBar.innerHTML = panelsWrap.innerHTML = '';

  strategies.forEach((s, i) => {
    // Tab button
    const btn = document.createElement('button');
    btn.className   = 'tab-btn' + (i === 0 ? ' active' : '');
    btn.textContent = s.strategy_label;
    btn.dataset.idx = i;
    btn.addEventListener('click', () => switchTab(i));
    tabsBar.appendChild(btn);

    // Panel
    const panel = document.createElement('div');
    panel.className = 'panel' + (i === 0 ? ' active' : '');
    panel.dataset.idx = i;
    panel.innerHTML = `
      <div class="panel-desc">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
          <circle cx="12" cy="12" r="10"/>
          <line x1="12" y1="8" x2="12" y2="12"/>
          <line x1="12" y1="16" x2="12.01" y2="16"/>
        </svg>
        ${esc(s.strategy_description)}
      </div>
      ${buildGrid(s.recommendations)}`;
    panelsWrap.appendChild(panel);
  });
}

function switchTab(idx) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', +b.dataset.idx === idx));
  document.querySelectorAll('.panel').forEach(p => p.classList.toggle('active', +p.dataset.idx === idx));
}

// ── BUILD BOOK GRID ──────────────────────────────────────
function buildGrid(recs) {
  if (!recs.length)
    return '<div class="empty-state">No recommendations available for this strategy.</div>';

  const maxScore = Math.max(...recs.map(r => r.score), 1e-9);

  return `<div class="book-grid">${recs.map((r, i) => {
    const pct      = Math.round((r.score / maxScore) * 100);
    const fallback = `${r.title} by ${r.author}`.toLowerCase();
    const hasDesc  = r.description && r.description.toLowerCase() !== fallback && r.description.length > 35;
    const tags     = (r.subjects || '').split(',').map(s => s.trim()).filter(Boolean).slice(0, 3)
                       .map(t => `<span class="tag">${esc(t)}</span>`).join('');

    // Encode book data into data attributes for the modal
    const bookData = encodeURIComponent(JSON.stringify({
      rank:          i + 1,
      isbn:          r.isbn,
      title:         r.title,
      author:        r.author,
      score:         r.score,
      scorePct:      pct,
      description:   r.description || '',
      subjects:      r.subjects    || '',
      ratingCount:   r.rating_count,
      bayesianRating:r.bayesian_rating,
    }));

    return `
      <div class="book-card" data-book="${attr(bookData)}">
        <div class="book-rank">${String(i + 1).padStart(2, '0')}</div>
        <div class="book-title">${esc(tc(r.title))}</div>
        <div class="book-author">${esc(tc(r.author))}</div>
        ${hasDesc ? `<div class="book-desc-preview">${esc(r.description)}</div>` : ''}
        ${tags    ? `<div class="book-tags">${tags}</div>` : ''}
        <div class="book-footer">
          <div class="score-track"><div class="score-fill" style="width:${pct}%"></div></div>
          <div class="book-meta">★ ${r.bayesian_rating.toFixed(1)} · ${fmt(r.rating_count)}</div>
        </div>
        <svg class="book-expand-hint" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
          <polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/>
          <line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/>
        </svg>
      </div>`;
  }).join('')}</div>`;
}

// ── MODAL ─────────────────────────────────────────────────
// Event delegation — handles cards added dynamically
document.addEventListener('click', e => {
  const card = e.target.closest('.book-card');
  if (card) openModal(card.dataset.book);
});

function openModal(encoded) {
  let book;
  try { book = JSON.parse(decodeURIComponent(encoded)); }
  catch { return; }

  const fallback = `${book.title} by ${book.author}`.toLowerCase();
  const hasDesc  = book.description &&
                   book.description.toLowerCase() !== fallback &&
                   book.description.length > 35;

  modalRank.textContent  = `#${book.rank}`;
  modalTitle.textContent = tc(book.title);
  modalAuthor.textContent= tc(book.author);

  modalStats.innerHTML = `
    <div class="modal-stat">
      <span class="modal-stat-label">Bayesian Avg</span>
      <span class="modal-stat-val">★ ${book.bayesianRating.toFixed(2)}</span>
    </div>
    <div class="modal-stat">
      <span class="modal-stat-label">Ratings</span>
      <span class="modal-stat-val">${fmt(book.ratingCount)}</span>
    </div>
    <div class="modal-stat">
      <span class="modal-stat-label">ISBN</span>
      <span class="modal-stat-val">${esc(book.isbn)}</span>
    </div>`;

  if (hasDesc) {
    modalDesc.textContent = book.description;
    modalDescSec.style.display = 'block';
  } else {
    modalDescSec.style.display = 'none';
  }

  const tags = (book.subjects || '').split(',').map(s => s.trim()).filter(Boolean)
    .map(t => `<span class="tag">${esc(t)}</span>`).join('');
  if (tags) {
    modalTags.innerHTML = tags;
    modalTagsSec.style.display = 'block';
  } else {
    modalTagsSec.style.display = 'none';
  }

  modalScoreBar.style.width = `${book.scorePct}%`;
  modalScoreVal.textContent  = `${book.scorePct}%`;

  modalBackdrop.classList.add('open');
  document.body.style.overflow = 'hidden';
}

function closeModal() {
  modalBackdrop.classList.remove('open');
  document.body.style.overflow = '';
}

$('modalClose').addEventListener('click', closeModal);

modalBackdrop.addEventListener('click', e => {
  if (e.target === modalBackdrop) closeModal();
});

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeModal();
});

// ── HELPERS ───────────────────────────────────────────────
function showError(msg) {
  errBanner.textContent = msg;
  errBanner.classList.add('visible');
  mainEl.classList.add('visible');
}

function attr(s) { return String(s || '').replace(/"/g, '&quot;'); }
function esc(s)  { return String(s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function fmt(n)  { return Number(n).toLocaleString(); }

const SMALL = new Set(['a','an','the','and','but','or','for','nor','on','at','to','by','up','of','in','is']);
function tc(s) {
  return String(s || '').toLowerCase().replace(/\b\w+/g, (w, i) =>
    (i === 0 || !SMALL.has(w)) ? w[0].toUpperCase() + w.slice(1) : w);
}
