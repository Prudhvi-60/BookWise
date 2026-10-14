const state = {
  featured: [],
  categories: [],
};

const elements = {
  searchInput: document.getElementById('searchInput'),
  searchButton: document.getElementById('searchButton'),
  featuredBooks: document.getElementById('featuredBooks'),
  recommendationList: document.getElementById('recommendationList'),
  categoryList: document.getElementById('categoryList'),
  chatInput: document.getElementById('chatInput'),
  chatButton: document.getElementById('chatButton'),
  chatMessages: document.getElementById('chatMessages'),
  bookCount: document.getElementById('bookCount'),
  categoryCount: document.getElementById('categoryCount'),
  avgRating: document.getElementById('avgRating'),
  naturalLanguageQuery: document.getElementById('naturalLanguageQuery'),
  authorInput: document.getElementById('authorInput'),
  excludeAuthorsInput: document.getElementById('excludeAuthorsInput'),
  yearMin: document.getElementById('yearMin'),
  yearMax: document.getElementById('yearMax'),
  minRating: document.getElementById('minRating'),
  minReviews: document.getElementById('minReviews'),
  pageMax: document.getElementById('pageMax'),
  resultCount: document.getElementById('resultCount'),
  recommendMode: document.getElementById('recommendMode'),
  strictToggle: document.getElementById('strictToggle'),
  genreFilterList: document.getElementById('genreFilterList'),
  activeFilters: document.getElementById('activeFilters'),
  resetFilters: document.getElementById('resetFilters'),
  generateRecommendations: document.getElementById('generateRecommendations'),
};

const DEFAULT_GENRES = [
  'Fantasy', 'Science Fiction', 'Mystery', 'Thriller', 'Romance',
  'Historical Fiction', 'Horror', 'Biography', 'Self-Help',
  'Philosophy', 'Psychology', 'Business', 'Technology', 'Adventure', 'Young Adult'
];

function escapeHTML(value) {
  return String(value).replace(/[&<>"']/g, character => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  })[character]);
}

function safeImageUrl(value) {
  try {
    const url = new URL(value);
    return ['http:', 'https:'].includes(url.protocol) ? url.href : '';
  } catch {
    return '';
  }
}

function createBookCard(book) {
  const title = book.title || 'Untitled';
  const authors = book.authors || 'Unknown author';
  const categories = book.categories ? book.categories.split(',').slice(0, 2).map(item => item.trim()).filter(Boolean) : [];
  const imageUrl = safeImageUrl(book.thumbnail || '');
  const year = book.published_year ? escapeHTML(book.published_year) : 'Unknown year';
  const rating = book.average_rating != null ? Number(book.average_rating).toFixed(1) : 'N/A';
  const ratingText = rating === 'N/A' ? 'Rating unavailable' : `★ ${escapeHTML(rating)}`;
  const matchReasons = Array.isArray(book.match_reasons) ? book.match_reasons.slice(0, 2).map(reason => `<li>${escapeHTML(reason)}</li>`).join('') : '';
  const score = typeof book.match_score === 'number' ? `Match ${Number(book.match_score).toFixed(2)}` : 'Catalogue pick';

  return `
    <article class="book-card">
      <div class="book-cover">
        ${imageUrl ? `<img class="book-image" src="${escapeHTML(imageUrl)}" alt="Cover of ${escapeHTML(title)}" loading="lazy" />` : '<span>Cover unavailable</span>'}
      </div>
      <div class="book-card-body">
        <div class="book-card-header">
          <h4>${escapeHTML(title)}</h4>
          <span class="score-pill">${score}</span>
        </div>
        <p>${escapeHTML(authors)}</p>
        <div class="meta-row">
          ${categories.map(item => `<span class="meta-badge">${escapeHTML(item)}</span>`).join('') || '<span class="meta-badge">General</span>'}
        </div>
        <div class="rating-line">
          <span>${ratingText}</span>
          <span>${escapeHTML(year)}</span>
        </div>
        ${matchReasons ? `<ul class="reason-list">${matchReasons}</ul>` : ''}
        <div class="card-actions">
          <button class="secondary-btn minimal" type="button">View details</button>
        </div>
      </div>
    </article>
  `;
}

document.addEventListener('error', event => {
  const image = event.target;
  if (!(image instanceof HTMLImageElement) || !image.classList.contains('book-image')) return;
  const cover = image.parentElement;
  image.remove();
  cover.textContent = 'Cover unavailable';
}, true);

function renderEmpty(target, message) {
  target.innerHTML = `<div class="empty-state">${message}</div>`;
}

function getSelectedGenres() {
  return Array.from(document.querySelectorAll('.genre-option input:checked')).map(item => item.value);
}

function updateActiveFiltersSummary() {
  const filters = [];
  const naturalLanguage = elements.naturalLanguageQuery.value.trim();
  if (naturalLanguage) filters.push(`Text: ${naturalLanguage.slice(0, 40)}`);
  const selectedGenres = getSelectedGenres();
  if (selectedGenres.length) filters.push(`Genres: ${selectedGenres.join(', ')}`);
  const authors = elements.authorInput.value.trim();
  if (authors) filters.push(`Authors: ${authors}`);
  const excludeAuthors = elements.excludeAuthorsInput.value.trim();
  if (excludeAuthors) filters.push(`Avoid: ${excludeAuthors}`);
  const yearMin = elements.yearMin.value.trim();
  const yearMax = elements.yearMax.value.trim();
  if (yearMin || yearMax) filters.push(`Years: ${yearMin || 'any'}-${yearMax || 'any'}`);
  const minRating = elements.minRating.value.trim();
  if (minRating) filters.push(`Min rating: ${minRating}`);
  const minReviews = elements.minReviews.value.trim();
  if (minReviews) filters.push(`Reviews: ${minReviews}+`);
  const pageMax = elements.pageMax.value.trim();
  if (pageMax) filters.push(`Max pages: ${pageMax}`);
  elements.activeFilters.textContent = filters.length ? `Active filters: ${filters.join(' • ')}` : 'No active filters yet.';
}

async function loadStats() {
  const response = await fetch('/api/stats');
  const data = await response.json();
  elements.bookCount.textContent = data.rows ?? 0;
  elements.categoryCount.textContent = data.category_count ?? 0;
  elements.avgRating.textContent = data.average_rating == null ? 'N/A' : Number(data.average_rating).toFixed(1);
}

async function loadFeaturedBooks() {
  const response = await fetch('/api/search?q=fiction');
  const data = await response.json();
  state.featured = data.books.slice(0, 4);
  elements.featuredBooks.innerHTML = state.featured.map(createBookCard).join('');
}

async function loadCategories() {
  const response = await fetch('/api/categories');
  const data = await response.json();
  state.categories = data.categories.slice(0, 20);
  const genreOptions = state.categories.length ? state.categories : DEFAULT_GENRES;
  elements.categoryList.innerHTML = genreOptions.map(category => `<button class="category-chip" data-category="${escapeHTML(category)}">${escapeHTML(category)}</button>`).join('');
  elements.genreFilterList.innerHTML = genreOptions.map(genre => `
    <label class="genre-option">
      <input type="checkbox" value="${escapeHTML(genre)}" />
      <span>${escapeHTML(genre)}</span>
    </label>
  `).join('');

  document.querySelectorAll('.category-chip').forEach(button => {
    button.addEventListener('click', () => {
      const category = button.dataset.category;
      fetchRecommendations(category, true);
    });
  });

  document.querySelectorAll('.genre-option input').forEach(input => {
    input.addEventListener('change', updateActiveFiltersSummary);
  });
}

async function fetchRecommendations(query, byGenre = false) {
  const endpoint = byGenre ? `/api/recommendations?genre=${encodeURIComponent(query)}&limit=6` : `/api/recommendations?q=${encodeURIComponent(query)}&limit=6`;
  const response = await fetch(endpoint);
  const data = await response.json();
  if (!data.books || !data.books.length) {
    renderEmpty(elements.recommendationList, 'No recommendations found for that query.');
    return;
  }
  elements.recommendationList.innerHTML = data.books.map(createBookCard).join('');
}

async function generateAdvancedRecommendations() {
  const params = new URLSearchParams();
  const text = elements.naturalLanguageQuery.value.trim();
  if (text) params.set('natural_language', text);
  const selectedGenres = getSelectedGenres();
  if (selectedGenres.length) params.set('genres', selectedGenres.join(','));
  const authorValue = elements.authorInput.value.trim();
  if (authorValue) params.set('authors', authorValue);
  const excludeAuthors = elements.excludeAuthorsInput.value.trim();
  if (excludeAuthors) params.set('exclude_authors', excludeAuthors);
  const yearMin = elements.yearMin.value.trim();
  if (yearMin) params.set('year_min', yearMin);
  const yearMax = elements.yearMax.value.trim();
  if (yearMax) params.set('year_max', yearMax);
  const minRating = elements.minRating.value.trim();
  if (minRating) params.set('min_rating', minRating);
  const minReviews = elements.minReviews.value.trim();
  if (minReviews) params.set('min_reviews', minReviews);
  const pageMax = elements.pageMax.value.trim();
  if (pageMax) params.set('page_max', pageMax);
  const resultCount = elements.resultCount.value;
  params.set('limit', resultCount);
  params.set('strict', String(elements.strictToggle.checked));
  params.set('mode', elements.recommendMode.value);

  const response = await fetch('/api/recommendations?' + params.toString());
  const data = await response.json();
  if (!data.books || !data.books.length) {
    renderEmpty(elements.recommendationList, 'No books matched those filters. Try widening your preferences or resetting the form.');
    return;
  }
  elements.recommendationList.innerHTML = data.books.map(createBookCard).join('');
}

async function searchBooks(query) {
  const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
  const data = await response.json();
  if (!data.books || !data.books.length) {
    renderEmpty(elements.featuredBooks, 'No books found for that search.');
    return;
  }
  elements.featuredBooks.innerHTML = data.books.slice(0, 4).map(createBookCard).join('');
  elements.recommendationList.innerHTML = data.books.slice(0, 6).map(createBookCard).join('');
}

function resetFilters() {
  elements.naturalLanguageQuery.value = '';
  elements.authorInput.value = '';
  elements.excludeAuthorsInput.value = '';
  elements.yearMin.value = '';
  elements.yearMax.value = '';
  elements.minRating.value = '';
  elements.minReviews.value = '';
  elements.pageMax.value = '';
  elements.resultCount.value = '10';
  elements.recommendMode.value = 'best_match';
  elements.strictToggle.checked = true;
  document.querySelectorAll('.genre-option input').forEach(input => {
    input.checked = false;
  });
  updateActiveFiltersSummary();
}

function addChatMessage(text, role = 'bot') {
  const msg = document.createElement('div');
  msg.className = `message ${role}`;
  msg.textContent = text;
  elements.chatMessages.appendChild(msg);
  elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

async function sendChatMessage() {
  const value = elements.chatInput.value.trim();
  if (!value) return;
  addChatMessage(value, 'user');
  elements.chatInput.value = '';

  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: value })
  });

  const data = await response.json();
  const reply = data.response || 'I could not understand that request.';
  addChatMessage(reply, 'bot');

  if (data.books && data.books.length) {
    elements.recommendationList.innerHTML = data.books.map(createBookCard).join('');
  }
}

elements.searchButton.addEventListener('click', () => {
  const value = elements.searchInput.value.trim();
  if (!value) return;
  searchBooks(value);
});

elements.searchInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') {
    elements.searchButton.click();
  }
});

elements.chatButton.addEventListener('click', sendChatMessage);

elements.chatInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') {
    sendChatMessage();
  }
});

document.getElementById('featuredRefresh').addEventListener('click', () => {
  loadFeaturedBooks();
});

elements.generateRecommendations.addEventListener('click', generateAdvancedRecommendations);
elements.resetFilters.addEventListener('click', resetFilters);

elements.naturalLanguageQuery.addEventListener('input', updateActiveFiltersSummary);
elements.authorInput.addEventListener('input', updateActiveFiltersSummary);
elements.excludeAuthorsInput.addEventListener('input', updateActiveFiltersSummary);
elements.yearMin.addEventListener('input', updateActiveFiltersSummary);
elements.yearMax.addEventListener('input', updateActiveFiltersSummary);
elements.minRating.addEventListener('input', updateActiveFiltersSummary);
elements.minReviews.addEventListener('input', updateActiveFiltersSummary);
elements.pageMax.addEventListener('input', updateActiveFiltersSummary);

async function init() {
  await loadStats();
  await loadFeaturedBooks();
  await loadCategories();
  updateActiveFiltersSummary();
  fetchRecommendations('fiction');
}

init();
