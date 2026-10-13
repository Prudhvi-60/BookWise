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
};

const fallbackImage = 'https://placehold.co/400x600/efe9ff/3d2a84?text=No+Cover';

function createBookCard(book) {
  const title = book.title || 'Untitled';
  const authors = book.authors || 'Unknown author';
  const categories = book.categories ? book.categories.split(',').slice(0, 2).map(item => item.trim()).filter(Boolean) : [];
  const imageUrl = book.thumbnail && /^https?:\/\//.test(book.thumbnail) ? book.thumbnail : fallbackImage;
  const year = book.published_year ? ` · ${book.published_year}` : '';
  const rating = book.average_rating ? Number(book.average_rating).toFixed(1) : 'N/A';

  return `
    <article class="book-card">
      <div class="book-cover">
        <img src="${imageUrl}" alt="${title}" loading="lazy" onerror="this.onerror=null;this.src='${fallbackImage}';" />
      </div>
      <div class="book-card-body">
        <h4>${title}</h4>
        <p>${authors}</p>
        <div class="meta-row">
          ${categories.map(item => `<span class="meta-badge">${item}</span>`).join('') || '<span class="meta-badge">General</span>'}
        </div>
        <div class="rating-line">
          <span>★ ${rating}</span>
          <span>${year}</span>
        </div>
      </div>
    </article>
  `;
}

function renderEmpty(target, message) {
  target.innerHTML = `<div class="empty-state">${message}</div>`;
}

async function loadStats() {
  const response = await fetch('/api/stats');
  const data = await response.json();
  elements.bookCount.textContent = data.rows ?? 0;
  elements.categoryCount.textContent = data.missing_pct ? Object.keys(data.missing_pct).length : 0;
  const ratings = await fetch('/api/search?q=fiction');
  const ratingData = await ratings.json();
  const values = ratingData.books.map(book => Number(book.average_rating || 0)).filter(v => !Number.isNaN(v));
  const avg = values.length ? (values.reduce((sum, v) => sum + v, 0) / values.length).toFixed(1) : '0.0';
  elements.avgRating.textContent = avg;
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
  state.categories = data.categories.slice(0, 12);
  elements.categoryList.innerHTML = state.categories.map(category => `<button class="category-chip" data-category="${category}">${category}</button>`).join('');

  document.querySelectorAll('.category-chip').forEach(button => {
    button.addEventListener('click', () => {
      const category = button.dataset.category;
      fetchRecommendations(category, true);
    });
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

async function init() {
  await loadStats();
  await loadFeaturedBooks();
  await loadCategories();
  fetchRecommendations('fiction');
}

init();
