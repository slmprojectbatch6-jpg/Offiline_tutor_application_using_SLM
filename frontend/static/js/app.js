/* ═══════════════════════════════════════════════════════════
   NCERT AI Tutor — Frontend Application Logic
   ═══════════════════════════════════════════════════════════ */

const API = '';   // Same-origin (FastAPI serves index.html)

// ── State ─────────────────────────────────────────────────────
const state = {
  token: localStorage.getItem('tutor_token') || null,
  user: JSON.parse(localStorage.getItem('tutor_user') || 'null'),
  currentSession: null,
  currentSubject: '',
  sessions: [],
  isGenerating: false,
  sourcesOpen: false,
};

// ── Startup ───────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  if (state.token && state.user) {
    launchApp();
  } else {
    showScreen('auth-screen');
  }
  checkSystemHealth();
});

// ── Screen Management ─────────────────────────────────────────
function showScreen(id) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  document.getElementById(id).classList.add('active');
}

function launchApp() {
  showScreen('app-screen');
  populateSidebar();
  showPage('dashboard');
  loadDashboard();
  loadRecentSessions();
}

// ── Page Navigation ───────────────────────────────────────────
function showPage(name) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.getElementById(`page-${name}`).classList.add('active');
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  const nav = document.getElementById(`nav-${name}`);
  if (nav) nav.classList.add('active');

  if (name === 'history') loadHistoryPage();
  if (name === 'subjects') loadSubjectsPage();
}

// ── Auth ──────────────────────────────────────────────────────
function switchTab(tab) {
  document.getElementById('login-form').classList.toggle('active', tab === 'login');
  document.getElementById('register-form').classList.toggle('active', tab === 'register');
  document.getElementById('tab-login').classList.toggle('active', tab === 'login');
  document.getElementById('tab-register').classList.toggle('active', tab === 'register');
}

async function handleLogin(e) {
  e.preventDefault();
  const username = document.getElementById('login-username').value.trim();
  const password = document.getElementById('login-password').value;
  const errorEl = document.getElementById('login-error');
  const btn = document.getElementById('login-btn');

  setButtonLoading(btn, true);
  errorEl.classList.add('hidden');

  try {
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);

    const res = await fetch(`${API}/api/auth/login`, { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail || 'Login failed');

    saveAuthData(data);
    launchApp();
    showToast('Welcome back, ' + data.user.full_name + '! 👋', 'success');
  } catch (err) {
    errorEl.textContent = err.message;
    errorEl.classList.remove('hidden');
  } finally {
    setButtonLoading(btn, false);
  }
}

async function handleRegister(e) {
  e.preventDefault();
  const errorEl = document.getElementById('register-error');
  const btn = document.getElementById('register-btn');

  const payload = {
    username: document.getElementById('reg-username').value.trim(),
    full_name: document.getElementById('reg-fullname').value.trim(),
    password: document.getElementById('reg-password').value,
    class_level: parseInt(document.getElementById('reg-class').value) || null,
  };

  setButtonLoading(btn, true);
  errorEl.classList.add('hidden');

  try {
    const res = await fetch(`${API}/api/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) {
      let errMsg = data.detail || 'Registration failed';
      if (Array.isArray(data.detail)) {
        errMsg = data.detail.map(err => `${err.loc.slice(-1)[0]}: ${err.msg}`).join(', ');
      }
      throw new Error(errMsg);
    }

    saveAuthData(data);
    launchApp();
    showToast('Account created! Welcome, ' + data.user.full_name + '! 🎉', 'success');
  } catch (err) {
    errorEl.textContent = err.message;
    errorEl.classList.remove('hidden');
  } finally {
    setButtonLoading(btn, false);
  }
}

function saveAuthData(data) {
  state.token = data.access_token;
  state.user = data.user;
  localStorage.setItem('tutor_token', state.token);
  localStorage.setItem('tutor_user', JSON.stringify(state.user));
}

function handleLogout() {
  state.token = null;
  state.user = null;
  state.currentSession = null;
  localStorage.removeItem('tutor_token');
  localStorage.removeItem('tutor_user');
  showScreen('auth-screen');
  showToast('Logged out successfully', 'info');
}

// ── Sidebar ───────────────────────────────────────────────────
function populateSidebar() {
  if (!state.user) return;
  document.getElementById('sidebar-username').textContent = state.user.full_name;
  document.getElementById('sidebar-class').textContent =
    state.user.class_level ? `Class ${state.user.class_level}` : 'Student';
  document.getElementById('user-avatar').textContent =
    state.user.full_name.charAt(0).toUpperCase();
}

function toggleSidebar() {
  document.getElementById('sidebar').classList.toggle('collapsed');
}

function setSubjectFilter(subject) {
  state.currentSubject = subject;
  document.querySelectorAll('.subject-pill').forEach(p =>
    p.classList.toggle('active', p.dataset.subject === subject));
  document.getElementById('subject-select').value = subject;
  onSubjectChange(subject);
  if (subject) showPage('chat');
}

function onSubjectChange(subject) {
  state.currentSubject = subject;
  const label = document.getElementById('chat-subject-label');
  const names = { history: '🏛️ History', polity: '⚖️ Political Science', geography: '🌏 Geography', economics: '📊 Economics', gk: '🧠 General Knowledge' };
  label.textContent = subject ? names[subject] : 'All Subjects';
  document.querySelectorAll('.subject-pill').forEach(p =>
    p.classList.toggle('active', p.dataset.subject === subject));
}

// ── Dashboard ─────────────────────────────────────────────────
async function loadDashboard() {
  try {
    const [statsRes, sugRes] = await Promise.all([
      apiFetch('/api/dashboard/stats'),
      apiFetch('/api/dashboard/suggestions'),
    ]);

    if (statsRes.ok) {
      const data = await statsRes.json();
      renderDashboardStats(data);
    }
    if (sugRes.ok) {
      const data = await sugRes.json();
      renderSuggestions(data);
    }

    loadQuickStart();
  } catch (err) {
    console.warn('Dashboard load error:', err);
  }
}

function renderDashboardStats(data) {
  // Welcome
  const name = state.user?.full_name?.split(' ')[0] || 'Student';
  const hour = new Date().getHours();
  const greeting = hour < 12 ? 'Good morning' : hour < 17 ? 'Good afternoon' : 'Good evening';
  document.getElementById('welcome-msg').textContent = `${greeting}, ${name}! 👋`;

  // Stat cards
  document.getElementById('stat-questions').textContent = data.total_messages || 0;
  document.getElementById('stat-sessions').textContent = data.total_sessions || 0;

  const topicsCount = (data.progress || []).reduce((s, p) => s + (p.topics_explored || 0), 0);
  document.getElementById('stat-topics').textContent = topicsCount;

  const strongest = data.most_active_subject;
  document.getElementById('stat-strong').textContent = strongest
    ? strongest.charAt(0).toUpperCase() + strongest.slice(1)
    : 'Keep studying!';

  // Progress bars
  const COLORS = { history: '#f59e0b', polity: '#6366f1', geography: '#10b981', economics: '#ef4444', gk: '#ec4899' };
  const subjectStats = data.subject_stats || {};
  const maxQ = Math.max(...Object.values(subjectStats), 1);
  const container = document.getElementById('subject-progress-bars');

  const subjects = ['history', 'polity', 'geography', 'economics', 'gk'];
  container.innerHTML = subjects.map(s => {
    const count = subjectStats[s] || 0;
    const pct = Math.round((count / maxQ) * 100);
    return `
      <div class="progress-item">
        <div class="progress-header">
          <span class="progress-label">${s.charAt(0).toUpperCase() + s.slice(1)}</span>
          <span class="progress-count">${count} questions</span>
        </div>
        <div class="progress-track">
          <div class="progress-fill" style="width:${pct}%;background:${COLORS[s]}"></div>
        </div>
      </div>`;
  }).join('');

  // Animate bars
  setTimeout(() => {
    document.querySelectorAll('.progress-fill').forEach(bar => {
      const w = bar.style.width;
      bar.style.width = '0';
      requestAnimationFrame(() => { bar.style.width = w; });
    });
  }, 50);
}

function renderSuggestions(data) {
  const container = document.getElementById('suggestions-list');
  const items = [];

  Object.entries(data.suggestions || {}).forEach(([subject, info]) => {
    (info.practice_questions || []).slice(0, 2).forEach(q => {
      items.push({ subject, question: q, isWeak: info.is_weak });
    });
  });

  if (!items.length) {
    container.innerHTML = '<p class="loading-text">Start asking questions to get personalized suggestions!</p>';
    return;
  }

  container.innerHTML = items.map(item => `
    <div class="suggestion-item" onclick="sendSuggestion('${item.subject}', ${JSON.stringify(item.question).replace(/'/g, "&#39;")})">
      <span class="suggestion-badge ${item.isWeak ? 'badge-weak' : 'badge-practice'}">${item.isWeak ? 'Review' : item.subject}</span>
      <span>${item.question}</span>
    </div>
  `).join('');
}

function sendSuggestion(subject, question) {
  state.currentSubject = subject;
  document.getElementById('subject-select').value = subject;
  onSubjectChange(subject);
  showPage('chat');
  setTimeout(() => sendMessageWithText(question), 200);
}

async function loadQuickStart() {
  try {
    const res = await apiFetch('/api/dashboard/leaderboard');
    if (!res.ok) return;
    const data = await res.json();
    const grid = document.getElementById('quick-start-grid');
    grid.innerHTML = (data.subjects || []).map(s => `
      <div class="quick-start-card" onclick="setSubjectFilter('${s.id}'); showPage('chat')" style="border-top: 3px solid ${s.color}">
        <div class="qs-icon">${s.icon}</div>
        <div class="qs-name">${s.name}</div>
        <div class="qs-desc">${s.description}</div>
      </div>
    `).join('');
  } catch (e) { }
}

// ── Chat ──────────────────────────────────────────────────────
async function startNewChat() {
  state.currentSession = null;
  document.getElementById('chat-messages').innerHTML = `
    <div class="chat-welcome">
      <div class="chat-welcome-icon">🤖</div>
      <h3>Hello! I'm your NCERT AI Tutor</h3>
      <p>Ask me anything about History, Political Science, Geography, Economics, or General Knowledge from Classes 6-10.</p>
      <div class="chat-starter-btns">
        <button class="starter-btn" onclick="sendStarterQuestion('What is the importance of the Constitution of India?')">⚖️ What is the Constitution?</button>
        <button class="starter-btn" onclick="sendStarterQuestion('Explain the Mughal Empire in India.')">🏛️ Mughal Empire</button>
        <button class="starter-btn" onclick="sendStarterQuestion('What are the major physical features of India?')">🌏 Physical Features</button>
        <button class="starter-btn" onclick="sendStarterQuestion('What is GDP and why is it important?')">📊 What is GDP?</button>
      </div>
    </div>`;
  document.getElementById('chat-session-title').textContent = 'New Chat';
  document.getElementById('sources-panel').classList.add('hidden');
  showPage('chat');
}

function sendStarterQuestion(q) {
  document.getElementById('chat-input').value = q;
  sendMessage();
}

function handleChatKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 150) + 'px';
}

async function sendMessage() {
  const input = document.getElementById('chat-input');
  const text = input.value.trim();
  if (!text || state.isGenerating) return;

  input.value = '';
  input.style.height = 'auto';
  await sendMessageWithText(text);
}

async function sendMessageWithText(text) {
  if (!text || state.isGenerating) return;

  state.isGenerating = true;
  document.getElementById('send-btn').disabled = true;

  // Remove welcome screen if present
  const welcome = document.querySelector('.chat-welcome');
  if (welcome) welcome.remove();

  // Add user message
  appendMessage('user', text);

  // Add typing indicator
  const typingId = 'typing-' + Date.now();
  appendTyping(typingId);

  try {
    const payload = {
      message: text,
      subject: state.currentSubject || null,
      session_id: state.currentSession,
    };

    const res = await apiFetch('/api/chat/ask', { method: 'POST', body: JSON.stringify(payload) });
    const data = await res.json();

    removeTyping(typingId);

    if (!res.ok) throw new Error(data.detail || 'Error getting response');

    // Save session
    if (data.session_id && !state.currentSession) {
      state.currentSession = data.session_id;
      await loadRecentSessions();
    }

    // Update session title
    document.getElementById('chat-session-title').textContent =
      text.length > 50 ? text.slice(0, 50) + '…' : text;

    // Show answer
    appendMessage('assistant', data.answer, {
      responseTime: data.response_time_ms,
      subject: data.subject,
      sources: data.sources,
    });

    // Show sources
    if (data.sources && data.sources.length > 0) {
      renderSources(data.sources);
    }

  } catch (err) {
    removeTyping(typingId);
    appendMessage('assistant', '⚠️ ' + err.message);
    showToast('Error: ' + err.message, 'error');
  } finally {
    state.isGenerating = false;
    document.getElementById('send-btn').disabled = false;
  }
}

function appendMessage(role, content, meta = {}) {
  const container = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = `message ${role}`;

  const avatar = role === 'user'
    ? (state.user?.full_name?.charAt(0).toUpperCase() || 'U')
    : '🤖';

  const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const formattedContent = formatMessageContent(content);

  let metaHtml = `<span class="msg-time">${timeStr}</span>`;
  if (meta.responseTime) metaHtml += `<span class="msg-time-badge">⚡ ${Math.round(meta.responseTime)}ms</span>`;
  if (meta.subject) metaHtml += `<span class="msg-time-badge">${meta.subject}</span>`;

  div.innerHTML = `
    <div class="msg-avatar">${avatar}</div>
    <div>
      <div class="msg-bubble">${formattedContent}</div>
      <div class="msg-meta">${metaHtml}</div>
    </div>`;

  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function formatMessageContent(text) {
  if (!text) return '';
  // Convert markdown-ish formatting
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`(.*?)`/g, '<code>$1</code>')
    .replace(/^- (.+)$/gm, '<li>$1</li>')
    .replace(/(<li>.*<\/li>\n?)+/g, match => `<ul>${match}</ul>`)
    .replace(/^\d+\. (.+)$/gm, '<li>$1</li>')
    .replace(/\n{2,}/g, '</p><p>')
    .replace(/\n/g, '<br />')
    .replace(/^(.+)$/, '<p>$1</p>');
}

function appendTyping(id) {
  const container = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = 'message assistant';
  div.id = id;
  div.innerHTML = `
    <div class="msg-avatar">🤖</div>
    <div>
      <div class="msg-bubble">
        <div class="typing-indicator">
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
        </div>
      </div>
    </div>`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function removeTyping(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

// ── Sources ───────────────────────────────────────────────────
function renderSources(sources) {
  const panel = document.getElementById('sources-panel');
  const content = document.getElementById('sources-content');

  content.innerHTML = sources.map(s => `
    <div class="source-chip">
      <div class="source-chip-header">
        <span class="source-subject">${s.subject || ''}</span>
        <span class="source-score">Score: ${(s.score * 100).toFixed(0)}%</span>
      </div>
      <div class="source-chapter">${s.chapter || 'General'}</div>
      <div class="source-preview">${s.preview || ''}</div>
    </div>
  `).join('');

  panel.classList.remove('hidden');
  state.sourcesOpen = true;
  document.getElementById('sources-toggle-icon').textContent = '▲';
}

function toggleSources() {
  const content = document.getElementById('sources-content');
  state.sourcesOpen = !state.sourcesOpen;
  content.style.display = state.sourcesOpen ? 'flex' : 'none';
  document.getElementById('sources-toggle-icon').textContent = state.sourcesOpen ? '▲' : '▼';
}

// ── Chat History ──────────────────────────────────────────────
async function loadRecentSessions() {
  try {
    const res = await apiFetch('/api/chat/sessions');
    if (!res.ok) return;
    const sessions = await res.json();
    state.sessions = sessions;

    const container = document.getElementById('recent-sessions');
    if (!sessions.length) {
      container.innerHTML = '<p class="session-empty">No chats yet</p>';
      return;
    }

    container.innerHTML = sessions.slice(0, 8).map(s => `
      <div class="session-item-wrap" style="display:flex;align-items:center;gap:4px;">
        <button class="session-item" onclick="loadSession(${s.session_id})" title="${s.title}" style="flex:1;">
          ${s.title || 'New Chat'}
        </button>
        <button onclick="deleteSession(${s.session_id})" style="background:none;border:none;color:var(--text-muted);cursor:pointer;padding:4px;font-size:0.9rem;border-radius:4px;flex-shrink:0;" title="Delete chat" onmouseover="this.style.background='var(--bg-card)'" onmouseout="this.style.background='none'">🗑️</button>
      </div>
    `).join('');
  } catch (e) { }
}

async function loadSession(sessionId) {
  state.currentSession = sessionId;
  showPage('chat');

  const container = document.getElementById('chat-messages');
  container.innerHTML = '<p style="text-align:center;color:var(--text-muted);padding:20px">Loading...</p>';

  try {
    const res = await apiFetch(`/api/chat/sessions/${sessionId}/messages`);
    if (!res.ok) throw new Error('Failed to load session');
    const messages = await res.json();

    container.innerHTML = '';
    if (!messages.length) {
      container.innerHTML = '<p style="text-align:center;color:var(--text-muted);padding:20px">Empty session</p>';
      return;
    }

    messages.forEach(msg => {
      appendMessage(msg.role, msg.content, {
        subject: msg.subject,
        responseTime: msg.response_time_ms,
      });
    });

    // Show last sources if available
    const lastAssistant = [...messages].reverse().find(m => m.role === 'assistant' && m.sources?.length);
    if (lastAssistant) renderSources(lastAssistant.sources);

    // Update title
    const session = state.sessions.find(s => s.session_id === sessionId);
    if (session) document.getElementById('chat-session-title').textContent = session.title;

  } catch (e) {
    showToast('Could not load session', 'error');
  }
}

async function deleteSession(sessionId) {
  if (!confirm('Are you sure you want to delete this chat session?')) return;
  try {
    const res = await apiFetch(`/api/chat/sessions/${sessionId}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to delete session');

    showToast('Chat session deleted', 'success');

    // Refresh history and recent sessions
    loadRecentSessions();
    if (document.getElementById('page-history').classList.contains('active')) {
      loadHistoryPage();
    }

    // Clear chat if it is the current one
    if (state.currentSession === sessionId) {
      startNewChat();
    }
  } catch (e) {
    showToast('Could not delete session', 'error');
  }
}

// ── History Page ──────────────────────────────────────────────
async function loadHistoryPage() {
  const container = document.getElementById('history-list');
  container.innerHTML = '<p style="color:var(--text-muted)">Loading...</p>';

  try {
    const res = await apiFetch('/api/chat/sessions');
    if (!res.ok) throw new Error();
    const sessions = await res.json();

    if (!sessions.length) {
      container.innerHTML = `
        <div class="history-empty">
          <div class="empty-icon">💬</div>
          <p>No chat history yet. Start asking questions!</p>
          <button class="btn-primary" onclick="showPage('chat')">Start Chatting</button>
        </div>`;
      return;
    }

    container.innerHTML = sessions.map(s => {
      const date = new Date(s.updated_at).toLocaleDateString('en-IN', {
        day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit'
      });
      return `
        <div class="history-card" onclick="loadSession(${s.session_id})">
          <div class="history-card-info">
            <div class="history-title">${s.title || 'New Chat'}</div>
            <div class="history-meta">
              <span>${date}</span>
              ${s.subject ? `<span class="history-subject-badge">${s.subject}</span>` : ''}
            </div>
          </div>
          <div style="display:flex;gap:8px;align-items:center;">
            <button class="history-open-btn" onclick="event.stopPropagation();deleteSession(${s.session_id})" style="color:var(--accent-red);border-color:rgba(239,68,68,0.3);" title="Delete chat">🗑️</button>
            <button class="history-open-btn" onclick="event.stopPropagation();loadSession(${s.session_id});showPage('chat')">Open →</button>
          </div>
        </div>`;
    }).join('');
  } catch (e) {
    container.innerHTML = '<p style="color:var(--text-muted)">Could not load history.</p>';
  }
}

// ── Subjects Page ─────────────────────────────────────────────
async function loadSubjectsPage() {
  const grid = document.getElementById('subjects-grid');
  try {
    const res = await apiFetch('/api/dashboard/leaderboard');
    if (!res.ok) return;
    const data = await res.json();

    grid.innerHTML = (data.subjects || []).map(s => `
      <div class="subject-card" style="--subject-color:${s.color}">
        <style>.subject-card[style*="${s.color}"]::before{background:${s.color}}</style>
        <div class="subject-icon">${s.icon}</div>
        <div class="subject-name">${s.name}</div>
        <div class="subject-desc">${s.description}</div>
        <div class="subject-chapters">
          ${(s.chapters || []).slice(0, 4).map(c => `<span class="chapter-tag">${c}</span>`).join('')}
        </div>
        <button class="subject-ask-btn" onclick="setSubjectFilter('${s.id}');showPage('chat')">
          Ask about ${s.name} →
        </button>
      </div>
    `).join('');
  } catch (e) { }
}

// ── System Health Check ───────────────────────────────────────
async function checkSystemHealth() {
  try {
    const res = await fetch(`${API}/api/health`);
    const data = await res.json();
    const statusEl = document.getElementById('system-status');
    const textEl = document.getElementById('status-text');

    if (!statusEl) return;

    const faissOk = data.faiss_index === 'ready';
    const phi3Ok = data.phi3_model === 'ready';

    if (faissOk && phi3Ok) {
      textEl.textContent = 'Fully Offline & Ready';
      statusEl.querySelector('.status-dot').style.background = 'var(--accent-green)';
    } else if (faissOk) {
      textEl.textContent = 'RAG Ready (Phi-3 missing)';
      statusEl.querySelector('.status-dot').style.background = 'var(--accent-orange)';
    } else {
      textEl.textContent = 'Setup Required';
      statusEl.querySelector('.status-dot').style.background = 'var(--accent-red)';
    }
  } catch (e) {
    const textEl = document.getElementById('status-text');
    if (textEl) textEl.textContent = 'Server Offline';
  }
}

// ── API Helper ────────────────────────────────────────────────
async function apiFetch(path, opts = {}) {
  const headers = {
    'Content-Type': 'application/json',
    ...(state.token ? { Authorization: `Bearer ${state.token}` } : {}),
    ...(opts.headers || {}),
  };
  return fetch(`${API}${path}`, { ...opts, headers });
}

// ── UI Helpers ────────────────────────────────────────────────
function setButtonLoading(btn, loading) {
  btn.querySelector('.btn-text').classList.toggle('hidden', loading);
  btn.querySelector('.btn-loader').classList.toggle('hidden', !loading);
  btn.disabled = loading;
}

function showToast(message, type = 'info') {
  const icons = { success: '✅', error: '❌', info: 'ℹ️' };
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${icons[type]}</span><span>${message}</span>`;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(20px)';
    toast.style.transition = '0.3s';
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}
