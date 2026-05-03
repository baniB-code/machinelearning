/**
 * Backend-backed auth for proposal portal pages.
 */

function escHtml(s) {
  if (!s) return "";
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function apiRequest(method, path, body) {
  const xhr = new XMLHttpRequest();
  xhr.open(method, path, false);
  xhr.setRequestHeader("Content-Type", "application/json");
  xhr.send(body ? JSON.stringify(body) : null);
  let data = {};
  try {
    data = xhr.responseText ? JSON.parse(xhr.responseText) : {};
  } catch {
    data = {};
  }
  return { status: xhr.status, data };
}

const Auth = (() => {
  function register(payload) {
    const res = apiRequest("POST", "/api/auth/register", payload);
    if (res.status >= 200 && res.status < 300) return res.data;
    return { ok: false, error: res.data.error || "Registration failed." };
  }

  function login(payload) {
    const res = apiRequest("POST", "/api/auth/login", payload);
    if (res.status >= 200 && res.status < 300) return res.data;
    return { ok: false, error: res.data.error || "Login failed." };
  }

  function logout() {
    apiRequest("POST", "/api/auth/logout", {});
  }

  function getSession() {
    const res = apiRequest("GET", "/api/auth/session");
    return res.data?.session || null;
  }

  function isLoggedIn() { return !!getSession(); }
  function isAdmin() { return getSession()?.role === "admin"; }
  function isClient() { return getSession()?.role === "client"; }
  function seedDefaults() { return true; }

  function getUserById(id) {
    return getAllUsers().find((u) => u.id === id) || null;
  }

  function getAllUsers() {
    const res = apiRequest("GET", "/api/users");
    return res.data?.users || [];
  }

  return { register, login, logout, getSession, isLoggedIn, isAdmin, isClient, seedDefaults, getUserById, getAllUsers };
})();

function requireAuth(requiredRole) {
  const sess = Auth.getSession();
  if (!sess) {
    window.location.href = "login.html";
    return false;
  }
  if (requiredRole && sess.role !== requiredRole) {
    window.location.href = sess.role === "admin" ? "admin-dashboard.html" : "client-complaints.html";
    return false;
  }
  return true;
}

document.addEventListener("DOMContentLoaded", () => {
  const session = Auth.getSession();
  const el = document.getElementById("navUser");
  if (el && session) {
    el.innerHTML = `
      <span style="font-size:0.82rem;color:var(--text-secondary);">👤 ${escHtml(session.name)}</span>
      <button onclick="Auth.logout(); window.location.href='login.html';" style="background:none;border:1px solid var(--glass-border);color:var(--text-muted);padding:0.3rem 0.7rem;border-radius:var(--radius);cursor:pointer;font-size:0.78rem;">Logout</button>
    `;
  }
});
