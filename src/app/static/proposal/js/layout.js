/**
 * CogniSense CX — Shared Layout (Sidebar + Topbar)
 * Call Layout.render('pageName') before DOMContentLoaded content renders.
 */
const Layout = (() => {

  const CLIENT_NAV = [
    { href: 'client-complaints.html',  icon: '📋', label: 'My Complaints' },
    { href: 'client-submit.html',      icon: '📝', label: 'Submit Complaint' },
    { href: 'login.html',              icon: '🚪', label: 'Logout',  special: 'logout' },
  ];

  const ADMIN_NAV = [
    { href: 'admin-dashboard.html',   icon: '📊', label: 'Dashboard' },
    { href: 'analyze.html',           icon: '🤖', label: 'AI Analyzer' },
    { href: 'admin-complaints.html',  icon: '📋', label: 'All Complaints' },
    { href: 'admin-messaging.html',   icon: '💬', label: 'Messaging' },
    { href: 'admin.html',             icon: '📈', label: 'Analytics' },
    { href: 'login.html',             icon: '🚪', label: 'Logout',  special: 'logout' },
  ];

  function render(activePage) {
    const session = Auth.getSession();
    if (!session) return;

    const nav = session.role === 'admin' ? ADMIN_NAV : CLIENT_NAV;
    const roleLabel = session.role === 'admin' ? '🛡️ Admin' : '👤 Client';
    const roleBadgeColor = session.role === 'admin' ? '#7c6ef5' : '#10d98a';

    const sidebarHTML = `
      <aside class="sidebar" id="sidebar">
        <div class="sidebar-brand">
          <div class="sb-logo">🧠</div>
          <div>
            <div class="sb-name">CogniSense CX</div>
            <div class="sb-tag">AI Intelligence</div>
          </div>
        </div>

        <div class="sb-user">
          <div class="sb-avatar">${session.name[0].toUpperCase()}</div>
          <div class="sb-user-info">
            <div class="sb-user-name">${escHtml(session.name)}</div>
            <div class="sb-role-badge" style="background:${roleBadgeColor}20;color:${roleBadgeColor};border:1px solid ${roleBadgeColor}40;">${roleLabel}</div>
          </div>
        </div>

        <nav class="sb-nav">
          ${nav.map(item => `
            <a href="${item.special === 'logout' ? '#' : item.href}"
               class="sb-link${activePage === item.href ? ' active' : ''}"
               ${item.special === 'logout' ? 'onclick="Auth.logout(); window.location.href=\'login.html\'; return false;"' : ''}
            >
              <span class="sb-icon">${item.icon}</span>
              <span>${item.label}</span>
            </a>
          `).join('')}
        </nav>
      </aside>

      <button class="sidebar-toggle" id="sidebarToggle" aria-label="Toggle sidebar">☰</button>
      <div class="sidebar-overlay" id="sidebarOverlay"></div>
    `;

    // Inject before page content
    const container = document.querySelector('.app-container') || document.body;
    const wrapper = document.createElement('div');
    wrapper.innerHTML = sidebarHTML;
    while (wrapper.firstChild) container.insertBefore(wrapper.firstChild, container.firstChild);

    // Toggle logic
    const toggle  = document.getElementById('sidebarToggle');
    const overlay = document.getElementById('sidebarOverlay');
    const sb      = document.getElementById('sidebar');
    if (toggle) toggle.addEventListener('click', () => { sb.classList.toggle('open'); overlay.classList.toggle('open'); });
    if (overlay) overlay.addEventListener('click', () => { sb.classList.remove('open'); overlay.classList.remove('open'); });
  }

  return { render };
})();
