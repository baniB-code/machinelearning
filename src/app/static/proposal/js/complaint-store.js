/**
 * Backend-backed complaint store facade.
 */

function storeApi(method, path, body) {
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
  if (xhr.status < 200 || xhr.status >= 300) {
    return { status: xhr.status, data: null };
  }
  return { status: xhr.status, data };
}

const ComplaintStore = (() => {
  function getAll() {
    const res = storeApi("GET", "/api/complaints");
    if (!res.data) return null;
    return res.data?.complaints || [];
  }

  function add(complaint) {
    const res = storeApi("POST", "/api/complaints", complaint);
    return res.data?.complaint || null;
  }

  function getById(id) {
    const all = getAll();
    if (!all) return null;
    return all.find((c) => c.id === id) || null;
  }

  function getByUser(uid) {
    const all = getAll();
    if (!all) return null;
    return all.filter((c) => c.userId === uid);
  }

  function updateStatus(id, status) {
    const res = storeApi("PATCH", `/api/complaints/${id}/status`, { status });
    return res.data?.complaint || null;
  }

  function getStats() {
    const all = getAll();
    if (!all) return null;
    const sentimentCounts = { Positive: 0, Negative: 0, Neutral: 0 };
    const categoryCounts = {};
    const priorityCounts = { High: 0, Medium: 0, Low: 0 };
    const statusCounts = { Pending: 0, "In Progress": 0, Resolved: 0, "Pending Analysis": 0 };
    const dailyTrend = {};
    all.forEach((c) => {
      if (c.sentiment?.label) sentimentCounts[c.sentiment.label] = (sentimentCounts[c.sentiment.label] || 0) + 1;
      const cat = c.classification?.category || c.category || "General";
      categoryCounts[cat] = (categoryCounts[cat] || 0) + 1;
      if (c.priority?.level) priorityCounts[c.priority.level] = (priorityCounts[c.priority.level] || 0) + 1;
      if (c.status) statusCounts[c.status] = (statusCounts[c.status] || 0) + 1;
      const day = (c.createdAt || c.timestamp || "").substring(0, 10);
      if (day) dailyTrend[day] = (dailyTrend[day] || 0) + 1;
    });
    return { total: all.length, sentimentCounts, categoryCounts, priorityCounts, statusCounts, dailyTrend };
  }

  function getMessages(complaintId) {
    const res = storeApi("GET", `/api/complaints/${complaintId}/messages`);
    if (!res.data) return null;
    return res.data?.messages || [];
  }

  function addMessage(complaintId, msg) {
    const res = storeApi("POST", `/api/complaints/${complaintId}/messages`, { text: msg.text });
    return res.data?.message || null;
  }

  function analyzeComplaint(complaintId) {
    const res = storeApi("POST", `/api/complaints/${complaintId}/analyze`, {});
    return res.data?.complaint || null;
  }

  function seedSamples() {
    return true;
  }

  return { getAll, getById, getByUser, add, updateStatus, getStats, getMessages, addMessage, analyzeComplaint, seedSamples };
})();

window.Store = ComplaintStore;
