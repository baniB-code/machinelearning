/**
 * Uses backend ML endpoint.
 */
const MLEngine = (() => {
  function analyze(text) {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/analyze", false);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.send(JSON.stringify({ text }));
    if (xhr.status < 200 || xhr.status >= 300) {
      return { error: "Failed to analyze complaint text." };
    }
    const payload = JSON.parse(xhr.responseText || "{}");
    if (!payload.ok) return { error: payload.error || "Analysis failed." };
    return {
      text,
      timestamp: new Date().toISOString(),
      sentiment: payload.result.sentiment,
      classification: payload.result.classification,
      priority: payload.result.priority,
      nlp: {},
    };
  }

  const CATEGORIES = {};
  return { analyze, CATEGORIES };
})();
