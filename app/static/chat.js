const form = document.getElementById("chat-form");
const input = document.getElementById("chat-input");
const log = document.getElementById("chat-log");

function addBubble(cls, text) {
  const div = document.createElement("div");
  div.className = `bubble ${cls}`;
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  return div;
}

// Tool activity is shown, not hidden: an agent that can act inside the product
// has to be auditable by the person whose ticket it is acting on.
function renderAction(action, before) {
  const args = Object.entries(action.arguments || {})
    .map(([k, v]) => `${k}=${v}`)
    .join(", ");
  const el = document.createElement("div");
  el.className = "action";
  el.textContent = action.result && action.result.error
    ? `⚠ ${action.tool}(${args}) — ${action.result.error}`
    : `⚙ ${action.tool}(${args})`;
  log.insertBefore(el, before);
  log.scrollTop = log.scrollHeight;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  input.value = "";
  addBubble("user", question);
  const answer = addBubble("assistant", "…");

  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let text = "";
  let sources = [];

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop(); // keep incomplete tail
    for (const ev of events) {
      const data = ev.replace(/^data: /, "").trim();
      if (!data || data === "[DONE]") continue;
      const msg = JSON.parse(data);
      if (msg.sources) sources = msg.sources;
      if (msg.action) renderAction(msg.action, answer);
      if (msg.delta) {
        text += msg.delta;
        answer.textContent = text;
        log.scrollTop = log.scrollHeight;
      }
    }
  }

  if (sources.length) {
    const src = document.createElement("div");
    src.className = "sources";
    // Distinct from the [file.md] citations inside the answer: those are the
    // model's claim, this is what retrieval actually put in the prompt.
    src.textContent = "Retrieved: " + [...new Set(sources.map((s) => s.source))].join(", ");
    answer.appendChild(src);
  }
});
