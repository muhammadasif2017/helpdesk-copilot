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
