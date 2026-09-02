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

async function refreshTicketQueue() {
  const rows = document.getElementById("ticket-rows");
  if (!rows) return;
  try {
    const res = await fetch(window.location.pathname, { headers: { "Cache-Control": "no-cache" } });
    const doc = new DOMParser().parseFromString(await res.text(), "text/html");
    const fresh = doc.getElementById("ticket-rows");
    if (fresh) rows.replaceChildren(...fresh.children);
  } catch {
    // A stale queue is a cosmetic problem; never let it break the chat panel.
  }
}

// State-changing actions are proposed, not performed. Nothing has happened when
// this renders — the assistant is asking the agent to approve it.
function renderProposal(proposal, before) {
  const args = Object.entries(proposal.arguments || {})
    .map(([k, v]) => `${k}=${v}`)
    .join(", ");

  const card = document.createElement("div");
  card.className = "proposal";

  const label = document.createElement("div");
  label.className = "proposal-label";
  label.textContent = `Needs your approval: ${proposal.tool}(${args})`;
  card.appendChild(label);

  const row = document.createElement("div");
  row.className = "proposal-actions";

  const settle = async (endpoint, verb) => {
    row.remove();
    label.textContent = `${verb}… ${proposal.tool}(${args})`;
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ proposal_id: proposal.id }),
    });
    const data = await res.json();
    if (data.error) {
      card.classList.add("proposal-error");
      label.textContent = `${proposal.tool}(${args}) — ${data.error}`;
    } else if (data.declined) {
      card.classList.add("proposal-declined");
      label.textContent = `Declined: ${proposal.tool}(${args})`;
    } else if (data.result && data.result.error) {
      card.classList.add("proposal-error");
      label.textContent = `Refused: ${proposal.tool}(${args}) — ${data.result.error}`;
    } else {
      card.classList.add("proposal-done");
      label.textContent = `Approved: ${proposal.tool}(${args})`;
      // An approved write changes ticket state, and the queue on the left was
      // rendered before it happened. Re-render just the rows so the agent is not
      // reading a stale status; reloading the page would take the chat with it.
      refreshTicketQueue();
    }
  };

  const approve = document.createElement("button");
  approve.type = "button";
  approve.textContent = "Approve";
  approve.addEventListener("click", () => settle("/api/approve", "Approving"));

  const decline = document.createElement("button");
  decline.type = "button";
  decline.className = "secondary";
  decline.textContent = "Decline";
  decline.addEventListener("click", () => settle("/api/decline", "Declining"));

  row.append(approve, decline);
  card.appendChild(row);
  log.insertBefore(card, before);
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
    body: JSON.stringify({
      question,
      ticket_id: Number(document.getElementById("ticket-select").value),
    }),
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
      if (msg.proposal) renderProposal(msg.proposal, answer);
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
