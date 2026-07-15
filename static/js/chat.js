/**
 * chat.js
 *
 * Drives the chat pane and the live XAI (LIME) insights panel.
 * Sends each user turn to POST /api/chat and renders.
 */

(function () {
  const SESSION_ID = window.MINDCARE_SESSION_ID;

  const messagesEl = document.getElementById("messages");
  const form = document.getElementById("chatForm");
  const input = document.getElementById("messageInput");
  const sendBtn = document.getElementById("sendBtn");

  const xaiPanel = document.getElementById("xaiPanel");
  const xaiToggleBtn = document.getElementById("xaiToggleBtn");
  const xaiIntent = document.getElementById("xaiIntent");
  const xaiConfidenceFill = document.getElementById("xaiConfidenceFill");
  const xaiConfidenceLabel = document.getElementById("xaiConfidenceLabel");
  const xaiWeights = document.getElementById("xaiWeights");

  if (!SESSION_ID) {
    window.location.href = "/";
    return;
  }

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function appendMessage(role, text, { crisis = false, intent = null, confidence = null } = {}) {
    const bubble = document.createElement("div");
    bubble.className = `msg msg-${role}` + (crisis ? " msg-crisis" : "");
    bubble.textContent = text;

    if (role === "assistant" && intent && !crisis) {
      const meta = document.createElement("div");
      meta.className = "msg-meta";
      const confPct = confidence !== null ? Math.round(confidence * 100) : null;
      meta.textContent = confPct !== null ? `intent: ${intent} (${confPct}%)` : `intent: ${intent}`;
      bubble.appendChild(meta);
    }

    messagesEl.appendChild(bubble);
    scrollToBottom();
  }

  function showTypingIndicator() {
    const el = document.createElement("div");
    el.className = "typing-indicator";
    el.id = "typingIndicator";
    el.textContent = "MindCare is typing...";
    messagesEl.appendChild(el);
    scrollToBottom();
  }

  function removeTypingIndicator() {
    const el = document.getElementById("typingIndicator");
    if (el) el.remove();
  }

  function updateXaiPanel(result) {
    if (result.crisis_tier && result.crisis_tier !== "none") {
      xaiIntent.textContent = "Crisis protocol";
      xaiConfidenceFill.style.width = "0%";
      xaiConfidenceLabel.textContent = "LIME explanations are skipped for crisis responses -- standard medical template triggered.";
      xaiWeights.innerHTML = "";
      return;
    }

    xaiIntent.textContent = result.intent || "—";

    const confidence = result.confidence || 0;
    xaiConfidenceFill.style.width = `${Math.round(confidence * 100)}%`;
    xaiConfidenceLabel.textContent = `Confidence: ${Math.round(confidence * 100)}%`;

    xaiWeights.innerHTML = "";
    const explanation = result.explanation;
    if (!explanation || !explanation.weights || explanation.weights.length === 0) {
      const empty = document.createElement("div");
      empty.className = "xai-empty";
      empty.textContent = "No weights generated for this response.";
      xaiWeights.appendChild(empty);
      return;
    }

    const maxAbs = Math.max(...explanation.weights.map((w) => Math.abs(w.weight)), 0.0001);

    explanation.weights.forEach((w) => {
      const row = document.createElement("div");
      row.className = "xai-weight-row";

      const token = document.createElement("span");
      token.className = "xai-weight-token";
      token.textContent = w.token;

      const track = document.createElement("span");
      track.className = "xai-weight-bar-track";

      const fill = document.createElement("span");
      const isPositive = w.weight >= 0;
      fill.className = `xai-weight-bar-fill ${isPositive ? "positive" : "negative"}`;
      
      // Compute percentage representation of weight strength
      const widthPct = Math.min(Math.abs(w.weight) / maxAbs, 1) * 100;
      fill.style.width = `${widthPct}%`;

      track.appendChild(fill);
      row.appendChild(token);
      row.appendChild(track);
      xaiWeights.appendChild(row);
    });
  }

  async function sendMessage(message) {
    appendMessage("user", message);
    
    // UI feedback reset
    input.value = "";
    input.style.height = "auto";
    sendBtn.disabled = true;
    showTypingIndicator();

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: SESSION_ID, message }),
      });

      removeTypingIndicator();

      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}));
        throw new Error(errBody.error || `Request failed with status ${response.status}`);
      }

      const result = await response.json();
      const isCrisis = result.crisis_tier && result.crisis_tier !== "none";

      appendMessage("assistant", result.reply, {
        crisis: isCrisis,
        intent: result.intent,
        confidence: result.confidence,
      });

      updateXaiPanel(result);
    } catch (err) {
      console.error("MindCare: chat request failed", err);
      removeTypingIndicator();
      appendMessage("assistant", "Sorry, something went wrong sending that message. Please try again.");
      
      // Restore user input so draft isn't lost on failure
      input.value = message;
    } finally {
      sendBtn.disabled = false;
      input.focus();
    }
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const message = input.value.trim();
    if (!message) return;
    sendMessage(message);
  });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 140)}px`;
  });

  xaiToggleBtn.addEventListener("click", () => {
    const isPressed = xaiToggleBtn.getAttribute("aria-pressed") === "true";
    xaiToggleBtn.setAttribute("aria-pressed", String(!isPressed));
    xaiPanel.classList.toggle("hidden", isPressed);
  });

  // Welcome user context on viewport initialization
  appendMessage(
    "assistant",
    "Hi, I'm MindCare. I'm here to listen and offer support. I'm not a substitute for professional care, but I'm glad you're here. What's on your mind today?"
  );
})();