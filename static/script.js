// =============================================================
// Edu-Query — Frontend Script
// Handles: chat UI, API calls, typing indicator, markdown render
// =============================================================

// ---- DOM References ----
const messagesWindow = document.getElementById("messagesWindow");
const welcomeScreen = document.getElementById("welcomeScreen");
const userInput = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");

// ---- State ----
let isWaiting = false; // Prevents double-sending while AI is responding


// ==============================================================
// SEND MESSAGE — main entry point
// ==============================================================
async function sendMessage() {
  const text = userInput.value.trim();
  if (!text || isWaiting) return;

  // Hide welcome screen on first message
  if (welcomeScreen) welcomeScreen.style.display = "none";

  // Render user bubble immediately
  appendBubble("user", text);

  // Clear input and reset height
  userInput.value = "";
  autoGrow(userInput);

  // Disable input while waiting
  setWaiting(true);

  // Show typing indicator
  const typingId = showTyping();

  try {
    // --- POST to Flask /chat endpoint ---
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text })
    });

    const data = await response.json();

    // Remove typing indicator
    removeTyping(typingId);

    if (data.error) {
      appendBubble("ai", "⚠️ Sorry, something went wrong. Please try again.", "general", null, null);
    } else {
      // Render AI response bubble
      appendBubble(
        "ai",
        data.response,
        data.category || "general",
        data.college,
        data.web_source
      );
    }

  } catch (err) {
    removeTyping(typingId);
    appendBubble("ai", "⚠️ Could not reach the server. Please check your connection.", "general", null, null);
    console.error("Fetch error:", err);
  }

  setWaiting(false);
}


// ==============================================================
// APPEND BUBBLE
// Creates and inserts a chat message bubble into the DOM
// Parameters:
//   role      : "user" | "ai"
//   text      : plain text or markdown-like string
//   category  : detected category (for AI bubbles)
//   college   : college name if detected (for AI bubbles)
//   webSource : URL of fetched website (for AI bubbles)
// ==============================================================
function appendBubble(role, text, category = "general", college = null, webSource = null) {

  // --- Outer message row ---
  const msgDiv = document.createElement("div");
  msgDiv.className = `message ${role}`;

  // --- Avatar ---
  const avatar = document.createElement("div");
  avatar.className = `avatar ${role}`;
  avatar.textContent = role === "ai" ? "🤖" : "👤";

  // --- Bubble wrapper ---
  const bubbleWrap = document.createElement("div");
  bubbleWrap.className = "bubble-wrap";

  // --- Meta row (only for AI messages) ---
  if (role === "ai") {
    const meta = document.createElement("div");
    meta.className = "bubble-meta";

    // Category tag
    const catTag = document.createElement("span");
    catTag.className = `cat-tag ${category}`;
    catTag.textContent = categoryEmoji(category) + " " + capitalize(category);
    meta.appendChild(catTag);

    // Web-fetched badge
    if (webSource) {
      const webBadge = document.createElement("span");
      webBadge.className = "web-badge";
      webBadge.textContent = "🌐 Live Data";
      meta.appendChild(webBadge);
    }

    bubbleWrap.appendChild(meta);
  }

  // --- Bubble body ---
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = renderMarkdown(text); // convert markdown-like text to HTML

  bubbleWrap.appendChild(bubble);

  // --- Source link (only if web data was fetched) ---
  if (role === "ai" && webSource) {
    const link = document.createElement("a");
    link.className = "source-pill";
    link.href = webSource;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.innerHTML = `🔗 Official Website${college ? " · " + college : ""}`;
    bubbleWrap.appendChild(link);
  }

  // Assemble
  msgDiv.appendChild(avatar);
  msgDiv.appendChild(bubbleWrap);
  messagesWindow.appendChild(msgDiv);

  // Smooth scroll to bottom after a tiny delay for rendering
  setTimeout(scrollToBottom, 50);
}


// ==============================================================
// TYPING INDICATOR
// Shows animated dots while waiting for AI response
// ==============================================================
function showTyping() {
  const id = "typing-" + Date.now();

  const msgDiv = document.createElement("div");
  msgDiv.className = "message ai";
  msgDiv.id = id;

  const avatar = document.createElement("div");
  avatar.className = "avatar ai";
  avatar.textContent = "🤖";

  const bubbleWrap = document.createElement("div");
  bubbleWrap.className = "bubble-wrap";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.style.padding = "0";

  const typing = document.createElement("div");
  typing.className = "typing-indicator";
  typing.innerHTML = "<span></span><span></span><span></span>";

  bubble.appendChild(typing);
  bubbleWrap.appendChild(bubble);
  msgDiv.appendChild(avatar);
  msgDiv.appendChild(bubbleWrap);

  messagesWindow.appendChild(msgDiv);
  setTimeout(scrollToBottom, 50);

  return id;
}

function removeTyping(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}


// ==============================================================
// MARKDOWN RENDERER
// Converts simple markdown to HTML for AI responses
// Supports: **bold**, *italic*, bullet lists, numbered lists,
//           line breaks, and basic paragraph splitting
// ==============================================================
function renderMarkdown(text) {
  if (!text) return "";

  // Escape HTML first to prevent XSS
  let html = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // --- Block: numbered lists (1. Item) ---
  html = html.replace(/^(\d+)\.\s+(.+)$/gm, "<li>$2</li>");
  html = html.replace(/(<li>.*<\/li>(\n|$))+/g, (match) => `<ol>${match}</ol>`);

  // --- Block: bullet lists (• or - or *) ---
  html = html.replace(/^[•\-\*]\s+(.+)$/gm, "<li>$1</li>");
  // Wrap consecutive <li> not already in <ol> in <ul>
  html = html.replace(/(?<!<\/ol>)(<li>.*?<\/li>\n?)+(?!<\/ul>)/g, (match) => {
    if (match.includes("<ol>")) return match;
    return `<ul>${match}</ul>`;
  });

  // --- Inline: **bold** ---
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

  // --- Inline: *italic* ---
  html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");

  // --- Inline: `code` ---
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

  // --- Paragraphs: double line break → <p> ---
  const blocks = html.split(/\n\n+/);
  html = blocks.map(block => {
    block = block.trim();
    if (!block) return "";
    // Don't wrap list/heading HTML in <p>
    if (block.startsWith("<ul>") || block.startsWith("<ol>") || block.startsWith("<li>")) {
      return block;
    }
    return `<p>${block.replace(/\n/g, "<br>")}</p>`;
  }).join("\n");

  return html;
}


// ==============================================================
// UTILITY FUNCTIONS
// ==============================================================

/** Send a pre-written quick question */
function sendQuick(text) {
  userInput.value = text;
  sendMessage();
}

/** Clear chat and restore welcome screen */
async function clearChat() {
  // Remove all messages
  const messages = messagesWindow.querySelectorAll(".message");
  messages.forEach(m => m.remove());

  // Show welcome screen again
  if (welcomeScreen) welcomeScreen.style.display = "block";

  // Reset server context
  try {
    await fetch("/clear", { method: "POST" });
    console.log("Context cleared on server.");
  } catch (err) {
    console.error("Error clearing context:", err);
  }
}

/** Scroll chat window to the bottom */
function scrollToBottom() {
  requestAnimationFrame(() => {
    messagesWindow.scrollTop = messagesWindow.scrollHeight;
  });
}

/** Enable/disable input while AI is responding */
function setWaiting(state) {
  isWaiting = state;
  sendBtn.disabled = state;
  userInput.disabled = state;
}

/** Auto-grow textarea as user types */
function autoGrow(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 120) + "px";
}

/** Handle keyboard: Enter = send, Shift+Enter = new line */
function handleKeyDown(event) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
}

/** Map category to a small emoji prefix */
function categoryEmoji(cat) {
  const map = {
    admissions: "🎯",
    fees: "💰",
    exams: "📝",
    campus: "🏛️",
    courses: "📖",
    placements: "🏆",
    general: "💬"
  };
  return map[cat] || "💬";
}

/** Capitalize first letter */
function capitalize(str) {
  if (!str) return "";
  return str.charAt(0).toUpperCase() + str.slice(1);
}