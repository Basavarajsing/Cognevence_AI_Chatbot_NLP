/**
 * AI Chatbot Frontend Interactive Logic
 * Handles real-time messaging, typing indicators, suggested questions,
 * conversation clearing, and export actions.
 */

document.addEventListener("DOMContentLoaded", () => {
    const chatMessages = document.getElementById("chatMessages");
    const chatForm = document.getElementById("chatForm");
    const userInput = document.getElementById("userInput");
    const sendBtn = document.getElementById("sendBtn");
    const suggestedContainer = document.getElementById("suggestedContainer");
    const clearChatBtn = document.getElementById("clearChatBtn");
    const exportDropdownBtn = document.getElementById("exportDropdownBtn");
    const exportDropdown = document.getElementById("exportDropdown");
    const sidebarToggleBtn = document.getElementById("sidebarToggleBtn");
    const sidebar = document.getElementById("sidebar");

    // Fetch and render smart suggested questions (Feature A)
    loadSuggestedQuestions();

    // Toggle Export Dropdown
    if (exportDropdownBtn && exportDropdown) {
        exportDropdownBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            exportDropdown.classList.toggle("open");
        });

        document.addEventListener("click", () => {
            exportDropdown.classList.remove("open");
        });
    }

    // Mobile Sidebar Toggle
    if (sidebarToggleBtn && sidebar) {
        sidebarToggleBtn.addEventListener("click", () => {
            sidebar.classList.toggle("open");
        });
    }

    // Chat Form Submission
    if (chatForm) {
        chatForm.addEventListener("submit", (e) => {
            e.preventDefault();
            const message = userInput.value.trim();
            if (message) {
                sendMessage(message);
            }
        });
    }

    // Clear Conversation History
    if (clearChatBtn) {
        clearChatBtn.addEventListener("click", async () => {
            if (confirm("Are you sure you want to clear the conversation history?")) {
                try {
                    const response = await fetch("/clear", { method: "POST" });
                    if (response.ok) {
                        chatMessages.innerHTML = "";
                        appendBotMessage({
                            response: "Conversation history cleared. How can I help you today?",
                            intent: "greeting",
                            confidence: 1.0,
                            is_fallback: false,
                            timestamp: getFormattedTime()
                        });
                    }
                } catch (err) {
                    console.error("Failed to clear chat:", err);
                }
            }
        });
    }

    /**
     * Load suggested questions from backend (Feature A)
     */
    async function loadSuggestedQuestions() {
        if (!suggestedContainer) return;
        try {
            const res = await fetch("/suggested");
            const data = await res.json();
            if (data.suggestions && data.suggestions.length > 0) {
                suggestedContainer.innerHTML = "";
                data.suggestions.forEach((q) => {
                    const chip = document.createElement("button");
                    chip.type = "button";
                    chip.className = "suggested-chip";
                    chip.innerHTML = `<span>💬</span> ${q}`;
                    chip.addEventListener("click", () => {
                        userInput.value = q;
                        sendMessage(q);
                    });
                    suggestedContainer.appendChild(chip);
                });
            }
        } catch (err) {
            console.warn("Could not load suggested questions:", err);
        }
    }

    /**
     * Send user query to Flask backend
     */
    async function sendMessage(text) {
        // Append user bubble
        appendUserMessage(text, getFormattedTime());
        userInput.value = "";
        userInput.disabled = true;
        sendBtn.disabled = true;

        // Show typing indicator
        const typingElem = showTypingIndicator();
        scrollToBottom();

        try {
            const res = await fetch("/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: text })
            });

            removeTypingIndicator(typingElem);

            if (!res.ok) {
                const errData = await res.json();
                appendBotMessage({
                    response: errData.error || "Sorry, an unexpected server error occurred.",
                    intent: "fallback",
                    confidence: 0.0,
                    is_fallback: true,
                    timestamp: getFormattedTime()
                });
            } else {
                const data = await res.json();
                appendBotMessage(data);
            }
        } catch (error) {
            removeTypingIndicator(typingElem);
            appendBotMessage({
                response: "Network error: Unable to reach the chatbot server.",
                intent: "fallback",
                confidence: 0.0,
                is_fallback: true,
                timestamp: getFormattedTime()
            });
        } finally {
            userInput.disabled = false;
            sendBtn.disabled = false;
            userInput.focus();
            scrollToBottom();
        }
    }

    /**
     * Append User Message Element
     */
    function appendUserMessage(text, time) {
        const msgDiv = document.createElement("div");
        msgDiv.className = "message user";
        msgDiv.innerHTML = `
            <div class="message-avatar">U</div>
            <div class="message-body">
                <div class="message-bubble">${escapeHtml(text)}</div>
                <div class="message-meta">
                    <span>${time}</span>
                </div>
            </div>
        `;
        chatMessages.appendChild(msgDiv);
        scrollToBottom();
    }

    /**
     * Append Bot Message Element
     */
    function appendBotMessage(data) {
        const msgDiv = document.createElement("div");
        msgDiv.className = "message bot";

        let badgeHtml = "";
        if (data.is_fallback) {
            badgeHtml = `<span class="fallback-badge">Fallback</span>`;
        } else if (data.intent) {
            const confPct = (data.confidence * 100).toFixed(0);
            badgeHtml = `
                <span class="intent-badge">${escapeHtml(data.intent)}</span>
                <span class="confidence-badge">${confPct}% conf</span>
            `;
        }

        msgDiv.innerHTML = `
            <div class="message-avatar">AI</div>
            <div class="message-body">
                <div class="message-bubble">${escapeHtml(data.response)}</div>
                <div class="message-meta">
                    <span>${data.timestamp || getFormattedTime()}</span>
                    ${badgeHtml}
                </div>
            </div>
        `;
        chatMessages.appendChild(msgDiv);
        scrollToBottom();
    }

    /**
     * Show Typing Indicator
     */
    function showTypingIndicator() {
        const typingDiv = document.createElement("div");
        typingDiv.className = "message bot typing-wrapper";
        typingDiv.innerHTML = `
            <div class="message-avatar">AI</div>
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        `;
        chatMessages.appendChild(typingDiv);
        return typingDiv;
    }

    function removeTypingIndicator(elem) {
        if (elem && elem.parentNode) {
            elem.parentNode.removeChild(elem);
        }
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function getFormattedTime() {
        const d = new Date();
        let hours = d.getHours();
        let minutes = d.getMinutes();
        const ampm = hours >= 12 ? "PM" : "AM";
        hours = hours % 12;
        hours = hours ? hours : 12;
        minutes = minutes < 10 ? "0" + minutes : minutes;
        return `${hours}:${minutes} ${ampm}`;
    }

    function escapeHtml(str) {
        if (!str) return "";
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
});
