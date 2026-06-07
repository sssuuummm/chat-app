/**
 * Chat App — Frontend logic.
 *
 * State:
 *   currentConvId  — active conversation ID (null = fresh chat, not yet saved)
 *   conversations  — cached list of conv summaries
 *   pendingImage   — { b64, mime, name } of image to send with next message
 *
 * Behavior:
 *   No auto-scroll — user keeps full scroll control.
 *   No streaming   — waits for full API response then renders.
 *   No sidebar     — history hidden in a modal, accessed via top-bar button.
 */

// ── State ────────────────────────────────────────────────────────────────────

let currentConvId = null;
let conversations = [];
let pendingImage = null;   // { b64, mime, name }
let isLoading = false;

// ── DOM refs ─────────────────────────────────────────────────────────────────

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const chatArea = $("#chat-area");
const emptyState = $("#empty-state");
const messagesContainer = $("#messages-container");
const inputArea = $("#input-area");
const messageInput = $("#message-input");
const btnSend = $("#btn-send");
const btnAttach = $("#btn-attach-image");
const fileInput = $("#file-input");
const imagePreviewBar = $("#image-preview-bar");
const imagePreviewThumb = $("#image-preview-thumb");
const btnRemoveImage = $("#btn-remove-image");
const btnHistory = $("#btn-history");
const btnSettings = $("#btn-settings");
const historyModal = $("#history-modal");
const settingsModal = $("#settings-modal");

// Settings fields
const settingDeepseekKey = $("#setting-deepseek-key");
const settingDeepseekModel = $("#setting-deepseek-model");
const settingVisionProvider = $("#setting-vision-provider");
const settingOpenaiKey = $("#setting-openai-key");
const settingGeminiKey = $("#setting-gemini-key");
const settingDoubaoKey = $("#setting-doubao-key");
const settingDoubaoModel = $("#setting-doubao-model");

// ── Init ─────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
    loadSettingsFromStorage();
    refreshConversationList();
    loadLastConversation();

    // If no saved conversation, show empty state with input available
    if (!currentConvId) {
        emptyState.style.display = "";
        messagesContainer.style.display = "none";
    }

    // Top bar buttons
    btnHistory.addEventListener("click", openHistory);
    btnSettings.addEventListener("click", openSettings);

    // History modal
    $("#btn-close-history").addEventListener("click", closeHistory);
    $("#btn-new-chat").addEventListener("click", () => { startNewChat(); closeHistory(); });
    $("#history-search").addEventListener("input", debounce(onHistorySearch, 200));
    historyModal.addEventListener("click", (e) => {
        if (e.target === historyModal) closeHistory();
    });

    // Settings modal
    $("#btn-save-settings").addEventListener("click", saveSettings);
    $("#btn-close-settings").addEventListener("click", closeSettings);
    settingsModal.addEventListener("click", (e) => {
        if (e.target === settingsModal) closeSettings();
    });

    // Message input
    btnSend.addEventListener("click", sendMessage);
    messageInput.addEventListener("keydown", onInputKeydown);
    messageInput.addEventListener("input", autoResizeInput);

    // Image
    btnAttach.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", onFileSelected);
    btnRemoveImage.addEventListener("click", removeImage);
    document.addEventListener("paste", onPaste);
});

// ── Conversation management ──────────────────────────────────────────────────

async function refreshConversationList(query = "") {
    try {
        let url = "/api/conversations";
        if (query) url += `?q=${encodeURIComponent(query)}`;
        const resp = await fetch(url);
        conversations = await resp.json();
        renderHistoryList();
    } catch (err) {
        console.error("Failed to load conversations:", err);
    }
}

function renderHistoryList() {
    const list = $("#history-list");
    if (!list) return;

    if (conversations.length === 0) {
        list.innerHTML = '<div class="history-empty">暂无历史记录</div>';
        return;
    }

    list.innerHTML = "";
    conversations.forEach((c) => {
        const el = document.createElement("div");
        el.className = "history-item" + (c.id === currentConvId ? " active" : "");
        el.innerHTML = `
            <div class="history-item-info">
                <div class="history-item-title">${escapeHtml(c.title || "新对话")}</div>
                <div class="history-item-meta">
                    <span>${c.message_count} 条消息</span>
                    <span>${formatDate(c.updated_at)}</span>
                </div>
            </div>
            <button class="history-item-delete" data-id="${c.id}" title="删除">🗑</button>
        `;
        el.addEventListener("click", (e) => {
            if (e.target.classList.contains("history-item-delete")) return;
            loadConversation(c.id);
            closeHistory();
        });
        list.appendChild(el);
    });

    // Delete buttons
    list.querySelectorAll(".history-item-delete").forEach((btn) => {
        btn.addEventListener("click", (e) => {
            e.stopPropagation();
            deleteConversation(btn.dataset.id);
        });
    });
}

async function loadConversation(convId) {
    try {
        const resp = await fetch(`/api/conversations/${convId}`);
        if (!resp.ok) throw new Error("Not found");
        const conv = await resp.json();

        currentConvId = convId;
        saveCurrentConvToStorage();
        renderMessages(conv.messages);
        showChatMode();
    } catch (err) {
        console.error("Failed to load conversation:", err);
    }
}

async function deleteConversation(convId) {
    if (!confirm("确定删除此对话？")) return;

    try {
        await fetch(`/api/conversations/${convId}`, { method: "DELETE" });
        if (currentConvId === convId) {
            currentConvId = null;
            saveCurrentConvToStorage();
            showEmptyState();
        }
        refreshConversationList();
    } catch (err) {
        console.error("Failed to delete conversation:", err);
    }
}

function startNewChat() {
    currentConvId = null;
    pendingImage = null;
    removeImage();
    saveCurrentConvToStorage();
    showChatMode();
    emptyState.style.display = "";
    messagesContainer.style.display = "none";
    messagesContainer.innerHTML = "";
    messageInput.value = "";
    messageInput.style.height = "auto";
    messageInput.focus();
}

// ── Message rendering ────────────────────────────────────────────────────────

function renderMessages(messages) {
    messagesContainer.innerHTML = "";

    if (!messages || messages.length === 0) {
        messagesContainer.innerHTML =
            '<div style="text-align:center;color:#999;padding:40px;">开始对话吧</div>';
        return;
    }

    messages.forEach((msg) => {
        appendMessageBubble(msg);
    });
}

function appendMessageBubble(msg) {
    const row = document.createElement("div");
    row.className = `message-row ${msg.role}`;

    const body = document.createElement("div");
    body.className = "message-body";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";

    let content = msg.content || "";
    content = renderMarkdownContent(content);

    if (msg.has_image && msg.image_b64) {
        const img = document.createElement("img");
        img.src = `data:${msg.image_mime || "image/png"};base64,${msg.image_b64}`;
        img.alt = "Sent image";
        img.addEventListener("click", () => {
            window.open(img.src, "_blank");
        });
        bubble.appendChild(img);
        if (content.trim()) {
            const textSpan = document.createElement("span");
            textSpan.innerHTML = content;
            bubble.appendChild(textSpan);
        }
    } else {
        bubble.innerHTML = content;
    }

    const time = document.createElement("div");
    time.className = "message-time";
    time.textContent = formatTime(msg.timestamp);

    body.appendChild(bubble);
    body.appendChild(time);
    row.appendChild(body);
    messagesContainer.appendChild(row);
}

function renderMarkdownContent(text) {
    if (!text) return "";

    let html = escapeHtml(text);

    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/__(.+?)__/g, "<strong>$1</strong>");

    html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");
    html = html.replace(/_(.+?)_/g, "<em>$1</em>");

    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
        return `<pre><code>${escapeHtml(code.trim())}</code></pre>`;
    });

    html = html.replace(/\n/g, "<br>");

    return html;
}

function appendLoadingBubble() {
    const row = document.createElement("div");
    row.className = "message-row assistant";
    row.id = "loading-bubble";

    const body = document.createElement("div");
    body.className = "message-body";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    bubble.innerHTML = '<div class="message-loader"><span></span><span></span><span></span></div>';

    body.appendChild(bubble);
    row.appendChild(body);
    messagesContainer.appendChild(row);
}

function removeLoadingBubble() {
    const el = document.getElementById("loading-bubble");
    if (el) el.remove();
}

// ── Message sending ──────────────────────────────────────────────────────────

async function sendMessage() {
    const text = messageInput.value.trim();
    if (!text && !pendingImage) return;
    if (isLoading) return;

    isLoading = true;
    btnSend.disabled = true;

    showChatMode();

    const userMsg = {
        role: "user",
        content: text || "",
        timestamp: new Date().toISOString(),
        has_image: !!pendingImage,
    };
    if (pendingImage) {
        userMsg.image_b64 = pendingImage.b64;
        userMsg.image_mime = pendingImage.mime;
    }
    appendMessageBubble(userMsg);

    messageInput.value = "";
    autoResizeInput();

    const imageData = pendingImage;
    pendingImage = null;
    removeImage();

    appendLoadingBubble();

    try {
        const body = {
            text: text,
            conversation_id: currentConvId || undefined,
        };
        if (imageData) {
            body.image_b64 = imageData.b64;
            body.image_mime = imageData.mime;
        }

        const resp = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.error || `HTTP ${resp.status}`);
        }

        const data = await resp.json();

        removeLoadingBubble();

        currentConvId = data.conversation_id;
        saveCurrentConvToStorage();

        appendMessageBubble(data.message);
        refreshConversationList();
    } catch (err) {
        removeLoadingBubble();
        const errorMsg = {
            role: "assistant",
            content: `**发送失败**：${escapeHtml(err.message)}`,
            timestamp: new Date().toISOString(),
        };
        appendMessageBubble(errorMsg);
    } finally {
        isLoading = false;
        btnSend.disabled = false;
        messageInput.focus();
    }
}

// ── Image handling ───────────────────────────────────────────────────────────

function onFileSelected(e) {
    const file = e.target.files[0];
    if (!file) return;
    encodeImageFile(file);
    fileInput.value = "";
}

function onPaste(e) {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (const item of items) {
        if (item.type.startsWith("image/")) {
            const file = item.getAsFile();
            encodeImageFile(file);
            break;
        }
    }
}

function encodeImageFile(file) {
    const reader = new FileReader();
    reader.onload = () => {
        const dataUrl = reader.result;
        const [header, b64] = dataUrl.split(",");
        const mime = header.match(/data:(.*);/)?.[1] || "image/png";
        pendingImage = {
            b64: b64,
            mime: mime,
            name: file.name || "image",
        };
        showImagePreview(dataUrl, file.name);
    };
    reader.readAsDataURL(file);
}

function showImagePreview(dataUrl, name) {
    imagePreviewThumb.src = dataUrl;
    const nameEl = imagePreviewBar.querySelector(".file-name");
    if (nameEl) nameEl.textContent = name || "图片";
    imagePreviewBar.style.display = "flex";
}

function removeImage() {
    pendingImage = null;
    imagePreviewBar.style.display = "none";
    imagePreviewThumb.src = "";
}

// ── UI modes ─────────────────────────────────────────────────────────────────

function showChatMode() {
    emptyState.style.display = "none";
    messagesContainer.style.display = "";
    inputArea.style.display = "";
}

function showEmptyState() {
    emptyState.style.display = "";
    messagesContainer.style.display = "none";
    messagesContainer.innerHTML = "";
    inputArea.style.display = "";
    messageInput.value = "";
}

// ── Input helpers ────────────────────────────────────────────────────────────

function onInputKeydown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

function autoResizeInput() {
    messageInput.style.height = "auto";
    messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + "px";
}

// ── History modal ────────────────────────────────────────────────────────────

function openHistory() {
    refreshConversationList();
    $("#history-search").value = "";
    historyModal.style.display = "";
}

function closeHistory() {
    historyModal.style.display = "none";
}

function onHistorySearch() {
    const query = $("#history-search").value.trim();
    refreshConversationList(query);
}

// ── Settings modal ───────────────────────────────────────────────────────────

function openSettings() {
    const saved = getSettingsFromStorage();
    settingDeepseekKey.value = saved.deepseek_api_key || "";
    settingDeepseekModel.value = saved.deepseek_model || "deepseek-chat";
    settingVisionProvider.value = saved.vision_provider || "none";
    settingOpenaiKey.value = saved.openai_api_key || "";
    settingGeminiKey.value = saved.gemini_api_key || "";
    settingDoubaoKey.value = saved.doubao_api_key || "";
    settingDoubaoModel.value = saved.doubao_vision_model || "doubao-vision-pro-32k";
    settingsModal.style.display = "";
}

function closeSettings() {
    settingsModal.style.display = "none";
}

async function saveSettings() {
    const settings = {
        deepseek_api_key: settingDeepseekKey.value.trim(),
        deepseek_model: settingDeepseekModel.value.trim() || "deepseek-chat",
        vision_provider: settingVisionProvider.value,
        openai_api_key: settingOpenaiKey.value.trim(),
        gemini_api_key: settingGeminiKey.value.trim(),
        doubao_api_key: settingDoubaoKey.value.trim(),
        doubao_vision_model: settingDoubaoModel.value.trim() || "doubao-vision-pro-32k",
    };

    localStorage.setItem("chat-settings", JSON.stringify(settings));

    try {
        await fetch("/api/config", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(settings),
        });
    } catch (err) {
        console.error("Failed to update server config:", err);
    }

    closeSettings();
    alert("设置已保存。API Key 仅在当前服务器运行期间有效，重启后需重新设置或写入 .env。");
}

function getSettingsFromStorage() {
    try {
        return JSON.parse(localStorage.getItem("chat-settings") || "{}");
    } catch {
        return {};
    }
}

function loadSettingsFromStorage() {
    // No longer auto-push to server — would overwrite .env / Render env vars.
    // Settings are only applied when user explicitly saves from the settings modal.
}

// ── Persistence ──────────────────────────────────────────────────────────────

function saveCurrentConvToStorage() {
    if (currentConvId) {
        localStorage.setItem("current-conv-id", currentConvId);
    } else {
        localStorage.removeItem("current-conv-id");
    }
}

function loadLastConversation() {
    const savedId = localStorage.getItem("current-conv-id");
    if (savedId) {
        loadConversation(savedId);
    }
}

// ── Utilities ────────────────────────────────────────────────────────────────

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

function formatDate(isoStr) {
    if (!isoStr) return "";
    try {
        const d = new Date(isoStr);
        const now = new Date();
        const isToday = d.toDateString() === now.toDateString();
        if (isToday) {
            return d.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
        }
        return d.toLocaleDateString("zh-CN", { month: "short", day: "numeric" });
    } catch {
        return isoStr;
    }
}

function formatTime(isoStr) {
    if (!isoStr) return "";
    try {
        return new Date(isoStr).toLocaleTimeString("zh-CN", {
            hour: "2-digit",
            minute: "2-digit",
        });
    } catch {
        return "";
    }
}

function debounce(fn, ms) {
    let timer;
    return (...args) => {
        clearTimeout(timer);
        timer = setTimeout(() => fn(...args), ms);
    };
}
