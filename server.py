"""
Chat App Backend — Flask server with multi-agent orchestration.

Architecture:
  User message (+ optional image)
       │
       ├─ No image ──→ DeepSeek ──→ response
       │
       └─ Has image ──→ DeepSeek generates vision query
                        ──→ Vision Agent processes image
                        ──→ DeepSeek sees vision result + user text
                        ──→ final response

Run:
  python server.py
  (reads .env for API keys; falls back to env vars)
"""

import base64
import io
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import requests
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

# ── App setup ────────────────────────────────────────────────────────────────

app = Flask(__name__, template_folder="templates", static_folder="static")
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "conversations"
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB

# ── Config from env / .env ───────────────────────────────────────────────────

def _load_dotenv(*paths: Path):
    """Minimal .env loader — no dependency on python-dotenv.
    Loads from each path in order; later paths take precedence for overriding."""
    for env_file in paths:
        if not env_file.exists():
            continue
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val

_load_dotenv(BASE_DIR / ".env")

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_THINKING = os.environ.get("DEEPSEEK_THINKING", "enabled")  # "enabled" | "disabled"
DEEPSEEK_REASONING_EFFORT = os.environ.get("DEEPSEEK_REASONING_EFFORT", "high")  # "low" | "medium" | "high"

VISION_PROVIDER = os.environ.get("VISION_PROVIDER", "none")  # "openai" | "gemini" | "doubao" | "none"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
DOUBAO_API_KEY = os.environ.get("DOUBAO_API_KEY", "")
DOUBAO_BASE_URL = os.environ.get("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
DOUBAO_VISION_MODEL = os.environ.get("DOUBAO_VISION_MODEL", "doubao-vision-pro-32k")

# ── Vision Agent: pluggable interface ────────────────────────────────────────

class VisionAgent:
    """Pluggable vision provider. Add new providers by implementing a method."""

    @staticmethod
    def describe(image_base64: str, mime_type: str, guidance: str = "") -> str:
        """Describe an image. `guidance` is a prompt from DeepSeek about what to look for."""
        if VISION_PROVIDER == "openai":
            return VisionAgent._describe_openai(image_base64, mime_type, guidance)
        elif VISION_PROVIDER == "gemini":
            return VisionAgent._describe_gemini(image_base64, mime_type, guidance)
        elif VISION_PROVIDER == "doubao":
            return VisionAgent._describe_doubao(image_base64, mime_type, guidance)
        else:
            return "[Vision Agent not configured. Set VISION_PROVIDER in .env]"

    @staticmethod
    def _describe_openai(image_base64: str, mime_type: str, guidance: str) -> str:
        if not OPENAI_API_KEY:
            return "[OpenAI API key not set]"
        data_url = f"data:{mime_type};base64,{image_base64}"
        prompt = guidance or "Please describe this image in detail, in Chinese."
        body = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url, "detail": "auto"}},
                    ],
                }
            ],
            "max_tokens": 1000,
        }
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json=body,
            timeout=60,
        )
        if resp.status_code != 200:
            return f"[Vision error: {resp.status_code} {resp.text[:300]}]"
        return resp.json()["choices"][0]["message"]["content"]

    @staticmethod
    def _describe_gemini(image_base64: str, mime_type: str, guidance: str) -> str:
        if not GEMINI_API_KEY:
            return "[Gemini API key not set]"
        prompt = guidance or "Please describe this image in detail, in Chinese."
        body = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": mime_type, "data": image_base64}},
                    ]
                }
            ]
        }
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
            json=body,
            timeout=60,
        )
        if resp.status_code != 200:
            return f"[Vision error: {resp.status_code} {resp.text[:300]}]"
        data = resp.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            return f"[Vision parse error: {json.dumps(data, ensure_ascii=False)[:500]}]"

    @staticmethod
    def _describe_doubao(image_base64: str, mime_type: str, guidance: str) -> str:
        """ByteDance 豆包 (Doubao) vision — OpenAI-compatible API via 火山引擎."""
        if not DOUBAO_API_KEY:
            return "[Doubao API key not set]"
        data_url = f"data:{mime_type};base64,{image_base64}"
        prompt = guidance or "请详细描述这张图片的内容，用中文。"
        body = {
            "model": DOUBAO_VISION_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "max_tokens": 1000,
        }
        resp = requests.post(
            f"{DOUBAO_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {DOUBAO_API_KEY}", "Content-Type": "application/json"},
            json=body,
            timeout=120,
        )
        if resp.status_code != 200:
            return f"[Doubao vision error: {resp.status_code} {resp.text[:300]}]"
        return resp.json()["choices"][0]["message"]["content"]


# ── DeepSeek API helper ──────────────────────────────────────────────────────

def call_deepseek(messages: list[dict], *, max_tokens: int = 2048, temperature: float = 0.7) -> str:
    """Call DeepSeek chat completion. Returns the assistant text."""
    if not DEEPSEEK_API_KEY:
        return "**错误**：未设置 DeepSeek API Key。请在 .env 文件中设置 DEEPSEEK_API_KEY，或在页面设置中填入。"
    body = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    # DeepSeek thinking mode (reasoning_content)
    if DEEPSEEK_THINKING == "enabled":
        body["thinking"] = {"type": "enabled"}
        body["reasoning_effort"] = DEEPSEEK_REASONING_EFFORT
    try:
        resp = requests.post(
            f"{DEEPSEEK_BASE_URL}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=120,
        )
        if resp.status_code != 200:
            body_text = resp.text[:500]
            print(f"[DeepSeek error] {resp.status_code}: {body_text}")
            return f"**API 错误** ({resp.status_code})：{body_text}"
        return resp.json()["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        return "**错误**：DeepSeek API 请求超时，请重试。"
    except requests.exceptions.ConnectionError:
        return "**错误**：无法连接 DeepSeek API，请检查网络。"


# ── Multi-agent orchestration ────────────────────────────────────────────────

def orchestrate(user_text: str, image_b64: Optional[str] = None, image_mime: Optional[str] = None) -> str:
    """
    Multi-agent pipeline:
      1. If image → DeepSeek decides what to ask the vision agent
      2. Vision agent answers DeepSeek's question
      3. DeepSeek produces final response with vision context
    """
    if image_b64 and VISION_PROVIDER != "none":
        # ── Step 1: DeepSeek generates a vision query ──
        guidance_prompt = (
            f"用户发来一条消息并附带了一张图片。用户消息：\n```\n{user_text}\n```\n\n"
            "请根据用户的意图，想一个最合适的问题去向视觉模型提问，"
            "让视觉模型描述图片中与用户问题有关的内容。"
            "只输出你要问视觉模型的问题本身（用中文），不要输出任何其他内容。"
            "最多3句话。"
        )
        guidance = call_deepseek(
            [{"role": "system", "content": "你是一个帮助分析用户意图的助手。只输出对视觉模型的提问。"},
             {"role": "user", "content": guidance_prompt}],
            max_tokens=256,
            temperature=0.3,
        )

        # ── Step 2: Vision agent answers ──
        vision_result = VisionAgent.describe(image_b64, image_mime, guidance.strip())

        # ── Step 3: DeepSeek produces final response ──
        final_prompt = (
            f"用户消息：{user_text}\n\n"
            f"用户附带了一张图片，以下是视觉模型根据你的提问「{guidance.strip()}」"
            f"对图片的描述：\n```\n{vision_result}\n```\n\n"
            "请结合图片描述，回答用户的问题。自然地引用图片中的信息，"
            "不要提到'根据图片描述'这样的字眼，直接回答问题即可。"
        )
        return call_deepseek([{"role": "user", "content": final_prompt}])

    elif image_b64 and VISION_PROVIDER == "none":
        # Vision not configured → tell DeepSeek there's an image but no vision agent
        hint = (
            f"{user_text}\n\n[系统提示：用户附带了一张图片，但视觉识别功能尚未配置。"
            "请在回复中告知用户：当前无法识别图片，如需识图功能请在 .env 中配置 VISION_PROVIDER。]"
        )
        return _chat_with_history(hint)

    else:
        # Text-only → straight to DeepSeek
        return _chat_with_history(user_text)


def _chat_with_history(user_text: str) -> str:
    """Simple pass-through to DeepSeek with no conversation history (stateless per-call)."""
    return call_deepseek([{"role": "user", "content": user_text}])


# ── Conversation storage ─────────────────────────────────────────────────────

def _tz_now() -> str:
    """ISO timestamp in local time (+08:00)."""
    tz = timezone(timedelta(hours=8))
    return datetime.now(tz).isoformat()


def load_conversation(conv_id: str) -> dict | None:
    path = DATA_DIR / f"{conv_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_conversation(conv: dict) -> None:
    path = DATA_DIR / f"{conv['id']}.json"
    path.write_text(json.dumps(conv, ensure_ascii=False, indent=2), encoding="utf-8")


def list_conversations() -> list[dict]:
    """Return all conversations sorted by updated_at desc (summary only)."""
    convs = []
    for p in sorted(DATA_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            c = json.loads(p.read_text(encoding="utf-8"))
            # Return summary: id, title, timestamps, message count, first few words of last msg
            convs.append({
                "id": c["id"],
                "title": c.get("title", "新对话"),
                "created_at": c.get("created_at", ""),
                "updated_at": c.get("updated_at", ""),
                "message_count": len(c.get("messages", [])),
                "preview": c["messages"][-1]["content"][:80] if c.get("messages") else "",
            })
        except Exception:
            continue
    return convs


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return app.send_static_file("index.html")  # will serve from static/


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Send a message, get AI reply. Body: { conversation_id?, text, image? (base64) }"""
    data = request.get_json(force=True, silent=True) or {}
    user_text = (data.get("text") or "").strip()
    image_b64 = data.get("image_b64") or None
    image_mime = data.get("image_mime") or "image/png"
    conv_id = data.get("conversation_id") or None

    if not user_text and not image_b64:
        return jsonify({"error": "Empty message"}), 400

    # Load or create conversation
    if conv_id:
        conv = load_conversation(conv_id)
        if not conv:
            conv = {
                "id": str(uuid.uuid4())[:12],
                "title": user_text[:40] or "图片消息",
                "created_at": _tz_now(),
                "updated_at": _tz_now(),
                "messages": [],
            }
    else:
        conv = {
            "id": str(uuid.uuid4())[:12],
            "title": user_text[:40] or "图片消息",
            "created_at": _tz_now(),
            "updated_at": _tz_now(),
            "messages": [],
        }

    # Add user message
    user_msg = {
        "role": "user",
        "content": user_text,
        "timestamp": _tz_now(),
        "has_image": bool(image_b64),
    }
    if image_b64:
        user_msg["image_b64"] = image_b64
        user_msg["image_mime"] = image_mime
    conv["messages"].append(user_msg)

    # Orchestrate multi-agent response
    ai_reply = orchestrate(user_text, image_b64, image_mime)

    # Add AI message
    ai_msg = {
        "role": "assistant",
        "content": ai_reply,
        "timestamp": _tz_now(),
    }
    conv["messages"].append(ai_msg)
    conv["updated_at"] = _tz_now()

    # Auto-title from first user message if still default
    if conv["title"] == "新对话" and conv["messages"]:
        first_user = next((m["content"] for m in conv["messages"] if m["role"] == "user"), "")
        conv["title"] = first_user[:40] if first_user else "新对话"

    save_conversation(conv)

    return jsonify({
        "conversation_id": conv["id"],
        "reply": ai_reply,
        "message": ai_msg,
    })


@app.route("/api/conversations", methods=["GET"])
def api_list_conversations():
    """List all conversations (summary). Query: ?q=search_term"""
    convs = list_conversations()
    q = (request.args.get("q") or "").strip().lower()
    if q:
        filtered = []
        for c in convs:
            # Search in title and preview
            if q in c["title"].lower() or q in c["preview"].lower():
                filtered.append(c)
                continue
            # Also search full message content for deeper matches
            full = load_conversation(c["id"])
            if full:
                for m in full.get("messages", []):
                    if q in m.get("content", "").lower():
                        filtered.append(c)
                        break
        convs = filtered
    return jsonify(convs)


@app.route("/api/conversations/<conv_id>", methods=["GET"])
def api_get_conversation(conv_id: str):
    """Get full conversation with all messages."""
    conv = load_conversation(conv_id)
    if not conv:
        return jsonify({"error": "Not found"}), 404
    return jsonify(conv)


@app.route("/api/conversations/<conv_id>", methods=["DELETE"])
def api_delete_conversation(conv_id: str):
    """Delete a conversation."""
    path = DATA_DIR / f"{conv_id}.json"
    if path.exists():
        path.unlink()
        return jsonify({"ok": True})
    return jsonify({"error": "Not found"}), 404


@app.route("/api/config", methods=["GET"])
def api_get_config():
    """Return current config status (no secrets)."""
    return jsonify({
        "deepseek_configured": bool(DEEPSEEK_API_KEY),
        "deepseek_model": DEEPSEEK_MODEL,
        "deepseek_thinking": DEEPSEEK_THINKING,
        "deepseek_reasoning_effort": DEEPSEEK_REASONING_EFFORT,
        "vision_provider": VISION_PROVIDER,
        "vision_configured": bool(
            (VISION_PROVIDER == "openai" and OPENAI_API_KEY)
            or (VISION_PROVIDER == "gemini" and GEMINI_API_KEY)
            or (VISION_PROVIDER == "doubao" and DOUBAO_API_KEY)
        ),
        "doubao_configured": bool(DOUBAO_API_KEY),
    })


@app.route("/api/config", methods=["POST"])
def api_set_config():
    """Update config at runtime (stored in session only, not persisted to .env)."""
    data = request.get_json(force=True, silent=True) or {}
    global DEEPSEEK_API_KEY, DEEPSEEK_MODEL, VISION_PROVIDER, OPENAI_API_KEY, GEMINI_API_KEY
    global DEEPSEEK_THINKING, DEEPSEEK_REASONING_EFFORT, DOUBAO_API_KEY, DOUBAO_VISION_MODEL
    if data.get("deepseek_api_key"):
        DEEPSEEK_API_KEY = data["deepseek_api_key"]
    if data.get("deepseek_model"):
        DEEPSEEK_MODEL = data["deepseek_model"]
    if data.get("deepseek_thinking") is not None:
        DEEPSEEK_THINKING = data["deepseek_thinking"]
    if data.get("deepseek_reasoning_effort") is not None:
        DEEPSEEK_REASONING_EFFORT = data["deepseek_reasoning_effort"]
    if data.get("vision_provider") is not None:
        VISION_PROVIDER = data["vision_provider"]
    if data.get("openai_api_key"):
        OPENAI_API_KEY = data["openai_api_key"]
    if data.get("gemini_api_key"):
        GEMINI_API_KEY = data["gemini_api_key"]
    if data.get("doubao_api_key"):
        DOUBAO_API_KEY = data["doubao_api_key"]
    if data.get("doubao_vision_model"):
        DOUBAO_VISION_MODEL = data["doubao_vision_model"]
    return jsonify({"ok": True})


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"[OK] DeepSeek configured: {bool(DEEPSEEK_API_KEY)}")
    print(f"[OK] Model: {DEEPSEEK_MODEL}")
    print(f"[OK] Thinking: {DEEPSEEK_THINKING} (effort={DEEPSEEK_REASONING_EFFORT})")
    print(f"[OK] Vision provider: {VISION_PROVIDER} (configured: {bool(OPENAI_API_KEY or GEMINI_API_KEY or DOUBAO_API_KEY)})")
    print(f"[OK] Conversations dir: {DATA_DIR}")
    print(f"--> http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
