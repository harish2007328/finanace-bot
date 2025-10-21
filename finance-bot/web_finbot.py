import re
import os
import uuid
import json
from flask import Flask, request, render_template_string, session, redirect, url_for, send_file
from markupsafe import Markup, escape

# Import finbot utilities and tools
from finbot import load_data, get_tool_call
from finbot import (
    get_summary, get_financial_total, get_top_spending_category,
    find_peak_spending_day_for_category, visualize_spending,
    find_transaction_date, get_financial_advice, add_transaction,
    get_balance, check_budgets, add_budget, check_goals, contribute_to_goal,
    calculate_savings_plan, add_savings_goal, identify_unnecessary_spending
)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")  # Replace in production

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CHAT_DIR = os.path.join(BASE_DIR, "chats")
MEDIA_DIR = os.path.join(BASE_DIR, "media")
PINNED_CHATS_FILE = os.path.join(CHAT_DIR, "pinned_chats.json")
os.makedirs(CHAT_DIR, exist_ok=True)
os.makedirs(MEDIA_DIR, exist_ok=True)

# Cache data per process
CACHE = {
    "data": None,
    "sessions": {}
}

def safe_filename(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', name or "")

def save_chat_history(chat_id, history):
    path = os.path.join(CHAT_DIR, f"{safe_filename(chat_id)}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False)

def load_pinned_chats():
    if not os.path.exists(PINNED_CHATS_FILE):
        return []
    try:
        with open(PINNED_CHATS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []

def save_pinned_chats(pinned_ids):
    with open(PINNED_CHATS_FILE, "w", encoding="utf-8") as f:
        json.dump(pinned_ids, f, ensure_ascii=False)

def load_chat_history(chat_id):
    try:
        path = os.path.join(CHAT_DIR, f"{safe_filename(chat_id)}.json")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def list_chat_sessions():
    pinned_ids = set(load_pinned_chats())
    sessions = []
    for fname in os.listdir(CHAT_DIR):
        if fname.endswith(".json") and fname != "pinned_chats.json":
            sessions.append(fname[:-5])

    pinned_sessions = [cid for cid in load_pinned_chats() if cid in sessions]
    unpinned_sessions = [cid for cid in sessions if cid not in pinned_ids]

    unpinned_sessions.sort(key=lambda cid: os.path.getmtime(os.path.join(CHAT_DIR, f"{cid}.json")), reverse=True)
    
    return pinned_sessions + unpinned_sessions

def markdown_to_html(text: str) -> str:
    if not text:
        return ""
    text = escape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Headings
    text = re.sub(r'^\s*### (.*)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*## (.*)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)

    # HR
    text = re.sub(r'(?:^|\n)-{3,}(?:\n|$)', r'\n<hr>\n', text, flags=re.MULTILINE)

    # Bold and Italic
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)

    # List items
    text = re.sub(r'^\s*-\s+(.*\S.*)$', r'<li>\1</li>', text, flags=re.MULTILINE)

    # Wrap consecutive <li> into <ul>
    def ul_wrap(m):
        # Correctly wrap list items without removing newlines between them.
        return f'<ul>{m.group(0)}</ul>'
    text = re.sub(r'(?:^|\n)(?:<li>.*?</li>\n?)+', ul_wrap, text, flags=re.DOTALL)

    # Double newlines => <br>
    text = re.sub(r'\n{2,}', r'<br>', text)
    return Markup(text)

def extract_image_filename(text: str):
    if not text:
        return None
    raw = re.sub(r'<[^>]+>', ' ', str(text))
    match = re.search(r'\*?([A-Za-z0-9_\-]+\.png)\*?', raw)
    if match:
        return safe_filename(match.group(1))
    return None

def get_tool_belt():
    return {
        "get_summary": get_summary,
        "get_financial_total": get_financial_total,
        "get_top_spending_category": get_top_spending_category,
        "find_peak_spending_day_for_category": find_peak_spending_day_for_category,
        "visualize_spending": visualize_spending,
        "find_transaction_date": find_transaction_date,
        "get_financial_advice": get_financial_advice,
        "add_transaction": add_transaction,
        "get_balance": get_balance,
        "check_budgets": check_budgets,
        "add_budget": add_budget,
        "check_goals": check_goals,
        "contribute_to_goal": contribute_to_goal,
        "calculate_savings_plan": calculate_savings_plan,
        "add_savings_goal": add_savings_goal,
        "identify_unnecessary_spending": identify_unnecessary_spending,
    }

def chatbot_response(user_query, current_data):
    try:
        plan = get_tool_call(user_query, current_data) or {}
    except Exception as e:
        return markdown_to_html(f"**Error planning tool call:** {e}")
    tool_name = plan.get("tool_name")
    arguments = plan.get("arguments") or {}
    if not isinstance(arguments, dict):
        arguments = {}

    if tool_name == "greeting_response":
        response = arguments.get("response", "Hi! How can I help?")
    else:
        tools = get_tool_belt()
        if tool_name in tools:
            fn = tools[tool_name]
            try:
                if "data" not in arguments:
                    arguments["data"] = current_data
                response = fn(**arguments)
            except Exception as e:
                response = f"An error occurred while running the tool: {e}"
        else:
            response = "I'm not sure how to do that. Please try rephrasing."
    return markdown_to_html(str(response))

@app.route("/media/<path:filename>")
def serve_media(filename):
    filename = safe_filename(filename)
    path = os.path.join(MEDIA_DIR, filename)
    if os.path.exists(path):
        return send_file(path, mimetype="image/png")
    return "File not found", 404

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>FinanceBot</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        :root {
            --bg: #ffffff;
            --panel: #f5f5f5;
            --border: #e0e0e0;
            --text: #000000;
            --text-muted: #666666;
            --brand: #007bff;
            --brand-dark: #0056b3;
            --radius-md: 8px;
            --pinned-bg: #e9ecef;
        }
        * { box-sizing: border-box; }
        html, body { height: 100%; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            margin: 0;
            background-color: var(--bg);
            color: var(--text);
            overflow: hidden;
        }

        /* Header */
        .header {
            position: fixed;
            inset: 0 0 auto 0;
            height: 60px;
            display: flex;
            align-items: center;
            z-index: 50;
            background-color: var(--panel);
            border-bottom: 1px solid var(--border);
            padding: 0 20px;
        }
        .brand-title h1 {
            font-size: 1.25rem;
            margin: 0;
            font-weight: 600;
        }

        /* Layout */
        .main {
            display: grid;
            grid-template-columns: 280px 1fr;
            height: 100vh;
            padding-top: 60px;
        }

        /* Sidebar */
        .sidebar {
            height: 100%;
            border-right: 1px solid var(--border);
            background-color: var(--panel);
            display: flex;
            flex-direction: column;
        }
        .sidebar-inner {
            flex: 1;
            overflow-y: auto;
            padding: 12px;
        }
        .sidebar-actions {
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border);
        }
        .btn-primary {
            width: 100%;
            padding: 10px;
            border-radius: var(--radius-md);
            border: 1px solid var(--brand);
            background-color: var(--brand);
            color: white;
            font-weight: 600;
            font-size: 1rem;
            cursor: pointer;
            transition: background-color .2s ease;
        }
        .btn-primary:hover { background-color: var(--brand-dark); }

        .sidebar-title {
            font-size: .8rem;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            padding: 16px 4px 8px 4px;
        }
        .sidebar-history { margin-top: 8px; display: grid; gap: 8px; }
        .sidebar-session {
            display: flex;
            align-items: center;
            justify-content: space-between;
            width: 100%;
            text-align: left;
            padding: 10px 12px;
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            cursor: pointer;
            transition: background-color .2s ease, border-color .2s ease;
        }
        .sidebar-session:hover { background-color: #e9ecef; }
        .sidebar-session.active {
            background-color: #e9ecef;
            border-color: var(--brand);
            font-weight: 600;
        }
        .sidebar-session.pinned {
            background-color: var(--pinned-bg);
        }
        .sidebar-session.active.pinned {
            background-color: #dde2e6;
        }
        .chat-title {
            overflow: hidden; white-space: nowrap; text-overflow: ellipsis;
            font-size: .95rem;
            flex: 1;
            padding-right: 8px;
        }
        .pin-indicator {
            display: inline-block;
            margin-right: 6px;
            font-size: .8em;
        }
        .chat-actions {
            display: flex;
            gap: 4px;
            opacity: 0;
            transition: opacity .2s ease;
        }
        .sidebar-session:hover .chat-actions {
            opacity: 1;
        }
        .chat-action-btn {
            background: none;
            border: none;
            cursor: pointer;
            padding: 4px;
            border-radius: 4px;
            font-size: 1rem;
            line-height: 1;
            color: var(--text-muted);
        }
        .chat-action-btn:hover {
            background-color: #dcdcdc;
            color: var(--text);
        }

        /* Chat area */
        .chat-section {
            display: flex;
            flex-direction: column;
            height: 100%;
            overflow: hidden; /* This ensures the container is constrained */
        }
        .chat-history {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
        }
        .chat-container {
            max-width: 800px;
            margin: 0 auto;
            width: 100%;
        }

        /* Welcome */
        .welcome {
            text-align: center;
            padding-top: 20vh;
        }
        .welcome h2 { font-size: 1.8rem; margin: 0; }
        .welcome p { color: var(--text-muted); font-size: 1.1rem; }

        /* Suggested Prompts */
        .suggestions-carousel {
            display: flex;
            gap: 16px;
            margin-top: 32px;
            justify-content: center;
            flex-wrap: wrap;
        }
        .suggestion-card {
            background-color: var(--panel);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 12px 16px;
            cursor: pointer;
            transition: background-color .2s ease, border-color .2s ease;
            font-size: .9rem;
            text-align: center;
            width: 220px;
        }
        .suggestion-card:hover {
            background-color: #e9ecef;
            border-color: var(--brand);
        }
        .suggestion-card span {
            color: var(--text-muted);
        }

        /* Bubbles */
        .bubble {
            max-width: 85%;
            padding: 12px 16px;
            margin-bottom: 16px;
            border-radius: 12px;
            font-size: 1rem;
            line-height: 1.6;
            word-wrap: break-word;
        }
        .user-bubble {
            margin-left: auto;
            background-color: #e9ecef;
            border-radius: 12px 12px 4px 12px;
        }
        .bot-bubble {
            margin-right: auto;
            background-color: var(--panel);
            border: 1px solid var(--border);
            border-radius: 12px 12px 12px 4px;
        }
        .bubble-head { font-weight: 700; margin-bottom: 4px; }
        .bot-bubble img {
            max-width: 100%;
            border-radius: var(--radius-md);
            border: 1px solid var(--border);
            margin-top: 10px;
        }

        /* Loading */
        .loading-bubble {
            display: inline-flex;
            gap: 6px;
            padding: 14px;
        }
        .dot-anim {
            width: 8px; height: 8px; background: var(--text-muted); border-radius: 50%;
            animation: blink 1.2s infinite;
        }
        .dot-anim:nth-child(2) { animation-delay: .2s; }
        .dot-anim:nth-child(3) { animation-delay: .4s; }
        @keyframes blink { 0%,80%,100%{opacity:.4} 40%{opacity:1} }

        /* Input bar */
        .input-bar {
            padding: 16px;
        }
        .input-wrap {
            display: flex;
            gap: 10px;
            max-width: 800px;
            margin: 0 auto;
        }
        .input-box {
            flex: 1;
            background: var(--bg);
            border: 2px solid var(--border);
            border-radius: var(--radius-md);
            padding: 10px 20px;
            font-size: 1rem;
            outline: none;
            transition: border-color .2s ease;
            height: 67px;
        }
        .input-box:focus { border-color: var(--brand); }
        .send-btn {
            padding: 10px 20px;
            border-radius: var(--radius-md);
            border: 1px solid var(--brand);
            background-color: var(--brand);
            color: white;
            font-weight: 600;
            cursor: pointer;
            transition: background-color .2s ease;
        }
        .send-btn:hover { background-color: var(--brand-dark); }
        
        /* Responsive */
        @media (max-width: 768px) {
            .main { grid-template-columns: 1fr; }
            .sidebar { display: none; }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="brand-title">
            <h1>FinanceBot</h1>
        </div>
    </div>

    <div class="main">
        <aside class="sidebar">
            <div class="sidebar-inner">
                <div class="sidebar-actions">
                    <button class="btn-primary" onclick="window.location.href='/?new_chat=1'">+ New Chat</button>
                </div>
                <div class="sidebar-title">Chats</div>
                <div class="sidebar-history" id="sidebar-history">
                    {% for cid in chat_sessions %}
                        {% set is_pinned = cid in pinned_chats %}
                        <div class="sidebar-session
                            {% if cid == chat_id %} active{% endif %}
                            {% if is_pinned %} pinned{% endif %}"
                            onclick="window.location.href='/?chat_id={{ cid }}'">
                            
                            <span class="chat-title">
                                {% if is_pinned %}
                                    <span class="pin-indicator">&#128204;</span>
                                {% endif %}
                                {{ get_chat_title(cid, is_pinned) }}
                            </span>

                            <span class="chat-actions" onclick="event.stopPropagation();">
                                <form method="post" action="/pin_chat/{{ cid }}" style="display:inline;">
                                    <button type="submit" class="chat-action-btn" title="{{ 'Unpin' if is_pinned else 'Pin' }}">
                                        {{ ('&#128205;' if is_pinned else '&#128204;') | safe }}
                                    </button>
                                </form>
                                <form method="post" action="/delete_chat/{{ cid }}" style="display:inline;" onsubmit="return confirm('Are you sure you want to delete this chat?');">
                                    <button type="submit" class="chat-action-btn" title="Delete">&#128465;</button>
                                </form>
                            </span>
                        </div>
                    {% endfor %}
                    {% if not chat_sessions %}
                        <div class="sidebar-session">
                            <div class="chat-title">No chats yet.</div>
                        </div>
                    {% endif %}
                </div>
            </div>
        </aside>

        <section class="chat-section">
            <div class="chat-history" id="chat-history">
                <div class="chat-container">
                    {% if chat_history %}
                        {% for msg in chat_history %}
                            {% if msg.role == 'user' %}
                                <div class="bubble user-bubble">
                                    <div class="bubble-head">You</div>
                                    <div class="bubble-content">{{ msg.text }}</div>
                                </div>
                            {% else %}
                                <div class="bubble bot-bubble">
                                    <div class="bubble-head">FinanceBot</div>
                                    <div class="bubble-content">
                                        {{ msg.text|safe }}
                                        {% if msg.image %}
                                            <img src="/{{ msg.image }}" alt="Generated Chart">
                                        {% endif %}
                                    </div>
                                </div>
                            {% endif %}
                        {% endfor %}
                    {% else %}
                        <div class="welcome">
                            <h2>Welcome to FinanceBot</h2>
                            <p>Your personal AI assistant for finance.</p>
                            <div class="suggestions-carousel">
                                <div class="suggestion-card" onclick="useSuggestion(this)">
                                    <span>“Summarize my spending this month”</span>
                                </div>
                                <div class="suggestion-card" onclick="useSuggestion(this)">
                                    <span>“Find my highest expense category”</span>
                                </div>
                                <div class="suggestion-card" onclick="useSuggestion(this)">
                                    <span>“Visualize my spending this month”</span>
                                </div>
                            </div>
                        </div>
                    {% endif %}
                </div>
            </div>

            <form id="chat-form" class="input-bar" autocomplete="off">
                <div class="input-wrap">
                    <input type="text" name="user_query" id="user_query" class="input-box" placeholder="Type your question..." autofocus required>
                    <input type="submit" class="send-btn" value="Ask">
                </div>
            </form>
        </section>
    </div>

    <script>
        function useSuggestion(card) {
            var promptText = card.querySelector('span').textContent.replace(/“|”/g, '');
            var input = document.getElementById('user_query');
            input.value = promptText;
            var form = document.getElementById('chat-form');
            if (form.requestSubmit) {
                form.requestSubmit();
            } else {
                form.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
            }
        }

        function scrollChatToBottom() {
            var chat = document.getElementById('chat-history');
            if (chat) {
                chat.scrollTop = chat.scrollHeight;
            }
        }
        window.onload = scrollChatToBottom;

        document.getElementById('chat-form').onsubmit = function(e) {
            e.preventDefault();
            var input = document.getElementById('user_query');
            var text = input.value.trim();
            if (!text) return;
            input.value = '';

            var chatContainer = document.querySelector('#chat-history .chat-container');
            if (!chatContainer) return;

            // Remove welcome message if it exists
            var welcome = chatContainer.querySelector('.welcome');
            if (welcome) {
                welcome.remove();
            }

            // Append user bubble
            var userBubble = document.createElement('div');
            userBubble.className = 'bubble user-bubble';
            userBubble.innerHTML = '<div class="bubble-head">You</div>'
                + '<div class="bubble-content">' + text.replace(/</g, "&lt;").replace(/>/g, "&gt;") + '</div>';
            chatContainer.appendChild(userBubble);

            // Loading bubble
            var loadingBubble = document.createElement('div');
            loadingBubble.className = 'bubble bot-bubble loading-bubble';
            loadingBubble.innerHTML = '<span class="dot-anim"></span><span class="dot-anim"></span><span class="dot-anim"></span>';
            chatContainer.appendChild(loadingBubble);
            scrollChatToBottom();

            // Perform POST
            fetch(`/?chat_id=${new URLSearchParams(window.location.search).get("chat_id") || ""}`, {
                method: "POST",
                headers: { "Content-Type": "application/x-www-form-urlencoded" },
                body: "user_query=" + encodeURIComponent(text)
            })
            .then(response => response.text())
            .then(html => {
                var parser = new DOMParser();
                var doc = parser.parseFromString(html, "text/html");
                var newChatContainer = doc.querySelector('#chat-history .chat-container');
                var newSidebar = doc.getElementById('sidebar-history');

                if (newChatContainer) {
                    chatContainer.innerHTML = newChatContainer.innerHTML;
                }
                var sidebar = document.getElementById('sidebar-history');
                if (sidebar && newSidebar) {
                    sidebar.innerHTML = newSidebar.innerHTML;
                }
                scrollChatToBottom();
            })
            .catch(() => {
                loadingBubble.textContent = "Error contacting server. Please try again.";
            });
        };
    </script>
</body>
</html>
"""

def get_or_create_chat(chat_id=None, new=False):
    """Unified chat session management"""
    if new or not chat_id:
        chat_id = str(uuid.uuid4())
        chat_history = []
        save_chat_history(chat_id, chat_history)
        return chat_id, chat_history
    
    try:
        chat_history = load_chat_history(chat_id)
        return chat_id, chat_history
    except Exception:
        return get_or_create_chat(new=True)

def get_chat_title(chat_id, is_pinned=False):
    history = load_chat_history(chat_id)
    for msg in history:
        if msg.get("role") == "user" and msg.get("text"):
            text = msg["text"]
            limit = 10 if is_pinned else 20
            return text[:limit] + ("..." if len(text) > limit else "")
    return "New Chat"

@app.route("/delete_chat/<chat_id>", methods=["POST"])
def delete_chat(chat_id):
    path = os.path.join(CHAT_DIR, f"{safe_filename(chat_id)}.json")
    if os.path.exists(path):
        os.remove(path)
    
    pinned_ids = load_pinned_chats()
    if chat_id in pinned_ids:
        pinned_ids.remove(chat_id)
        save_pinned_chats(pinned_ids)

    if session.get("chat_id") == chat_id:
        session.pop("chat_id", None)
        return redirect(url_for("home", new_chat=1))
        
    return redirect(url_for("home"))

@app.route("/pin_chat/<chat_id>", methods=["POST"])
def pin_chat(chat_id):
    pinned_ids = load_pinned_chats()
    if chat_id in pinned_ids:
        pinned_ids.remove(chat_id)
    else:
        pinned_ids.insert(0, chat_id)
    save_pinned_chats(pinned_ids)
    return redirect(url_for("home", chat_id=chat_id))

@app.route("/", methods=["GET", "POST"])
def home():
    if CACHE["data"] is None:
        try:
            CACHE["data"] = load_data()
        except Exception as e:
            print(f"Error loading data: {e}")
            CACHE["data"] = {}

    if request.method == "POST":
        chat_id = session.get("chat_id")
        if not chat_id:
            chat_id, _ = get_or_create_chat(new=True)
            session["chat_id"] = chat_id

        chat_history = load_chat_history(chat_id)
        user_query = (request.form.get("user_query") or "").strip()

        if user_query:
            chat_history.append({"role": "user", "text": user_query})
            try:
                bot_response = chatbot_response(user_query, CACHE["data"])
                image_filename = extract_image_filename(bot_response)
                image_path = None

                if image_filename:
                    source_path = os.path.join(BASE_DIR, image_filename)
                    if os.path.exists(source_path):
                        new_name = f"chart_{uuid.uuid4().hex[:8]}.png"
                        dest_path = os.path.join(MEDIA_DIR, new_name)
                        os.rename(source_path, dest_path)
                        image_path = f"media/{new_name}"
                
                chat_history.append({
                    "role": "bot",
                    "text": bot_response,
                    "image": image_path
                })
                save_chat_history(chat_id, chat_history)
            except Exception as e:
                print(f"Error processing message: {e}")
                chat_history.append({
                    "role": "bot",
                    "text": "Sorry, I encountered an error.",
                    "image": None
                })
    
    query_chat_id = request.args.get("chat_id")
    new_chat = request.args.get("new_chat") == "1"

    if new_chat:
        chat_id, chat_history = get_or_create_chat(new=True)
    else:
        chat_id, chat_history = get_or_create_chat(
            chat_id=query_chat_id or session.get("chat_id")
        )
    session["chat_id"] = chat_id

    pinned_list = load_pinned_chats()
    return render_template_string(
        HTML_TEMPLATE,
        chat_history=chat_history,
        chat_sessions=list_chat_sessions(),
        chat_id=chat_id,
        get_chat_title=get_chat_title,
        pinned_chats=pinned_list
    )

if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 5000))
    print(f"FinanceBot running locally: http://{host}:{port}/")
    app.run(debug=True, host=host, port=port)