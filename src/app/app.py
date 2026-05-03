from pathlib import Path
from functools import lru_cache, wraps
import json
import os
import re
import sqlite3
import time

import joblib
import pandas as pd
from flask import (
    Flask,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT_DIR / "dataset" / "complaints.csv"
MODEL_PATH = ROOT_DIR / "models" / "issue_classifier.joblib"
ENCODER_PATH = ROOT_DIR / "models" / "label_encoder.joblib"
MODEL_COMPARISON_PATH = ROOT_DIR / "outputs" / "reports" / "model_comparison.txt"
PROPOSAL_DB_PATH = ROOT_DIR / "data" / "portal.db"

TEXT_COLUMN = "Consumer complaint narrative"
TARGET_COLUMN = "Issue"
LIGHTWEIGHT_DASHBOARD_MAX_MB = 250
DASHBOARD_CACHE_TTL_SECONDS = 60
POSITIVE_TERMS = {"resolved", "helpful", "satisfied", "thank", "fixed", "refund processed"}
NEGATIVE_TERMS = {
    "fraud",
    "scam",
    "unauthorized",
    "charged",
    "complaint",
    "harassed",
    "threat",
    "refused",
    "error",
    "incorrect",
    "dispute",
    "delay",
}
URGENT_TERMS = {
    "urgent",
    "asap",
    "immediately",
    "lawsuit",
    "legal action",
    "police",
    "identity theft",
    "fraud",
    "unauthorized",
    "harassed",
}
TRIAGE_KEYWORD_WEIGHTS = {
    # Legal/regulatory risk
    "legal": 16,
    "lawsuit": 18,
    "attorney": 12,
    "court": 12,
    "regulator": 10,
    "complaint filed": 10,
    # Safety and security risk
    "safety": 18,
    "unsafe": 18,
    "identity theft": 20,
    "data breach": 20,
    "fraud": 18,
    "scam": 14,
    "unauthorized": 14,
    "threat": 14,
    "harassed": 12,
    # Financial harm and resolution blockers
    "refund": 10,
    "chargeback": 10,
    "charged twice": 12,
    "overcharged": 10,
    "dispute": 10,
    "stolen": 12,
    "locked account": 12,
    "account frozen": 12,
    "foreclosure": 14,
}
ESCALATION_PHRASES = {"urgent", "asap", "immediately", "right now", "today", "now"}

app = Flask(__name__)
app.secret_key = "cognisense-cx-dev-key"

DEFAULT_USER = {"username": "admin", "password": "admin123"}
dashboard_cache = {"summary": None, "cached_at": 0.0}
LANDING_SECTIONS = {
    "analytics": {
        "title": "Analytics Overview",
        "description": "Track dataset volume, model readiness, and top complaint patterns in one view.",
        "bullets": [
            "Instant dashboard summary for quick project demos.",
            "Lightweight mode keeps the app responsive on very large datasets.",
            "Focuses on the metrics most useful for presentation.",
        ],
    },
    "models": {
        "title": "Model Pipeline",
        "description": "TF-IDF plus Logistic Regression powers fast and interpretable text classification.",
        "bullets": [
            "Memory-safe chunked training for large CSV files.",
            "Saved artifacts include classifier and label encoder.",
            "Evaluation outputs include classification report and confusion matrix.",
        ],
    },
    "insights": {
        "title": "Customer Insights",
        "description": "Convert complaint narratives into categories that reveal operational pain points.",
        "bullets": [
            "Highlights recurring complaint themes for decision-making.",
            "Supports faster triage and summary of customer issues.",
            "Turns unstructured text into presentation-ready insights.",
        ],
    },
    "explore": {
        "title": "Explore Features",
        "description": "Try prediction, upload workflows, and health endpoints before your presentation.",
        "bullets": [
            "Use /warmup before demo to reduce first-load latency.",
            "Test predictions using sample complaint narratives.",
            "Verify deployment readiness via the health endpoint.",
        ],
    },
}

# Consumer-facing reference cards (Encyclopedia) — aligned with common CFPB-style themes.
ISSUE_REFERENCE = [
    {
        "title": "Debt collection",
        "signals": "Repeated calls, threats of legal action, wrong balance, or collector contacting others about your debt.",
        "action": "Log every call, validate the debt in writing, and escalate to the CFPB if harassment or misrepresentation continues.",
    },
    {
        "title": "Credit reporting",
        "signals": "Accounts you do not recognize, incorrect balances, outdated negative marks, or disputes ignored after 30 days.",
        "action": "File disputes with each bureau with evidence; keep certified-mail receipts and follow up until tradelines are corrected.",
    },
    {
        "title": "Mortgage / loan servicing",
        "signals": "Misapplied payments, escrow surprises, slow loss-mitigation responses, or unclear payoff quotes.",
        "action": "Request a complete payment history in writing and escalate through the servicer’s appeals process with dates and amounts.",
    },
    {
        "title": "Credit card / prepaid",
        "signals": "Unauthorized charges, denied fraud claims, surprise fees, or rewards not posting as advertised.",
        "action": "Dispute in writing within billing timelines; keep charge receipts and note every call reference number.",
    },
    {
        "title": "Bank account / transfers",
        "signals": "ACH errors, holds without explanation, Zelle scams, or ATM deposit delays affecting bills.",
        "action": "Ask for provisional credit rules, submit a written fraud affidavit, and document timelines for Regulation E claims.",
    },
    {
        "title": "Student loan",
        "signals": "Income-driven repayment miscounts, wrong servicer, cosigner release denied without clear criteria.",
        "action": "Request NSLDS history, keep IDR certification records, and appeal with a clear month-by-month payment table.",
    },
    {
        "title": "Checking / savings",
        "signals": "Overdraft stacking, early direct-deposit reversals, or branch errors not reversed after investigation.",
        "action": "Request the fee schedule used, compare to your opt-in status, and ask for a written investigation outcome.",
    },
    {
        "title": "Vehicle / title loan",
        "signals": "Repo timing disputes, insurance-force-placement, or payoff amounts that do not match amortization schedules.",
        "action": "Demand a full amortization schedule and state notices; verify insurance coverage dates against lender letters.",
    },
]


def get_db_connection() -> sqlite3.Connection:
    PROPOSAL_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(PROPOSAL_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_portal_db() -> None:
    conn = get_db_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin','client')),
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            user_name TEXT NOT NULL,
            client_name TEXT NOT NULL,
            title TEXT NOT NULL,
            text TEXT NOT NULL,
            status TEXT NOT NULL,
            sentiment_json TEXT,
            classification_json TEXT,
            priority_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            complaint_id INTEGER NOT NULL,
            sender_id INTEGER NOT NULL,
            sender_name TEXT NOT NULL,
            sender_role TEXT NOT NULL,
            text TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(complaint_id) REFERENCES complaints(id),
            FOREIGN KEY(sender_id) REFERENCES users(id)
        )
        """
    )
    default_users = [
        ("Admin", "admin@cognisense.com", "admin123", "admin"),
        ("Jane Doe", "jane@demo.com", "client123", "client"),
        ("Carlos Vega", "carlos@demo.com", "client123", "client"),
    ]
    for name, email, password, role in default_users:
        conn.execute(
            """
            INSERT OR IGNORE INTO users(name, email, password, role, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (name, email, password, role, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        )
    conn.commit()
    conn.close()


init_portal_db()


@app.before_request
def start_request_timer():
    g.start_time = time.perf_counter()


@app.after_request
def log_request_timing(response):
    start_time = getattr(g, "start_time", None)
    if start_time is not None:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        print(f"[REQ] {request.method} {request.path} -> {response.status_code} in {elapsed_ms:.1f}ms")
    return response


def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return func(*args, **kwargs)

    return wrapper


def portal_auth_required(required_role: str | None = None):
    user_id = session.get("portal_user_id")
    role = session.get("portal_user_role")
    if not user_id:
        return False
    if required_role and role != required_role:
        return False
    return True


def parse_json_field(value: str | None):
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def serialize_complaint(row: sqlite3.Row) -> dict:
    return {
        "id": f"c_{row['id']}",
        "db_id": row["id"],
        "userId": f"u_{row['user_id']}",
        "userName": row["user_name"],
        "clientName": row["client_name"],
        "title": row["title"],
        "description": row["text"],
        "text": row["text"],
        "status": row["status"],
        "sentiment": parse_json_field(row["sentiment_json"]),
        "classification": parse_json_field(row["classification_json"]),
        "priority": parse_json_field(row["priority_json"]),
        "createdAt": row["created_at"],
        "timestamp": row["created_at"],
        "updatedAt": row["updated_at"],
    }


@lru_cache(maxsize=1)
def load_model_assets():
    if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
        return None, None
    return joblib.load(MODEL_PATH), joblib.load(ENCODER_PATH)


def parse_model_accuracy_percent() -> int | None:
    """Read held-out accuracy for the selected model from training comparison output."""
    if not MODEL_COMPARISON_PATH.exists():
        return None
    text = MODEL_COMPARISON_PATH.read_text(encoding="utf-8")
    selected = "LogisticRegression"
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("- selected model"):
            parts = line.split(":", 1)
            if len(parts) > 1:
                selected = parts[1].strip()
            break
    acc_re = re.compile(r"accuracy=([\d.]+)")
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("- ") or "accuracy=" not in line or ":" not in line:
            continue
        name_part = line[2 : line.index(":")].strip()
        if name_part != selected:
            continue
        m = acc_re.search(line)
        if m:
            return int(round(float(m.group(1)) * 100))
    return None


def compute_class_probabilities(
    model, label_encoder, narrative: str, top_k: int = 20
) -> list[tuple[str, float]]:
    """Return (label, probability) sorted descending; empty if unavailable."""
    input_df = pd.DataFrame([{TEXT_COLUMN: narrative}])
    if not hasattr(model, "predict_proba"):
        return []
    try:
        proba = model.predict_proba(input_df)[0]
    except Exception:
        return []
    classes = getattr(model, "classes_", None)
    if classes is None or len(classes) != len(proba):
        return []
    try:
        names = label_encoder.inverse_transform(classes)
    except Exception:
        return []
    pairs = list(zip(names, (float(p) for p in proba)))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:top_k]


def compute_dashboard_summary():
    summary = {
        "total_rows": 0,
        "has_model": MODEL_PATH.exists(),
        "top_issues": [],
        "dataset_size_mb": 0,
        "dashboard_note": "",
        "model_comparison_note": "",
    }
    if MODEL_COMPARISON_PATH.exists():
        lines = [ln.strip() for ln in MODEL_COMPARISON_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]
        selected_line = next((ln for ln in lines if ln.lower().startswith("- selected model")), "")
        summary["model_comparison_note"] = selected_line.replace("- ", "") if selected_line else "Model comparison available in outputs/reports/model_comparison.txt"
    if DATASET_PATH.exists():
        file_size_mb = round(DATASET_PATH.stat().st_size / (1024 * 1024), 2)
        summary["dataset_size_mb"] = file_size_mb

        # Avoid blocking login/dashboard for very large files.
        if file_size_mb > LIGHTWEIGHT_DASHBOARD_MAX_MB:
            summary["dashboard_note"] = (
                "Dataset is very large, so dashboard uses lightweight mode for speed. "
                "Detailed issue counts are skipped."
            )
            summary["total_rows"] = "Large dataset (count skipped)"
        else:
            df = pd.read_csv(DATASET_PATH, low_memory=False)
            summary["total_rows"] = len(df)
            if TARGET_COLUMN in df.columns:
                issue_counts = df[TARGET_COLUMN].value_counts().head(10)
                summary["top_issues"] = issue_counts.items()
    return summary


def get_dashboard_summary():
    now = time.time()
    if (
        dashboard_cache["summary"] is not None
        and now - dashboard_cache["cached_at"] <= DASHBOARD_CACHE_TTL_SECONDS
    ):
        return dashboard_cache["summary"]

    summary = compute_dashboard_summary()
    dashboard_cache["summary"] = summary
    dashboard_cache["cached_at"] = now
    return summary


def infer_sentiment(text: str) -> tuple[str, float]:
    lowered = text.lower()
    pos_hits = sum(1 for t in POSITIVE_TERMS if t in lowered)
    neg_hits = sum(1 for t in NEGATIVE_TERMS if t in lowered)
    score = (pos_hits - neg_hits) / max(1, pos_hits + neg_hits)
    if score > 0.2:
        return "Positive", score
    if score < -0.2:
        return "Negative", score
    return "Neutral", score


def _count_term_occurrences(text: str, term: str) -> int:
    if " " in term:
        return text.count(term)
    return len(re.findall(rf"\b{re.escape(term)}\b", text))


def infer_urgency(text: str, sentiment_label: str, sentiment_score: float = 0.0) -> tuple[str, int]:
    lowered = text.lower()

    weighted_keyword_score = 0
    for term, weight in TRIAGE_KEYWORD_WEIGHTS.items():
        hits = _count_term_occurrences(lowered, term)
        if hits:
            # Diminishing returns for repeated mention of the same term.
            weighted_keyword_score += min(2, hits) * weight

    escalation_hits = sum(1 for phrase in ESCALATION_PHRASES if phrase in lowered)
    urgent_hits = sum(1 for t in URGENT_TERMS if t in lowered)
    exclamations = text.count("!")
    all_caps_tokens = sum(1 for token in text.split() if len(token) >= 4 and token.isupper())

    sentiment_risk = max(0.0, -sentiment_score)
    sentiment_component = int(round(sentiment_risk * 30))
    keyword_component = min(55, weighted_keyword_score)
    escalation_component = min(20, escalation_hits * 6 + urgent_hits * 4)
    intensity_component = min(12, exclamations * 2 + all_caps_tokens * 2)

    score = keyword_component + escalation_component + sentiment_component + intensity_component
    if sentiment_label == "Negative":
        score += 8
    score = min(100, score)

    if score >= 70:
        return "High", score
    if score >= 40:
        return "Medium", score
    return "Low", score


@app.route("/")
def root():
    if session.get("logged_in"):
        return redirect(url_for("home"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        if username == DEFAULT_USER["username"] and password == DEFAULT_USER["password"]:
            session["logged_in"] = True
            return redirect(url_for("home"))
        flash("Invalid credentials. Try admin / admin123", "error")
    return render_template("login.html")


@app.route("/landing/<section>")
def landing_info(section: str):
    section_data = LANDING_SECTIONS.get(section)
    if section_data is None:
        return redirect(url_for("login"))
    return render_template("landing_info.html", section=section_data)


@app.route("/proposal")
def proposal_home():
    return redirect(url_for("proposal_page", page="index.html"))


@app.route("/proposal/<path:page>")
def proposal_page(page: str):
    if page.startswith("css/") or page.startswith("js/"):
        return send_from_directory(ROOT_DIR / "src" / "app" / "static" / "proposal", page)
    if not page.endswith(".html"):
        return "Unsupported asset", 404
    return render_template(f"proposal/{page}")


@app.route("/api/auth/session")
def api_auth_session():
    if not session.get("portal_user_id"):
        return jsonify({"session": None})
    return jsonify(
        {
            "session": {
                "id": f"u_{session['portal_user_id']}",
                "name": session.get("portal_user_name"),
                "email": session.get("portal_user_email"),
                "role": session.get("portal_user_role"),
            }
        }
    )


@app.route("/api/auth/login", methods=["POST"])
def api_auth_login():
    payload = request.get_json(silent=True) or {}
    email = (payload.get("email") or "").strip().lower()
    password = payload.get("password") or ""
    conn = get_db_connection()
    user = conn.execute(
        "SELECT id, name, email, role FROM users WHERE email = ? AND password = ?",
        (email, password),
    ).fetchone()
    conn.close()
    if not user:
        return jsonify({"ok": False, "error": "Invalid email or password."}), 401
    session["portal_user_id"] = user["id"]
    session["portal_user_name"] = user["name"]
    session["portal_user_email"] = user["email"]
    session["portal_user_role"] = user["role"]
    return jsonify(
        {
            "ok": True,
            "user": {
                "id": f"u_{user['id']}",
                "name": user["name"],
                "email": user["email"],
                "role": user["role"],
            },
        }
    )


@app.route("/api/auth/register", methods=["POST"])
def api_auth_register():
    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "").strip()
    email = (payload.get("email") or "").strip().lower()
    password = payload.get("password") or ""
    role = (payload.get("role") or "client").strip().lower()
    if role not in {"client", "admin"}:
        role = "client"
    if not name or not email or len(password) < 6:
        return jsonify({"ok": False, "error": "Invalid registration input."}), 400
    conn = get_db_connection()
    try:
        cur = conn.execute(
            "INSERT INTO users(name, email, password, role, created_at) VALUES (?, ?, ?, ?, ?)",
            (name, email, password, role, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"ok": False, "error": "Email already registered."}), 409
    user_id = cur.lastrowid
    conn.close()
    session["portal_user_id"] = user_id
    session["portal_user_name"] = name
    session["portal_user_email"] = email
    session["portal_user_role"] = role
    return jsonify({"ok": True, "user": {"id": f"u_{user_id}", "name": name, "email": email, "role": role}})


@app.route("/api/auth/logout", methods=["POST"])
def api_auth_logout():
    for key in ["portal_user_id", "portal_user_name", "portal_user_email", "portal_user_role"]:
        session.pop(key, None)
    return jsonify({"ok": True})


@app.route("/api/users")
def api_users():
    if not portal_auth_required():
        return jsonify({"error": "Unauthorized"}), 401
    conn = get_db_connection()
    rows = conn.execute("SELECT id, name, email, role, created_at FROM users ORDER BY id ASC").fetchall()
    conn.close()
    return jsonify(
        {
            "users": [
                {
                    "id": f"u_{r['id']}",
                    "name": r["name"],
                    "email": r["email"],
                    "role": r["role"],
                    "createdAt": r["created_at"],
                }
                for r in rows
            ]
        }
    )


@app.route("/api/complaints", methods=["GET"])
def api_get_complaints():
    if not portal_auth_required():
        return jsonify({"error": "Unauthorized"}), 401
    role = session.get("portal_user_role")
    user_id = session.get("portal_user_id")
    conn = get_db_connection()
    if role == "admin":
        rows = conn.execute("SELECT * FROM complaints ORDER BY id DESC").fetchall()
    else:
        rows = conn.execute("SELECT * FROM complaints WHERE user_id = ? ORDER BY id DESC", (user_id,)).fetchall()
    conn.close()
    return jsonify({"complaints": [serialize_complaint(r) for r in rows]})


@app.route("/api/complaints", methods=["POST"])
def api_create_complaint():
    if not portal_auth_required():
        return jsonify({"error": "Unauthorized"}), 401
    payload = request.get_json(silent=True) or {}
    title = (payload.get("title") or "").strip()
    text = (payload.get("text") or payload.get("description") or "").strip()
    if not title or len(text) < 3:
        return jsonify({"error": "Title and complaint text are required."}), 400
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    role = session.get("portal_user_role")
    client_name = (payload.get("clientName") or session.get("portal_user_name") or "Client").strip()
    owner_user_id = session["portal_user_id"]
    owner_user_name = session.get("portal_user_name", "Client")
    if role == "admin":
        # Allow admin to create proactive outbound tickets while keeping ownership explicit.
        owner_user_name = client_name
    conn = get_db_connection()
    cur = conn.execute(
        """
        INSERT INTO complaints(
            user_id, user_name, client_name, title, text, status,
            sentiment_json, classification_json, priority_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            owner_user_id,
            owner_user_name,
            client_name,
            title,
            text,
            "Pending Analysis",
            None,
            None,
            None,
            now,
            now,
        ),
    )
    complaint_id = cur.lastrowid
    conn.execute(
        """
        INSERT INTO messages(complaint_id, sender_id, sender_name, sender_role, text, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            complaint_id,
            session["portal_user_id"],
            session.get("portal_user_name", "Client"),
            "admin" if role == "admin" else "client",
            text,
            now,
        ),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM complaints WHERE id = ?", (complaint_id,)).fetchone()
    conn.close()
    return jsonify({"ok": True, "complaint": serialize_complaint(row)})


@app.route("/api/complaints/<complaint_id>/status", methods=["PATCH"])
def api_update_complaint_status(complaint_id: str):
    if not portal_auth_required("admin"):
        return jsonify({"error": "Unauthorized"}), 401
    db_id = int(complaint_id.replace("c_", ""))
    status = (request.get_json(silent=True) or {}).get("status", "").strip()
    if not status:
        return jsonify({"error": "Status is required"}), 400
    conn = get_db_connection()
    conn.execute(
        "UPDATE complaints SET status = ?, updated_at = ? WHERE id = ?",
        (status, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), db_id),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM complaints WHERE id = ?", (db_id,)).fetchone()
    conn.close()
    return jsonify({"ok": True, "complaint": serialize_complaint(row)})


@app.route("/api/complaints/<complaint_id>/messages", methods=["GET"])
def api_get_messages(complaint_id: str):
    if not portal_auth_required():
        return jsonify({"error": "Unauthorized"}), 401
    db_id = int(complaint_id.replace("c_", ""))
    conn = get_db_connection()
    complaint = conn.execute("SELECT * FROM complaints WHERE id = ?", (db_id,)).fetchone()
    if complaint is None:
        conn.close()
        return jsonify({"error": "Complaint not found"}), 404
    if session.get("portal_user_role") != "admin" and complaint["user_id"] != session.get("portal_user_id"):
        conn.close()
        return jsonify({"error": "Forbidden"}), 403
    rows = conn.execute(
        """
        SELECT id, sender_id, sender_name, sender_role, text, created_at
        FROM messages WHERE complaint_id = ? ORDER BY id ASC
        """,
        (db_id,),
    ).fetchall()
    conn.close()
    return jsonify(
        {
            "messages": [
                {
                    "id": f"m_{r['id']}",
                    "senderId": f"u_{r['sender_id']}",
                    "senderName": r["sender_name"],
                    "senderRole": r["sender_role"],
                    "text": r["text"],
                    "time": r["created_at"],
                    "timestamp": r["created_at"],
                }
                for r in rows
            ]
        }
    )


@app.route("/api/complaints/<complaint_id>/messages", methods=["POST"])
def api_post_message(complaint_id: str):
    if not portal_auth_required():
        return jsonify({"error": "Unauthorized"}), 401
    db_id = int(complaint_id.replace("c_", ""))
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Message text is required"}), 400
    role = session.get("portal_user_role")
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    conn = get_db_connection()
    complaint = conn.execute("SELECT * FROM complaints WHERE id = ?", (db_id,)).fetchone()
    if complaint is None:
        conn.close()
        return jsonify({"error": "Complaint not found"}), 404
    if role != "admin" and complaint["user_id"] != session.get("portal_user_id"):
        conn.close()
        return jsonify({"error": "Forbidden"}), 403
    sender_name = session.get("portal_user_name", "User")
    sender_role = "admin" if role == "admin" else "client"
    cur = conn.execute(
        """
        INSERT INTO messages(complaint_id, sender_id, sender_name, sender_role, text, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (db_id, session["portal_user_id"], sender_name, sender_role, text, now),
    )
    if role == "admin" and complaint["status"] in {"Pending", "Pending Analysis"}:
        conn.execute("UPDATE complaints SET status = ?, updated_at = ? WHERE id = ?", ("In Progress", now, db_id))
    conn.commit()
    conn.close()
    return jsonify(
        {
            "ok": True,
            "message": {
                "id": f"m_{cur.lastrowid}",
                "senderId": f"u_{session['portal_user_id']}",
                "senderName": sender_name,
                "senderRole": sender_role,
                "text": text,
                "time": now,
                "timestamp": now,
            },
        }
    )


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    if not portal_auth_required():
        return jsonify({"error": "Unauthorized"}), 401
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Text is required"}), 400
    model, label_encoder = load_model_assets()
    prediction = "General Inquiry"
    if model is not None and label_encoder is not None:
        pred_index = model.predict(pd.DataFrame([{TEXT_COLUMN: text}]))[0]
        prediction = label_encoder.inverse_transform([pred_index])[0]
    sentiment_label, sentiment_score = infer_sentiment(text)
    urgency_label, urgency_score = infer_urgency(text, sentiment_label, sentiment_score)
    result = {
        "sentiment": {
            "label": sentiment_label,
            "emoji": {"Positive": "😊", "Negative": "😠", "Neutral": "😐"}[sentiment_label],
            "cssClass": sentiment_label.lower(),
            "confidence": int(round((abs(sentiment_score) * 50) + 50)),
            "score": round(sentiment_score, 3),
        },
        "classification": {
            "category": prediction,
            "icon": "🗂️",
            "confidence": 80,
        },
        "priority": {
            "level": urgency_label,
            "icon": {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[urgency_label],
            "cssClass": urgency_label.lower(),
            "score": urgency_score,
        },
    }
    return jsonify({"ok": True, "result": result})


@app.route("/api/complaints/<complaint_id>/analyze", methods=["POST"])
def api_analyze_complaint(complaint_id: str):
    if not portal_auth_required("admin"):
        return jsonify({"error": "Unauthorized"}), 401
    db_id = int(complaint_id.replace("c_", ""))
    conn = get_db_connection()
    complaint = conn.execute("SELECT * FROM complaints WHERE id = ?", (db_id,)).fetchone()
    if complaint is None:
        conn.close()
        return jsonify({"error": "Complaint not found"}), 404
    text = complaint["text"]
    model, label_encoder = load_model_assets()
    prediction = "General Inquiry"
    if model is not None and label_encoder is not None:
        pred_index = model.predict(pd.DataFrame([{TEXT_COLUMN: text}]))[0]
        prediction = label_encoder.inverse_transform([pred_index])[0]
    sentiment_label, sentiment_score = infer_sentiment(text)
    urgency_label, urgency_score = infer_urgency(text, sentiment_label, sentiment_score)
    sentiment = {
        "label": sentiment_label,
        "emoji": {"Positive": "😊", "Negative": "😠", "Neutral": "😐"}[sentiment_label],
        "cssClass": sentiment_label.lower(),
        "confidence": int(round((abs(sentiment_score) * 50) + 50)),
        "score": round(sentiment_score, 3),
    }
    classification = {"category": prediction, "icon": "🗂️", "confidence": 80}
    priority = {
        "level": urgency_label,
        "icon": {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[urgency_label],
        "cssClass": urgency_label.lower(),
        "score": urgency_score,
    }
    conn.execute(
        """
        UPDATE complaints
        SET sentiment_json = ?, classification_json = ?, priority_json = ?, status = ?, updated_at = ?
        WHERE id = ?
        """,
        (
            json.dumps(sentiment),
            json.dumps(classification),
            json.dumps(priority),
            "Pending",
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            db_id,
        ),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM complaints WHERE id = ?", (db_id,)).fetchone()
    conn.close()
    return jsonify({"ok": True, "complaint": serialize_complaint(row)})


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    return redirect(url_for("home"))


@app.route("/home")
@login_required
def home():
    summary = get_dashboard_summary()
    accuracy_pct = parse_model_accuracy_percent()
    return render_template("home.html", summary=summary, accuracy_pct=accuracy_pct)


@app.route("/encyclopedia")
@login_required
def encyclopedia():
    return render_template("encyclopedia.html", issues=ISSUE_REFERENCE)


@app.route("/history")
@login_required
def history():
    return render_template("history.html")


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    prediction = None
    sentiment_label = None
    sentiment_score = None
    urgency_label = None
    urgency_score = None
    probability_rows: list[dict[str, float | str]] = []
    history_payload = None
    narrative = ""
    model, label_encoder = load_model_assets()
    if request.method == "POST":
        narrative = request.form.get("narrative", "").strip()
        if not narrative:
            flash("Please enter complaint text.", "error")
        elif model is None or label_encoder is None:
            flash("Model not found. Run training first.", "error")
        else:
            input_df = pd.DataFrame([{TEXT_COLUMN: narrative}])
            pred_index = model.predict(input_df)[0]
            prediction = label_encoder.inverse_transform([pred_index])[0]
            sentiment_label, sentiment_score = infer_sentiment(narrative)
            urgency_label, urgency_score = infer_urgency(
                narrative, sentiment_label, sentiment_score
            )
            pairs = compute_class_probabilities(model, label_encoder, narrative, top_k=25)
            probability_rows = [
                {"label": str(lbl), "pct": round(prob * 100.0, 2)} for lbl, prob in pairs
            ]
            top_prob = pairs[0][1] if pairs else 0.0
            history_payload = {
                "id": f"{time.time():.0f}_{os.urandom(3).hex()}",
                "prediction": prediction,
                "confidence": round(float(top_prob) * 100.0, 2),
                "snippet": narrative[:220],
                "sentiment": sentiment_label,
                "urgency": urgency_label,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            }
    return render_template(
        "predict.html",
        narrative=narrative,
        prediction=prediction,
        sentiment_label=sentiment_label,
        sentiment_score=sentiment_score,
        urgency_label=urgency_label,
        urgency_score=urgency_score,
        probability_rows=probability_rows,
        history_payload=history_payload,
    )


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        file_obj = request.files.get("csv_file")
        if file_obj is None or file_obj.filename == "":
            flash("Please choose a CSV file.", "error")
            return redirect(url_for("upload"))

        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        file_obj.save(DATASET_PATH)
        # Uploaded dataset may trigger future retraining; keep cache behavior explicit.
        load_model_assets.cache_clear()
        dashboard_cache["summary"] = None
        dashboard_cache["cached_at"] = 0.0
        flash(f"Dataset uploaded: {DATASET_PATH.name}", "success")
        return redirect(url_for("home"))

    return render_template("upload.html")


@app.route("/health")
def health():
    model, encoder = load_model_assets()
    return jsonify(
        {
            "status": "ok",
            "model_ready": model is not None and encoder is not None,
            "dataset_exists": DATASET_PATH.exists(),
            "dataset_size_mb": round(DATASET_PATH.stat().st_size / (1024 * 1024), 2)
            if DATASET_PATH.exists()
            else 0,
        }
    )


@app.route("/warmup")
def warmup():
    load_model_assets.cache_clear()
    model, encoder = load_model_assets()
    # Warm dashboard cache too for smoother first paint.
    dashboard_cache["summary"] = compute_dashboard_summary()
    dashboard_cache["cached_at"] = time.time()
    return jsonify(
        {
            "status": "warmed",
            "model_ready": model is not None and encoder is not None,
            "dashboard_cached": True,
        }
    )


if __name__ == "__main__":
    # Faster and more stable localhost defaults.
    init_portal_db()
    model, encoder = load_model_assets()
    if model is not None and encoder is not None:
        print("Model cache warmed.")

    if os.getenv("USE_WAITRESS", "1") == "1":
        try:
            from waitress import serve

            serve(app, host="127.0.0.1", port=5000, threads=8)
        except ImportError:
            print("Waitress not installed, falling back to Flask dev server.")
            app.run(
                debug=os.getenv("FLASK_DEBUG", "0") == "1",
                use_reloader=False,
                threaded=True,
            )
    else:
        app.run(
            debug=os.getenv("FLASK_DEBUG", "0") == "1",
            use_reloader=False,
            threaded=True,
        )
