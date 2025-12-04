from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3
from functools import wraps
import csv
import io
import base64

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = "super_secret_key"
DB_NAME = "database.db"


def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as db:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL
            )
            """
        )

        db.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                input_type TEXT NOT NULL,
                original_text TEXT NOT NULL,
                cleaned_text TEXT,
                sentiment_label TEXT NOT NULL,
                score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )
        db.commit()


def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    return wrapper


def has_arabic(text: str) -> bool:
    for ch in text:
        if "\u0600" <= ch <= "\u06FF":
            return True
    return False


positive_ar = [
    "اعجبني",
    "أعجبني",
    "حلو",
    "جيد",
    "ممتاز",
    "رائع",
    "مريح",
    "سريع",
    "جميل",
    "ممتازة",
    "افضل",
    "أفضل",
    "احب",
]

negative_ar = [
    "سيئ",
    "سئ",
    "سء",
    "سيء",
    "لم يعجبني",
    "لا يعجبني",
    "سيئة",
    "بطئ",
    "بطيء",
    "غالي",
    "سعره غالي",
    "رديء",
    "رديئة",
    "تجربة سيئة",
]

positive_en = [
    "i like",
    "i like this",
    "like this",
    "like it",
    "good",
    "great",
    "excellent",
    "perfect",
    "love",
    "loved",
    "like",
    "liked",
    "amazing",
    "fast",
    "nice",
    "happy",
    "satisfied",
]

negative_en = [
    "bad",
    "very bad",
    "terrible",
    "awful",
    "slow",
    "hate",
    "dislike",
    "poor",
    "worst",
    "angry",
    "upset",
    "not good",
    "disappointed",
]


def simple_sentiment(text: str):
    text_lower = text.lower().strip()

    if has_arabic(text_lower):
        pos_list = positive_ar
        neg_list = negative_ar
    else:
        pos_list = positive_en
        neg_list = negative_en

    score = 0
    for word in pos_list:
        if word in text_lower:
            score += 1
    for word in neg_list:
        if word in text_lower:
            score -= 1

    if score > 0:
        label = "Positive"
    elif score < 0:
        label = "Negative"
    else:
        label = "Neutral"

    return label, float(score)


def build_chart_base64(positive, negative, neutral):
    labels = ["Positive", "Negative", "Neutral"]
    values = [positive, negative, neutral]

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(labels, values)
    ax.set_ylabel("Count")

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()

        if not username or not email or not password:
            return render_template(
                "register.html", error="Please fill in all required fields."
            )

        db = get_db()
        exists = db.execute(
            "SELECT id FROM users WHERE email = ?", (email,)
        ).fetchone()
        if exists:
            return render_template("register.html", error="Email already exists.")

        db.execute(
            "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
            (username, email, password),
        )
        db.commit()
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()

        db = get_db()
        user = db.execute(
            "SELECT * FROM users WHERE email = ? AND password = ?",
            (email, password),
        ).fetchone()

        if user:
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["email"] = user["email"]
            return redirect(url_for("index"))

        return render_template(
            "login.html", error="Incorrect email or password."
        )

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    return render_template("index.html")


@app.route("/analyze_text", methods=["POST"])
@login_required
def analyze_text_route():
    feedback = request.form.get("feedback", "").strip()
    if not feedback:
        return render_template(
            "index.html", error_text="Please enter a customer feedback sentence."
        )

    label, score = simple_sentiment(feedback)

    db = get_db()
    db.execute(
        """
        INSERT INTO analyses (user_id, input_type, original_text, cleaned_text,
                              sentiment_label, score)
        VALUES (?, 'text', ?, ?, ?, ?)
        """,
        (session["user_id"], feedback, feedback, label, score),
    )
    db.commit()

    return render_template(
        "index.html",
        feedback=feedback,
        sentiment_label=label,
        sentiment_score=score,
    )


@app.route("/analyze_file", methods=["POST"])
@login_required
def analyze_file_route():
    file = request.files.get("feedback_file")
    if not file or file.filename == "":
        return render_template(
            "index.html",
            error_file="Please upload a .txt file that contains feedback.",
        )

    filename = file.filename.lower()
    if not filename.endswith(".txt"):
        return render_template(
            "index.html",
            error_file="Only .txt files are supported in this section.",
        )

    content = file.read().decode("utf-8", errors="ignore")
    lines = [line.strip() for line in content.splitlines() if line.strip()]

    if not lines:
        return render_template(
            "index.html",
            error_file="The uploaded file is empty.",
        )

    details = []
    pos_count = neg_count = neu_count = 0
    db = get_db()

    for line in lines:
        label, score = simple_sentiment(line)

        if label == "Positive":
            pos_count += 1
        elif label == "Negative":
            neg_count += 1
        else:
            neu_count += 1

        db.execute(
            """
            INSERT INTO analyses (user_id, input_type, original_text, cleaned_text,
                                  sentiment_label, score)
            VALUES (?, 'txt', ?, ?, ?, ?)
            """,
            (session["user_id"], line, line, label, score),
        )

        details.append({"text": line, "label": label, "score": score})

    db.commit()

    total = len(lines)
    pos_percent = round((pos_count / total) * 100, 1)
    neg_percent = round((neg_count / total) * 100, 1)
    neu_percent = round((neu_count / total) * 100, 1)

    chart_b64 = build_chart_base64(pos_count, neg_count, neu_count)

    summary = {
        "total": total,
        "positive": pos_count,
        "negative": neg_count,
        "neutral": neu_count,
        "pos_percent": pos_percent,
        "neg_percent": neg_percent,
        "neu_percent": neu_percent,
        "chart": chart_b64,
    }

    return render_template(
        "index.html",
        file_summary=summary,
        file_details=details,
    )


@app.route("/analyze_csv", methods=["POST"])
@login_required
def analyze_csv_route():
    file = request.files.get("csv_file")
    if not file or file.filename == "":
        return render_template(
            "index.html",
            error_file="Please upload a CSV file that contains feedback.",
        )

    filename = file.filename.lower()
    if not filename.endswith(".csv"):
        return render_template(
            "index.html",
            error_file="Only .csv files are supported here.",
        )

    stream = io.StringIO(file.read().decode("utf-8", errors="ignore"))
    reader = csv.reader(stream)

    lines = []
    for row in reader:
        for cell in row:
            text = cell.strip()
            if text:
                lines.append(text)

    if not lines:
        return render_template(
            "index.html",
            error_file="The uploaded CSV file is empty.",
        )

    details = []
    pos_count = neg_count = neu_count = 0
    db = get_db()

    for line in lines:
        label, score = simple_sentiment(line)

        if label == "Positive":
            pos_count += 1
        elif label == "Negative":
            neg_count += 1
        else:
            neu_count += 1

        db.execute(
            """
            INSERT INTO analyses (user_id, input_type, original_text, cleaned_text,
                                  sentiment_label, score)
            VALUES (?, 'csv', ?, ?, ?, ?)
            """,
            (session["user_id"], line, line, label, score),
        )

        details.append({"text": line, "label": label, "score": score})

    db.commit()

    total = len(lines)
    pos_percent = round((pos_count / total) * 100, 1)
    neg_percent = round((neg_count / total) * 100, 1)
    neu_percent = round((neu_count / total) * 100, 1)

    chart_b64 = build_chart_base64(pos_count, neg_count, neu_count)

    summary = {
        "total": total,
        "positive": pos_count,
        "negative": neg_count,
        "neutral": neu_count,
        "pos_percent": pos_percent,
        "neg_percent": neg_percent,
        "neu_percent": neu_percent,
        "chart": chart_b64,
    }

    return render_template(
        "index.html",
        file_summary=summary,
        file_details=details,
    )


@app.route("/history")
@login_required
def history():
    db = get_db()
    rows = db.execute(
        """
        SELECT original_text, sentiment_label, score, input_type, created_at
        FROM analyses
        WHERE user_id = ?
        ORDER BY created_at DESC
        """,
        (session["user_id"],),
    ).fetchall()

    return render_template("history.html", analyses=rows)


@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/guide")
def guide():
    return render_template("guide.html")



if __name__ == "__main__":
    init_db()
    app.run(debug=True)
