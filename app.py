from flask import Flask, render_template, request, jsonify
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import json
import re

nltk.download('stopwords', quiet=True)

app = Flask(__name__)

stop_words = set(stopwords.words('english'))

# ─── Skill taxonomy ───────────────────────────────────────────────────────────
SKILLS_LIST = [
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "r", "go", "rust", "kotlin", "swift",
    # ML / AI
    "machine learning", "deep learning", "nlp", "computer vision", "reinforcement learning",
    "neural networks", "transformers", "llm", "generative ai", "prompt engineering",
    # Data Science
    "data science", "data analysis", "statistics", "feature engineering", "data wrangling",
    "exploratory data analysis", "a/b testing", "hypothesis testing",
    # Frameworks / Libraries
    "tensorflow", "pytorch", "keras", "scikit-learn", "xgboost", "lightgbm",
    "numpy", "pandas", "matplotlib", "seaborn", "plotly", "huggingface",
    # Web
    "flask", "django", "fastapi", "react", "vue", "node.js", "rest api", "graphql",
    # Cloud / MLOps
    "aws", "gcp", "azure", "docker", "kubernetes", "mlflow", "airflow", "ci/cd",
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    # General
    "git", "linux", "agile", "scrum"
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    """Lowercase, remove punctuation, strip stopwords."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)


def extract_text(file) -> str:
    """Extract raw text from PDF or plain-text file."""
    if file.filename.lower().endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        return " ".join(page.extract_text() or "" for page in reader.pages)
    return file.read().decode("utf-8", errors="ignore")


def extract_skills(text: str) -> list[str]:
    """Return skills found in text (multi-word safe)."""
    text_lower = text.lower()
    return [skill for skill in SKILLS_LIST if skill in text_lower]


def compute_similarity(resume_text: str, job_text: str) -> float:
    """TF-IDF cosine similarity, 0-100."""
    r_clean = preprocess(resume_text)
    j_clean = preprocess(job_text)
    if not r_clean.strip() or not j_clean.strip():
        return 0.0
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([r_clean, j_clean])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(float(score) * 100, 2)


def feedback_for(missing: list[str]) -> list[str]:
    suggestions_map = {

        # ─── Programming ───
        "python": "Build a small backend project using Flask or FastAPI (e.g., REST API with auth) and deploy it.",
        "java": "Create a Spring Boot API (CRUD + database) and showcase it with proper folder structure.",
        "javascript": "Build an interactive web app (DOM + API integration) and host it on GitHub Pages.",
        "typescript": "Convert a JS project to TypeScript and implement strict typing for better reliability.",
        "c++": "Solve DSA problems (arrays, graphs) and push solutions to GitHub with explanations.",
        "c#": "Build a .NET Web API with authentication and connect it to a SQL database.",
        "go": "Develop a simple microservice in Go and expose REST endpoints.",
        "rust": "Create a CLI tool using Rust and publish it with proper documentation.",

        # ─── AI / ML ───
        "machine learning": "Train a model (classification/regression) using scikit-learn and explain results.",
        "deep learning": "Build a neural network using PyTorch/TensorFlow (image or text task).",
        "nlp": "Use HuggingFace Transformers to build a chatbot or text classifier.",
        "computer vision": "Create an image classifier or object detector using OpenCV or YOLO.",
        "transformers": "Fine-tune a transformer model (BERT/GPT) on a custom dataset.",
        "llm": "Integrate OpenAI or open-source LLM APIs into a real-world app (chatbot, summarizer).",
        "generative ai": "Build a content generator (text/image) using diffusion or LLM APIs.",
        "prompt engineering": "Design structured prompts and build a prompt-based app (chat or automation tool).",

        # ─── Data Science ───
        "data science": "Work on an end-to-end data project (cleaning → modeling → insights).",
        "data analysis": "Analyze a dataset using Pandas and present insights with visualizations.",
        "statistics": "Apply statistical tests (A/B testing, hypothesis testing) in a real dataset.",
        "feature engineering": "Improve model performance by creating meaningful features.",
        "data wrangling": "Clean messy datasets and document preprocessing steps clearly.",
        "exploratory data analysis": "Perform EDA with plots and summarize key findings.",
        "a/b testing": "Design an A/B test scenario and analyze conversion results.",

        # ─── ML Tools ───
        "tensorflow": "Train and deploy a TensorFlow model with saved checkpoints.",
        "pytorch": "Build and train a custom neural network using PyTorch.",
        "scikit-learn": "Implement ML pipelines (scaling, model, evaluation).",
        "xgboost": "Use XGBoost for structured data and compare results with baseline models.",
        "lightgbm": "Train a LightGBM model and tune hyperparameters.",
        "huggingface": "Use HuggingFace pipelines for NLP tasks (classification, QA).",

        # ─── Data Libraries ───
        "numpy": "Use NumPy for matrix operations in ML or data processing.",
        "pandas": "Handle datasets using Pandas (groupby, joins, cleaning).",
        "matplotlib": "Visualize trends and patterns using Matplotlib.",
        "seaborn": "Create statistical plots (heatmaps, distributions).",
        "plotly": "Build interactive dashboards using Plotly.",

        # ─── Backend / Web ───
        "flask": "Develop a REST API using Flask and deploy it.",
        "django": "Build a full-stack web app using Django (auth + admin panel).",
        "fastapi": "Create a high-performance API using FastAPI with async endpoints.",
        "react": "Build a React app with API integration and deploy it.",
        "vue": "Create a frontend project using Vue with components and routing.",
        "node.js": "Develop a backend service using Node.js and Express.",
        "rest api": "Design REST APIs with proper endpoints, status codes, and docs.",
        "graphql": "Build a GraphQL API and query data efficiently.",

        # ─── Cloud / DevOps ───
        "aws": "Deploy an app on AWS (EC2 or S3) and configure basic services.",
        "gcp": "Deploy a backend on Google Cloud Run or App Engine.",
        "azure": "Host an application using Azure App Services.",
        "docker": "Containerize your app using Docker and run it locally.",
        "kubernetes": "Deploy a containerized app on Kubernetes (Minikube).",
        "mlflow": "Track ML experiments using MLflow.",
        "airflow": "Build a simple DAG pipeline using Apache Airflow.",
        "ci/cd": "Set up CI/CD using GitHub Actions for automated deployment.",

        # ─── Databases ───
        "sql": "Write complex SQL queries (joins, subqueries) and optimize them.",
        "mysql": "Design a relational schema and build queries using MySQL.",
        "postgresql": "Use PostgreSQL with indexing and advanced queries.",
        "mongodb": "Build a NoSQL app using MongoDB collections.",
        "redis": "Implement caching using Redis in a backend app.",
        "elasticsearch": "Build a search feature using Elasticsearch.",

        # ─── Tools ───
        "git": "Maintain clean commits and use branching strategies in GitHub.",
        "linux": "Use Linux commands and shell scripting for automation.",
        "agile": "Work in sprints and manage tasks using Agile practices.",
        "scrum": "Participate in sprint planning and daily standups in projects.",
    }

    results = []
    for skill in missing:
        if skill in suggestions_map:
            results.append(suggestions_map[skill])
        else:
            results.append(f"Work on a hands-on project involving {skill} and showcase it on GitHub.")

    return results[:5]


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    job_desc = request.form.get('job_desc', '').strip()
    files    = request.files.getlist('files')

    if not job_desc:
        return jsonify({"error": "Job description is required."}), 400
    if not files or all(f.filename == '' for f in files):
        return jsonify({"error": "Please upload at least one resume."}), 400

    job_skills = extract_skills(job_desc)
    results    = []

    for file in files:
        if file.filename == '':
            continue

        resume_text = extract_text(file)
        score       = compute_similarity(resume_text, job_desc)
        res_skills  = extract_skills(resume_text)

        matched = sorted(set(res_skills) & set(job_skills))
        missing = sorted(set(job_skills) - set(res_skills))

        results.append({
            "name":       file.filename,
            "score":      score,
            "matched":    matched,
            "missing":    missing,
            "feedback":   feedback_for(missing),
            "all_skills": res_skills,
        })

    # Sort by score descending, assign rank
    results.sort(key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)
