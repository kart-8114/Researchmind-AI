import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Suppress Hugging Face warnings and disable experimental Xet storage on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"
from flask import Flask, render_template, request, flash, redirect, url_for, send_file, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import faiss
import re
import traceback
import kagglehub
import pandas as pd
from sentence_transformers import SentenceTransformer
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import numpy as np
import io
from sqlalchemy.pool import NullPool
# Import Classical AI Modules
from recommendation import generate_dynamic_plans, solve_resource_csp
from planner_v2 import htn_project_roadmap, generate_experiment_design
from rules import forward_chain, forward_chain_project
from research_ml import ResearchKnowledgeAgent
from openai_service import is_available as openai_available, analyze_paper, generate_project_candidates, answer_question, extract_research_gaps

app = Flask(__name__)
# CHANGED SECRET KEY to clear old "Please log in" session messages
app.secret_key = os.getenv("SECRET_KEY", "researchmind_pro_v3_secure_key")

def enrich_research_gaps(gaps):
    enriched = []
    for gap in gaps:
        lower = gap.lower()
        if "computational" in lower or "efficiency" in lower or "latency" in lower or "speed" in lower or "compute" in lower:
            desc = "Existing models achieve high accuracy at the cost of heavy computational requirements, making real-time or resource-constrained deployment challenging."
            impact = "Limits applicability on edge devices, embedded systems, and latency-critical environments."
            mitigation = "Investigate model quantization, pruning, distillation, or efficient architectural redesign to maintain accuracy while reducing FLOPs."
        elif "generalization" in lower or "dataset" in lower or "robustness" in lower or "diversity" in lower:
            desc = "Models trained on benchmark datasets often overfit to domain-specific distributions and fail to generalize under distribution shifts."
            impact = "Degraded real-world performance and vulnerability to adversarial examples or unseen data domains."
            mitigation = "Incorporate domain adaptation, cross-domain zero-shot evaluation, data augmentation, or robust training regimes."
        elif "scalability" in lower or "production" in lower:
            desc = "Algorithms scale poorly with data volume or model parameter growth due to quadratic complexity or memory bottlenecks."
            impact = "Impractical for enterprise-scale or large-scale production environments."
            mitigation = "Implement distributed training strategies, memory-efficient attention mechanisms, or indexing optimizations (e.g., FAISS)."
        else:
            desc = f"Specific limitation identified regarding {gap.lower()}, requiring specialized architectural refinement."
            impact = "Restricts the overall performance ceiling and robustness of baseline models."
            mitigation = "Design targeted ablation experiments and introduce domain-specific evaluation metrics."
            
        enriched.append({
            "title": gap,
            "description": desc,
            "impact": impact,
            "mitigation": mitigation
        })
    return enriched

@app.route("/collections/<int:collection_id>/analyze")
@login_required
def analyze_collection(collection_id):
    """Phase 4 Upgrade: Full Intelligence Orchestrator with Experiment Design"""
    collection = db.get_or_404(Collection, collection_id)
    if collection.user_id != current_user.id:
        return "Unauthorized", 403
        
    papers = collection.papers
    if not papers:
        flash("Collection is empty!")
        return redirect(url_for('history', collection=collection_id))
        
    # 1. Gap Detection Logic (Evidence-Based / OpenAI)
    all_limitations = " ".join([p.limitation for p in papers if p.limitation])
    all_algos = ",".join([p.algos for p in papers if p.algos])
    domain = papers[0].domain
    
    gaps = []
    if openai_available():
        try:
            corpus_text = all_limitations + " " + " ".join([p.summary for p in papers])
            gaps = extract_research_gaps(corpus_text)
        except Exception as e:
            print(f"OpenAI collection gap extraction failed: {e}")
            
    if not gaps:
        if any(x in all_limitations.lower() for x in ["latency", "speed", "slow", "resource", "compute", "gpu", "expensive"]): 
            gaps.append("Computational efficiency vs. accuracy trade-off")
        if any(x in all_limitations.lower() for x in ["data", "diversity", "bias"]): 
            gaps.append("Limited dataset generalization and robustness")
        if "scalability" not in all_limitations.lower(): 
            gaps.append("Large-scale production deployment feasibility")
        if not gaps: 
            gaps.append("Domain-specific architectural optimization")
    
    # 2. Forward Chaining Reasoning
    project_title = forward_chain(domain, all_algos, all_limitations)
    
    # 3. Dynamic Best-First Search for plans
    ranked_plans = generate_dynamic_plans(domain, gaps)
    
    # 4. CSP + Backtracking (Resource Configurator)
    assignment = None
    if request.args.get('run_csp'):
        constraints = {
            "gpu": request.args.get("gpu"),
            "budget": request.args.get("budget")
        }
        selected_plan = ranked_plans[0]
        assignment = solve_resource_csp(selected_plan, constraints)
        
    # 5. HTN Planning (Unit 4) for the project roadmap
    selected_for_roadmap = ranked_plans[0] if ranked_plans else {"title": project_title}
    roadmap = htn_project_roadmap(
        selected_for_roadmap.get("title", project_title),
        domain,
        str(all_algos).split(","),
        []
    )
    
    # 6. Experiment Design (Phase 4 Final)
    experiment = generate_experiment_design(gaps[0], domain, ranked_plans[0])
    
    return render_template(
        "intelligence_dashboard.html", 
        collection=collection, 
        gaps=gaps, 
        enriched_gaps=enrich_research_gaps(gaps),
        project_title=project_title,
        ranked_plans=ranked_plans,
        assignment=assignment,
        roadmap=roadmap,
        experiment=experiment
    )

# Initialize Knowledge-Based Agent
knowledge_agent = ResearchKnowledgeAgent()
try:
    knowledge_agent.load_or_initialize()
except Exception as e:
    print(f"Agent Initialization Error: {e}")

# ============================================================
# MYSQL DATABASE CONFIGURATION
# ============================================================

app.config["SQLALCHEMY_DATABASE_URI"] = (
    "mysql+pymysql://researchmind_user:"
    "ResearchMind%40123"
    "@localhost:3306/researchmind"
)

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Configure engine for MySQL robustness
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    "pool_pre_ping": True,
    "pool_recycle": 1800,
    "pool_size": 10,
    "max_overflow": 20,
}
# Set max upload size to 50MB
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'pdf'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --- Database Models ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    papers = db.relationship('ResearchPaper', backref='owner', lazy=True)
    collections = db.relationship('Collection', backref='owner', lazy=True)
    tags = db.relationship('Tag', backref='owner', lazy=True)

class Collection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    papers = db.relationship('ResearchPaper', backref='collection', lazy=True)

# Many-to-Many association for Papers and Tags
paper_tags = db.Table('paper_tags',
    db.Column('paper_id', db.Integer, db.ForeignKey('research_paper.id'), primary_key=True),
    db.Column('tag_id', db.Integer, db.ForeignKey('tag.id'), primary_key=True)
)

class Tag(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class ResearchPaper(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable=False)
    summary = db.Column(db.Text, nullable=False)
    domain = db.Column(db.String(100))
    algos = db.Column(db.String(500))
    datasets = db.Column(db.String(500))
    metrics = db.Column(db.String(500))
    
    # New Technical Columns
    problem = db.Column(db.Text)
    limitation = db.Column(db.Text)
    results = db.Column(db.Text)
    future_work = db.Column(db.Text)
    
    # Existing Metadata Fields
    doi = db.Column(db.String(100))
    published_date = db.Column(db.String(50))
    journal = db.Column(db.String(200))
    citations_count = db.Column(db.Integer, default=0)
    bibtex = db.Column(db.Text)
    
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    collection_id = db.Column(db.Integer, db.ForeignKey('collection.id'))
    tags = db.relationship('Tag', secondary=paper_tags, backref=db.backref('papers', lazy='dynamic'))

# Graceful Database Initialization
try:
    from sqlalchemy import text, inspect
    with app.app_context():
        db.create_all()
        # Direct auto-migration via exception handling to guarantee columns exist across platforms/cases
        new_columns = {
            "problem": "TEXT",
            "limitation": "TEXT",
            "results": "TEXT",
            "future_work": "TEXT",
            "doi": "VARCHAR(100)",
            "published_date": "VARCHAR(50)",
            "journal": "VARCHAR(200)",
            "citations_count": "INT DEFAULT 0",
            "bibtex": "TEXT"
        }
        with db.engine.begin() as conn:
            for col, col_type in new_columns.items():
                try:
                    conn.execute(text(f"ALTER TABLE research_paper ADD COLUMN {col} {col_type}"))
                except Exception:
                    # Ignore errors if column already exists
                    pass
        print("✅ Database tables and columns verified/created successfully.")
except Exception as e:
    print("❌ Database Connection/Migration Error!")
    print(f"Details: {e}")
    print("Please ensure your MySQL server is running and the credentials are correct.")

@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, int(user_id))
    except Exception:
        return None

# --- AI Models Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Summarization
SUMM_MODEL = "sshleifer/distilbart-cnn-12-6"
print(f"Loading Summarizer: {SUMM_MODEL}...")
summ_tokenizer = AutoTokenizer.from_pretrained(SUMM_MODEL)
summ_model = AutoModelForSeq2SeqLM.from_pretrained(SUMM_MODEL).to(device)

# 2. Classification
CLASS_MODEL = "typeform/distilbert-base-uncased-mnli"
print(f"Loading Classifier: {CLASS_MODEL}...")
classifier = pipeline("zero-shot-classification", model=CLASS_MODEL, device=0 if torch.cuda.is_available() else -1)

# 3. RAG Embeddings
print("Loading Embedding Model...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# 4. LLM Extraction (Phase 2 Upgrade)
print("Loading LLM Extraction Pipeline...")
# Using model/tokenizer directly to avoid pipeline task validation issues
extraction_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
extraction_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(device)

# Global stores for RAG session
vector_store = {} 

from crossref.restful import Works

# --- Helper Functions ---
def fetch_doi_metadata(title):
    """Phase 2: Auto-fetch DOI and Metadata via CrossRef"""
    try:
        works = Works()
        # Query and take the first matching result
        results = works.query(title)
        for item in results:
            return {
                'doi': item.get('DOI'),
                'journal': item.get('container-title', [None])[0],
                'published_date': str(item.get('published-print', {}).get('date-parts', [[None]])[0][0]),
                'citations': item.get('is-referenced-by-count', 0)
            }
    except Exception as e:
        print(f"DOI Fetch error: {e}")
    return {}

def extract_entities_llm(text):
    """Phase 2 Upgrade: Comprehensive technical extraction using LLM"""
    prompt = (
        f"Analyze this research abstract and extract the following: "
        f"1. Problem addressed, 2. Primary algorithm, 3. Main dataset, 4. Evaluation metrics, "
        f"5. Key limitation, 6. Future work. "
        f"Format as JSON: {{'problem': '', 'algo': '', 'dataset': '', 'metrics': '', 'limitation': '', 'future': ''}}. "
        f"Abstract: {text[:800]}"
    )
    try:
        inputs = extraction_tokenizer(prompt, return_tensors="pt").to(device)
        outputs = extraction_model.generate(**inputs, max_length=150)
        raw_text = extraction_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Simple heuristic parsing (since we use a small T5 model, it might not follow JSON perfectly)
        # We try to split by known markers if it fails JSON parsing
        return raw_text 
    except Exception as e:
        print(f"LLM Extraction error: {e}")
        return "Extraction failed"

def generate_summary(text):
    if not text.strip():
        return "No text available to summarize."
    
    # Chunking for long papers
    max_chunk = 1024
    chunks = [text[i:i + max_chunk] for i in range(0, len(text), max_chunk)]
    
    partial_summaries = []
    print(f"Summarizing in {len(chunks)} chunks...")
    
    for chunk in chunks[:2]: # Limit to first 2 chunks for extreme speed during live demo
        inputs = summ_tokenizer(chunk, return_tensors="pt", max_length=1024, truncation=True).to(device)
        summary_ids = summ_model.generate(
            inputs["input_ids"], 
            max_new_tokens=150, 
            min_new_tokens=50, 
            num_beams=1, # Greedy decoding for maximum performance
            early_stopping=True
        )
        partial_summaries.append(summ_tokenizer.decode(summary_ids[0], skip_special_tokens=True))
    
    if len(partial_summaries) > 1:
        final_input = " ".join(partial_summaries)
        inputs = summ_tokenizer(final_input, return_tensors="pt", max_length=1024, truncation=True).to(device)
        summary_ids = summ_model.generate(inputs["input_ids"], max_new_tokens=200, num_beams=1)
        return summ_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return partial_summaries[0] if partial_summaries else "Summary generation failed."

def extract_entities(text):
    """Smart heuristic-based extraction for Algorithms, Data Structures, and Datasets."""
    # 1. Syllabus-Aligned Algorithms
    algo_keywords = [
        "Dijkstra", "Breadth First Search", "Depth First Search", "BFS", "DFS", "Sorting", 
        "Merge Sort", "Quick Sort", "Binary Search", "Dynamic Programming", "Recursion",
        "Backtracking", "Greedy Algorithm", "Prim's", "Kruskal's", "A* Search", "Bellman-Ford",
        "Knowledge-Based Agent", "Forward Chaining", "Constraint Satisfaction", "CSP"
    ]
    
    # 2. Dataset Keywords
    dataset_keywords = [
        "MNIST", "ImageNet", "CIFAR", "COCO", "GLUE", "SQuAD", "IMDb", "Kaggle", 
        "PASCAL VOC", "UCI Repository", "MIMIC", "Common Crawl", "WordNet", "Cityscapes"
    ]

    # Find keywords (case-insensitive)
    found_algos = [w for w in algo_keywords if re.search(r'\b' + re.escape(w) + r'\b', text, re.IGNORECASE)]
    found_datasets = [w for w in dataset_keywords if re.search(r'\b' + re.escape(w) + r'\b', text, re.IGNORECASE)]

    # 3. Pattern Matching for "unknown" datasets/algos
    patterns = {
        "algo": [r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(?:algorithm|method|model|approach)'],
        "dataset": [r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(?:dataset|data set|repository|corpus)'],
        "metric": [r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(?:metric|score|measure|index)']
    }
    
    # 4. Evaluation Metrics Keywords
    metric_keywords = ["Accuracy", "Precision", "Recall", "F1-Score", "mAP", "IOU", "BLEU", "ROUGE", "METEOR", "AUC", "ROC", "MSE", "MAE", "RMSE", "Perplexity"]
    found_metrics = [w for w in metric_keywords if re.search(r'\b' + re.escape(w) + r'\b', text, re.IGNORECASE)]

    for label, p_list in patterns.items():
        for p in p_list:
            matches = re.findall(p, text)
            if label == "algo": found_algos.extend(matches)
            elif label == "dataset": found_datasets.extend(matches)
            else: found_metrics.extend(matches)

    # Clean and limit results
    final_algos = sorted(list(set(found_algos)), key=len, reverse=True)[:6]
    final_datasets = sorted(list(set(found_datasets)), key=len, reverse=True)[:4]
    final_metrics = sorted(list(set(found_metrics)), key=len, reverse=True)[:5]
    
    return final_algos, final_datasets, final_metrics

def get_research_plan(domain):
    roadmap = [f"1. Setup environment for {domain} research.", "2. Data preprocessing.", "3. Baseline implementation.", "4. Optimization.", "5. Evaluation."]
    gaps = ["Computational complexity.", "Dataset diversity.", "Interpretability.", "Edge-case performance."]
    return roadmap, gaps

def setup_rag(paper_id, full_text):
    # Phase 2: Overlapping chunking for better context
    chunk_size = 500
    overlap = 100
    chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size - overlap)]
    
    if not chunks:
        return
    embeddings = embed_model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    vector_store[paper_id] = (index, chunks)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Auth Routes ---
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        # Modern SQLAlchemy 2.0 syntax
        user = db.session.execute(db.select(User).filter_by(username=username)).scalar_one_or_none()
        if user:
            flash("Username already exists")
            return redirect(url_for("register"))
        new_user = User(username=username, password=generate_password_hash(password))
        db.session.add(new_user)
        db.session.commit()
        flash("Account created! Please login.")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        # Modern SQLAlchemy 2.0 syntax
        user = db.session.execute(db.select(User).filter_by(username=username)).scalar_one_or_none()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for("home"))
        flash("Invalid credentials")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# --- App Routes ---
@app.route("/")
@login_required
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
@login_required
def upload():
    # Robust check for files
    if not request.files:
        flash("No file part in the request")
        return redirect(url_for('home'))

    pdf = request.files.get('pdf') or request.files.get('pdfs')
    if not pdf or pdf.filename == '':
        flash('No file selected for upload')
        return redirect(url_for('home'))

    if pdf and allowed_file(pdf.filename):
        filename = secure_filename(pdf.filename)
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        pdf.save(path)

        # Extraction with 10-page limit for stability
        full_text = ""
        analysis_text = ""
        try:
            with fitz.open(path) as doc:
                page_count = len(doc)
                # Read all for RAG (limited to first 15 pages for extreme demo performance speed)
                for i in range(min(15, page_count)):
                    page_text = doc[i].get_text()
                    full_text += page_text
                    if i < 10:
                        analysis_text += page_text
        except Exception as e:
            print(f"Extraction error: {e}")
            traceback.print_exc()
            flash(f"Error reading PDF content: {str(e)}")
            return redirect(url_for('home'))

        if not analysis_text.strip():
            flash("No text could be extracted from this PDF")
            return redirect(url_for('home'))

        # Core Analysis
        short_text = analysis_text[:3000].strip()
        if not short_text:
            flash("Extracted text is too short for AI analysis.")
            return redirect(url_for('home'))

        # ------------------------------------------------------------
        # OpenAI layer: language understanding only.
        # Classical syllabus algorithms remain local and explainable.
        # ------------------------------------------------------------
        openai_analysis = None
        if openai_available():
            try:
                print("Running OpenAI literature analysis...")
                openai_analysis = analyze_paper(analysis_text)
            except Exception as e:
                print(f"OpenAI analysis failed; using local fallbacks: {e}")

        # Summary: OpenAI first, local Transformer fallback.
        summary = "Summary generation failed."
        if openai_analysis:
            summary = openai_analysis.get("summary") or summary
        else:
            try:
                summary = generate_summary(analysis_text)
            except Exception as e:
                print(f"Summary step failed: {e}")

        # Domain: OpenAI first, then local knowledge agent/classifier.
        knowledge_domain, confidence = "General AI", 0.0
        try:
            knowledge_domain, confidence = knowledge_agent.infer_domain_category(analysis_text[:2000])
        except Exception as e:
            print(f"Agent Domain Inference step failed: {e}")

        domain = (openai_analysis or {}).get("domain") or knowledge_domain
        try:
            candidate_labels = ["Computer Vision", "Natural Language Processing", "Healthcare", "Robotics", "Finance", "Security"]
            class_res = classifier(short_text[:1000], candidate_labels)
            if not openai_analysis and class_res and 'labels' in class_res:
                domain = class_res['labels'][0]
        except Exception as e:
            print(f"Local classification step failed: {e}")

        # Structured technical facts from OpenAI; heuristic extractor remains a fallback.
        if openai_analysis:
            algos_list = openai_analysis.get("algorithms", [])
            datasets_list = openai_analysis.get("datasets", [])
            metrics_list = openai_analysis.get("metrics", [])
            problem_ext = openai_analysis.get("problem", "Not explicitly stated")
            limitations_list = openai_analysis.get("limitations", [])
            limitation_ext = "; ".join(limitations_list) if limitations_list else "Not explicitly stated"
            future_list = openai_analysis.get("future_work", [])
            future_ext = "; ".join(future_list) if future_list else "Not explicitly stated"
            algos_ext = ", ".join(algos_list) if algos_list else "N/A"
            datasets_ext = ", ".join(datasets_list) if datasets_list else "N/A"
            metrics_ext = ", ".join(metrics_list) if metrics_list else "N/A"
        else:
            print("OpenAI unavailable; using local heuristic entity extraction.")
            found_algos, found_datasets, found_metrics = extract_entities(analysis_text)
            algos_ext = ", ".join(found_algos) if found_algos else "N/A"
            datasets_ext = ", ".join(found_datasets) if found_datasets else "N/A"
            metrics_ext = ", ".join(found_metrics) if found_metrics else "N/A"
            problem_ext = "Extracted via local heuristic analysis."
            limitation_ext = "Computational and generalization trade-offs identified."
            future_ext = "Further optimization and benchmark evaluation."

        print("Fetching Related Literature...")
        recommended_papers = []
        try:
            recommended_papers = knowledge_agent.get_related_literature(analysis_text[:2000])
        except Exception as e:
            print(f"Recommendations step failed: {e}")

        print("Fetching Metadata (CrossRef)...")
        metadata = {}
        try:
            metadata = fetch_doi_metadata(filename)
        except Exception as e:
            print(f"Metadata step failed: {e}")

        paper = ResearchPaper(
            filename=filename[:195],
            summary=summary,
            domain=domain[:95],
            algos=algos_ext[:495],
            datasets=datasets_ext[:495],
            metrics=metrics_ext[:495],
            problem=problem_ext,
            limitation=limitation_ext,
            future_work=future_ext,
            doi=(metadata.get('doi') or "N/A")[:95],
            journal=(metadata.get('journal') or "N/A")[:195],
            published_date=(metadata.get('published_date') or "N/A")[:45],
            citations_count=metadata.get('citations', 0),
            user_id=current_user.id
        )
        db.session.add(paper)
        db.session.commit()

        print("Setting up RAG index...")
        try:
            setup_rag(paper.id, full_text)
        except Exception as e:
            print(f"RAG step failed: {e}")

        # --- Syllabus-Aligned AI Integration ---
        print("Running classical AI algorithms with OpenAI outputs...")

        # 1. OpenAI generates candidates; Forward Chaining selects one using extracted facts.
        gaps = []
        if openai_analysis:
            gaps = openai_analysis.get("research_gaps", []) or []
        if not gaps and openai_available():
            try:
                gaps = extract_research_gaps(analysis_text)
            except Exception as e:
                print(f"OpenAI gap extraction failed: {e}")
        if not gaps:
            combined_check_text = f"{limitation_ext} {analysis_text} {filename}".lower()
            if any(x in combined_check_text for x in ["latency", "speed", "slow", "resource", "compute", "gpu", "expensive"]):
                gaps.append("Computational efficiency vs. accuracy trade-off")
            if any(x in combined_check_text for x in ["data", "diversity", "bias", "generalization"]):
                gaps.append("Limited dataset generalization")
            if not gaps:
                gaps.append("Domain-specific optimization")

        ai_candidates = []
        if openai_available():
            try:
                candidate_data = generate_project_candidates(
                    domain,
                    [x.strip() for x in algos_ext.split(',') if x.strip() and x.strip() != "N/A"],
                    [x.strip() for x in datasets_ext.split(',') if x.strip() and x.strip() != "N/A"],
                    [x.strip() for x in metrics_ext.split(',') if x.strip() and x.strip() != "N/A"],
                    gaps,
                    [x.strip() for x in limitation_ext.split(';') if x.strip()]
                )
                ai_candidates = candidate_data.get("candidates", [])
            except Exception as e:
                print(f"OpenAI project candidate generation failed: {e}")

        if not ai_candidates:
            ai_candidates = [
                {
                    "title": f"Optimized {domain} Architecture for {gaps[0] if gaps else 'Performance Scaling'}",
                    "problem_statement": f"Addressing {gaps[0] if gaps else 'computational and generalization bottlenecks'} in {domain}.",
                    "algorithms": [x.strip() for x in algos_ext.split(',') if x.strip() and x.strip() != "N/A"] or ["Baseline Method"],
                    "datasets": [x.strip() for x in datasets_ext.split(',') if x.strip() and x.strip() != "N/A"] or ["Benchmark Dataset"],
                    "metrics": [x.strip() for x in metrics_ext.split(',') if x.strip() and x.strip() != "N/A"] or ["Accuracy"],
                    "future_scope": ["Cross-domain evaluation", "Deployment scaling"],
                    "rationale": "Locally synthesized project candidate addressing extracted literature gaps."
                },
                {
                    "title": f"Robust Data Generalization Framework for {domain}",
                    "problem_statement": "Mitigating dataset distribution shift and robustness issues.",
                    "algorithms": [x.strip() for x in algos_ext.split(',') if x.strip() and x.strip() != "N/A"] or ["Ensemble Model"],
                    "datasets": [x.strip() for x in datasets_ext.split(',') if x.strip() and x.strip() != "N/A"] or ["Standard Dataset"],
                    "metrics": [x.strip() for x in metrics_ext.split(',') if x.strip() and x.strip() != "N/A"] or ["F1-Score"],
                    "future_scope": ["Adversarial stress-testing"],
                    "rationale": "Locally synthesized project candidate focusing on generalization and robustness."
                }
            ]

        algo_list = [x.strip() for x in algos_ext.split(',') if x.strip() and x.strip() != "N/A"]
        dataset_list = [x.strip() for x in datasets_ext.split(',') if x.strip() and x.strip() != "N/A"]
        metric_list = [x.strip() for x in metrics_ext.split(',') if x.strip() and x.strip() != "N/A"]
        limitation_list = [x.strip() for x in limitation_ext.split(';') if x.strip()]

        selected_project, fc_candidates = forward_chain_project(
            domain, algo_list, dataset_list, metric_list, gaps, limitation_list, ai_candidates
        )
        project_title = selected_project.get("title", forward_chain(domain, algos_ext, limitation_ext))

        # 2. Best-First Search ranks the OpenAI candidates using the classical heuristic.
        ranked_plans = generate_dynamic_plans(domain, gaps, fc_candidates)

        # 3. HTN Planning creates the hierarchical implementation roadmap.
        roadmap = htn_project_roadmap(
            project_title,
            domain,
            selected_project.get("algorithms") or algo_list,
            selected_project.get("datasets") or dataset_list
        )

        # 4. Experiment design is retained as a deterministic project-support module.
        experiment = None
        try:
            if ranked_plans:
                experiment = generate_experiment_design(gaps[0], domain, ranked_plans[0])
        except Exception as e:
            print(f"Experiment generation failed: {e}")

        # 5. Optional CSP + Backtracking resource configurator.
        assignment = None
        if request.args.get('run_csp') and ranked_plans:
            constraints = {
                "gpu": request.args.get("gpu"),
                "budget": request.args.get("budget")
            }
            assignment = solve_resource_csp(ranked_plans[0], constraints)

        return render_template(
            "result.html",
            paper=paper,
            project_title=project_title,
            selected_project=selected_project,
            fc_candidates=fc_candidates,
            ranked_plans=ranked_plans,
            roadmap=roadmap,
            assignment=assignment,
            gaps=gaps,
            enriched_gaps=enrich_research_gaps(gaps),
            experiment=experiment,
            algos=paper.algos,
            datasets=paper.datasets,
            metrics=paper.metrics,
            knowledge_domain=knowledge_domain,
            confidence=confidence,
            recommended_papers=recommended_papers,
            openai_enabled=openai_available(),
            openai_analysis=openai_analysis
        )
    else:
        flash("Invalid file type. Only PDFs are allowed.")
        return redirect(url_for('home'))

@app.route("/chat", methods=["POST"])
@login_required
def chat():
    data = request.json or {}
    paper_id = data.get("paper_id")
    query = data.get("query")

    if not paper_id or not query:
        return jsonify({"response": "Missing paper ID or query."})

    try:
        paper_id = int(paper_id)
    except (TypeError, ValueError):
        return jsonify({"response": "Invalid paper ID."})

    if paper_id not in vector_store:
        return jsonify({"response": "Context lost. Please re-upload the paper to chat."})

    try:
        index, chunks = vector_store[paper_id]
        query_vector = embed_model.encode([query])
        k = min(4, len(chunks))
        _, indices = index.search(np.array(query_vector).astype('float32'), k=k)
        context = "\n\n".join(chunks[idx] for idx in indices[0] if 0 <= idx < len(chunks))

        if openai_available():
            response = answer_question(query, context)
        else:
            response = f"OpenAI is not configured. Relevant paper context: {context[:800]}..."
        return jsonify({"response": response})
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({"response": "An error occurred during chat processing."})

@app.route("/download_report/<int:paper_id>")
@login_required
def download_report(paper_id):
    # Modern Flask-SQLAlchemy 3.0 syntax
    paper = db.get_or_404(ResearchPaper, paper_id)
    if paper.user_id != current_user.id:
        return "Unauthorized", 403
        
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.setFont("Helvetica-Bold", 16)
    p.drawString(100, 750, f"Research Report: {paper.filename}")
    p.setFont("Helvetica", 12)
    p.drawString(100, 720, f"Domain: {paper.domain}")
    p.drawString(100, 700, "Summary:")
    p.setFont("Helvetica", 10)
    
    textobject = p.beginText(100, 680)
    lines = paper.summary.split('.')
    for line in lines:
        if line.strip():
            textobject.textLine(line.strip() + ".")
    p.drawText(textobject)
    
    # Add Technical Facts Section
    curr_y = textobject.getY() - 40
    if curr_y < 100: # Simple page overflow check
        p.showPage()
        curr_y = 750
        
    p.setFont("Helvetica-Bold", 12)
    p.drawString(100, curr_y, "Technical Insights:")
    p.setFont("Helvetica", 10)
    curr_y -= 20
    p.drawString(120, curr_y, f"Algorithms: {(paper.algos or 'N/A')[:80]}")
    curr_y -= 15
    p.drawString(120, curr_y, f"DOI: {paper.doi or 'N/A'}")
    curr_y -= 15
    p.drawString(120, curr_y, f"Journal: {(paper.journal or 'N/A')[:80]}")
    
    p.showPage()
    p.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name=f"Report_{paper.filename}.pdf", mimetype='application/pdf')

@app.route("/ai_ingestion", methods=["POST"])
@login_required
def ai_ingestion():
    try:
        topic = request.form.get("topic", "").strip()
        if not topic:
            flash("Please enter a valid research topic.")
            return redirect(url_for('home'))
            
        print(f"Ingesting paper generation request for topic: {topic}")
        
        # Determine domain via zero-shot classification over the topic text
        domain = "General AI"
        try:
            candidate_labels = ["Computer Vision", "Natural Language Processing", "Healthcare", "Robotics", "Finance", "Security"]
            class_res = classifier(topic, candidate_labels)
            if class_res and 'labels' in class_res:
                domain = class_res['labels'][0]
        except Exception as e:
            print(f"Ingestion Classification error: {e}")

        # Synthesize standard structured context based on domain properties
        if domain == "Computer Vision":
            summary = f"This research explores advanced approaches to '{topic}'. By focusing on visual feature extractors and real-time processing components, the proposed system sets a new standard for high-fidelity spatial understanding."
            algos = "BFS,DFS,Heuristic Search"
            datasets = "Unit 2 Search Space"
            metrics = "Efficiency,Accuracy"
        elif domain == "Natural Language Processing":
            summary = f"This research presents an intensive analysis of '{topic}'. The investigation covers sequence learning optimizations, contextual tokenization strategies, and robust attention network fine-tuning configurations."
            algos = "Knowledge-Based Agent,Forward Chaining"
            datasets = "Unit 3 Knowledge Base"
            metrics = "Inference Speed,Correctness"
        elif domain == "Healthcare":
            summary = f"This study evaluates methodology parameters for '{topic}'. The text structures predictive architectures targeting high clinical integrity while prioritizing local processing constraints."
            algos = "A* Search,CSP"
            datasets = "Constrained Research Environment"
            metrics = "Constraint Satisfaction,Precision"
        else:
            summary = f"This exploration develops actionable frameworks addressing the core themes of '{topic}'. The investigation introduces modular execution algorithms tailored to modern production pipelines."
            algos = "Hierarchical Planning,Best-First Search"
            datasets = "Unit 4 Planning Graph"
            metrics = "Goal Achievement,F1-Score"

        paper = ResearchPaper(
            filename=f"AI_Generated_{topic.replace(' ', '_')[:40]}.pdf",
            summary=summary,
            domain=domain,
            algos=algos,
            datasets=datasets,
            metrics=metrics,
            user_id=current_user.id
        )
        db.session.add(paper)
        db.session.commit()
        
        flash(f"Successfully generated structured research record for: {topic}!")
        return redirect(url_for('history'))
        
    except Exception as e:
        print(f"AI Ingestion Error: {e}")
        traceback.print_exc()
        flash(f"Failed to ingest research topic: {str(e)}")
        return redirect(url_for('home'))

import arxiv

@app.route("/arxiv/search")
@login_required
def search_arxiv():
    query = request.args.get('query')
    if not query:
        return render_template("arxiv_search.html", results=[])
    
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=10,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    results = []
    for res in client.results(search):
        results.append({
            'title': res.title,
            'summary': res.summary,
            'url': res.pdf_url,
            'published': res.published.year,
            'entry_id': res.entry_id.split('/')[-1]
        })
        
    return render_template("arxiv_search.html", results=results, query=query)

@app.route("/arxiv/import", methods=["POST"])
@login_required
def import_arxiv():
    title = request.form.get("title")
    summary = request.form.get("summary")
    url = request.form.get("url")
    
    # Process like a normal paper
    domain, _ = knowledge_agent.infer_domain_category(summary)
    
    paper = ResearchPaper(
        filename=f"arXiv_{title[:50]}.pdf",
        summary=summary,
        domain=domain,
        user_id=current_user.id
    )
    db.session.add(paper)
    db.session.commit()
    
    flash(f"Paper '{title}' imported from arXiv!")
    return redirect(url_for('history'))

@app.route("/history")
@login_required
def history():
    # Fetch collections and tags for filters
    collections = db.session.execute(
        db.select(Collection).filter_by(user_id=current_user.id)
    ).scalars().all()
    
    tags = db.session.execute(
        db.select(Tag).filter_by(user_id=current_user.id)
    ).scalars().all()

    # Filtering logic
    query = db.select(ResearchPaper).filter_by(user_id=current_user.id)
    
    collection_id = request.args.get('collection')
    if collection_id:
        query = query.filter_by(collection_id=collection_id)
        
    tag_id = request.args.get('tag')
    if tag_id:
        query = query.join(ResearchPaper.tags).filter(Tag.id == tag_id)
        
    search = request.args.get('search')
    if search:
        query = query.filter(ResearchPaper.filename.contains(search) | ResearchPaper.summary.contains(search))

    papers = db.session.execute(query).scalars().all()
    
    return render_template("history.html", papers=papers, collections=collections, tags=tags)

@app.route("/collections/create", methods=["POST"])
@login_required
def create_collection():
    name = request.form.get("name")
    desc = request.form.get("description")
    if name:
        new_coll = Collection(name=name, description=desc, user_id=current_user.id)
        db.session.add(new_coll)
        db.session.commit()
        flash(f"Collection '{name}' created!")
    return redirect(url_for('history'))

@app.route("/tags/create", methods=["POST"])
@login_required
def create_tag():
    name = request.form.get("name")
    if name:
        new_tag = Tag(name=name, user_id=current_user.id)
        db.session.add(new_tag)
        db.session.commit()
        flash(f"Tag '{name}' created!")
    return redirect(url_for('history'))

@app.route("/papers/<int:paper_id>/citation")
@login_required
def get_citation(paper_id):
    paper = db.get_or_404(ResearchPaper, paper_id)
    if paper.user_id != current_user.id:
        return "Unauthorized", 403
        
    year = paper.published_date or "2024"
    journal = paper.journal or "arXiv preprint"
    title = paper.filename.replace(".pdf", "").replace("AI_Generated_", "").replace("_", " ")
    
    citations = {
        'apa': f"Author, A. ({year}). {title}. {journal}.",
        'ieee': f"[1] A. Author, \"{title},\" {journal}, {year}.",
        'bibtex': f"@article{{key,\n  title={{{title}}},\n  author={{Author, A.}},\n  journal={{{journal}}},\n  year={{{year}}}\n}}"
    }
    return jsonify(citations)

@app.route("/papers/<int:paper_id>/organize", methods=["POST"])
@login_required
def organize_paper(paper_id):
    paper = db.get_or_404(ResearchPaper, paper_id)
    if paper.user_id != current_user.id:
        return "Unauthorized", 403
        
    coll_id = request.form.get("collection_id")
    if coll_id:
        paper.collection_id = coll_id if coll_id != "None" else None
        
    tag_ids = request.form.getlist("tag_ids")
    if tag_ids:
        paper.tags = []
        for tid in tag_ids:
            tag = db.session.get(Tag, int(tid))
            if tag and tag.user_id == current_user.id:
                paper.tags.append(tag)
                
    db.session.commit()
    flash("Paper organization updated!")
    return redirect(url_for('history'))

@app.route("/compare", methods=["POST"])
@login_required
def compare_papers():
    paper_ids = request.form.getlist("paper_ids")
    if not paper_ids:
        flash("Select papers to compare!")
        return redirect(url_for('history'))
        
    papers = []
    for pid in paper_ids:
        paper = db.get_or_404(ResearchPaper, int(pid))
        if paper.user_id == current_user.id:
            papers.append(paper)
            
    return render_template("compare.html", papers=papers)
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Non-interactive backend

@app.route("/knowledge_graph")
@login_required
def knowledge_graph():
    """Phase 4: Visualize relationships between papers"""
    papers = db.session.execute(
        db.select(ResearchPaper).filter_by(user_id=current_user.id)
    ).scalars().all()
    
    G = nx.Graph()
    for p in papers:
        G.add_node(p.id, label=p.filename[:15], domain=p.domain)
        # Connect papers in the same collection
        if p.collection_id:
            for other in papers:
                if other.id != p.id and other.collection_id == p.collection_id:
                    G.add_edge(p.id, other.id, type='collection')
        
        # Connect papers with same domain
        for other in papers:
            if other.id != p.id and other.domain == p.domain:
                G.add_edge(p.id, other.id, type='domain')
                
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, labels={node: data['label'] for node, data in G.nodes(data=True)}, 
            node_color='skyblue', node_size=2000, font_size=10, font_weight='bold')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return send_file(img, mimetype='image/png')

@app.route("/api/trends")
@login_required
def get_trends():
    """API for corpus-level analysis (Slide 10 visualization)."""
    # Modern SQLAlchemy 2.0 syntax
    papers = db.session.execute(
        db.select(ResearchPaper).filter_by(user_id=current_user.id)
    ).scalars().all()
    
    # 1. Domain Distribution
    domains = {}
    for p in papers:
        domains[p.domain] = domains.get(p.domain, 0) + 1
        
    # 2. Algorithm Frequency
    algo_counts = {}
    for p in papers:
        if p.algos:
            # Safer parsing: handle empty strings or malformed data
            raw_algos = [a.strip() for a in p.algos.split(',') if a.strip()]
            for a in raw_algos:
                if len(a) > 2: # Filter out noise
                    algo_counts[a] = algo_counts.get(a, 0) + 1
    
    # Sort and take top 5
    top_algos = dict(sorted(algo_counts.items(), key=lambda x: x[1], reverse=True)[:5])
    
    return jsonify({
        "domains": domains,
        "top_algos": top_algos,
        "count": len(papers)
    })

if __name__ == "__main__":
    with app.app_context():
        try:
            db.session.execute(db.text("SELECT 1"))
            print()
            print("==========================================")
            print("       RESEARCHMIND AI")
            print("==========================================")
            print("MYSQL CONNECTION : SUCCESS")
            print("DATABASE         : researchmind")
            print("USER             : researchmind_user")
            print("==========================================")
            print()
        except Exception as e:
            print()
            print("==========================================")
            print("MYSQL CONNECTION : FAILED")
            print("==========================================")
            print(e)
            print()
            raise

    app.run(debug=True, use_reloader=True)
