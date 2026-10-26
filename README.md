# ResearchMind AI — Evidence-Grounded Multi-Paper Research Assistant

ResearchMind AI is an advanced, explainable, evidence-grounded research assistant designed to ingest scientific literature, extract structured facts with precise page provenance, perform cross-paper comparison, and orchestrate classical AI reasoning (Forward Chaining, Best-First Search heuristic planning, and HTN task decomposition).

---

## 🚀 Key Features

1. **Page-Preserving PDF Ingestion & Provenance**
   - Extracts structured facts (Problem, Algorithms, Datasets, Metrics, Limitations, Future Work) from research papers using PyMuPDF and optional OpenAI structured outputs.
   - Tracks exact page references (`page_number`) and supporting text excerpts (`excerpt`) for every extracted claim.

2. **Cross-Paper Evidence Comparison Matrix**
   - Compare multiple papers in a collection side-by-side across core research dimensions, highlighting agreements, conflicts, missing evidence, and source page provenance.

3. **Evidence-Backed Gap Inference with Forward Chaining**
   - Derives candidate research gaps from cross-paper evidence and applies explicit, inspectable Forward Chaining rules (`rules.py`) with full antecedent tracking and rule traceability.

4. **Explainable Best-First Search & Heuristic Planning**
   - Ranks research plans using a transparent Best-First Search heuristic:
     $$\mathbf{h = 0.4F + 0.4N + 0.2R}$$
     - **Feasibility (`F`)**: Evaluated dynamically based on extracted algorithms and datasets.
     - **Novelty (`N`)**: Evaluated based on identified research gaps and limitations.
     - **Resource Suitability (`R`)**: Evaluated based on metrics presence and domain requirements.

5. **Honest Experiment Design & HTN Roadmaps**
   - Generates structured experiment specifications validated against extracted evidence, accompanied by Hierarchical Task Network (HTN) milestone and task decomposition.

---

## 🛠️ Installation & Setup

### 1. Clone & Setup Virtual Environment
```powershell
git clone https://github.com/kart-8114/Researchmind-AI.git
cd Researchmind-AI
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your configuration:
```powershell
Copy-Item .env.example .env
```
Set your MySQL database credentials and optional `OPENAI_API_KEY`.

### 3. Initialize MySQL Database
Make sure MySQL is running, then create the database:
```sql
CREATE DATABASE researchmind;
```

---

## ▶️ How to Run

Start the Flask application:
```powershell
python app.py
```

Open your browser at:
```text
http://127.0.0.1:5000
```

---

## 🧪 Running Tests
Execute the automated unit test suite:
```powershell
python test_researchmind.py
```
