# ResearchMind AI — OpenAI + Syllabus Algorithms

This version follows the requested architecture:

| Component | Implementation |
|---|---|
| PDF extraction | PyMuPDF |
| Paper understanding | OpenAI Responses API |
| Summary | OpenAI |
| Information extraction | OpenAI Structured Outputs |
| Research-gap analysis | OpenAI |
| Project recommendation | OpenAI candidates + **Forward Chaining** |
| Research-plan ranking | **Best-First Search** |
| Project roadmap | **HTN Planning** |
| Similar papers | Local semantic/RAG infrastructure + arXiv/knowledge corpus |
| Web research | arXiv API + CrossRef metadata |
| Paper Q&A | Sentence-Transformer retrieval + OpenAI |

## 1. Install

Use Python 3.10+ (your existing Python 3.14 environment is supported by the project dependencies if all packages install correctly).

```powershell
cd ResearchMindAI
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Configure `.env`

Copy `.env.example` to `.env` and add your own values:

```text
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-5.6-luna

MYSQL_USER=root
MYSQL_PASSWORD=your_mysql_password
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=researchmind

SECRET_KEY=change_this_secret_key
```

Never commit `.env` or an API key to GitHub.

## 3. Run MySQL

Create the database before starting the Flask application:

```sql
CREATE DATABASE researchmind;
```

Make sure MySQL is running and the credentials in `.env` are correct.

## 4. Run

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## 5. What happens after a PDF upload?

```text
PDF
 ↓
PyMuPDF text extraction
 ↓
OpenAI analysis
 ├── summary
 ├── domain
 ├── algorithms
 ├── datasets
 ├── metrics
 ├── limitations
 └── research gaps
 ↓
OpenAI generates 3 project candidates
 ↓
Forward Chaining selects a candidate from extracted facts
 ↓
Best-First Search ranks research plans
 ↓
HTN decomposes the selected project into implementation tasks
 ↓
Dashboard
```

## 6. Why OpenAI does not replace the syllabus algorithms

OpenAI is used for language understanding. The classical algorithms are implemented locally so they can be demonstrated during the AI-subject viva:

- **Forward Chaining:** selects a project using explicit IF-condition/fact rules.
- **Best-First Search:** ranks project candidates with a heuristic priority queue.
- **HTN Planning:** decomposes the selected project into hierarchical tasks.

## 7. Fallback behavior

If `OPENAI_API_KEY` is missing or the OpenAI package is not installed, the project falls back to its existing local NLP/Transformer components where possible. This makes local development easier, but the OpenAI features are enabled only after configuring the API key.
