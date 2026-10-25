import pandas as pd
import numpy as np
import os
import kagglehub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class ResearchKnowledgeAgent:
    """
    Unit 3: Knowledge-Based Agent.
    Maintains a corpus of research facts and performs domain inference.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.classifier = LogisticRegression(max_iter=1000)
        self.corpus_df = None
        self.tfidf_matrix = None
        self.is_trained = False
        self.model_path = "instance/research_knowledge.pkl"
        
    def load_or_initialize(self):
        """Attempts to load knowledge base from cache or initializes from dataset."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.vectorizer = data['vectorizer']
                    self.classifier = data['classifier']
                    self.corpus_df = data['corpus_df']
                    self.tfidf_matrix = data['tfidf_matrix']
                    self.is_trained = True
                print("✅ Knowledge-Based Agent initialized from cache.")
                return
            except Exception as e:
                print(f"⚠️ Initialization failed: {e}. Re-indexing knowledge...")

        try:
            print("📥 Fetching research corpus for knowledge representation...")
            path = kagglehub.dataset_download("sumitm004/arxiv-scientific-research-papers-dataset")
            csv_file = None
            for file in os.listdir(path):
                if file.endswith(".csv"):
                    csv_file = os.path.join(path, file)
                    break
            
            if not csv_file:
                print("❌ No knowledge source found.")
                return

            df = pd.read_csv(csv_file).dropna(subset=['title', 'summary', 'category'])
            # Use a larger subset and shuffle for better knowledge coverage
            self.corpus_df = df.sample(n=min(10000, len(df)), random_state=42).copy()
            self.corpus_df['text'] = self.corpus_df['title'] + " " + self.corpus_df['summary']
            
            print("⚙️ Structuring Knowledge Representation...")
            X = self.vectorizer.fit_transform(self.corpus_df['text'])
            y = self.corpus_df['category']
            self.classifier.fit(X, y)
            
            self.tfidf_matrix = X
            self.is_trained = True
            
            # Cache the knowledge structure
            os.makedirs("instance", exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'vectorizer': self.vectorizer,
                    'classifier': self.classifier,
                    'corpus_df': self.corpus_df,
                    'tfidf_matrix': self.tfidf_matrix
                }, f)
            print("✅ Knowledge-Based Agent fully indexed.")
            
        except Exception as e:
            print(f"❌ Knowledge indexing failed: {e}")

    def infer_domain_category(self, text):
        """Unit 3: Inference based on structured knowledge."""
        if not self.is_trained:
            return "General AI", 0.0
        
        X = self.vectorizer.transform([text])
        probs = self.classifier.predict_proba(X)[0]
        max_idx = np.argmax(probs)
        return self.classifier.classes_[max_idx], probs[max_idx]

    def get_related_literature(self, text, top_n=5):
        """Unit 2: Heuristic similarity search across known knowledge."""
        if not self.is_trained or self.corpus_df is None:
            return []
        
        query_vec = self.vectorizer.transform([text])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        related_indices = similarities.argsort()[-top_n:][::-1]
        
        results = []
        for idx in related_indices:
            row = self.corpus_df.iloc[idx]
            arxiv_id = row.get('id') if 'id' in row else None
            title = row['title']
            
            # Bulletproof fallback url format
            url = f"https://arxiv.org/search/?query={title}&searchtype=title"
            if arxiv_id:
                try:
                    str_id = f"{arxiv_id}"
                    if '-' in str_id:
                        url = f"https://arxiv.org/abs/{str_id.replace('-', '/')}"
                    else:
                        url = f"https://arxiv.org/abs/{str_id}"
                except Exception:
                    pass
                
            results.append({
                "title": title,
                "category": row['category'],
                "similarity": float(similarities[idx]),
                "url": url
            })
        return results
