import re
from research_ml import ResearchKnowledgeAgent

def extract_entities(text):
    """Smart heuristic-based extraction for Algorithms, Data Structures, and Datasets."""
    algo_keywords = [
        "Dijkstra", "Breadth First Search", "Depth First Search", "BFS", "DFS", "Sorting", 
        "Merge Sort", "Quick Sort", "Binary Search", "Dynamic Programming", "Recursion",
        "Backtracking", "Greedy Algorithm", "Prim's", "Kruskal's", "A* Search", "Bellman-Ford",
        "Knowledge-Based Agent", "Forward Chaining", "Constraint Satisfaction", "CSP"
    ]
    
    dataset_keywords = [
        "MNIST", "ImageNet", "CIFAR", "COCO", "GLUE", "SQuAD", "IMDb", "Kaggle", 
        "PASCAL VOC", "UCI Repository", "MIMIC", "Common Crawl", "WordNet", "Cityscapes"
    ]

    found_algos = [w for w in algo_keywords if re.search(r'\b' + re.escape(w) + r'\b', text, re.IGNORECASE)]
    found_datasets = [w for w in dataset_keywords if re.search(r'\b' + re.escape(w) + r'\b', text, re.IGNORECASE)]

    patterns = {
        "algo": [r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(?:algorithm|method|model|approach)'],
        "dataset": [r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(?:dataset|data set|repository|corpus)'],
        "metric": [r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(?:metric|score|measure|index)']
    }
    
    metric_keywords = ["Accuracy", "Precision", "Recall", "F1-Score", "mAP", "IOU", "BLEU", "ROUGE", "METEOR", "AUC", "ROC", "MSE", "MAE", "RMSE", "Perplexity"]
    found_metrics = [w for w in metric_keywords if re.search(r'\b' + re.escape(w) + r'\b', text, re.IGNORECASE)]

    for label, p_list in patterns.items():
        for p in p_list:
            matches = re.findall(p, text)
            if label == "algo": found_algos.extend(matches)
            elif label == "dataset": found_datasets.extend(matches)
            else: found_metrics.extend(matches)

    final_algos = sorted(list(set(found_algos)), key=len, reverse=True)[:6]
    final_datasets = sorted(list(set(found_datasets)), key=len, reverse=True)[:4]
    final_metrics = sorted(list(set(found_metrics)), key=len, reverse=True)[:5]
    
    return final_algos, final_datasets, final_metrics
