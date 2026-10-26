from reasoning.best_first_search import best_first_search_plans

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


def generate_dynamic_plans(domain, gaps, ai_candidates=None, paper_features=None):
    """Build Best-First candidates from OpenAI suggestions or local fallback and rank them using dynamic evidence-based scores."""
    candidates = []
    paper_features = paper_features or {}
    algos = paper_features.get("algos", "")
    datasets = paper_features.get("datasets", "")
    metrics = paper_features.get("metrics", "")
    citations = paper_features.get("citations", 0)

    for i, candidate in enumerate(ai_candidates or []):
        c = dict(candidate)
        c.setdefault("description", c.get("problem_statement", "AI-generated research direction."))
        c.setdefault("id", f"openai_plan_{i}")
        
        feas = 0.6
        if algos and algos != "N/A": feas += 0.15
        if datasets and datasets != "N/A": feas += 0.1
        c.setdefault("feasibility", round(min(0.95, feas), 2))
        
        nov = 0.65 + (0.05 * i)
        if "efficiency" in str(gaps).lower() or "generalization" in str(gaps).lower():
            nov += 0.1
        c.setdefault("novelty", round(min(0.95, nov), 2))
        
        res = 0.6
        if metrics and metrics != "N/A": res += 0.2
        if citations > 5: res += 0.1
        c.setdefault("resource_suitability", round(min(0.95, res), 2))
        
        candidates.append(c)

    if not candidates:
        for i, gap in enumerate(gaps[:3]):
            feas = 0.5 + (0.08 * i)
            if algos and algos != "N/A": feas += 0.15
            
            nov = 0.85 - (0.08 * i)
            
            res = 0.65
            if datasets and datasets != "N/A": res += 0.15

            candidates.append({
                "title": f"Addressing Gap: {gap}",
                "description": f"Proposed research to solve the {gap.lower()} limitation in {domain}.",
                "feasibility": round(min(0.95, feas), 2),
                "novelty": round(min(0.95, nov), 2),
                "resource_suitability": round(min(0.95, res), 2),
                "id": f"fallback_plan_{i}"
            })

    return best_first_search_plans(candidates)


def generate_experiment_design(gap, domain, selected_plan):
    """Automated Experiment Design Generator."""
    domain_metrics = {
        "Computer Vision": ["mAP (Mean Average Precision)", "IoU (Intersection over Union)", "Inference Latency (FPS)", "Parameters/FLOPs"],
        "Natural Language Processing": ["BLEU Score", "ROUGE-L", "Perplexity", "F1-Score / Execution Time"],
        "Healthcare": ["AUC-ROC", "Sensitivity/Specificity", "F1-Score", "False Positive Rate"]
    }
    
    domain_datasets = {
        "Computer Vision": "MS COCO / ImageNet-1K / Cityscapes",
        "Natural Language Processing": "GLUE Benchmark / SQuAD v2.0 / IMDB",
        "Healthcare": "MIMIC-IV Clinical Database / BioASQ / CheXpert"
    }
    
    domain_baselines = {
        "Computer Vision": ["YOLOv8-Large / ResNet-152 Baseline", "SOTA Dense Transformer Networks"],
        "Natural Language Processing": ["Vanilla BERT-Base / RoBERTa", "Standard GPT-2 Fine-tuned Baselines"],
        "Healthcare": ["Standard Logistic Regression / Random Forest", "Vanilla 3D-UNet / DenseNet-121"]
    }

    metrics = domain_metrics.get(domain, ["Accuracy", "F1-Score", "Inference Cost / Latency"])
    dataset = domain_datasets.get(domain, "Standard Domain Benchmark Dataset")
    baselines = domain_baselines.get(domain, ["Standard SOTA Baseline Methods"])

    if "bottleneck" in gap.lower() or "performance" in gap.lower() or "real-time" in gap.lower() or "efficiency" in gap.lower() or "compute" in gap.lower() or "optimization" in gap.lower():
        research_question = "Can a lightweight approach maintain comparable accuracy with lower computational cost?"
        strong_difference = "Unlike standard approaches that optimize solely for accuracy, this explicitly targets the accuracy-compute Pareto frontier."
    elif "dataset" in gap.lower() or "generalization" in gap.lower():
        research_question = f"How effectively does the proposed {selected_plan['title']} generalize across multi-domain target datasets?"
        strong_difference = "Moves beyond localized single-dataset validation by incorporating cross-domain zero-shot evaluation protocols."
    else:
        research_question = f"How can we systematically bridge the discovered '{gap}' limitation using a structured {selected_plan['title']} design?"
        strong_difference = "Provides a first-of-its-kind direct optimization loop mapped from automated literature evidence."

    design = {
        "research_question": research_question,
        "proposed_method": f"{selected_plan['title']} Architecture Upgrade",
        "baseline_methods": baselines,
        "dataset": dataset,
        "evaluation_metrics": metrics,
        "experiments_to_perform": [
            f"1. Baseline vs. Proposed Method benchmark evaluation on {dataset}.",
            "2. Ablation study isolating individual architectural components.",
            "3. Stress-testing under constrained resource/data regimes."
        ],
        "comparison_criteria": "Statistical significance (p-value < 0.05), Pareto efficiency mapping.",
        "expected_observations": "The proposed method is expected to maintain task accuracy parity within 1-2% while achieving a 2x reduction in computational overhead.",
        "strong_difference": strong_difference,
        "hypothesis": f"Using the {selected_plan['title']} approach will directly address the '{gap}' gap in {domain}.",
        "target_improvement": "15-20% relative efficiency increase"
    }
    return design
