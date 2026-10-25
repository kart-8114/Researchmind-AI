import heapq

def a_star_milestones(start_state, goal_state, domain):
    """
    Unit 2: A* Search for finding the most efficient research milestone sequence.
    Now with simulated cost (GPU/Time) and reward (Novelty), influenced by research domain.
    """
    milestones = ["Dataset", "Preprocessing", "Baseline", "Proposed Method", "Evaluation", "Comparison"]
    
    # Domain-specific cost multipliers
    domain_modifiers = {
        "Computer Vision": {"cost": 1.5, "reward": 1.2},  # High compute cost
        "Natural Language Processing": {"cost": 1.3, "reward": 1.1},
        "Healthcare": {"cost": 1.8, "reward": 1.5}  # High sensitivity/verification cost
    }
    mod = domain_modifiers.get(domain, {"cost": 1.0, "reward": 1.0})

    # Milestone-specific properties
    node_properties = {
        "Dataset": {"cost": 1 * mod["cost"], "reward": 2},
        "Preprocessing": {"cost": 2 * mod["cost"], "reward": 1},
        "Baseline": {"cost": 3 * mod["cost"], "reward": 3},
        "Proposed Method": {"cost": 5 * mod["cost"], "reward": 8 * mod["reward"]},
        "Evaluation": {"cost": 4 * mod["cost"], "reward": 4},
        "Comparison": {"cost": 2 * mod["cost"], "reward": 5}
    }

    def heuristic(state):
        try:
            # h(n): distance to goal in the ideal sequence
            return len(milestones) - milestones.index(state)
        except: return 10

    open_set = []
    # (priority, current_node, path, total_cost)
    heapq.heappush(open_set, (heuristic(start_state), start_state, [], 0))
    visited = {} # Stores min cost to reach a node

    while open_set:
        f, current, path, cost = heapq.heappop(open_set)
        
        if current == goal_state:
            return path + [current]
            
        if current in visited and visited[current] <= cost:
            continue
        visited[current] = cost
        
        curr_idx = milestones.index(current) if current in milestones else -1
        if curr_idx < len(milestones) - 1:
            neighbor = milestones[curr_idx + 1]
            # g(n): cost of current path
            step_cost = node_properties.get(neighbor, {"cost": 1})["cost"]
            new_cost = cost + step_cost
            h = heuristic(neighbor)
            heapq.heappush(open_set, (new_cost + h, neighbor, path + [current], new_cost))
            
    return milestones

def hierarchical_roadmap(milestones):
    """
    Unit 4: Hierarchical Planning.
    Decomposes milestones into granular implementation sub-tasks.
    """
    task_library = {
        "Dataset": [
            "Search repositories (Kaggle, UCI, arXiv)",
            "Verify license and ethical compliance",
            "Perform exploratory data analysis (EDA)",
            "Generate descriptive statistics"
        ],
        "Preprocessing": [
            "Clean missing values and outliers",
            "Apply normalization/standardization",
            "Implement domain-specific augmentation",
            "Serialize preprocessed tensors"
        ],
        "Baseline": [
            "Clone repository of SOTA implementation",
            "Configure environment dependencies",
            "Execute baseline training on subset",
            "Validate results against reported metrics"
        ],
        "Proposed Method": [
            "Implement core novel architecture",
            "Apply optimization (Quantization/Pruning)",
            "Fine-tune hyperparameters (Grid/Random Search)",
            "Document architectural innovations"
        ],
        "Evaluation": [
            "Run full-scale training session",
            "Calculate primary and secondary metrics",
            "Perform cross-validation (k=5)",
            "Generate error confusion matrices"
        ],
        "Comparison": [
            "Generate comparative visualizations",
            "Perform statistical significance tests",
            "Conduct ablation studies",
            "Finalize research paper draft"
        ]
    }
    
    roadmap = {}
    for ms in milestones:
        roadmap[ms] = task_library.get(ms, ["General research task"])
        
    return roadmap

def generate_experiment_design(gap, domain, selected_plan):
    """
    Phase 4: Automated Experiment Design Generator.
    Converts identified research gaps into structured, explainable experiment design specifications.
    """
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
    dataset = domain_datasets.get(domain, "Standard Domain Benchmark Dataset (via Extracted Metadata)")
    baselines = domain_baselines.get(domain, ["Standard SOTA Baseline Methods"])

    # Generate specialized Research Question based on the gap text
    if "bottleneck" in gap.lower() or "performance" in gap.lower() or "real-time" in gap.lower() or "efficiency" in gap.lower() or "compute" in gap.lower() or "optimization" in gap.lower():
        research_question = "Can a lightweight approach maintain comparable accuracy with lower computational cost?"
        strong_difference = "Unlike standard approaches that optimize solely for accuracy, this explicitly targets the accuracy-compute Pareto frontier to produce explainable, lightweight designs."
    elif "dataset" in gap.lower() or "generalization" in gap.lower():
        research_question = f"How effectively does the proposed {selected_plan['title']} generalize across multi-domain target datasets without encountering catastrophic forgetting?"
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
        "comparison_criteria": "Statistical significance (p-value < 0.05), Pareto efficiency mapping (Accuracy vs. Resource usage).",
        "expected_observations": f"The proposed method is expected to maintain task accuracy parity within 1-2% while achieving a 2x reduction in computational overhead.",
        "strong_difference": strong_difference,
        "hypothesis": f"Using the {selected_plan['title']} approach will directly address the '{gap}' gap in {domain} by optimizing underlying feature representations.",
        "target_improvement": "15-20% relative efficiency increase"
    }
    
    return design


def htn_project_roadmap(project_title, domain, algorithms=None, datasets=None):
    """Unit 4: Hierarchical Task Network planning for a recommended project.

    Top-level task -> subtasks. The structure is deterministic and explainable.
    """
    algorithms = algorithms or []
    datasets = datasets or []
    dataset_text = ", ".join(datasets[:3]) if datasets else "Select a suitable benchmark dataset"
    algorithm_text = ", ".join(algorithms[:3]) if algorithms else "Select an algorithm from the literature"

    return {
        "Project Definition": [
            f"Finalize project scope for: {project_title}",
            f"Confirm research domain: {domain}",
            "Define research question and success criteria"
        ],
        "Literature Review": [
            "Review uploaded papers and similar arXiv papers",
            "Verify extracted algorithms, datasets and metrics",
            "Document the identified research gap"
        ],
        "Dataset": [
            f"Select dataset: {dataset_text}",
            "Check license, quality and class distribution",
            "Perform exploratory data analysis"
        ],
        "Method Development": [
            f"Implement baseline using: {algorithm_text}",
            "Implement the proposed improvement",
            "Tune parameters and document the design"
        ],
        "Evaluation": [
            "Run baseline and proposed experiments",
            "Calculate the selected evaluation metrics",
            "Perform error analysis and comparison"
        ],
        "Finalization": [
            "Prepare result visualizations",
            "Document limitations and future scope",
            "Prepare the final research report"
        ]
    }
