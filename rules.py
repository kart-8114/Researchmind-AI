"""Classical AI reasoning used by ResearchMind AI.

The OpenAI model proposes candidate ideas. These deterministic rules perform
Forward Chaining over extracted facts so the project still contains a
syllabus-aligned reasoning algorithm rather than delegating everything to an LLM.
"""


def forward_chain(domain, algorithms, limitations):
    """Backward-compatible syllabus Forward Chaining rule engine."""
    facts = {
        "is_vision": domain == "Computer Vision",
        "is_nlp": domain == "Natural Language Processing",
        "uses_search": any(x.lower() in str(algorithms).lower() for x in ["bfs", "dfs", "heuristic", "a*", "best-first"]),
        "uses_inference": any(x.lower() in str(algorithms).lower() for x in ["agent", "forward chaining", "reasoning"]),
        "limited_data": any(x in str(limitations).lower() for x in ["data", "diversity", "small dataset", "generalization"]),
        "limited_speed": any(x in str(limitations).lower() for x in ["latency", "speed", "slow", "compute"])
    }

    if facts["is_vision"] and facts["limited_data"]:
        return "Unit 3: Knowledge-Based Agent Scaling"
    if facts["is_nlp"] and facts["limited_speed"]:
        return "Unit 4: Efficient Planning for Low-Latency Inference"
    if facts["is_vision"] and facts["uses_search"]:
        return "Unit 2: Heuristic Search Optimization"
    if facts["is_nlp"] and facts["uses_inference"]:
        return "Unit 3: Advanced Forward Chaining Reasoning"
    return "Unit 1: General Intelligent Agent Scaling"


def forward_chain_project(domain, algorithms, datasets, metrics, gaps, limitations, candidates):
    """Select one OpenAI-generated candidate using explicit Forward Chaining rules.

    Each rule adds a score to candidates when its antecedent facts are true.
    This is deliberately transparent for an AI-subject viva.
    """
    alg_text = " ".join(algorithms or []).lower()
    data_text = " ".join(datasets or []).lower()
    metric_text = " ".join(metrics or []).lower()
    gap_text = " ".join(gaps or []).lower()
    limit_text = " ".join(limitations or []).lower()

    scored = []
    for candidate in candidates or []:
        title = candidate.get("title", "AI Research Project")
        candidate_text = " ".join([
            title,
            candidate.get("problem_statement", ""),
            " ".join(candidate.get("algorithms", [])),
            " ".join(candidate.get("datasets", [])),
        ]).lower()

        score = 0.0
        reasons = []

        # Rule 1: domain compatibility
        if domain.lower() in candidate_text:
            score += 0.25
            reasons.append("Domain matches the extracted research domain")
        else:
            score += 0.10

        # Rule 2: extracted algorithms are useful to the project
        if any(a.strip().lower() in candidate_text for a in algorithms if a.strip()):
            score += 0.20
            reasons.append("Uses an algorithm found in the literature")

        # Rule 3: extracted datasets are useful to the project
        if any(d.strip().lower() in candidate_text for d in datasets if d.strip()):
            score += 0.15
            reasons.append("Uses an extracted or related dataset")

        # Rule 4: project addresses an identified gap/limitation
        if any(word in candidate_text for word in ["efficien", "generaliz", "robust", "scalab", "latency", "resource", "optimization", "gap"]):
            if any(word in gap_text or word in limit_text for word in ["efficien", "generaliz", "robust", "scalab", "latency", "resource", "optimization", "data"]):
                score += 0.20
                reasons.append("Addresses an extracted research gap or limitation")

        # Rule 5: evaluation compatibility
        if any(m.strip().lower() in candidate_text for m in metrics if m.strip()):
            score += 0.10
            reasons.append("Evaluation metric is supported by the literature")

        # Rule 6: knowledge-based fallback facts
        if "search" in alg_text or "tree" in data_text:
            if "search" in candidate_text:
                score += 0.10
                reasons.append("Search-related fact triggered the rule")

        candidate = dict(candidate)
        candidate["forward_chain_score"] = round(min(score, 1.0), 3)
        candidate["forward_chain_reasons"] = reasons
        scored.append(candidate)

    if not scored:
        return {
            "title": f"AI-Assisted {domain} Research Project",
            "problem_statement": "Generate a project from the analyzed literature.",
            "algorithms": algorithms,
            "datasets": datasets,
            "metrics": metrics,
            "future_scope": ["Expand the literature corpus", "Evaluate on additional datasets"],
            "rationale": "No OpenAI candidates were available; deterministic fallback used.",
            "forward_chain_score": 0.0,
            "forward_chain_reasons": ["Fallback rule"]
        }, []

    scored.sort(key=lambda x: x["forward_chain_score"], reverse=True)
    return scored[0], scored
