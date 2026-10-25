import heapq


def best_first_search_plans(candidate_plans):
    """Best-First Search using a heuristic priority queue with explicit score breakdown.

    h(n) = 0.4 * feasibility + 0.4 * novelty + 0.2 * resource_suitability.
    """
    heap = []
    for plan in candidate_plans:
        feas = float(plan.get("feasibility", 0.5))
        nov = float(plan.get("novelty", 0.5))
        res = float(plan.get("resource_suitability", 0.5))
        score = (feas * 0.4) + (nov * 0.4) + (res * 0.2)
        
        plan = dict(plan)
        plan["heuristic_score"] = round(score, 3)
        plan["score_components"] = {
            "feasibility": {"weight": 0.4, "value": feas, "weighted": round(feas * 0.4, 3)},
            "novelty": {"weight": 0.4, "value": nov, "weighted": round(nov * 0.4, 3)},
            "resource_suitability": {"weight": 0.2, "value": res, "weighted": round(res * 0.2, 3)}
        }
        plan_id = str(plan.get("id", plan.get("title", "plan")))
        # Deterministic tie-breaking: negative score, then plan_id ascending
        heapq.heappush(heap, (-score, plan_id, plan))

    ranked = []
    while heap:
        _, _, plan = heapq.heappop(heap)
        ranked.append(plan)
    return ranked


def generate_dynamic_plans(domain, gaps, ai_candidates=None):
    """Build Best-First candidates from OpenAI suggestions and rank them locally."""
    candidates = []

    for i, candidate in enumerate(ai_candidates or []):
        c = dict(candidate)
        c.setdefault("description", c.get("problem_statement", "AI-generated research direction."))
        c.setdefault("id", f"openai_plan_{i}")
        c.setdefault("feasibility", 0.6)
        c.setdefault("novelty", 0.6)
        c.setdefault("resource_suitability", 0.6)
        candidates.append(c)

    if not candidates:
        for i, gap in enumerate(gaps[:3]):
            candidates.append({
                "title": f"Addressing Gap: {gap}",
                "description": f"Proposed research to solve the {gap.lower()} limitation in {domain}.",
                "feasibility": 0.5 + (0.1 * i),
                "novelty": 0.9 - (0.1 * i),
                "resource_suitability": 0.7,
                "id": f"fallback_plan_{i}"
            })

    return best_first_search_plans(candidates)


# Unit 2: BFS / DFS for graph traversal

def bfs_related_nodes(graph, start_node, max_depth=2):
    visited = {start_node}
    queue = [(start_node, 0)]
    result = []

    while queue:
        node, depth = queue.pop(0)
        if depth > 0:
            result.append(node)
        if depth >= max_depth:
            continue
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
    return result


# Unit 3: CSP + Backtracking for resource allocation

def solve_resource_csp(selected_plan, constraints):
    hardware_options = ["High GPU", "Medium GPU", "CPU Only"]
    budget_constraints = {"High GPU": 100, "Medium GPU": 50, "CPU Only": 0}

    def is_valid(assignment):
        hw = assignment.get("hardware")
        if hw:
            if constraints.get("gpu") == "Low" and hw == "High GPU":
                return False
            if constraints.get("budget") == "Zero" and budget_constraints[hw] > 0:
                return False
        return True

    assignment = {}
    for hw in hardware_options:
        assignment["hardware"] = hw
        if is_valid(assignment):
            return assignment
    return {"hardware": "CPU Only", "status": "Sub-optimal (Backtracked)"}
