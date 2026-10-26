import heapq

def a_star_milestones(start_state, goal_state, domain):
    milestones = ["Dataset", "Preprocessing", "Baseline", "Proposed Method", "Evaluation", "Comparison"]
    domain_modifiers = {
        "Computer Vision": {"cost": 1.5, "reward": 1.2},
        "Natural Language Processing": {"cost": 1.3, "reward": 1.1},
        "Healthcare": {"cost": 1.8, "reward": 1.5}
    }
    mod = domain_modifiers.get(domain, {"cost": 1.0, "reward": 1.0})

    def heuristic(state):
        try:
            return len(milestones) - milestones.index(state)
        except Exception:
            return 10

    open_set = []
    heapq.heappush(open_set, (heuristic(start_state), start_state, [], 0))
    visited = {}

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
            step_cost = 2 * mod["cost"]
            new_cost = cost + step_cost
            h = heuristic(neighbor)
            heapq.heappush(open_set, (new_cost + h, neighbor, path + [current], new_cost))
            
    return milestones


def htn_project_roadmap(project_title, domain, algorithms=None, datasets=None):
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
