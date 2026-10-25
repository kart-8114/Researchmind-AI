import unittest
from rules import forward_chain, forward_chain_project
from recommendation import best_first_search_plans, solve_resource_csp
from planner_v2 import generate_experiment_design, htn_project_roadmap


class TestResearchMindAI(unittest.TestCase):

    def test_forward_chain_basic(self):
        result = forward_chain("Computer Vision", ["YOLO"], ["small dataset", "latency"])
        self.assertIn("Unit", result)

    def test_forward_chain_project_rules(self):
        domain = "Computer Vision"
        algorithms = ["CNN"]
        datasets = ["ImageNet"]
        metrics = ["Accuracy"]
        gaps = ["Computational efficiency vs accuracy trade-off"]
        limitations = ["high compute cost"]
        candidates = [{
            "title": "Efficient CNN for Computer Vision",
            "problem_statement": "Addressing computational efficiency in Computer Vision using CNN on ImageNet.",
            "algorithms": ["CNN"],
            "datasets": ["ImageNet"],
            "metrics": ["Accuracy"],
            "future_scope": ["Deployment"]
        }]
        selected, scored = forward_chain_project(domain, algorithms, datasets, metrics, gaps, limitations, candidates)
        self.assertEqual(selected["title"], "Efficient CNN for Computer Vision")
        self.assertGreater(selected["forward_chain_score"], 0.0)
        self.assertTrue(len(selected["forward_chain_reasons"]) > 0)

    def test_best_first_search_ranking(self):
        plans = [
            {"id": "plan_1", "title": "Plan A", "feasibility": 0.5, "novelty": 0.5, "resource_suitability": 0.5},
            {"id": "plan_2", "title": "Plan B", "feasibility": 0.9, "novelty": 0.8, "resource_suitability": 0.9}
        ]
        ranked = best_first_search_plans(plans)
        self.assertEqual(ranked[0]["id"], "plan_2")
        self.assertIn("score_components", ranked[0])

    def test_experiment_design(self):
        plan = {"title": "Test Plan"}
        design = generate_experiment_design("Computational efficiency vs. accuracy trade-off", "Computer Vision", plan)
        self.assertIn("research_question", design)
        self.assertIn("proposed_method", design)
        self.assertIn("evaluation_metrics", design)

    def test_htn_roadmap(self):
        roadmap = htn_project_roadmap("Test Project", "Computer Vision", ["CNN"], ["ImageNet"])
        self.assertIn("Project Definition", roadmap)
        self.assertIn("Dataset", roadmap)
        self.assertIn("Evaluation", roadmap)


if __name__ == "__main__":
    unittest.main()
