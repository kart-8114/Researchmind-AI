import unittest
from services.planner import htn_project_roadmap, a_star_milestones

class TestPlanner(unittest.TestCase):
    def test_htn_roadmap(self):
        roadmap = htn_project_roadmap("Test Title", "Computer Vision", ["YOLO"], ["COCO"])
        self.assertIn("Dataset", roadmap)
        self.assertIn("Evaluation", roadmap)

    def test_astar_milestones(self):
        path = a_star_milestones("Dataset", "Evaluation", "Computer Vision")
        self.assertTrue(len(path) > 0)

if __name__ == "__main__":
    unittest.main()
