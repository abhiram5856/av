import unittest
from backend.models.concern_scorer import MultimodalConcernScorer

class TestMultimodalConcernScore(unittest.TestCase):
    def setUp(self):
        self.scorer = MultimodalConcernScorer()
        
    def test_high_confidence_strong_env(self):
        env = {"overall_compatibility": "Strongly Compatible"}
        result = self.scorer.calculate_concern(
            disease_name="tomato_late_blight",
            is_healthy=False,
            ml_confidence=0.95,
            lesion_area_ratio=0.45,
            environmental_evidence=env,
            growth_stage="Flowering",
            image_quality_status="acceptable"
        )
        # Expected: conf (33.25) + visual (25) + env (20) + growth (10) + quality (10) = 98.25
        self.assertTrue(result["score"] > 90.0)
        self.assertEqual(result["level"], "Critical Attention Required")
        self.assertFalse(result["environmental_conflict"])
        
    def test_high_confidence_env_conflict(self):
        env = {"overall_compatibility": "Incompatible"}
        result = self.scorer.calculate_concern(
            disease_name="tomato_late_blight",
            is_healthy=False,
            ml_confidence=0.91,
            lesion_area_ratio=0.45,
            environmental_evidence=env,
            growth_stage="Flowering",
            image_quality_status="acceptable"
        )
        # Expected: conf (31.85) + visual (25) + env (0) + growth (10) + quality (10) = 76.85
        self.assertTrue(result["score"] >= 75.0)
        self.assertTrue(result["environmental_conflict"])
        self.assertIn("Environmental conditions are incompatible with expected disease habitat.", result["limiting_factors"])

    def test_low_confidence(self):
        env = {"overall_compatibility": "Insufficient Evidence"}
        result = self.scorer.calculate_concern(
            disease_name="tomato_late_blight",
            is_healthy=False,
            ml_confidence=0.30,
            lesion_area_ratio=0.05,
            environmental_evidence=env,
            growth_stage="Unknown",
            image_quality_status="poor"
        )
        # Expected: conf (10.5) + visual (5) + env (0) + growth (0) + quality (0) = 15.5
        self.assertEqual(result["level"], "Insufficient Evidence")
        self.assertEqual(result["evidence_quality"], "Poor")
        
    def test_missing_env_data(self):
        env = {"overall_compatibility": "Insufficient Evidence"}
        result = self.scorer.calculate_concern(
            disease_name="tomato_late_blight",
            is_healthy=False,
            ml_confidence=0.85,
            lesion_area_ratio=0.20,
            environmental_evidence=env,
            growth_stage="Vegetative",
            image_quality_status="acceptable"
        )
        self.assertEqual(result["evidence_quality"], "Insufficient")
        # Should NOT be 'Insufficient Evidence' level because confidence is high (0.85 > 0.6)
        self.assertNotEqual(result["level"], "Insufficient Evidence")
        
    def test_poor_image_quality(self):
        env = {"overall_compatibility": "Compatible"}
        result = self.scorer.calculate_concern(
            disease_name="tomato_late_blight",
            is_healthy=False,
            ml_confidence=0.80,
            lesion_area_ratio=0.15,
            environmental_evidence=env,
            growth_stage="Flowering",
            image_quality_status="poor"
        )
        self.assertEqual(result["evidence_quality"], "Poor")
        self.assertIn("Poor image quality reduces diagnostic reliability.", result["limiting_factors"])

    def test_unknown_growth_stage(self):
        env = {"overall_compatibility": "Compatible"}
        result = self.scorer.calculate_concern(
            disease_name="tomato_late_blight",
            is_healthy=False,
            ml_confidence=0.80,
            lesion_area_ratio=0.15,
            environmental_evidence=env,
            growth_stage="Unknown",
            image_quality_status="acceptable"
        )
        self.assertIn("Growth stage is unknown.", result["limiting_factors"])

    def test_healthy_prediction(self):
        env = {"overall_compatibility": "Partially Compatible"}
        result = self.scorer.calculate_concern(
            disease_name="tomato_healthy",
            is_healthy=True,
            ml_confidence=0.98,
            lesion_area_ratio=0.01,
            environmental_evidence=env,
            growth_stage="Flowering",
            image_quality_status="acceptable"
        )
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["level"], "Low Concern")

    def test_strong_gradcam_evidence(self):
        env = {"overall_compatibility": "Compatible"}
        result = self.scorer.calculate_concern(
            disease_name="tomato_late_blight",
            is_healthy=False,
            ml_confidence=0.85,
            lesion_area_ratio=0.55,
            environmental_evidence=env,
            growth_stage="Flowering",
            image_quality_status="acceptable"
        )
        self.assertEqual(result["visual_evidence_level"], "Strong")

    def test_weak_gradcam_evidence(self):
        env = {"overall_compatibility": "Compatible"}
        result = self.scorer.calculate_concern(
            disease_name="tomato_late_blight",
            is_healthy=False,
            ml_confidence=0.85,
            lesion_area_ratio=0.05,
            environmental_evidence=env,
            growth_stage="Flowering",
            image_quality_status="acceptable"
        )
        self.assertEqual(result["visual_evidence_level"], "Weak")
        
    def test_deterministic(self):
        env = {"overall_compatibility": "Compatible"}
        result1 = self.scorer.calculate_concern(
            disease_name="tomato_late_blight",
            is_healthy=False,
            ml_confidence=0.80,
            lesion_area_ratio=0.15,
            environmental_evidence=env,
            growth_stage="Flowering",
            image_quality_status="acceptable"
        )
        result2 = self.scorer.calculate_concern(
            disease_name="tomato_late_blight",
            is_healthy=False,
            ml_confidence=0.80,
            lesion_area_ratio=0.15,
            environmental_evidence=env,
            growth_stage="Flowering",
            image_quality_status="acceptable"
        )
        self.assertEqual(result1, result2)

if __name__ == "__main__":
    unittest.main()
