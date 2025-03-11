import unittest
import numpy as np
import os
import joblib

# Import functions from your domain_classifier module
from domain_classifier import (
    train_domain_classifier,
    load_domain_classifier,
    predict_domain,
    evaluate_domain_classifier,
    MODEL_PATH
)

# Create a dummy dataset for testing using textual data
dummy_texts = [
    "This is a legal document regarding contracts.",
    "The financial report shows significant growth.",
    "Medical records indicate patient recovery.",
    "A legal case was heard in court.",
    "Educational materials for college students.",
    "The financial statements were audited.",
    "Legal documents require precise language.",
    "Medical procedures require careful attention.",
    "Educational reports show improvement.",
    "Financial analysis indicates risks."
]
# Map domains to integer labels: legal:0, financial:1, medical:2, educational:3
dummy_labels = [0, 1, 2, 0, 3, 1, 0, 2, 3, 1]

class TestDomainClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Train the classifier using textual inputs.
        cls.classifier = train_domain_classifier(dummy_texts, dummy_labels)

    def test_train_and_save(self):
        # Check that the model file was created
        self.assertTrue(os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}")
        # Load the model and check if it's successfully loaded.
        loaded_model = load_domain_classifier(num_labels=len(set(dummy_labels)))
        self.assertIsNotNone(loaded_model)
    
    def test_predict_domain(self):
        # Use a dummy text input for prediction.
        dummy_text = "The contract was legally binding."
        # Call predict_domain with the text and the classifier instance.
        predicted_label = predict_domain(dummy_text, self.classifier)
        # Check that a label (an integer) is returned.
        self.assertIsInstance(predicted_label, int)
    
    def test_evaluate_classifier(self):
        # Evaluate classifier on dummy data.
        report = evaluate_domain_classifier(dummy_texts, dummy_labels, self.classifier)
        self.assertIsInstance(report, dict)
    
    @classmethod
    def tearDownClass(cls):
        # Optionally, remove the model file after tests.
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)

if __name__ == "__main__":
    unittest.main()
