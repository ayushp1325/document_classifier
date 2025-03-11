import os
import joblib
import logging
import numpy as np
from typing import Any, List, Tuple, Union, Optional, Dict, Set
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report
from scipy.sparse import spmatrix
import faiss
import json
from datetime import datetime
import os

MODEL_DIR = "models"
MODEL_FILENAME = "domain_classifier.pkl"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Type aliases and constants
FeatureMatrix = Union[np.ndarray, spmatrix]
CONFIDENCE_THRESHOLD = 0.7
BERT_MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 512  # Maximum token length for BERT

# Directory structure
MODEL_DIR = "models"
BERT_MODEL_PATH = os.path.join(MODEL_DIR, "bert_domain_classifier")
EMBEDDINGS_PATH = os.path.join(MODEL_DIR, "domain_embeddings")
UNKNOWN_DOCS_PATH = os.path.join(MODEL_DIR, "unknown_domains")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

# Create necessary directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UNKNOWN_DOCS_PATH, exist_ok=True)
os.makedirs(EMBEDDINGS_PATH, exist_ok=True)

class DocumentDataset(Dataset):
    """Dataset for document classification with BERT"""
    
    def __init__(self, texts: List[str], labels: Optional[List[int]] = None, tokenizer=None):
        """
        Initialize the dataset with texts and optional labels
        
        Args:
            texts: List of document texts
            labels: Optional list of numeric labels (for training)
            tokenizer: BERT tokenizer (if None, one will be loaded)
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained(BERT_MODEL_NAME)
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encodings['input_ids'][0],
            'attention_mask': encodings['attention_mask'][0]
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
            
        return item

class BERTDomainClassifier:
    """Domain classifier using BERT for document classification"""
    
    def __init__(self):
        """Initialize the BERT domain classifier"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.label_encoder = None  # For mapping between class names and indices
        self.index = None  # FAISS index for similarity search
        self.embeddings = None  # Document embeddings
        self.embedding_labels = None  # Labels for embeddings
        self.metadata = {
            "model_version": "1.0.0",
            "last_trained": None,
            "num_classes": 0,
            "classes": [],
            "num_documents": 0,
            "unknown_documents": 0
        }
        logger.info(f"Initialized BERT domain classifier (using device: {self.device})")
    
    def _load_or_initialize_model(self, num_labels: int):
        """Load existing model or initialize a new one"""
        if os.path.exists(BERT_MODEL_PATH) and os.path.isdir(BERT_MODEL_PATH):
            try:
                logger.info(f"Loading BERT model from {BERT_MODEL_PATH}")
                self.model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
                self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
                
                # Load metadata
                if os.path.exists(METADATA_PATH):
                    with open(METADATA_PATH, 'r') as f:
                        self.metadata = json.load(f)
                        
                logger.info(f"Loaded model version {self.metadata['model_version']} "
                           f"trained on {self.metadata['num_documents']} documents "
                           f"with {self.metadata['num_classes']} classes")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}", exc_info=True)
                logger.warning("Initializing new model instead")
                self._initialize_new_model(num_labels)
        else:
            logger.info("No existing model found, initializing new one")
            self._initialize_new_model(num_labels)
    
    def _initialize_new_model(self, num_labels: int):
        """Initialize a new BERT model"""
        logger.info(f"Initializing new BERT model with {num_labels} labels")
        self.model = BertForSequenceClassification.from_pretrained(
            BERT_MODEL_NAME, 
            num_labels=num_labels
        )
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
        
    def _save_model(self):
        """Save the model and metadata"""
        try:
            # Save model and tokenizer
            self.model.save_pretrained(BERT_MODEL_PATH)
            self.tokenizer.save_pretrained(BERT_MODEL_PATH)
            
            # Update metadata
            self.metadata["last_trained"] = datetime.now().isoformat()
            
            # Save metadata
            with open(METADATA_PATH, 'w') as f:
                json.dump(self.metadata, f, indent=2)
                
            logger.info(f"Model and metadata saved to {MODEL_DIR}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}", exc_info=True)
            raise
    
    def _initialize_faiss_index(self, dimension: int):
        """Initialize FAISS index for similarity search"""
        self.index = faiss.IndexFlatL2(dimension)
        logger.info(f"Initialized FAISS index with dimension {dimension}")
    
    def _extract_embeddings(self, texts: List[str]) -> np.ndarray:
        """Extract BERT embeddings for a list of texts"""
        self.model.eval()
        dataset = DocumentDataset(texts, tokenizer=self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=8)
        
        all_embeddings = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                # Use [CLS] token embedding from last layer as document embedding
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)
    
    def train(self, texts: List[str], labels: List[str], epochs: int = 3):
        """
        Train the BERT domain classifier
        
        Args:
            texts: List of document texts
            labels: List of domain labels
            epochs: Number of training epochs
        """
        logger.info(f"Starting training on {len(texts)} documents with {len(set(labels))} unique labels")
        
        # Create label encoder (map string labels to integers)
        unique_labels = sorted(list(set(labels)))
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        id_to_label = {i: label for i, label in enumerate(unique_labels)}
        
        # Store label mapping
        self.label_encoder = {
            "label_to_id": label_to_id,
            "id_to_label": id_to_label
        }
        
        # Convert string labels to ids
        label_ids = [label_to_id[label] for label in labels]
        
        # Load or initialize model
        self._load_or_initialize_model(len(unique_labels))
        
        # Move model to device
        self.model.to(self.device)
        
        # Create dataset and dataloader
        dataset = DocumentDataset(texts, label_ids, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        # Prepare optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            logger.info(f"Starting epoch {epoch+1}/{epochs}")
            running_loss = 0.0
            
            for i, batch in enumerate(dataloader):
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/10:.4f}")
                    running_loss = 0.0
        
        # Update metadata
        self.metadata["model_version"] = "1.0.0"  # Increment on major changes
        self.metadata["num_classes"] = len(unique_labels)
        self.metadata["classes"] = unique_labels
        self.metadata["num_documents"] = len(texts)
        
        # Save model
        self._save_model()
        
        # Generate and store embeddings for similarity search
        logger.info("Generating document embeddings for similarity search")
        embeddings = self._extract_embeddings(texts)
        
        # Initialize FAISS index
        self._initialize_faiss_index(embeddings.shape[1])
        
        # Add embeddings to index
        self.index.add(embeddings)
        self.embeddings = embeddings
        self.embedding_labels = label_ids
        
        # Save embeddings and labels
        np.save(os.path.join(EMBEDDINGS_PATH, "embeddings.npy"), embeddings)
        np.save(os.path.join(EMBEDDINGS_PATH, "labels.npy"), np.array(label_ids))
        joblib.dump(self.label_encoder, os.path.join(EMBEDDINGS_PATH, "label_encoder.pkl"))
        
        logger.info(f"Training completed. Model saved to {BERT_MODEL_PATH}")
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict domain for a document
        
        Args:
            text: Document text
            
        Returns:
            Tuple of (predicted_label, confidence_score)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please train or load the model first.")
        
        self.model.eval()
        
        # Tokenize the text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        ).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Get probabilities
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)
        
        # Convert to numpy
        confidence = confidence.item()
        predicted_idx = predicted_idx.item()
        
        # Get the predicted label
        predicted_label = self.label_encoder["id_to_label"][predicted_idx]
        
        # Check confidence threshold
        if confidence < CONFIDENCE_THRESHOLD:
            # Store unknown document for future training
            self._store_unknown_document(text, confidence)
            return "unknown", confidence
        
        return predicted_label, confidence
    
    def predict_with_similarity(self, text: str) -> Tuple[str, float]:
        """
        Predict domain using similarity search
        
        Args:
            text: Document text
            
        Returns:
            Tuple of (predicted_label, similarity_score)
        """
        if self.index is None or self.embeddings is None:
            raise ValueError("Similarity index not initialized. Please train the model first.")
        
        # Extract embedding for the text
        embedding = self._extract_embeddings([text])
        
        # Search for similar documents
        distances, indices = self.index.search(embedding, 5)  # Get top 5 matches
        
        # Get labels for similar documents
        similar_labels = [self.embedding_labels[idx] for idx in indices[0]]
        
        # Get most common label
        from collections import Counter
        label_counts = Counter(similar_labels)
        most_common_label_id = label_counts.most_common(1)[0][0]
        
        # Calculate similarity score (inverse of distance)
        similarity_score = 1.0 / (1.0 + distances[0][0])
        
        # Convert label ID to string
        predicted_label = self.label_encoder["id_to_label"][most_common_label_id]
        
        # Check confidence threshold
        if similarity_score < CONFIDENCE_THRESHOLD:
            # Store unknown document for future training
            self._store_unknown_document(text, similarity_score)
            return "unknown", similarity_score
        
        return predicted_label, similarity_score
    
    def _store_unknown_document(self, text: str, confidence: float):
        """Store unknown document for future training"""
        try:
            # Create a unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"unknown_{timestamp}_{abs(hash(text[:20]))}.txt"
            filepath = os.path.join(UNKNOWN_DOCS_PATH, filename)
            
            # Store document with metadata
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"CONFIDENCE: {confidence}\n")
                f.write(f"TIMESTAMP: {datetime.now().isoformat()}\n")
                f.write("\n--- DOCUMENT TEXT ---\n\n")
                f.write(text)
            
            # Update metadata
            self.metadata["unknown_documents"] = self.metadata.get("unknown_documents", 0) + 1
            
            logger.info(f"Stored unknown document to {filepath}")
        except Exception as e:
            logger.error(f"Error storing unknown document: {str(e)}", exc_info=True)
    
    def load(self):
        """Load the BERT classifier model"""
        try:
            if not os.path.exists(BERT_MODEL_PATH):
                raise FileNotFoundError(f"Model directory not found at {BERT_MODEL_PATH}")
            
            # Load model and tokenizer
            logger.info(f"Loading BERT model from {BERT_MODEL_PATH}")
            self.model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
            self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
            
            # Move model to device
            self.model.to(self.device)
            
            # Load metadata
            if os.path.exists(METADATA_PATH):
                with open(METADATA_PATH, 'r') as f:
                    self.metadata = json.load(f)
            
            # Load label encoder
            label_encoder_path = os.path.join(EMBEDDINGS_PATH, "label_encoder.pkl")
            if os.path.exists(label_encoder_path):
                self.label_encoder = joblib.load(label_encoder_path)
            
            # Load embeddings and initialize FAISS index
            embeddings_path = os.path.join(EMBEDDINGS_PATH, "embeddings.npy")
            labels_path = os.path.join(EMBEDDINGS_PATH, "labels.npy")
            
            if os.path.exists(embeddings_path) and os.path.exists(labels_path):
                self.embeddings = np.load(embeddings_path)
                self.embedding_labels = np.load(labels_path)
                
                # Initialize and populate FAISS index
                self._initialize_faiss_index(self.embeddings.shape[1])
                self.index.add(self.embeddings)
                
                logger.info(f"Loaded {len(self.embeddings)} document embeddings for similarity search")
            
            logger.info(f"Model loaded successfully (version: {self.metadata.get('model_version', 'unknown')})")
            logger.info(f"Classes: {self.metadata.get('classes', [])}")
            
            return self
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            raise
    
    def evaluate(self, texts: List[str], true_labels: List[str]) -> Dict:
        """
        Evaluate the model on test data
        
        Args:
            texts: List of document texts
            true_labels: List of true domain labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please train or load the model first.")
        
        logger.info(f"Evaluating model on {len(texts)} documents")
        
        predicted_labels = []
        confidence_scores = []
        
        for text in texts:
            try:
                label, confidence = self.predict(text)
                predicted_labels.append(label)
                confidence_scores.append(confidence)
            except Exception as e:
                logger.error(f"Error predicting label for document: {str(e)}")
                predicted_labels.append("error")
                confidence_scores.append(0.0)
        
        # Calculate metrics
        report = classification_report(true_labels, predicted_labels, output_dict=True)
        
        # Add average confidence
        report["average_confidence"] = sum(confidence_scores) / len(confidence_scores)
        
        # Calculate unknown rate
        unknown_count = sum(1 for label in predicted_labels if label == "unknown")
        report["unknown_rate"] = unknown_count / len(predicted_labels)
        
        # Log results
        logger.info(f"Evaluation results: Accuracy: {report['accuracy']:.4f}, "
                   f"Unknown rate: {report['unknown_rate']:.4f}, "
                   f"Avg confidence: {report['average_confidence']:.4f}")
        
        return report
    
    def retrain_with_new_data(self, new_texts: List[str], new_labels: List[str], 
                             existing_texts: Optional[List[str]] = None,
                             existing_labels: Optional[List[str]] = None):
        """
        Retrain the model with new data, optionally including existing data
        
        Args:
            new_texts: List of new document texts
            new_labels: List of new document labels
            existing_texts: Optional list of existing document texts
            existing_labels: Optional list of existing document labels
        """
        # Combine new and existing data if provided
        if existing_texts is not None and existing_labels is not None:
            all_texts = existing_texts + new_texts
            all_labels = existing_labels + new_labels
            logger.info(f"Retraining with {len(existing_texts)} existing documents and {len(new_texts)} new documents")
        else:
            all_texts = new_texts
            all_labels = new_labels
            logger.info(f"Retraining with {len(new_texts)} new documents")
        
        # Train the model
        self.train(all_texts, all_labels)
        
        logger.info("Model retrained successfully")

# Main functions that wrap the class methods
def train_domain_classifier(features: List[str], labels: List[str]) -> BERTDomainClassifier:
    """
    Train a domain classifier using provided texts and labels.
    
    Args:
        features: List of document texts
        labels: List of domain labels corresponding to the texts
        
    Returns:
        BERTDomainClassifier: Trained classifier model
        
    Raises:
        ValueError: If features or labels are invalid
        Exception: If training fails
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Validate inputs
        if features is None or len(features) == 0:
            raise ValueError("Features list cannot be empty")
        
        if labels is None or len(labels) == 0:
            raise ValueError("Labels list cannot be empty")
            
        if len(features) != len(labels):
            raise ValueError(f"Number of documents ({len(features)}) must match number of labels ({len(labels)})")
        
        # Create and train the classifier
        classifier = BERTDomainClassifier()
        classifier.train(features, labels)
        
        return classifier
        
    except ValueError as e:
        logger.error(f"Invalid input for classifier training: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error during classifier training: {str(e)}", exc_info=True)
        raise

def load_domain_classifier() -> BERTDomainClassifier:
    """
    Load the previously saved domain classifier from disk.
    
    Returns:
        BERTDomainClassifier: Loaded classifier model
    
    Raises:
        FileNotFoundError: If the model file doesn't exist
        Exception: If there's an issue loading the model
    """
    logger = logging.getLogger(__name__)
    
    try:
        classifier = BERTDomainClassifier()
        return classifier.load()
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading domain classifier: {str(e)}", exc_info=True)
        raise

def predict_domain(text: str) -> Tuple[str, float]:
    """
    Predict the domain label for a given text using the trained classifier.
    
    Args:
        text: The text to classify
        
    Returns:
        Tuple[str, float]: Predicted domain label and confidence score
        
    Raises:
        ValueError: If text is empty
        FileNotFoundError: If model file is not found
        Exception: If prediction fails
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Validate inputs
        if not text or not text.strip():
            logger.warning("Empty text provided for prediction")
            raise ValueError("Text cannot be empty")
        
        # Load the classifier
        classifier = load_domain_classifier()
        
        # Make prediction with both methods
        try:
            # Try similarity-based prediction first
            label, confidence = classifier.predict_with_similarity(text)
        except Exception as e:
            logger.warning(f"Similarity-based prediction failed: {str(e)}. Falling back to standard prediction.")
            # Fall back to standard prediction
            label, confidence = classifier.predict(text)
        
        logger.info(f"Predicted domain: {label} with confidence: {confidence:.4f}")
        
        return label, confidence
        
    except ValueError as e:
        logger.error(f"Invalid input for domain prediction: {str(e)}")
        raise
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error predicting domain: {str(e)}", exc_info=True)
        raise

def evaluate_domain_classifier(test_texts: List[str], true_labels: List[str]) -> Dict:
    """
    Evaluate the performance of the trained domain classifier.
    
    Args:
        test_texts: List of document texts for testing
        true_labels: List of true domain labels corresponding to the test texts
        
    Returns:
        Dict: Dictionary containing evaluation metrics
        
    Raises:
        ValueError: If inputs are invalid
        FileNotFoundError: If model file is not found
        Exception: If evaluation fails
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Validate inputs
        if test_texts is None or len(test_texts) == 0:
            raise ValueError("Test texts cannot be empty")
            
        if true_labels is None or len(true_labels) == 0:
            raise ValueError("True labels cannot be empty")
            
        if len(test_texts) != len(true_labels):
            raise ValueError(f"Number of test texts ({len(test_texts)}) must match number of labels ({len(true_labels)})")
        
        # Load the classifier
        classifier = load_domain_classifier()
        
        # Evaluate
        results = classifier.evaluate(test_texts, true_labels)
        
        # Print the results
        print("\nClassification Report:")
        for class_name, metrics in results.items():
            if isinstance(metrics, dict):
                print(f"\n{class_name}:")
                for metric_name, value in metrics.items():
                    print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"{class_name}: {metrics:.4f}")
        
        return results
        
    except ValueError as e:
        logger.error(f"Invalid input for classifier evaluation: {str(e)}")
        raise
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error evaluating domain classifier: {str(e)}", exc_info=True)
        raise

def update_model_with_new_data(new_texts: List[str], new_labels: List[str]) -> BERTDomainClassifier:
    """
    Update the existing model with new labeled data
    
    Args:
        new_texts: List of new document texts
        new_labels: List of new domain labels
        
    Returns:
        BERTDomainClassifier: Updated classifier model
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Load existing classifier
        classifier = load_domain_classifier()
        
        # Retrain with new data
        classifier.retrain_with_new_data(new_texts, new_labels)
        
        return classifier
        
    except Exception as e:
        logger.error(f"Error updating model: {str(e)}", exc_info=True)
        raise
