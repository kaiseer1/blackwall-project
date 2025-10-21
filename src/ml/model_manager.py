"""
Machine Learning model management for BlackWall
Handles training, loading, and inference of threat detection models
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from src.utils.logger import get_logger


class ModelManager:
    """
    Manages ML models for threat detection.
    Supports multiple algorithms and ensemble methods.
    """

    def __init__(self, model_dir: str = "models"):
        """
        Initialize model manager.

        Args:
            model_dir: Directory to store trained models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.scaler = StandardScaler()
        self.logger = get_logger('blackwall.ml', log_file='logs/ml.log')

        self.model_info = {
            'status': 'not_loaded',
            'version': '4.0.0',
            'last_updated': None,
            'algorithm': None,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'training_samples': 0,
            'feature_count': 0
        }

    def train_model(
        self,
        dataset_path: Optional[str] = None,
        model_type: str = "RandomForest",
        force: bool = False,
        use_ensemble: bool = False,
        balance_classes: bool = True
    ) -> bool:
        """
        Train a machine learning model.

        Args:
            dataset_path: Path to training dataset CSV
            model_type: Model type (RandomForest, GradientBoosting, or Ensemble)
            force: Force retraining even if model exists
            use_ensemble: Use ensemble of multiple models
            balance_classes: Apply SMOTE for class balancing

        Returns:
            True if training successful, False otherwise
        """
        try:
            model_path = self.model_dir / "blackwall_model.joblib"
            scaler_path = self.model_dir / "scaler.joblib"

            # Check if model exists
            if model_path.exists() and not force:
                self.logger.info("Model already exists. Use force=True to retrain.")
                return self.load_model()

            # Load dataset
            self.logger.info(f"Loading dataset from {dataset_path or 'default paths'}")
            X, y = self._load_dataset(dataset_path)

            if X is None or y is None:
                self.logger.error("Failed to load dataset")
                return False

            self.logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Scale features
            self.logger.info("Scaling features...")
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Handle class imbalance with SMOTE
            if balance_classes and len(np.unique(y_train)) > 1:
                self.logger.info("Applying SMOTE for class balancing...")
                try:
                    smote = SMOTE(random_state=42)
                    X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
                    self.logger.info(f"After SMOTE: {X_train_scaled.shape[0]} samples")
                except Exception as e:
                    self.logger.warning(f"SMOTE failed: {e}. Continuing without balancing.")

            # Train model
            if use_ensemble:
                self.logger.info("Training ensemble model...")
                model = self._create_ensemble_model()
            else:
                self.logger.info(f"Training {model_type} model...")
                model = self._create_model(model_type)

            # Train with progress tracking
            start_time = time.time()
            model.fit(X_train_scaled, y_train)
            training_time = time.time() - start_time

            self.logger.info(f"Training completed in {training_time:.2f} seconds")

            # Evaluate model
            self.logger.info("Evaluating model...")
            y_pred = model.predict(X_test_scaled)

            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }

            self.logger.info(f"Model Performance:")
            self.logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
            self.logger.info(f"  Precision: {metrics['precision']:.4f}")
            self.logger.info(f"  Recall:    {metrics['recall']:.4f}")
            self.logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")

            # Cross-validation
            self.logger.info("Performing cross-validation...")
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, n_jobs=-1)
            self.logger.info(f"Cross-validation scores: {cv_scores}")
            self.logger.info(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

            # Save model and scaler
            self.logger.info(f"Saving model to {model_path}")
            joblib.dump(model, model_path)
            joblib.dump(self.scaler, scaler_path)

            # Update model info
            self.model = model
            self.model_info.update({
                'status': 'loaded',
                'algorithm': model_type if not use_ensemble else 'Ensemble',
                'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'training_samples': X_train_scaled.shape[0],
                'feature_count': X_train_scaled.shape[1],
                **metrics
            })

            return True

        except Exception as e:
            self.logger.error(f"Error training model: {e}", exc_info=True)
            return False

    def _load_dataset(self, dataset_path: Optional[str] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load and preprocess dataset"""
        try:
            # Try provided path or default paths
            paths_to_try = []
            if dataset_path:
                paths_to_try.append(dataset_path)

            paths_to_try.extend([
                "Sampled_Dataset_Example_cleaned.csv",
                "datasets/Sampled_Dataset_Example_cleaned.csv",
                "datasets/Sampled_Dataset_Example.csv",
                "datasets/Final_Preprocessed_Dataset_Sample.csv"
            ])

            data = None
            for path in paths_to_try:
                if os.path.exists(path):
                    self.logger.info(f"Loading dataset from {path}")
                    data = pd.read_csv(path)
                    break

            if data is None:
                self.logger.error("No dataset file found")
                return None, None

            # Find label column
            label_columns = ['Label', 'label', 'class', 'Class', 'target', 'Target']
            label_col = None
            for col in label_columns:
                if col in data.columns:
                    label_col = col
                    break

            if label_col is None:
                self.logger.error("No label column found in dataset")
                return None, None

            # Separate features and labels
            y = data[label_col]
            X = data.drop(label_col, axis=1)

            # Drop non-essential columns
            cols_to_drop = ['Flow ID', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Timestamp']
            X = X.drop([col for col in cols_to_drop if col in X.columns], axis=1)

            # Handle missing values
            X = X.fillna(X.median())

            # Convert to numeric
            X = X.select_dtypes(include=[np.number])

            # Handle infinite values
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(0)

            return X.values, y.values

        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}", exc_info=True)
            return None, None

    def _create_model(self, model_type: str):
        """Create a single model"""
        if model_type == "RandomForest":
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=25,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
        elif model_type == "GradientBoosting":
            return GradientBoostingClassifier(
                n_estimators=150,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _create_ensemble_model(self):
        """Create an ensemble of multiple models"""
        rf_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=25,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )

        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=7,
            learning_rate=0.1,
            random_state=42
        )

        ensemble = VotingClassifier(
            estimators=[('rf', rf_model), ('gb', gb_model)],
            voting='soft',
            n_jobs=-1
        )

        return ensemble

    def load_model(self) -> bool:
        """Load trained model from disk"""
        try:
            model_path = self.model_dir / "blackwall_model.joblib"
            scaler_path = self.model_dir / "scaler.joblib"

            if not model_path.exists():
                self.logger.error(f"Model file not found: {model_path}")
                return False

            self.logger.info(f"Loading model from {model_path}")
            self.model = joblib.load(model_path)

            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)

            self.model_info['status'] = 'loaded'
            self.model_info['last_updated'] = datetime.fromtimestamp(
                model_path.stat().st_mtime
            ).strftime("%Y-%m-%d %H:%M:%S")

            return True

        except Exception as e:
            self.logger.error(f"Error loading model: {e}", exc_info=True)
            return False

    def predict(self, features: np.ndarray) -> Optional[Tuple[int, float]]:
        """
        Make prediction on feature vector.

        Args:
            features: Feature vector (numpy array)

        Returns:
            Tuple of (prediction, confidence) or None on error
        """
        try:
            if self.model is None:
                self.logger.error("No model loaded")
                return None

            # Scale features
            features_scaled = self.scaler.transform(features)

            # Predict
            prediction = self.model.predict(features_scaled)[0]

            # Get confidence (probability)
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_scaled)[0]
                confidence = float(probabilities.max())
            else:
                confidence = 1.0

            return int(prediction), confidence

        except Exception as e:
            self.logger.error(f"Error making prediction: {e}", exc_info=True)
            return None

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return self.model_info.copy()
