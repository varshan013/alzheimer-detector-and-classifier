import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

from src.components.data_ingestion import load_generators
from src.config.configuration import path_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:

    def __init__(self, model_path=None):
        """
        Loads model from artifacts folder unless a specific path is given.
        """
        self.model_path = model_path or path_config.model_path

        logger.info(f"Loading model from: {self.model_path}")

        self.model = tf.keras.models.load_model(self.model_path)
        logger.info("Model loaded successfully!")

    def evaluate(self):
        """
        Evaluate model on test dataset.
        """
        logger.info("Loading test data generator...")
        _, _, test_gen = load_generators()

        logger.info("Running predictions on test dataset...")
        y_pred = self.model.predict(test_gen)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true = test_gen.classes

        acc = np.mean(y_pred_labels == y_true)
        logger.info(f"Test Accuracy: {acc * 100:.2f}%")

        target_names = list(test_gen.class_indices.keys())

        logger.info("Generating classification report...")
        report = classification_report(
            y_true,
            y_pred_labels,
            target_names=target_names
        )
        logger.info(f"\nClassification Report:\n{report}")

        cm = confusion_matrix(y_true, y_pred_labels)
        logger.info(f"Confusion Matrix:\n{cm}")

        return acc, cm, report
