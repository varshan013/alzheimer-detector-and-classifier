# src/pipelines/training_pipeline.py
import mlflow
import mlflow.tensorflow
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluator
from src.utils.logger import get_logger
from src.config.configuration import mlflow_config

logger = get_logger(__name__)


def run_training_pipeline():

    logger.info("Initializing MLflow...")

    # Set MLflow tracking
    mlflow.set_tracking_uri(mlflow_config.tracking_uri)
    mlflow.set_experiment(mlflow_config.experiment_name)

    logger.info("Starting training pipeline inside MLflow run...")

    with mlflow.start_run(run_name="Alzheimer_Training_Run"):

        trainer = ModelTrainer()

        logger.info("🚀 Training started...")
        trainer.train()
        logger.info("Training completed.")

        # --- EVALUATION ---
        logger.info("📊 Starting Evaluation...")
        evaluator = ModelEvaluator()
        acc, cm, report = evaluator.evaluate()

        # --- LOG METRICS TO MLFLOW ---
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("mild_precision", report['MildDemented']['precision'])
        mlflow.log_metric("moderate_precision", report['ModerateDemented']['precision'])
        mlflow.log_metric("non_precision", report['NonDemented']['precision'])
        mlflow.log_metric("verymild_precision", report['VeryMildDemented']['precision'])

        logger.info(f"Final Test Accuracy: {acc*100:.2f}%")
        logger.info("Pipeline finished successfully.")

        print("\n📌 MLflow logging complete! View UI using:")
        print("👉 mlflow ui --backend-store-uri artifacts/mlflow\n")


if __name__ == "__main__":
    run_training_pipeline()
