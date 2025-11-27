import os
from dataclasses import dataclass

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # project root

@dataclass
class DataConfig:
    split_data_dir: str = os.path.join(ROOT_DIR, "alz_split_dataset")
    img_size: tuple = (224, 224)
    batch_size: int = 32

@dataclass
class TrainConfig:
    num_classes: int = 4
    epochs_phase1: int = 20
    epochs_phase2: int = 10
    lr_phase1: float = 5e-3
    lr_phase2: float = 1e-5

@dataclass
class PathConfig:
    model_path: str = os.path.join(ROOT_DIR, "artifacts", "models", "alz_model.h5")
    logs_dir: str = os.path.join(ROOT_DIR, "artifacts", "logs")

data_config = DataConfig()
train_config = TrainConfig()
path_config = PathConfig()

mlflow_config = SimpleNamespace(
    experiment_name="Alzheimer_Detector",
    tracking_uri=os.path.join(ARTIFACT_DIR, "mlflow")
)
