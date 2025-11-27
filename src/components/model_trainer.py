import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

from src.config.configuration import data_config, train_config, path_config
from src.components.data_ingestion import load_generators
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:

    def __init__(self):
        self.model = None

    # -------------------------------------------------------
    # BUILD MODEL
    # -------------------------------------------------------
    def build_model(self):
        logger.info("🧠 Building EfficientNetB0 model...")

        base = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(*data_config.img_size, 3)
        )

        # PARTIAL UNFREEZE — allow last 60 layers to train
        base.trainable = True
        for layer in base.layers[:-60]:
            layer.trainable = False

        trainable_count = len([l for l in base.layers if l.trainable])
        logger.info(f"🔧 Trainable layers in base model: {trainable_count}")

        # Custom classification head
        x = layers.GlobalAveragePooling2D()(base.output)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(train_config.num_classes, activation="softmax")(x)

        self.model = models.Model(inputs=base.input, outputs=outputs)
        logger.info("✅ Model built successfully!")

    # -------------------------------------------------------
    # PHASE 1 TRAINING
    # -------------------------------------------------------
    def train_phase1(self, train_gen, val_gen):
        logger.info("🚀 Starting Phase 1 Training (Partial Unfreeze)...")

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(train_config.lr_phase1),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.2),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=path_config.model_path,
                save_best_only=True,
                monitor="val_accuracy",
                mode="max"
            ),
        ]

        self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=train_config.epochs_phase1,
            callbacks=callbacks
        )

        logger.info("✅ Phase 1 Training Completed.")

    # -------------------------------------------------------
    # FINE-TUNE PHASE
    # -------------------------------------------------------
    def fine_tune(self, train_gen, val_gen):
        logger.info("🔧 Starting Phase 2 Fine-Tuning (Unfreeze All Layers)...")

        base_model = self.model.layers[0]
        base_model.trainable = True  # unfreeze entire EfficientNet

        logger.info("🔓 Entire EfficientNet model is now trainable.")

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(train_config.lr_phase2),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.2),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=path_config.model_path,
                save_best_only=True,
                monitor="val_accuracy",
                mode="max"
            ),
        ]

        self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=train_config.epochs_phase2,
            callbacks=callbacks
        )

        logger.info("🎯 Phase 2 Fine-Tuning Completed.")

    # -------------------------------------------------------
    # FULL TRAINING PIPELINE
    # -------------------------------------------------------
    def train(self):
        logger.info("📥 Loading data generators...")
        train_gen, val_gen, _ = load_generators()

        logger.info("📌 Starting Complete Training Pipeline...")

        self.build_model()
        self.train_phase1(train_gen, val_gen)
        self.fine_tune(train_gen, val_gen)

        logger.info(f"🎉 Training Completed — Best model saved at: {path_config.model_path}")
