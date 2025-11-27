import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from src.config.configuration import data_config, train_config, path_config

def load_generators():

    # -----------------------------
    # TRAIN DATAGEN (with augment + preprocess)
    # -----------------------------
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,  # 🔥 critical for EfficientNet
        rotation_range=5,
        width_shift_range=0.02,
        height_shift_range=0.02,
        zoom_range=0.02,
        horizontal_flip=True,
    )

    # -----------------------------
    # TEST/VAL DATAGEN (only preprocess)
    # -----------------------------
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input  # 🔥 must use same preprocess
    )

    # -----------------------------
    # TRAIN GENERATOR
    # -----------------------------
    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_config.split_data_dir, "train"),
        target_size=data_config.img_size,
        batch_size=data_config.batch_size,
        class_mode="categorical",
        color_mode="rgb",        # 🔥 force proper RGB conversion
        shuffle=True
    )

    # -----------------------------
    # VALIDATION GENERATOR
    # -----------------------------
    val_gen = test_datagen.flow_from_directory(
        os.path.join(data_config.split_data_dir, "val"),
        target_size=data_config.img_size,
        batch_size=data_config.batch_size,
        class_mode="categorical",
        color_mode="rgb"         # 🔥 force RGB
    )

    # -----------------------------
    # TEST GENERATOR
    # -----------------------------
    test_gen = test_datagen.flow_from_directory(
        os.path.join(data_config.split_data_dir, "test"),
        target_size=data_config.img_size,
        batch_size=data_config.batch_size,
        class_mode="categorical",
        color_mode="rgb",        # 🔥 force RGB
        shuffle=False            # needed for evaluation consistency
    )

    return train_gen, val_gen, test_gen
