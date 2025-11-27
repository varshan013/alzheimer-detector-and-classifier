import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib.pyplot as plt
from src.config.configuration import data_config, path_config


class AlzheimerPredictor:
    def __init__(self, model_path=None):
        model_path = model_path or path_config.model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print("🔍 Loading trained model...")
        self.model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")

        # IMPORTANT: order must match your training class_indices
        # {'MildDemented':0, 'ModerateDemented':1, 'NonDemented':2, 'VeryMildDemented':3}
        self.class_names = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

        # Short class descriptions (non-diagnostic)
        self.class_descriptions = {
            "NonDemented": (
                "No significant dementia-related structural patterns detected in this MRI. "
                "This does not replace a professional medical evaluation."
            ),
            "VeryMildDemented": (
                "Patterns consistent with very mild cognitive impairment. Changes may be subtle "
                "and early. Clinical correlation is recommended."
            ),
            "MildDemented": (
                "Patterns consistent with mild dementia. There may be noticeable memory or "
                "cognitive changes affecting some daily activities."
            ),
            "ModerateDemented": (
                "Patterns consistent with moderate dementia. Cognitive decline may significantly "
                "impact daily functioning and independence."
            ),
        }

        # Very high-level, safe recommendations (not medical advice)
        self.recommendations = {
            "NonDemented": [
                "Maintain a healthy lifestyle (sleep, exercise, nutrition).",
                "Continue regular health checkups as advised by your doctor.",
                "If any symptoms appear, consult a medical professional."
            ],
            "VeryMildDemented": [
                "Consider scheduling a clinical evaluation with a neurologist or specialist.",
                "Monitor memory, attention, and daily functioning over time.",
                "Discuss results with a qualified healthcare provider."
            ],
            "MildDemented": [
                "A full clinical assessment is recommended.",
                "Discuss support for memory and daily activities with a specialist.",
                "Talk with family/caregivers about monitoring and assistance."
            ],
            "ModerateDemented": [
                "Seek a comprehensive neurological and cognitive assessment.",
                "Plan for assistance with daily living and safety supervision.",
                "Follow up regularly with a dementia specialist or neurologist."
            ],
        }

    # -----------------------------
    # Preprocess image (must match training)
    # -----------------------------
    def preprocess_image(self, img_path):
        """Load and preprocess image for EfficientNet."""
        img = image.load_img(img_path, target_size=data_config.img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  # 🔥 same as training
        return img_array

    # -----------------------------
    # Grad-CAM heatmap generation
    # -----------------------------
    def _make_gradcam_heatmap(self, img_array, last_conv_layer_name="top_conv"):
        # Build a model that maps the input image to activations of the last conv layer
        # and the final predictions
        last_conv_layer = self.model.get_layer(last_conv_layer_name)
        grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [last_conv_layer.output, self.model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            pred_index = tf.argmax(predictions[0])
            pred_score = predictions[:, pred_index]

        # Compute gradients of top predicted class wrt last conv layer
        grads = tape.gradient(pred_score, conv_outputs)

        # Global average pooling on gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the conv outputs with pooled grads
        conv_outputs = conv_outputs[0]  # remove batch dim
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

        # Relu + normalize to [0,1]
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        return heatmap

    def _save_gradcam_overlay(self, img_path, heatmap, output_path, alpha=0.4):
        """Save Grad-CAM overlay image."""
        # Load original image (for visualization)
        orig_img = image.load_img(img_path)
        orig_size = orig_img.size  # (width, height)

        # Resize heatmap to match original image
        heatmap_resized = tf.image.resize(
            heatmap[..., np.newaxis],
            (orig_size[1], orig_size[0])
        ).numpy().squeeze()

        plt.figure(figsize=(4, 4))
        plt.imshow(orig_img)
        plt.imshow(heatmap_resized, cmap="jet", alpha=alpha)
        plt.axis("off")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close()

    # -----------------------------
    # Main prediction method
    # -----------------------------
    def predict(self, img_path, generate_gradcam=True, gradcam_dir="gradcams"):
        """
        Returns:
        {
          "class": str,
          "confidence": float (0-100),
          "description": str,
          "recommendations": [str, ...],
          "gradcam_path": str or None
        }
        """
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Preprocess for model
        img_array = self.preprocess_image(img_path)

        # Predict
        preds = self.model.predict(img_array)
        predicted_class_idx = int(np.argmax(preds))
        confidence = float(np.max(preds))

        class_name = self.class_names[predicted_class_idx]
        description = self.class_descriptions.get(class_name, "")
        recos = self.recommendations.get(class_name, [])

        gradcam_path = None
        if generate_gradcam:
            heatmap = self._make_gradcam_heatmap(img_array)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            gradcam_dir = gradcam_dir or "gradcams"
            gradcam_path = os.path.join(gradcam_dir, f"{base_name}_gradcam.png")
            self._save_gradcam_overlay(img_path, heatmap, gradcam_path)

        return {
            "class": class_name,
            "confidence": round(confidence * 100, 2),
            "description": description,
            "recommendations": recos,
            "gradcam_path": gradcam_path
        }
