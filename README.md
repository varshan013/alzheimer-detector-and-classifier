Alzheimer Detector & MRI Classification
Using EfficientNetB0, Grad-CAM, Flask UI, MLflow & DVC

This project is an end-to-end deep learning system that analyzes MRI brain scans to classify Alzheimer's stages:

Non-Demented

Very Mild Demented

Mild Demented

Moderate Demented

It includes model training (TensorFlow), explainability (Grad-CAM), experiment tracking (MLflow), data versioning (DVC), and a Flask web interface for real-time predictions.

🚀 Features
🔍 1. Deep Learning Model

EfficientNetB0 backbone

Transfer learning + fine-tuning

97% test accuracy

🧾 2. Explainability (Grad-CAM)

Automatically generates heatmaps showing what part of the brain the model used for prediction.

🌐 3. Flask Web App

Upload MRI image

Get predicted class

Confidence score

Grad-CAM visualization

Stage-based medical recommendations

📊 4. MLflow Tracking

Tracks metrics (accuracy, loss)

Stores models

Experiment comparison dashboard

📦 5. DVC Integration

Version control for datasets

Reproducible machine learning pipeline

.dvc tracking for model file


📁 Project Structure
AlzheimerDetector/
│── app/                     # Flask UI
│   ├── static/              # images, uploads, gradcams
│   ├── templates/           # HTML pages
│   └── __init__.py
│
│── artifacts/               # MLflow + trained models
│
│── src/
│   ├── components/          # model training, evaluation, prediction
│   ├── config/              # config + paths
│   ├── utils/               # logger, helpers
│   └── pipelines/           # training pipeline
│
│── notebook/                # Jupyter notebook experiments
│── scripts/                 # utility scripts
│── gradcams/                # raw generated heatmaps
│── alz_split_dataset/       # dataset (not pushed to GitHub)
│── dvc.yaml
│── main.py
│── README.md
│── requirements.txt

🧠 Model Architecture
🔹 Base Model

EfficientNetB0 (pretrained on ImageNet)

🔹 Custom Classification Head
GlobalAveragePooling2D
Dropout(0.3)
Dense(128, relu)
Dropout(0.2)
Dense(4, softmax)

🔹 Training Strategy
Phase	Description	LR
Phase 1	Partial Unfreeze (last 60 layers trainable)	1e-3
Phase 2	Full Fine-tuning (all layers trainable)	1e-5
📊 Model Performance
Class	Precision	Recall	F1	Support
Mild Demented	0.96	0.94	0.95	1000
Moderate Demented	1.00	0.99	0.99	1000
Non-Demented	0.96	0.89	0.92	1281
Very Mild Demented	0.87	0.95	0.91	1121
⭐ Final Test Accuracy: 93.90%
⭐ Peak Validation Accuracy: 97%+
🔥 Grad-CAM Explainability

Your app generates a heatmap showing where the model is looking.

Example:

gradcams/
├── mild_001_gradcam.png
├── nonDem_124_gradcam.png
└── moderateDem40_gradcam.png


These heatmaps are also shown inside the result page of Flask.

⚙️ Installation
git clone https://github.com/YOUR_USERNAME/AlzheimerDetector.git
cd AlzheimerDetector
pip install -r requirements.txt

▶️ Run Training Pipeline
python -m src.pipelines.training_pipeline

🌐 Run Flask Web App
python main.py


Then open:
👉 http://localhost:5000

Upload MRI image → View detected stage + Grad-CAM.

📦 DVC Tracking
dvc init
dvc add alz_split_dataset/
git add alz_split_dataset.dvc .gitignore
git commit -m "Track dataset with DVC"

📈 MLflow Tracking

Start MLflow UI:

mlflow ui --backend-store-uri artifacts/mlflow


Open:
👉 http://localhost:5000

Track runs, parameters, metrics, and artifacts.

🛠 Tech Stack
Category	Tools
Deep Learning	TensorFlow, EfficientNetB0
Explainability	Grad-CAM
MLOps	MLflow, DVC
Backend	Python, Flask
Deployment Ready	Docker 
Logging	Custom Logger
