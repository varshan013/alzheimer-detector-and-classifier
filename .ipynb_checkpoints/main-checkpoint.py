import os
from werkzeug.utils import secure_filename
from src.config.configuration import data_config, train_config, path_config
from prediction_pipeline import AlzheimerPredictor
from flask import Flask, render_template, request, redirect, url_for
from src.components.prediction_pipeline import AlzheimerPredictor

# --- Flask setup ---
app = Flask(__name__)

# folders
UPLOAD_FOLDER = os.path.join("static", "uploads")
GRADCAM_FOLDER = os.path.join("static", "gradcams")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["GRADCAM_FOLDER"] = GRADCAM_FOLDER

# --- Load model once (global) ---
# use the path you tested earlier ("alz_model.h5")
predictor = AlzheimerPredictor("alz_model.h5")


# ---------- ROUTES ----------

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # no file
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(upload_path)

            # run prediction + gradcam (saved inside static/gradcams)
            result = predictor.predict(
                upload_path,
                generate_gradcam=True,
                gradcam_dir=app.config["GRADCAM_FOLDER"],
            )

            # build URL for gradcam image (relative to /static)
            gradcam_path = result.get("gradcam_path")
            gradcam_url = None
            if gradcam_path is not None:
                rel_to_static = os.path.relpath(gradcam_path, "static")
                gradcam_url = url_for(
                    "static",
                    filename=rel_to_static.replace(os.sep, "/")
                )

            # also show original image
            upload_url = url_for( "static", filename=f"uploads/{filename}")
            
            return render_template(
                "result.html",
                predicted_class=result["class"],
                confidence=result["confidence"],
                description=result["description"],
                recommendations=result["recommendations"],
                gradcam_url=gradcam_url,
                # optional: show original if you later move uploads into static
                orig_url=None
            )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
