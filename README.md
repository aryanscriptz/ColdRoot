# ColdRoot

**ColdRoot** is an AI-generated image detection web application designed to analyze uploaded images and estimate whether they are likely to be synthetically generated.

The project uses the **`Smogy/SMOGY-Ai-images-detector`** model and a multi-view analysis pipeline to improve the robustness of its predictions across different image compositions and transformations.

## ✨ Key Features

* **AI Image Detection** — Estimates whether an uploaded image is AI-generated.
* **Multi-View Analysis** — Analyzes each image from three different perspectives.
* **Median-Based Scoring** — Combines the three predictions using the median score to reduce the influence of unusual artifacts in a single view.
* **Out-of-Domain Detection** — Designed to evaluate images from generators beyond the model's primary training distribution, including DALL·E, FLUX, Imagen, and Stable Diffusion.
* **Visual Analysis** — Supports heatmap generation for inspecting potentially significant regions of an image.
* **Web-Based Interface** — Simple browser-based workflow for uploading and analyzing images.

## 🔬 Detection Methodology

ColdRoot uses a **Tri-View Processing Pipeline** to make its prediction less dependent on a single image representation.

### 1. Original Image

The uploaded image is analyzed in its original form.

### 2. Horizontal Flip

A horizontally mirrored version of the image is analyzed to provide a second independent view.

### 3. Center Crop

A center-cropped version focuses the detector on the central visual region of the image.

### Median Scoring

The three prediction scores are combined using the **median value**:

```text
Original Image ───────┐
                      │
Horizontal Flip ──────┼──→ AI Detector ──→ Median Score
                      │
Center Crop ──────────┘
```

Using the median helps reduce the effect of an unusually high or low prediction from one individual view.

## 🧠 Model

ColdRoot currently uses:

**Smogy/SMOGY-Ai-images-detector**

The model is integrated through the Hugging Face Transformers ecosystem and is used to estimate the likelihood that an image is AI-generated.

ColdRoot is intended as an **assistive detection tool**, not as definitive proof of image authenticity. Detection performance can vary depending on image source, compression, resizing, editing, and the generation model used.

## 🏗️ Project Structure

```text
ColdRoot/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── uploads/
│   └── User-uploaded images
│
├── heatmaps/
│   └── Generated visual analyses
│
├── templates/
│   └── index.html
│
└── static/
    ├── style.css
    └── app.js
```

### File Description

| File / Directory       | Purpose                                                                    |
| ---------------------- | -------------------------------------------------------------------------- |
| `app.py`               | Flask backend and image-detection logic                                    |
| `requirements.txt`     | Python dependencies                                                        |
| `templates/index.html` | Main web interface                                                         |
| `static/style.css`     | Frontend styling                                                           |
| `static/app.js`        | Client-side functionality                                                  |
| `uploads/`             | Temporary uploaded-image storage                                           |
| `heatmaps/`            | Generated heatmap outputs                                                  |
| `.gitignore`           | Prevents unnecessary files and sensitive/runtime data from being committed |

## ⚙️ Processing Pipeline

```text
Image Upload
     ↓
Input Validation
     ↓
Image Preprocessing
     ↓
┌──────────────┬──────────────┬──────────────┐
│ Original     │ Horizontal   │ Center Crop  │
│ Image        │ Flip         │              │
└──────────────┴──────────────┴──────────────┘
        ↓             ↓              ↓
        └─────────────┼──────────────┘
                      ↓
               AI Detection Model
                      ↓
               Three Predictions
                      ↓
                Median Scoring
                      ↓
             Final AI Confidence
                      ↓
              Result + Heatmap
```

## 🛠️ Technology Stack

* **Python**
* **Flask**
* **PyTorch**
* **Hugging Face Transformers**
* **Pillow**
* **HTML5**
* **CSS3**
* **JavaScript**

 🚀
## 📊 Example Workflow

1. Upload an image.
2. ColdRoot validates and preprocesses the image.
3. Three image views are generated.
4. Each view is passed through the AI-image detector.
5. The three predictions are combined using median scoring.
6. ColdRoot displays the final AI-generation confidence.
7. A visual heatmap can be generated for additional analysis.

## 🎯 Intended Use

ColdRoot can be used for:

* AI-generated image screening
* Academic and research projects
* Media verification workflows
* Demonstrations of AI-content detection
* Exploring robustness across different image-generation systems



## 🔮 Future Improvements

Potential future improvements include:

* Evaluation across a larger range of image generators
* Additional image-forensic features
* Improved heatmap visualization
* Ensemble-based detection
* Better calibration of confidence scores
* Continuous evaluation on authentic and synthetic image datasets
* GPU acceleration for faster inference
* Support for additional media formats

## 📌 Project Status

**ColdRoot is an actively developed AI-image detection project.**

The current version focuses on image-level AI-generation detection using a Transformer-based detector and multi-view inference.
