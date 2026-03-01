# 🍽️ NutriScan AI

> **Precision fuel for high-performers.** AI-powered food detection and real-time nutritional analysis — no manual typing, no hidden fluff.

![NutriScan AI](https://images.unsplash.com/photo-1546069901-ba9599a7e63c?auto=format&fit=crop&w=1200&q=80)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Running the App](#running-the-app)
- [How It Works](#how-it-works)
- [API Reference](#api-reference)
- [Supported Foods](#supported-foods)
- [Troubleshooting](#troubleshooting)

---

## Overview

NutriScan AI is a Flask web application that lets you **upload or capture a photo of any meal** and instantly receive:

- A list of detected food items (powered by YOLOv8-World open-vocabulary detection)
- Per-food nutritional breakdown: **Calories, Protein, Carbs, Fat**
- An AI-generated verdict about your meal
- An annotated image showing bounding boxes around detected foods

The app runs a two-stage pipeline:

```
Image → YOLOv8-World (Food Detection) → Open Food Facts API / distilgpt2 (Nutrition) → Result
```

---

## Features

| Feature | Description |
|---|---|
| 🎯 Open-vocabulary detection | YOLOv8-World detects any food class you define — no retraining needed |
| 🔬 Multi-tier nutrition lookup | Tries Open Food Facts API first, falls back to AI inference |
| 📷 Camera capture | Live webcam capture directly in the browser |
| 🖼️ Drag & Drop upload | Supports PNG, JPG, JPEG up to 100 MB |
| ⚡ Lightweight AI | Uses `distilgpt2` (82 MB) — loads in seconds on CPU |
| 🌐 Clean web UI | Neubrutalist design with two pages: Landing + Analyzer |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.10, Flask |
| Food Detection | Ultralytics YOLOv8-World (`yolov8s-world.pt`) |
| Nutrition AI | HuggingFace `distilgpt2` + Open Food Facts API |
| Image Processing | OpenCV, PIL, NumPy |
| Frontend | HTML5, Vanilla CSS, Vanilla JS |
| Environment | Conda (`food-detection-and-nutritional-value`) |

---

## Project Structure

```
Food-Detection-And-Nutrition-Analysis/
├── app.py                          # Flask server — routes & API endpoints
├── main_fe.py                      # Orchestration: detection → nutrition pipeline
├── main.py                         # CLI entry point (batch processing)
│
├── food_detection_and_nutritional_value/
│   └── modeling/
│       ├── LLAMA.py                # Nutrition intelligence (AI + API lookup)
│       └── YOLOPredict.py          # YOLOv8-World food detection
│
├── templates/
│   ├── index.html                  # Landing page (/)
│   └── analyze.html                # Analyzer dashboard (/analyze)
│
├── uploads/
│   └── raw/                        # Uploaded images saved here temporarily
│
├── models/                         # (Optional) Custom model weights
├── data/                           # Sample data / test images
├── notebooks/                      # Jupyter notebooks for experimentation
├── docs/                           # Additional documentation
│
├── yolov8s-world.pt               # YOLOv8-World model weights (auto-downloaded)
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Project metadata
└── README.md                      # This file
```

---

## Prerequisites

Before you start, make sure you have these installed:

### 1. Conda (Anaconda or Miniconda)
Download from: https://docs.conda.io/en/latest/miniconda.html

Verify installation:
```bash
conda --version
# Expected: conda 23.x.x or higher
```

### 2. Git
```bash
git --version
# Expected: git version 2.x.x
```

### 3. macOS (Apple Silicon or Intel)
- The app runs fully on **CPU** — no GPU required
- Tested on macOS Sonoma 14+ with Apple M-series chips
- Minimum 8 GB RAM recommended (16 GB for faster inference)

---

## Installation & Setup

### Step 1 — Clone the Repository

```bash
git clone https://github.com/IqraSayed2/Food-Detection-And-Nutrition-Analysis.git
cd Food-Detection-And-Nutrition-Analysis
```

### Step 2 — Create the Conda Environment

```bash
conda create -n food-detection-and-nutritional-value python=3.10 -y
```

Activate the environment:
```bash
conda activate food-detection-and-nutritional-value
```

> ⚠️ **Important:** Always activate this environment before running any commands.

### Step 3 — Install Dependencies

```bash
pip install flask werkzeug pillow opencv-python numpy torch torchvision ultralytics transformers requests tabulate
```

Or if a `requirements.txt` works on your system:
```bash
pip install -r requirements.txt
```

> **Note for Apple Silicon (M1/M2/M3):** If you face issues with `torch`, install the macOS-optimised version:
> ```bash
> pip install torch torchvision torchaudio
> ```

### Step 4 — Download Model Weights

The YOLOv8-World model downloads automatically on first run. But to pre-download it:

```bash
conda run -n food-detection-and-nutritional-value python -c "from ultralytics import YOLO; YOLO('yolov8s-world.pt')"
```

The `distilgpt2` nutrition model also downloads automatically from HuggingFace on first run (~82 MB).

---

## Running the App

### Quick Start

```bash
conda activate food-detection-and-nutritional-value
python app.py
```

### Using Conda Run (without activating)

```bash
conda run -n food-detection-and-nutritional-value python app.py
```

### Expected Output on Startup

```
Loading Lightweight Nutritional Intelligence: distilgpt2
tokenizer_config.json: 100%|████████| ...
* Running on http://127.0.0.1:5001
* Debug mode: on
```

> The first startup takes **1–3 minutes** as models are loaded into memory. Subsequent requests are fast.

### Open in Browser

Navigate to:

| Page | URL |
|---|---|
| 🏠 Landing Page | http://127.0.0.1:5001/ |
| 🔬 Analyzer | http://127.0.0.1:5001/analyze |

---

## How It Works

### Stage 1 — Food Detection (YOLOv8-World)

`food_detection_and_nutritional_value/modeling/YOLOPredict.py`

- Uses **YOLOv8s-World**, an open-vocabulary object detection model
- Custom food classes are set explicitly:
  ```python
  ['french fries', 'pancake', 'lettuce', 'strawberry', 'blueberry',
   'burger', 'pizza', 'sandwich', 'salad', 'bread', 'egg', 'banana', 'apple']
  ```
- Confidence threshold: `0.01` (low, to maximize detection sensitivity)
- Returns bounding boxes + class names + confidence scores

### Stage 2 — Nutritional Analysis (Multi-tier)

`food_detection_and_nutritional_value/modeling/LLAMA.py`

The system tries these in order:

1. **Open Food Facts API** — real verified nutritional data:
   ```
   GET https://world.openfoodfacts.org/cgi/search.pl?search_terms={food}&action=process&json=1
   ```
   Tries the exact name, synonyms, and the core ingredient word.

2. **distilgpt2 AI Inference** — if API returns no data, the model generates nutrient estimates using a structured prompt:
   ```
   Food Facts: Pancake
   Nutrients (100g):
   Calories:
   ```

3. **Deterministic Hash Fallback** — if AI output is unparseable, generates unique realistic values using `md5(food_name)` — ensures different foods always get different numbers.

### Stage 3 — Image Annotation & Response

`main_fe.py`

- Draws bounding boxes + labels on the detected image using OpenCV
- Encodes annotated image to base64
- Returns JSON:
  ```json
  {
    "detected_items": ["pancake", "banana"],
    "nutrition_info": "Pancake\nCalories: 227\nProtein: 6g\nCarbs: 34g\nFat: 8g",
    "annotated_image": "<base64 string>",
    "detections_raw": [{"class": "pancake", "score": 0.87, "bbox": [...]}]
  }
  ```

---

## API Reference

### `POST /detect`

Accepts a food image and returns detection results.

**Request (multipart form):**
```bash
curl -X POST http://127.0.0.1:5001/detect \
  -F "file=@/path/to/your/meal.jpg"
```

**Request (base64, from camera):**
```
Content-Type: application/x-www-form-urlencoded
Body: image=data:image/jpeg;base64,<base64string>
```

**Response:**
```json
{
  "detected_items": ["burger", "fries"],
  "nutrition_info": "Burger\nCalories: 542\nProtein: 28g\nCarbs: 41g\nFat: 26g\n\nFrench Fries\nCalories: 312\nProtein: 3g\nCarbs: 41g\nFat: 15g",
  "annotated_image": "<base64 encoded JPEG>",
  "detections_raw": [
    {"class": "burger", "score": 0.91, "bbox": [x1, y1, x2, y2]},
    {"class": "fries",  "score": 0.78, "bbox": [x1, y1, x2, y2]}
  ]
}
```

---

## Supported Foods

The detector is configured to identify these food items out of the box:

| Category | Items |
|---|---|
| Fast Food | Burger, French Fries, Pizza, Sandwich |
| Produce | Banana, Apple, Lettuce, Strawberry, Blueberry |
| Breakfast | Pancake, Egg, Bread |
| Other | Salad |

> **To add more foods**, edit `YOLOPredict.py` and append to the classes list:
> ```python
> self.model.set_classes([
>     'french fries', 'pancake', ..., 'sushi', 'pasta'  # add here
> ])
> ```
> YOLOv8-World supports **any natural language food description** without retraining.

---

## Troubleshooting

### ❌ `ModuleNotFoundError: No module named 'ultralytics'`
```bash
conda activate food-detection-and-nutritional-value
pip install ultralytics
```

### ❌ `ConnectionRefusedError` when accessing the site
The server is still loading models. Wait ~2 minutes and refresh.

### ❌ `Food not detected` even with clear image
- Lower confidence threshold in `YOLOPredict.py`:
  ```python
  self.conf_threshold = 0.005  # even lower
  ```
- Make sure the food is in the supported classes list
- Try a clearer, better-lit photo

### ❌ `Calories: ??` in results
The Open Food Facts API returned no data for that specific food. The AI fallback will still generate a result. This is normal for niche or brand-specific foods.

### ❌ `anaconda-cloud-auth` errors in terminal
These are cosmetic warnings from Conda's plugin system and **do not affect** the app's functionality. You can safely ignore them.

### ❌ Server already running / port in use
```bash
pkill -f "python app.py"
python app.py
```

---

## Development Notes

- **Port:** App runs on `5001` (not the default Flask `5000`)
- **Upload limit:** 100 MB per image
- **Supported formats:** PNG, JPG, JPEG
- **Model cache:** HuggingFace models are cached at `~/.cache/huggingface/`
- **YOLO weights:** Stored in the project root directory

---

## License

This project is part of an academic AI/ML demonstration. All third-party models (YOLOv8, distilgpt2) are subject to their respective licenses.

- [Ultralytics YOLOv8 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)
- [distilgpt2 License (MIT)](https://huggingface.co/distilgpt2)
- [Open Food Facts License (Open Database License)](https://openfoodfacts.org/data)

---

*Built with ⚡ by the NutriScan AI team.*
