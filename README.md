# Food-Detection-And-Nutrition-Analysis
This repository combines the power of YOLO v11 and LLaMA 3.2 1B to provide an intelligent food detection and nutritional analysis system

# food-detection-and-nutritional-value



### Features:
Food Detection: Utilize YOLO v11 for accurate and efficient food item recognition in images or live video streams.
Nutritional Information Generation: Leverage LLaMA 3.2 1B to provide detailed nutritional insights for the detected food items, including calories, macronutrients, and more.
Seamless Integration: Easily adaptable for real-time applications like mobile apps, websites, and IoT devices.
### Use Cases:
Personal health tracking
Meal planning and dietary analysis
Smart kitchen and restaurant automation
### Technologies:
YOLO v11: State-of-the-art object detection model fine-tuned for food recognition.
LLaMA 3.2 1B: A powerful language model for generating nutritional descriptions and contextual information.
Feel free to clone this repository and contribute! 🚀



## Project Organization


## Setup Instructions

### 1. Clone the Repository

```
git clone <this-repo-url>
cd Food-Detection-And-Nutrition-Analysis
```

### 2. (Recommended) Create and Activate a Conda Environment

```
conda create --name food-detection-and-nutritional-value python=3.10 -y
conda activate food-detection-and-nutritional-value
```

### 3. Install Python Dependencies

```
pip install -r requirements.txt
```

### 4. Download/Place Model File

Ensure the YOLO model file exists at `models/last (1) (1).pt`.

### 5. Prepare Data

- For batch processing, place your images in `data/raw/archive (1)` (create the folder if it does not exist).
- For web uploads, images will be uploaded via the web interface.

---

## Usage

### Option 1: Run the Web Application (Flask)

This will start a web interface for uploading images and getting food detection and nutrition analysis.

```
python app.py
```

Then open your browser and go to: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

### Option 2: Run Batch/Image Processing Script

This will process all images in the specified folder and print results to the console.

```
python main.py
```

---

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         food_detection_and_nutritional_value and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── food_detection_and_nutritional_value   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes food_detection_and_nutritional_value a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------
