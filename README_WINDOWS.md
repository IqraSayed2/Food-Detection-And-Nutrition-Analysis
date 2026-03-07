# 🖥️ NutriScan AI - Complete Windows Setup Guide

This guide is written for **complete beginners** who have nothing installed on their Windows PC. By following these steps in order, you will download everything needed to run this project from absolute scratch.

---

## 🛑 Stage 1: Install Required Software

### 1. Download & Install Git
Git allows you to download (clone) the code from GitHub.
1. Go to: [https://git-scm.com/download/win](https://git-scm.com/download/win)
2. Click on **"64-bit Git for Windows Setup"**.
3. Run the downloaded `.exe` file.
4. Keep clicking **Next** to install with all the default settings until it finishes.

### 2. Download & Install Miniconda (Python)
Miniconda manages Python and keeps the project isolated so it doesn't break other things on your PC.
1. Go to: [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
2. Download the **"Miniconda3 Windows 64-bit"** installer.
3. Run the downloaded installer.
4. Click **Next** -> **I Agree** -> **Just Me** (or All Users) -> **Next**.
5. When you reach "Advanced Options", you can leave everything as default and click **Install**.

---

## 🚀 Stage 2: Download the Project

1. Click on the **Start Menu** (Windows icon) on your taskbar.
2. Search for **"Anaconda Prompt"** and open it. (A black command-line window will appear).
3. We are going to download the code to your Desktop. Type the following command and press **Enter**:
   ```cmd
   cd Desktop
   ```
4. Now, download the project code by typing this command and pressing **Enter**:
   ```cmd
   git clone https://github.com/IqraSayed2/Food-Detection-And-Nutrition-Analysis.git
   ```
5. Enter the project folder:
   ```cmd
   cd Food-Detection-And-Nutrition-Analysis
   ```

---

## ⚙️ Stage 3: Setup the Environment

Keep your **Anaconda Prompt** open and make sure you are in the `Food-Detection-And-Nutrition-Analysis` folder. 

1. Create a dedicated Python environment for this project (press **Enter** and type **y** if it asks to proceed):
   ```cmd
   conda create -n food-env python=3.10 -y
   ```

2. Activate the environment (you must do this every time you want to run the project):
   ```cmd
   conda activate food-env
   ```
   *(You should see `(food-env)` at the beginning of your command prompt line).*

3. **Install the dependencies**. 
   Since this project uses AI models, we need PyTorch. Run this command first to get PyTorch (this might take a few minutes as it downloads heavy files):
   ```cmd
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   
4. Next, install all the other required Python packages:
   ```cmd
   pip install flask werkzeug pillow opencv-python numpy ultralytics transformers requests tabulate pandas
   ```
   
*(Note: We use this manual pip command instead of `requirements.txt` because the current requirements file is strictly tied to a specific system setup, and doing it manually ensures it works perfectly on your fresh Windows install).*

---

## ▶️ Stage 4: Run the Application

You are almost ready!

1. Still inside your **Anaconda Prompt** (with `(food-env)` activated), start the web server:
   ```cmd
   python app.py
   ```
   
2. The first time you run this, it will automatically download some required AI models (`yolov8s-world.pt` and `distilgpt2`). **This might take 1–3 minutes** depending on your internet speed. Wait until you see a message like:
   ```text
   * Running on http://127.0.0.1:5001
   ```

3. Open your web browser (Chrome, Edge, Firefox, etc.) and go to this address:
   **[http://127.0.0.1:5001](http://127.0.0.1:5001)**

🎉 **Congratulations! The app is now running on your Windows machine.**

---

## 🛠️ How to run the app again next time?
If you close the application and want to run it again tomorrow, simply do this:
1. Open **Anaconda Prompt** from your Start Menu.
2. Navigate to your project folder:
   ```cmd
   cd Desktop\Food-Detection-And-Nutrition-Analysis
   ```
3. Activate the environment:
   ```cmd
   conda activate food-env
   ```
4. Start the server:
   ```cmd
   python app.py
   ```
5. Open **http://127.0.0.1:5001** in your browser.
