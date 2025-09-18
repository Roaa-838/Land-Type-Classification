# 🌍 Land Type Classification using Sentinel-2 (EuroSAT)

## 📌 Problem Statement

Accurate land type classification is critical for applications such as **agriculture monitoring, urban planning, water resource management, and environmental studies**.

This project builds a **deep learning model** to classify **land cover types in Egypt and similar regions** using **Sentinel-2 satellite imagery**.  
The goal is to classify input tiles into one of 10 categories:

* 🌾 AnnualCrop
* 🌲 Forest
* 🌿 HerbaceousVegetation
* 🛣️ Highway
* 🏭 Industrial
* 🐄 Pasture
* 🌳 PermanentCrop
* 🏘️ Residential
* 🌊 River
* 🌅 SeaLake

Users can interact with the model through a **Streamlit web app** that accepts satellite tiles and outputs the **top-2 predicted land uses**.

---

## 📊 Dataset

### Context
This project uses the **EuroSAT dataset (RGB)** and the **EuroSATallBands dataset (Sentinel-2 spectral bands)**, both based on Sentinel-2 satellite imagery, for land use and land cover classification.  
Both datasets are too large to store in GitHub. They must be downloaded manually or inside Google Colab.

### Content
* **EuroSAT (RGB):** JPG images with Red, Green, and Blue bands.
* **EuroSATallBands:** `.tif` images with all 13 spectral bands from Sentinel-2.
* Each image is **64×64 pixels**.

**Classes (10):**
AnnualCrop | Forest | HerbaceousVegetation | Highway | Industrial | Pasture | PermanentCrop | Residential | River | SeaLake

---

### 🔹 Local Setup

* Download the datasets from Kaggle:
  - [EuroSAT (RGB)](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset)
  - [EuroSATallBands](https://www.kaggle.com/datasets/apollo2506/eurosatallbands-dataset)

* Extract them into the following folder structure:
```
data/
├── raw/
│   ├── EuroSAT/
│   │   ├── AnnualCrop/
│   │   ├── Forest/
│   │   ├── HerbaceousVegetation/
│   │   ├── Highway/
│   │   ├── Industrial/
│   │   ├── Pasture/
│   │   ├── PermanentCrop/
│   │   ├── Residential/
│   │   ├── River/
│   │   ├── SeaLake/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── validation.csv
│   │   ├── label_map.json
│   │
│   ├── EuroSATallBands/
│   │   ├── *.tif
│   │   ├── metadata files
│
├── interim/       <- intermediate data (cleaned/augmented)
├── processed/     <- final datasets ready for training

```
⚠️ data/raw/ and all .zip files are excluded from GitHub via .gitignore.

**Files included:**

* `train.csv`, `validation.csv`, `test.csv` → Image splits
* `label_map.json` → Class mappings

**Csv structures:**
* `Filename` → the relative path of the image --> use as the *image column*
* `Label` → numeric class ID (0–9)
* `ClassName` → the human-readable class (AnnualCrop, Forest, …) -> use as the *label column*

⚠️ Note: Drop the **index column** when loading CSVs.

🔹 Google Colab Setup
Run the following inside your notebook:
```

⚠️ `data/raw/` and all `.zip` files are excluded from GitHub via `.gitignore`.

**Files included:**
* `train.csv`, `validation.csv`, `test.csv` → Image splits
* `label_map.json` → Class mappings

⚠️ Note: Drop the **index column** when loading CSVs.

---

### 🔹 Google Colab Setup

Run the following inside your notebook:

```bash
# Install Kaggle CLI
!pip install kaggle

# Upload Kaggle API key (kaggle.json)
from google.colab import files
files.upload()  # upload kaggle.json

# Move kaggle.json to the right place
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Create data folders
!mkdir -p data/raw data/interim data/processed

# Download EuroSAT (RGB)
!kaggle datasets download -d apollo2506/eurosat-dataset -p data/raw/
!unzip -q data/raw/eurosat-dataset.zip -d data/raw/

# Download EuroSATallBands
!kaggle datasets download -d apollo2506/eurosatallbands-dataset -p data/raw/
!unzip -q data/raw/eurosatallbands-dataset.zip -d data/raw/

```
---

### 🔹 Important Notes for Team Members

* Do not push raw datasets to GitHub.
The .gitignore file already excludes data/raw/ and large files (>100MB).
```
# Ignore raw data and large files
data/raw/
*.zip
*.tif
*.csv
*.json
```
* Only push:
    * Preprocessing / augmentation scripts
    * Processed datasets (if small enough, otherwise provide a link)
    * Model weights (if <100MB, otherwise upload to Google Drive/Dropbox and share the link)
* This keeps the repository lightweight, professional, and reproducible.

---

### Acknowledgements

Dataset reference:
Helber, Patrick; Bischke, Benjamin; Dengel, Andreas; Borth, Damian.
**EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification**.
*IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019.*

📂 [Dataset Link – EuroSAT on Kaggle](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset)

### License

📝 **CC0 – Public Domain**

---

## ⚙️ Methodology

The project follows this ML pipeline:

```mermaid

flowchart TD
    A[Data Collection] --> B[Preprocessing & Cleaning]
    B --> C[Exploratory Data Analysis]
    C --> D[Feature Engineering & Augmentation]
    D --> E[Model Training (CNN/ResNet/EfficientNet)]
    E --> F[Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)]
    F --> G[Deployment via Streamlit Web App]

```

### Key Steps:

1. **Data Preparation**

   * Resize & normalize spectral bands
   * Ensure consistent labeling
   * Augment images (cropping, flipping, rotation)

2. **Model Training**

   * Baseline CNN
   * Transfer Learning (ResNet / EfficientNet)
   * Loss: Cross-entropy
   * Optimizer: Adam

3. **Evaluation**

   * Metrics: Accuracy, Precision, Recall, F1-score
   * Class-wise performance
   * Confusion matrix

4. **Deployment**

   * Streamlit app for uploading images
   * Return top-2 predicted land classes

---

## 📈 Results

| Metric    | Score |
| --------- | ----- |
| Accuracy  | XX%   |
| Precision | XX%   |
| Recall    | XX%   |
| F1-score  | XX%   |

### Confusion Matrix

📸 (Add plot here once training is complete)

---

## 🖥️ Web Application

Built using **Streamlit**.

* Upload an image tile
* Get **top-2 predicted classes** with probabilities

📸 *Screenshots will go here*

---

## 🚀 How to Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/your-username/LandType-Classification.git
cd LandType-Classification
```

### 2. Create environment & install dependencies

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run training

```bash
python train.py
```

### 4. Launch Streamlit app

```bash
streamlit run app.py
```

---

## 📂 Repository Structure

```
LandType-Classification/
│── data/                # Dataset (or link in README)
│── notebooks/           # Jupyter notebooks for EDA & prototyping
│── src/                 # Source code (models, preprocessing, utils)
│── app.py               # Streamlit app
│── train.py             # Training script
│── requirements.txt     # Dependencies
│── README.md            # Project documentation
```

---

## 📌 Best Practices Implemented

✔️ Experiment tracking with MLflow / Weights & Biases
✔️ `requirements.txt` for reproducibility
✔️ GitHub Projects & Issues used for task tracking
✔️ Unit tests for preprocessing functions
✔️ Deployment on **Streamlit Cloud / Hugging Face Spaces**

---
