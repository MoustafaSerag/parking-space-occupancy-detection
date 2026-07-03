# 🚗 Parking Space Occupancy Classification

A machine learning project that classifies parking spaces as **Empty** or **Occupied** using image classification and a **Support Vector Machine (SVM)**.

The project demonstrates a complete machine learning workflow, including image preprocessing, dataset preparation, model training, hyperparameter tuning with GridSearchCV, model evaluation, and model serialization.

---

## 📌 Features

- Binary image classification
- Parking space occupancy prediction
- Image preprocessing and resizing
- Feature extraction using flattened image vectors
- Support Vector Machine (SVM) classifier
- Hyperparameter tuning with GridSearchCV
- Model evaluation using accuracy score
- Trained model serialization with Pickle

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| Machine Learning | Scikit-learn |
| Image Processing | scikit-image |
| Numerical Computing | NumPy |
| Model | Support Vector Machine (SVM) |
| Hyperparameter Optimization | GridSearchCV |
| Model Serialization | Pickle |

---

## 📂 Project Structure

```
parking-space-occupancy-classification/
│
├── clf-data/
│   ├── empty/
│   └── not_empty/
│
├── main.py
├── model.p
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔄 Machine Learning Pipeline

```
Dataset
   │
   ▼
Load Images
   │
   ▼
Resize Images (15 × 15)
   │
   ▼
Flatten Images
   │
   ▼
Train/Test Split
   │
   ▼
GridSearchCV
   │
   ▼
Train SVM Classifier
   │
   ▼
Evaluate Accuracy
   │
   ▼
Save Trained Model
```

---

## 📊 Dataset

The dataset contains two classes:

- 🟢 Empty
- 🚗 Not Empty

Each image is resized to **15 × 15 pixels** before being converted into a one-dimensional feature vector for training.

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/MoustafaSerag/parking-space-occupancy-detection.git
cd parking-space-occupancy-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Training

Run the training script:

```bash
python main.py
```

The script will:

- Load the dataset
- Train an SVM classifier
- Search for the best hyperparameters using GridSearchCV
- Evaluate model accuracy
- Save the trained model as `model.p`

---

## 📈 Model Evaluation

The model is evaluated using:

- Accuracy Score

---

## 🔮 Future Improvements

- Deep Learning implementation with PyTorch
- Convolutional Neural Networks (CNNs)
- Real-time parking occupancy detection
- Webcam or CCTV integration
- Web dashboard for monitoring parking availability
- Multi-class parking analysis

---

## 👨‍💻 Author

**Moustafa Serag**

GitHub: https://github.com/MoustafaSerag
