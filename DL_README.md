# 🧠 Internship Task 2 — Deep Learning Project

**Author:** Kartikay Verma  
**Task:** Image Classification using TensorFlow / Keras (CNN)  
**Dataset:** CIFAR-10 (10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

---

## 🏗️ Model Architecture
- 3 × Conv Blocks (Conv2D → BatchNorm → MaxPool → Dropout)
- Filter sizes: 32 → 64 → 128
- Dense(256) → Dropout → Output(10, softmax)
- Total params: ~668K

## ⚙️ Techniques Used
- Batch Normalization, Dropout (regularization)
- Adam optimizer, Sparse Categorical Crossentropy
- EarlyStopping + ReduceLROnPlateau callbacks
- Data Augmentation (flip, shift, rotate)

## 📊 Output Files (in `results/`)
| File | Description |
|------|-------------|
| `01_sample_images.png` | Grid of training images |
| `02_training_curves.png` | Accuracy & Loss over epochs |
| `03_confusion_matrix.png` | Heatmap of predictions |
| `04_per_class_accuracy.png` | Per-class bar chart |
| `05_predictions_showcase.png` | 20 sample predictions |
| `cifar10_cnn_model.keras` | Saved model |

## 🚀 Setup & Run
```bash
pip install tensorflow scikit-learn matplotlib seaborn
python deep_learning_classifier.py
```

> **Note:** Uses real CIFAR-10 data when internet is available. Falls back to synthetic data for demonstration.
