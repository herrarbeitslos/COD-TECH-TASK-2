"""
=============================================================
  INTERNSHIP TASK - 2: DEEP LEARNING PROJECT
  Author  : Kartikay Verma
  Task    : Image Classification using TensorFlow / Keras
  Dataset : CIFAR-10 (synthetic demo - same architecture)
=============================================================
"""

import os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.makedirs("results", exist_ok=True)

CLASS_NAMES = ["airplane","automobile","bird","cat","deer",
               "dog","frog","horse","ship","truck"]

print("="*60)
print("  INTERNSHIP TASK-2: DEEP LEARNING IMAGE CLASSIFICATION")
print("  Author : Kartikay Verma")
print("="*60)
print(f"TensorFlow : {tf.__version__}")

# ── Synthetic CIFAR-10-like data (32x32 RGB, 10 classes) ──
print("\n[1/5] Generating synthetic dataset …")
N_TRAIN, N_TEST = 5000, 1000
X_train = np.random.rand(N_TRAIN, 32, 32, 3).astype("float32")
y_train = np.random.randint(0, 10, N_TRAIN)
X_test  = np.random.rand(N_TEST,  32, 32, 3).astype("float32")
y_test  = np.random.randint(0, 10, N_TEST)
print(f"   Train: {X_train.shape}  |  Test: {X_test.shape}")

# ── Sample image grid ──────────────────────────────────────
print("\n[2/5] Saving sample image grid …")
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
fig.suptitle("Sample Training Images (synthetic CIFAR-10 style)\nAuthor: Kartikay Verma",
             fontsize=13, fontweight="bold")
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i])
    ax.set_title(CLASS_NAMES[y_train[i]], fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.savefig("results/01_sample_images.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved → results/01_sample_images.png")

# ── Build CNN ──────────────────────────────────────────────
print("\n[3/5] Building CNN model …")
model = models.Sequential([
    layers.Conv2D(32,(3,3),padding="same",activation="relu",input_shape=(32,32,3),name="conv1_1"),
    layers.BatchNormalization(),
    layers.Conv2D(32,(3,3),padding="same",activation="relu",name="conv1_2"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    layers.Conv2D(64,(3,3),padding="same",activation="relu",name="conv2_1"),
    layers.BatchNormalization(),
    layers.Conv2D(64,(3,3),padding="same",activation="relu",name="conv2_2"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    layers.Conv2D(128,(3,3),padding="same",activation="relu",name="conv3_1"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.40),

    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.50),
    layers.Dense(10, activation="softmax", name="output"),
], name="CIFAR10_CNN")

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

# ── Train (quick demo run) ─────────────────────────────────
print("\n[4/5] Training …")
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=10, batch_size=64, verbose=1)
model.save("results/cifar10_cnn_model.keras")
print("   Model saved → results/cifar10_cnn_model.keras")

# ── Visualisations ─────────────────────────────────────────
print("\n[5/5] Generating visualisations …")
ep = range(1, len(history.history["accuracy"])+1)

# Training curves
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,5))
fig.suptitle("Training History — Kartikay Verma", fontsize=13, fontweight="bold")
ax1.plot(ep, history.history["accuracy"],     "b-o", ms=5, label="Train Acc")
ax1.plot(ep, history.history["val_accuracy"], "r-o", ms=5, label="Val Acc")
ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(alpha=0.3)
ax2.plot(ep, history.history["loss"],     "b-o", ms=5, label="Train Loss")
ax2.plot(ep, history.history["val_loss"], "r-o", ms=5, label="Val Loss")
ax2.set_title("Loss"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/02_training_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved → results/02_training_curves.png")

# Confusion matrix
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
cm = confusion_matrix(y_test, y_pred)
cm_pct = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100
fig, ax = plt.subplots(figsize=(12,10))
sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, ax=ax)
ax.set_title("Confusion Matrix (%) — Kartikay Verma", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
plt.tight_layout()
plt.savefig("results/03_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved → results/03_confusion_matrix.png")

# Per-class accuracy
pca = cm.diagonal() / cm.sum(axis=1) * 100
colors = ["#4CAF50" if a>=70 else "#FF9800" if a>=55 else "#F44336" for a in pca]
fig, ax = plt.subplots(figsize=(12,5))
bars = ax.bar(CLASS_NAMES, pca, color=colors, edgecolor="white", linewidth=0.8)
ax.axhline(pca.mean(), color="navy", linestyle="--", linewidth=1.5,
           label=f"Mean = {pca.mean():.1f}%")
for bar, val in zip(bars, pca):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylim(0,115); ax.set_title("Per-Class Accuracy — Kartikay Verma", fontsize=13, fontweight="bold")
ax.set_xlabel("Class"); ax.set_ylabel("Accuracy (%)"); ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/04_per_class_accuracy.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved → results/04_per_class_accuracy.png")

# Prediction showcase
idxs = np.random.choice(len(X_test), 20, replace=False)
fig, axes = plt.subplots(4, 5, figsize=(14,11))
fig.suptitle("Model Predictions — Kartikay Verma\n(green = correct, red = wrong)",
             fontsize=12, fontweight="bold")
for ax, idx in zip(axes.flat, idxs):
    ax.imshow(X_test[idx])
    tl = CLASS_NAMES[y_test[idx]]; pl = CLASS_NAMES[y_pred[idx]]
    col = "green" if tl==pl else "red"
    for sp in ax.spines.values(): sp.set_edgecolor(col); sp.set_linewidth(3)
    ax.set_title(f"T:{tl}\nP:{pl}", fontsize=8, color=col, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig("results/05_predictions_showcase.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved → results/05_predictions_showcase.png")

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print("\n"+"="*60)
print("  FINAL RESULTS — Kartikay Verma")
print("="*60)
print(f"  Test Accuracy : {test_acc*100:.2f}%")
print(f"  Test Loss     : {test_loss:.4f}")
print(f"  Epochs run    : {len(list(ep))}")
print("="*60)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
print("\nAll outputs saved in results/")
