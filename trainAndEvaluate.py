import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, utils, callbacks
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import tensorflow as tf

# Hyperparameters 
EPOCHS = 10
BATCH_SIZE = 64
MODEL_PATH = "cifar10_cnn_model.keras"
NUM_CLASSES = 10
IMG_SHAPE = (32, 32, 3)

# Load data (CIFAR-10) 
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = y_train.flatten()
y_test = y_test.flatten()

# Normalisasi & One-hot 
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train_cat = utils.to_categorical(y_train, NUM_CLASSES)
y_test_cat = utils.to_categorical(y_test, NUM_CLASSES)

# Build CNN Model 
def build_model(input_shape=IMG_SHAPE, num_classes=NUM_CLASSES):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Compile model 
model = build_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Callback 
cb = [callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]

# Train model 
history = model.fit(
    x_train, y_train_cat,
    validation_data=(x_test, y_test_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=cb
)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=2)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2%}")

# Confusion Matrix & Report
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test, y_pred)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - CIFAR10 CNN')
plt.show()

# Training & Validation Plots
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training & Validation Loss')

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training & Validation Accuracy')
plt.tight_layout()
plt.show()

# visualisasi prediksi contoh
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
n_examples = 6
plt.figure(figsize=(12,4))
idxs = np.random.choice(len(x_test), n_examples, replace=False)
for i, idx in enumerate(idxs):
    plt.subplot(1, n_examples, i+1)
    plt.imshow(x_test[idx])
    plt.title(f"T:{class_names[y_test[idx]]}\nP:{class_names[y_pred[idx]]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Save model
model.save(MODEL_PATH)
print(f"\n Model trained & saved to {MODEL_PATH}")