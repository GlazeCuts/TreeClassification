# TreeClassification
This project implements a deep learning model to classify images of 30+ tree and plant species using Convolutional Neural Networks (CNN) and EfficientNetB0 with TensorFlow/Keras. The dataset contains real-world images and includes significant class imbalance, which is handled using data augmentation and class weighting.

1. Dataset Preparation
Images were collected and structured in directories by class.

.git and unrelated folders were removed from dataset directory.

Duplicate images were filtered out.

Outliers were removed based on image dimension anomalies (too small or too large).

2. Data Preprocessing
Normalization: Pixel values were scaled to [0, 1] using:

rescale=1./255
Image Resizing: All images resized to 224x224 to match EfficientNetB0's input shape.

3. Data Augmentation
Implemented with ImageDataGenerator to improve model generalization:

ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)
4. Class Imbalance Handling
Used collections.Counter to get per-class image count.

Applied class weights during training via:

model.fit(..., class_weight=class_weight_dict)
This prevents the model from overfitting to more common classes (e.g., "other") and improves minority class performance.

5. CNN Architecture
Two models were tested:

🔹 A Basic CNN:

Conv2D → MaxPooling → Conv2D → MaxPooling → Flatten → Dense(256) → Dropout → Output
🔹 Transfer Learning with EfficientNetB0:

EfficientNetB0 (pretrained, unfrozen) →
GlobalAveragePooling →
Dense(128, relu) →
Dropout(0.3) →
Dense(num_classes, softmax)
Fine-tuning was enabled by setting base_model.trainable = True

Lower learning rate (0.0001) used to prevent catastrophic forgetting.

6. Model Training
Training done on Google Colab with GPU or TPU support.

Epochs: 10 (can be increased based on performance)

Monitored accuracy and loss on both training and validation sets.

7. 💾 Model Saving
Final model saved in HDF5 format:
model.save('tree_species_classifier.h5')
✅ Summary
The final pipeline effectively:

Handles image normalization

Balances class distribution using class_weight

Uses transfer learning for strong feature extraction

Achieves much better accuracy than a base CNN on a diverse 30-class dataset
