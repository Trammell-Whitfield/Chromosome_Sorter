import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import cv2
import os

# Step 1: Load and prepare the pre-existing dataset
def load_existing_dataset(data_path):
    images = []
    labels = []
    for chromosome_type in os.listdir(data_path):
        type_path = os.path.join(data_path, chromosome_type)
        if os.path.isdir(type_path):
            for image_file in os.listdir(type_path):
                img = cv2.imread(os.path.join(type_path, image_file), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(chromosome_type)
    return np.array(images), np.array(labels)

# Load the pre-existing dataset
data_path = 'path_to_your_existing_dataset'
X, y = load_existing_dataset(data_path)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Step 2: Image Preprocessing
def preprocess_image(image):
    resized = cv2.resize(image, (100, 100))  # Adjust size as needed
    normalized = resized / 255.0
    return normalized.reshape(100, 100, 1)  # Add channel dimension

X_train = np.array([preprocess_image(img) for img in X_train])
X_test = np.array([preprocess_image(img) for img in X_test])

# Step 3: Define and train the model (or load a pre-trained model)
def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Either train a new model or load a pre-trained one
train_new_model = False  # Set to True if you want to train a new model

if train_new_model:
    model = create_model((100, 100, 1), len(le.classes_))
    history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)
    model.save('chromosome_model.h5')
else:
    model = load_model('chromosome_model.h5')

# Step 4: Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# Step 5: Function to distinguish chromosomes
def distinguish_chromosomes(images):
    preprocessed = np.array([preprocess_image(img) for img in images])
    predictions = model.predict(preprocessed)
    predicted_labels = le.inverse_transform(np.argmax(predictions, axis=1))
    return predicted_labels

# Example usage
'''
def load_new_images(new
'''