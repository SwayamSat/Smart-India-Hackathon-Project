import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set the path to your dataset
dataset_path = 'dataset'

# Parameters
img_height, img_width = 128, 128
batch_size = 32
num_classes = 2  # Drone and bird
epochs = 30
learning_rate = 0.0001
def wav_to_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=None)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    spectrogram = librosa.util.normalize(spectrogram)
    return spectrogram

# Load and preprocess the data
X = []
y = []

for label in ['bird', 'drone']:
    class_folder = os.path.join(dataset_path, label)
    if os.path.isdir(class_folder):
        for file_name in os.listdir(class_folder):
            if file_name.endswith('.wav'):
                file_path = os.path.join(class_folder, file_name)
                spectrogram = wav_to_spectrogram(file_path)
                
                # Resize or pad the spectrogram to fit the desired dimensions
                if spectrogram.shape[1] > img_width:
                    spectrogram = spectrogram[:, :img_width]
                else:
                    padding = img_width - spectrogram.shape[1]
                    spectrogram = np.pad(spectrogram, ((0, 0), (0, padding)), mode='constant')

                X.append(spectrogram)
                y.append(label)

# Convert lists to numpy arrays
X = np.array(X)
X = X[..., np.newaxis]  # Add channel dimension
y = np.array(y)

# Encode labels and convert to one-hot
y = np.array([0 if label == 'bird' else 1 for label in y])
y = to_categorical(y, num_classes=num_classes)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Normalize the data
X_train = X_train / np.max(X_train)
X_test = X_test / np.max(X_test)

# Build the CNN model with batch normalization and dropout
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.show()

# Save the model
model.save('trainedCNNModel.h5')
