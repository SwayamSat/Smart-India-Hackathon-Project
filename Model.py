import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define constants
IMG_HEIGHT, IMG_WIDTH = 525, 700
BATCH_SIZE = 16
MAX_EPOCHS = 50  # Maximum number of epochs
PATIENCE = 10    # Number of epochs with no improvement after which training will be stopped
NUM_CLASSES = 2

def create_model():
    """
    Create and compile the CNN model for imbalanced dataset.
    
    Returns:
    tf.keras.Model: Compiled CNN model
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

def create_data_generators():
    """
    Create data generators for training, validation, and testing.
    
    Returns:
    tuple: (train_generator, val_generator, test_generator)
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        'dataset',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=['bird', 'drone'],
        subset='training'
    )
    
    val_generator = train_datagen.flow_from_directory(
        'dataset',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=['bird', 'drone'],
        subset='validation'
    )
    
    test_generator = test_datagen.flow_from_directory(
        'dataset',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=['bird', 'drone'],
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

def compute_class_weights(train_generator):
    """
    Compute class weights for imbalanced dataset.
    
    Args:
    train_generator: Training data generator
    
    Returns:
    dict: Class weights
    """
    y_train = train_generator.classes
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    return dict(enumerate(class_weights))

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss.
    
    Args:
    history: Training history object
    """
    metrics = ['accuracy', 'auc', 'loss']
    plt.figure(figsize=(15, 5))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        plt.plot(history.history[metric], label=f'Training {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
        plt.title(f'Training and Validation {metric.capitalize()}')
        plt.legend()
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    # Create data generators
    train_generator, val_generator, test_generator = create_data_generators()
    
    # Compute class weights
    class_weights = compute_class_weights(train_generator)
    print("Class weights:", class_weights)
    
    # Create and compile the model
    model = create_model()
    model.summary()
    
    # Define early stopping callback
    early_stopping = callbacks.EarlyStopping(
        monitor='val_auc',  # Monitor validation AUC
        mode='max',         # We want to maximize AUC
        patience=PATIENCE,
        verbose=1,
        restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        epochs=MAX_EPOCHS,
        validation_data=val_generator,
        class_weight=class_weights,
        callbacks=[early_stopping]
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate the model
    test_loss, test_accuracy, test_auc = model.evaluate(test_generator)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    # Save the model
    model.save('trainedCNNModel.h5')
    print("Model saved as 'trainedCNNModel.h5'")
    
    # Make predictions on test data
    predictions = model.predict(test_generator)
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    
    # Print some example predictions
    print("\nExample predictions:")
    for i in range(5):
        true_class = test_generator.classes[i]
        predicted_class = predicted_classes[i]
        confidence = predictions[i][0] if predicted_class == 1 else 1 - predictions[i][0]
        print(f"True class: {'drone' if true_class else 'bird'}, "
              f"Predicted class: {'drone' if predicted_class else 'bird'}, "
              f"Confidence: {confidence:.4f}")

if __name__ == "__main__":
    main() 