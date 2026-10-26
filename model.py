import tensorflow as tf
from tensorflow.keras import layers, models

def build_liveness_cnn(input_shape=(128, 128, 3)):
    """
    Constructs a custom custom CNN architecture optimized for mobile 
    and edge liveness authentication systems.
    """
    model = models.Sequential([
        # Block 1: Feature Extraction
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2: Texture & Depth Analysis
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3: Micro-surface anomaly tracking
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Block 4: Dense Classification Layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), # Regularization to prevent overfitting to background noise
        layers.Dense(1, activation='sigmoid') # Binary probability output (0 = Spoof, 1 = Real)
    ])
    
    return model

if __name__ == "__main__":
    # Print out structural layer summaries to verify the model layout
    liveness_network = build_liveness_cnn()
    liveness_network.summary()
