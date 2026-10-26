import os
import numpy as np
import tensorflow as tf
from model import build_liveness_cnn

def execute_training_pipeline():
    X_path = 'data/processed/X_train.npy'
    y_path = 'data/processed/y_train.npy'
    
    # Trigger data engineering step if binary matrices are missing from local directories
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print("Data artifacts not found. Starting preprocessing pipeline first...")
        from dataset_pipeline import FaceLivenessDataPipeline
        pipeline = FaceLivenessDataPipeline()
        pipeline.generate_simulation_dataset(num_samples=60)
        pipeline.process_and_vectorize_dataset()

    print("Loading engineered numpy data matrices into deep learning execution memory...")
    X_train = np.load(X_path)
    y_train = np.load(y_path)
    
    # Initialize your custom CNN model architecture
    model = build_liveness_cnn(input_shape=(128, 128, 3))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    print("Beginning custom CNN training loops across presentation attack datasets...")
    model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=16,
        validation_split=0.2
    )
    
    model.save('liveness_model.h5')
    print("Training phase finalized. Weights matrix exported as 'liveness_model.h5'.")

if __name__ == "__main__":
    execute_training_pipeline()
