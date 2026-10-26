import os
import numpy as np
import cv2

class FaceLivenessDataPipeline:
    def __init__(self, data_dir="data", img_size=(128, 128)):
        self.data_dir = data_dir
        self.img_size = img_size
        self.raw_real_dir = os.path.join(data_dir, "raw", "real")
        self.raw_spoof_dir = os.path.join(data_dir, "raw", "spoof")
        self.processed_dir = os.path.join(data_dir, "processed")
        
        # Ensure structural project directory layout exists
        os.makedirs(self.raw_real_dir, exist_ok=True)
        os.makedirs(self.raw_spoof_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def generate_simulation_dataset(self, num_samples=100):
        """
        Populates raw project folders with mock image structures to verify 
        the validation integrity of the pipeline engine locally.
        """
        print(f"Generating localized mock image templates for validation arrays...")
        for i in range(num_samples):
            # Simulate real faces (higher pixel variation in color channels)
            real_img = np.random.randint(50, 255, size=(256, 256, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(self.raw_real_dir, f"real_face_{i:04d}.jpg"), real_img)
            
            # Simulate spoof attacks (lower illumination gradients typical of 2D prints)
            spoof_img = np.random.randint(10, 180, size=(256, 256, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(self.raw_spoof_dir, f"spoof_attack_{i:04d}.jpg"), spoof_img)
        print("Mock baseline collection directories successfully created.")

    def process_and_vectorize_dataset(self):
        """
        Ingests image structures, resizes facial matrices to custom target dimensions, 
        and applies float normalization for optimal CNN layer convergence.
        """
        X = []
        y = []
        
        # Class Configurations: 1 = Real Face (Bona Fide), 0 = Spoof Attack (PAD)
        categories = [(self.raw_real_dir, 1), (self.raw_spoof_dir, 0)]
        
        print("Executing spatial resizing and pixel preprocessing routines...")
        for path, label in categories:
            for file_name in os.listdir(path):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(path, file_name)
                    
                    # Read image with OpenCV
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    # Resize to input layer shape matching model.py configuration (128x128)
                    img_resized = cv2.resize(img, self.img_size)
                    
                    # Transform matrix into spatial color array
                    X.append(img_resized)
                    y.append(label)
        
        # Convert arrays to numpy arrays and perform pixel matrix normalization [0, 1]
        X_array = np.array(X, dtype=np.float32) / 255.0
        y_array = np.array(y, dtype=np.float32).reshape(-1, 1)
        
        # Serialize vectors to disk to accelerate model compilation loops
        np.save(os.path.join(self.processed_dir, "X_train.npy"), X_array)
        np.save(os.path.join(self.processed_dir, "y_train.npy"), y_array)
        
        print(f"\n--- Data Layer Serialization Successful ---")
        print(f"Processed Feature Matrix Shape: {X_array.shape} (Images, Width, Height, Channels)")
        print(f"Processed Target Labels Shape: {y_array.shape} (Binary Classification Vectors)")
        print(f"Saved binary arrays into: {self.processed_dir}")

if __name__ == "__main__":
    pipeline = FaceLivenessDataPipeline()
    # 1. Prepare raw visual directories
    pipeline.generate_simulation_dataset(num_samples=50)
    # 2. Run data engineering transformations
    pipeline.process_and_vectorize_dataset()
