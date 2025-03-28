import numpy as np
import cv2
import os
import shutil
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# Step 1: Extract frames efficiently with resizing
def extract_frames(video_path, output_folder="frames", target_size=(128, 128)):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, target_size)  # Resize to target size
        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        frame_vector = frame_gray.flatten()
        frames.append(frame_vector)
        cv2.imwrite(os.path.join(output_folder, f"frame_{frame_count}.jpg"), frame_gray)
        frame_count += 1

    cap.release()
    return np.array(frames, dtype=np.float32), frame_count

# Step 2: Incremental PCA from Scratch
class IncrementalPCA:
    def __init__(self, n_components, batch_size=100):
        self.n_components = n_components
        self.batch_size = batch_size
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None

    def fit(self, X):
        n_samples, n_features = X.shape
        self.mean_ = np.zeros(n_features)
        self.components_ = np.zeros((self.n_components, n_features))
        self.explained_variance_ = np.zeros(self.n_components)

        # Process data in batches
        for i in tqdm(range(0, n_samples, self.batch_size), desc="Fitting PCA"):
            batch = X[i:i + self.batch_size]
            batch_mean = np.mean(batch, axis=0)
            batch_centered = batch - batch_mean

            # Update mean incrementally
            self.mean_ = (i * self.mean_ + batch_mean * len(batch)) / (i + len(batch))

            # Update covariance matrix incrementally
            if i == 0:
                cov_matrix = np.dot(batch_centered.T, batch_centered) / len(batch)
            else:
                cov_matrix = (i * cov_matrix + np.dot(batch_centered.T, batch_centered)) / (i + len(batch))

        # Perform eigenvalue decomposition on the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        idx = np.argsort(eigenvalues)[::-1]  # Sort eigenvalues in descending order
        self.components_ = eigenvectors[:, idx[:self.n_components]].T
        self.explained_variance_ = eigenvalues[idx[:self.n_components]]

    def transform(self, X):
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

# Step 3: Select key frames based on PCA variance
def select_key_frames(transformed_data, num_key_frames):
    frame_variances = np.var(transformed_data, axis=1)
    key_frame_indices = np.argsort(frame_variances)[-num_key_frames:]
    return sorted(key_frame_indices)

# Step 4: Generate summarized video
def create_summary_video(frames_folder, key_frame_indices, output_video="summary.avi", fps=5):
    key_frames = [cv2.imread(os.path.join(frames_folder, f"frame_{i}.jpg")) for i in key_frame_indices]
    height, width = key_frames[0].shape[:2]
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    for frame in key_frames:
        out.write(frame)

    out.release()

# Step 5: Evaluate the summary using SSIM
def evaluate_summary(frames_folder, key_frame_indices):
    reference_frame = cv2.imread(os.path.join(frames_folder, "frame_0.jpg"), cv2.IMREAD_GRAYSCALE)
    similarities = []

    for i in key_frame_indices:
        current_frame = cv2.imread(os.path.join(frames_folder, f"frame_{i}.jpg"), cv2.IMREAD_GRAYSCALE)
        score, _ = ssim(reference_frame, current_frame, full=True)
        similarities.append(score)

    avg_ssim = np.mean(similarities)
    print(f"Average SSIM Score: {avg_ssim:.4f}")

    plt.plot(key_frame_indices, similarities, marker='o')
    plt.xlabel("Key Frame Index")
    plt.ylabel("SSIM Score")
    plt.title("SSIM Scores of Key Frames")
    plt.show()

# Step 6: Main Execution
video_path = "/content/umcp.mpg"  # Change this to your video path
frames_data, total_frames = extract_frames(video_path)
print("Total Frames used", total_frames)

num_components = min(50, total_frames)
ipca = IncrementalPCA(n_components=num_components, batch_size=100)
ipca.fit(frames_data)
transformed_data = ipca.transform(frames_data)

num_key_frames = min(999, total_frames)
key_frame_indices = select_key_frames(transformed_data, num_key_frames)
print(f"Selected key frames: {key_frame_indices}")

create_summary_video("frames", key_frame_indices)
evaluate_summary("frames", key_frame_indices)

# Cleanup extracted frames
shutil.rmtree("frames")
