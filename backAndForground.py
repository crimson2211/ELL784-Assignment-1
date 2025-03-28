import numpy as np
import cv2
from google.colab.patches import cv2_imshow
import os

def gaussian_pdf(value, mean, std_dev):
    """Compute Gaussian probability density function."""
    return (1 / (np.sqrt(2 * np.pi) * std_dev)) * np.exp(-0.5 * ((value - mean) / std_dev) ** 2)

def normalize_gaussian_weights(weights):
    """Ensure Gaussian weights sum to 1."""
    return weights / np.sum(weights, axis=0)

def reorder_gaussian_components(array, indices):
    """Rearrange Gaussian components based on given indices."""
    return np.take_along_axis(array, indices, axis=0)

class GMMBackgroundSubtractor:
    def __init__(self, video_source, learning_rate, threshold, output_path):
        self.video_source = video_source
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.output_path = output_path

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Read the first frame
        _, frame = self.video_source.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.height, self.width = frame_gray.shape

        # Initialize GMM parameters
        self.means = np.zeros((3, self.height, self.width), dtype=np.float64)
        self.means[1] = frame_gray

        self.variances = np.full((3, self.height, self.width), 5.0)
        self.weights = np.array([0.0, 0.0, 1.0]).reshape(3, 1, 1) * np.ones((3, self.height, self.width))
        self.background_model = np.zeros((self.height, self.width), dtype=np.uint8)

        self.std_devs = np.sqrt(self.variances)
        self.scaled_std_devs = 2.5 * self.std_devs

        # Initialize video writers
        self.init_video_writers(frame.shape)

    def init_video_writers(self, frame_shape):
        """Initialize video writers for saving output videos."""
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.original_writer = cv2.VideoWriter(
            os.path.join(self.output_path, 'original_output.avi'), fourcc, 30.0,
            (frame_shape[1], frame_shape[0]), True
        )
        self.background_writer = cv2.VideoWriter(
            os.path.join(self.output_path, 'background_output.avi'), fourcc, 30.0,
            (self.width, self.height), False
        )
        self.foreground_writer = cv2.VideoWriter(
            os.path.join(self.output_path, 'foreground_output.avi'), fourcc, 30.0,
            (self.width, self.height), False
        )

    def process_video(self):
        """Process the video frame-by-frame."""
        while self.video_source.isOpened():
            ret, frame = self.video_source.read()
            if not ret:
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            diff = np.abs(frame_gray - self.means)
            weight_ratio = self.weights / self.std_devs

            matched = diff <= self.scaled_std_devs
            unmatched = ~matched

            high_weight_bg = self.weights[2] > self.threshold
            med_weight_bg = (self.weights[1] + self.weights[2] > self.threshold) & ~high_weight_bg

            strong_bg = np.logical_and(high_weight_bg, matched[2])
            weak_bg = np.logical_and(med_weight_bg, (matched[1] | matched[2]))

            for i in range(3):
                update_factor = self.learning_rate * gaussian_pdf(
                    frame_gray[matched[i]], self.means[i][matched[i]], self.std_devs[i][matched[i]]
                )
                var_update = update_factor * (frame_gray[matched[i]] - self.means[i][matched[i]]) ** 2

                self.means[i][matched[i]] = (1 - update_factor) * self.means[i][matched[i]] + update_factor * frame_gray[matched[i]]
                self.variances[i][matched[i]] = (1 - update_factor) * self.variances[i][matched[i]] + var_update
                self.weights[i][matched[i]] = (1 - self.learning_rate) * self.weights[i][matched[i]] + self.learning_rate
                self.weights[i][unmatched[i]] *= (1 - self.learning_rate)

            new_bg_pixels = ~np.any(matched, axis=0)
            self.means[0][new_bg_pixels] = frame_gray[new_bg_pixels]
            self.variances[0][new_bg_pixels] = 500
            self.weights[0][new_bg_pixels] = 0.1

            self.weights = normalize_gaussian_weights(self.weights)

            sorted_indices = np.argsort(weight_ratio, axis=0)
            self.means = reorder_gaussian_components(self.means, sorted_indices)
            self.variances = reorder_gaussian_components(self.variances, sorted_indices)
            self.weights = reorder_gaussian_components(self.weights, sorted_indices)

            self.background_model[weak_bg] = frame_gray[weak_bg]
            self.background_model[strong_bg] = frame_gray[strong_bg]

            foreground_mask = self.extract_foreground(frame_gray)
            self.save_video_frames(frame, self.background_model, foreground_mask)
            self.display_results(frame_gray, self.background_model, foreground_mask)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.video_source.release()
        self.original_writer.release()
        self.background_writer.release()
        self.foreground_writer.release()
        cv2.destroyAllWindows()

    def extract_foreground(self, frame):
        """Perform background subtraction and post-process the foreground mask."""
        frame = frame.astype(np.uint8)  # Ensure frame is uint8
        self.background_model = self.background_model.astype(np.uint8)  # Ensure background model is uint8

        foreground = cv2.absdiff(frame, self.background_model)
        _, mask = cv2.threshold(foreground, 50, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=1)

        return mask

    def save_video_frames(self, original_frame, background_frame, foreground_mask):
        """Save frames to output videos."""
        self.original_writer.write(original_frame)
        self.background_writer.write(background_frame)
        self.foreground_writer.write(foreground_mask)

    def display_results(self, original, background, foreground):
        """Display video processing frames."""
        cv2_imshow(original)
        cv2_imshow(background)
        cv2_imshow(foreground)

if __name__ == '__main__':
    video = cv2.VideoCapture("/content/umcp.mpg")
    segmenter = GMMBackgroundSubtractor(video, learning_rate=0.01, threshold=0.75, output_path="/content/output")
    segmenter.process_video()
