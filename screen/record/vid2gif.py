import cv2
from PIL import Image

class VideoToGifConverter:
    def __init__(self, video_path, gif_path, frame_duration=50, frame_rate_reduction=2, resize_factor=0.2):
        self.video_path = video_path
        self.gif_path = gif_path
        self.frame_duration = frame_duration
        self.frame_rate_reduction = frame_rate_reduction
        self.resize_factor = resize_factor

    def convert_mp4_to_gif(self):
        video_capture = cv2.VideoCapture(self.video_path)
        frames = []
        frame_count = 0

        still_reading, image = video_capture.read()
        while still_reading:
            if frame_count % self.frame_rate_reduction == 0:
                # Convert the image from BGR to RGB (OpenCV uses BGR by default)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Resize the image
                new_size = (int(image_rgb.shape[1] * self.resize_factor), int(image_rgb.shape[0] * self.resize_factor))
                resized_image = cv2.resize(image_rgb, new_size)
                # Convert the image to a PIL Image
                pil_image = Image.fromarray(resized_image)
                frames.append(pil_image)

            # Read the next frame
            still_reading, image = video_capture.read()
            frame_count += 1

        if frames:
            # Save the frames as a GIF
            frames[0].save(self.gif_path, format="GIF", append_images=frames[1:],
                           save_all=True, duration=self.frame_duration, loop=0)
            print(f"GIF saved to {self.gif_path}")
        else:
            print("No frames were captured.")

# Example usage
if __name__ == "__main__":
    converter = VideoToGifConverter("D:/playground/egzersiz/salakprojects/artoid/ar_physics/planets.mp4", "output_gif.gif", frame_rate_reduction=3, resize_factor=0.9)
    converter.convert_mp4_to_gif()
