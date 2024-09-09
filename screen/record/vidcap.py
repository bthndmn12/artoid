import pygame
import cv2
import numpy as np
import os
import time

class VidCap:
    def __init__(self, filename, resolution=(800, 600), fps=30, file_type="avi"):
        self.filename = filename
        self.resolution = resolution
        self.fps = fps
        self.file_type = file_type
        self.codec_dict = {
            "avi": 'DIVX',
            "mp4": 'MP4V'
        }
        self.codec = cv2.VideoWriter_fourcc(*self.codec_dict[self.file_type])
        self.video = cv2.VideoWriter(self.filename, self.codec, self.fps, self.resolution)
        self.recording = False
        self.start_time = None
        self.frames = []
        self.dts = []

    def start(self):
        self.recording = True
        self.start_time = time.time()

    def stop(self):
        self.recording = False
        self.save()

    def record(self, screen):
        if self.recording:
            frame = pygame.surfarray.array3d(screen)
            frame = np.rot90(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.video.write(frame)
            self.frames.append(frame)
            self.dts.append(time.time() - self.start_time)

    def save(self):
        self.video.release()
        self.print_recording_info()

    def print_recording_info(self):
        frame_num = len(self.frames)
        dt_sum = sum(self.dts)
        average_dt = dt_sum / frame_num
        memory_usage_approx = frame_num * self.resolution[0] * self.resolution[1] * 3  # Assuming 3 bytes per pixel (RGB)

        print("Total time:", dt_sum, "s")
        print("Average time per frame:", average_dt, "s")
        print("Number of frames:", frame_num)
        print("Memory usage approximation:", memory_usage_approx / 1000, "KB")
        print("Video saved to:", self.filename)

    def get_frame(self):
        return self.frames[-1] if self.frames else None

    def get_fps(self):
        return self.fps

    def set_fps(self, fps):
        self.fps = fps

    def get_resolution(self):
        return self.resolution

# # Example usage
# if __name__ == "__main__":
#     pygame.init()
#     screen = pygame.display.set_mode((800, 600))
#     clock = pygame.time.Clock()
#     vidcap = VidCap("./output.mp4", resolution=(800,600), fps=30)

#     running = True
#     vidcap.start()

#     while running:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False
#             if event.type == pygame.KEYDOWN:
#                 if event.key == pygame.K_ESCAPE:
#                     running = False

#         screen.fill((0, 0, 0))
#         pygame.draw.circle(screen, (255, 0, 0), (400, 300), 50)
#         pygame.display.flip()

#         vidcap.record(screen)
#         clock.tick(30)

#     vidcap.stop()
#     pygame.quit()