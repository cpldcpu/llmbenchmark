import numpy as np

class Camera:
    def __init__(self, position, yaw=0.0, pitch=0.0):
        self.position = np.array(position, dtype=np.float32)
        self.yaw = yaw
        self.pitch = pitch

    def get_direction(self):
        # Convert yaw and pitch to a direction vector
        yaw_rad = np.radians(self.yaw)
        pitch_rad = np.radians(self.pitch)
        x = np.cos(pitch_rad) * np.cos(yaw_rad)
        y = np.sin(pitch_rad)
        z = np.cos(pitch_rad) * np.sin(yaw_rad)
        return np.array([x, y, z], dtype=np.float32)

class Scene:
    def __init__(self):
        self.camera = Camera([0, 1, 5])
        # Placeholder for objects and lights
        self.objects = []
        self.lights = []
