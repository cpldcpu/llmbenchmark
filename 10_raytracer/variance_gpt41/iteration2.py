import numpy as np
from PIL import Image

WIDTH = 800
HEIGHT = 600

# Basic vector operations
def normalize(v):
    return v / np.linalg.norm(v)

def reflect(I, N):
    return I - 2 * np.dot(I, N) * N

# Ray, Sphere, and Plane classes
class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

class Sphere:
    def __init__(self, center, radius, color, shininess=64):
        self.center = np.array(center, dtype=np.float32)
        self.radius = radius
        self.color = np.array(color, dtype=np.float32)
        self.shininess = shininess

    def intersect(self, ray):
        oc = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius*self.radius
        discr = b*b - 4*a*c
        if discr < 0:
            return None
        sqrt_discr = np.sqrt(discr)
        t1 = (-b - sqrt_discr) / (2*a)
        t2 = (-b + sqrt_discr) / (2*a)
        t = min(t1, t2)
        if t < 0:
            t = max(t1, t2)
        if t < 0:
            return None
        hit_point = ray.origin + ray.direction * t
        normal = normalize(hit_point - self.center)
        return t, hit_point, normal, self.color, self.shininess

class Plane:
    def __init__(self, point, normal, color, shininess=1):
        self.point = np.array(point, dtype=np.float32)
        self.normal = normalize(np.array(normal, dtype=np.float32))
        self.color = np.array(color, dtype=np.float32)
        self.shininess = shininess

    def intersect(self, ray):
        denom = np.dot(self.normal, ray.direction)
        if abs(denom) < 1e-6:
            return None
        t = np.dot(self.point - ray.origin, self.normal) / denom
        if t < 0:
            return None
        hit_point = ray.origin + ray.direction * t
        return t, hit_point, self.normal, self.color, self.shininess

# Scene setup
# Spheres of various colors
spheres = [
    Sphere([0, 1, 8], 2, [1, 0.5, 0.5], shininess=128),           # Pink
    Sphere([-3.5, 0.5, 6], 1.5, [0.4, 1, 0.4], shininess=24),     # Green
    Sphere([3.5, 0.8, 6], 1.8, [0.4, 0.7, 1], shininess=64),      # Blue
    Sphere([0, -9001, 8], 9000, [1, 1, 1], shininess=8),          # Floor (giant sphere)
]

# Plane for wall
walls = [
    Plane([0, 0, 18], [0, 0, -1], [0.8, 0.8, 0.8], shininess=4),  # Rear wall
]

objects = spheres + walls

# Many colored lights
lights = [
    # Y, X, Z, Color
    {'pos': np.array([-6, 7, 0]), 'color': np.array([3, 0.3, 0.1])},
    {'pos': np.array([6, 6, 5]), 'color': np.array([0.2, 1.8, 1.4])},
    {'pos': np.array([0, 9, 10]), 'color': np.array([1.9, 2, 0.05])},
    {'pos': np.array([-4, 4, 12]), 'color': np.array([0.2, 0.8, 2])},
    {'pos': np.array([4, 3, 13]), 'color': np.array([2, 0.3, 1.6])},
]

background = np.array([0.05, 0.08, 0.14])

# Camera
fov = np.pi/2
camera_origin = np.array([0, 1, -4])

def trace_ray(ray, depth=0):
    min_t = np.inf
    hit_info = None
    obj_hit = None

    for obj in objects:
        result = obj.intersect(ray)
        if result is not None:
            t, hit_point, normal, color, shininess = result
            if t < min_t:
                min_t = t
                hit_info = (hit_point, normal, color, shininess)
                obj_hit = obj

    if hit_info is None:
        return background
    hit_point, normal, color, shininess = hit_info

    # Ambient
    result_color = 0.15 * color

    for light in lights:
        to_light = light['pos'] - hit_point
        to_light_dir = normalize(to_light)
        dist_to_light = np.linalg.norm(to_light)

        # Shadow: shoot a ray to the light, check if any object blocks it
        shadow_ray = Ray(hit_point + normal * 0.001, to_light_dir)
        shadow = False
        for obj in objects:
            result = obj.intersect(shadow_ray)
            if result is not None:
                t_shadow, _, _, _, _ = result
                if 0 < t_shadow < dist_to_light:
                    shadow = True
                    break
        if shadow:
            continue

        # Diffuse
        diff_intensity = np.dot(normal, to_light_dir)
        if diff_intensity > 0:
            result_color += color * light['color'] * diff_intensity * 0.26

        # Specular (Blinn-Phong)
        view_dir = normalize(camera_origin - hit_point)
        half_vec = normalize(to_light_dir + view_dir)
        spec_intensity = np.dot(normal, half_vec)
        if spec_intensity > 0:
            result_color += light['color'] * (spec_intensity ** shininess) * 0.15

    # Clamp
    return np.clip(result_color, 0, 1)

# Render loop
image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
aspect_ratio = WIDTH / HEIGHT

for y in range(HEIGHT):
    for x in range(WIDTH):
        u = (2*(x + 0.5) / WIDTH - 1) * aspect_ratio * np.tan(fov/2)
        v = (1 - 2*(y + 0.5) / HEIGHT) * np.tan(fov/2)
        dir = normalize(np.array([u, v, 1]))
        ray = Ray(camera_origin, dir)
        color = trace_ray(ray)
        image[y, x] = color

# Save image
img8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
im = Image.fromarray(img8, "RGB")
im.save("raytraced_scene2.png")
print("Rendered saved to colorful_lights_raytraced.png")
