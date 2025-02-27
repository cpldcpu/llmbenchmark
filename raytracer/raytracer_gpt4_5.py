import numpy as np
from PIL import Image

WIDTH, HEIGHT = 800, 600
MAX_DEPTH = 3

# Define a sphere class
class Sphere:
    def __init__(self, center, radius, color, emissive=0):
        self.center = np.array(center)
        self.radius = radius
        self.color = np.array(color)
        self.emissive = emissive

    def intersect(self, ray_origin, ray_direction):
        oc = ray_origin - self.center
        a = np.dot(ray_direction, ray_direction)
        b = 2.0 * np.dot(oc, ray_direction)
        c = np.dot(oc, oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return None
        sqrtd = np.sqrt(discriminant)
        t0 = (-b - sqrtd) / (2*a)
        t1 = (-b + sqrtd) / (2*a)
        tmin = min(t0, t1)
        if tmin < 0:
            tmin = max(t0, t1)
        if tmin < 0:
            return None
        hit_point = ray_origin + ray_direction * tmin
        normal = (hit_point - self.center) / self.radius
        return hit_point, normal

# Raytracing function
def trace(ray_origin, ray_direction, objects, depth=MAX_DEPTH):
    nearest_obj = None
    min_dist = np.inf
    intersection_data = None
    for obj in objects:
        hit = obj.intersect(ray_origin, ray_direction)
        if hit:
            point, normal = hit
            dist = np.linalg.norm(point - ray_origin)
            if dist < min_dist:
                min_dist = dist
                nearest_obj = obj
                intersection_data = (point, normal)

    if nearest_obj is None:
        return np.array([0, 0, 0])  # Background color

    hit_point, normal = intersection_data
    color = nearest_obj.emissive * nearest_obj.color
    if depth <=0 or nearest_obj.emissive > 0:
        return color

    # Ambient color
    ambient = np.array([0.05, 0.05, 0.05])
    color += nearest_obj.color * ambient

    # Light contribution
    for obj in objects:
        if obj.emissive > 0:
            to_light = obj.center - hit_point
            distance_light = np.linalg.norm(to_light)
            to_light = to_light / distance_light
            shadow_hit = False
            for other in objects:
                if other is nearest_obj or other is obj:
                    continue
                shadow_intersection = other.intersect(hit_point + 1e-4*normal, to_light)
                if shadow_intersection:
                    shadow_hit = True
                    break
            if not shadow_hit:
                brightness = max(np.dot(normal, to_light), 0)
                contribution = obj.emissive * brightness * obj.color
                color += nearest_obj.color * contribution

    color = np.clip(color, 0, 1)
    return color

# Scene setup
objects = [
    Sphere([0, -10004, -20], 10000, [0.2, 0.2, 0.2]),  # Ground plane
    Sphere([0, 0, -20], 4, [1, 0.32, 0.36]),
    Sphere([5, -1, -15], 2, [0.9, 0.76, 0.46]),
    Sphere([5, 0, -25], 3, [0.65, 0.77, 0.97]),
    Sphere([-5.5, 0, -15], 3, [0.9, 0.9, 0.9]),  # Mirror-like object

    # Emissive spheres as colorful light sources
    Sphere([0, 20, -30], 3, [1, 0, 0], emissive=3),  # Red light
    Sphere([-10, 10, -20], 2, [0, 1, 0], emissive=3),  # Green light
    Sphere([10, 10, -20], 2, [0, 0, 1], emissive=3),  # Blue light
    Sphere([0, -2, -15], 1, [1, 1, 0], emissive=4),   # Yellow bright sphere
]

# Render loop
aspect_ratio = WIDTH / HEIGHT
scene_img = Image.new("RGB", (WIDTH, HEIGHT))
pixels = scene_img.load()

fov = np.pi / 3  # field of view
for y in range(HEIGHT):
    for x in range(WIDTH):
        xx = (2 * (x + 0.5) / WIDTH - 1) * np.tan(fov / 2) * aspect_ratio
        yy = (1 - 2 * (y + 0.5) / HEIGHT) * np.tan(fov / 2)
        ray_direction = np.array([xx, yy, -1])
        ray_direction /= np.linalg.norm(ray_direction)
        color = trace(np.array([0, 0, 0]), ray_direction, objects)
        pixels[x, y] = tuple((np.clip(color, 0, 1) * 255).astype(np.uint8))

    print(f"Row {y+1}/{HEIGHT} completed")

scene_img.save('colorful_scene.png')
print("Rendering done! Saved as 'colorful_scene.png'")