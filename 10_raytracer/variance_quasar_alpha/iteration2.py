import numpy as np
from PIL import Image

WIDTH, HEIGHT = 800, 600
MAX_DEPTH = 3
BACKGROUND_COLOR = np.array([0.05, 0.05, 0.05])

class Sphere:
    def __init__(self, center, radius, color, specular=50):
        self.center = center
        self.radius = radius
        self.color = color
        self.specular = specular

    def intersect(self, orig, dir):
        OC = orig - self.center
        b = 2 * np.dot(dir, OC)
        c = np.dot(OC, OC) - self.radius**2
        disc = b * b - 4 * c
        if disc < 0:
            return np.inf, None
        sqrt_disc = np.sqrt(disc)
        t1 = (-b - sqrt_disc) / 2
        t2 = (-b + sqrt_disc) / 2
        if t1 > 0.001:
            return t1, orig + dir * t1
        if t2 > 0.001:
            return t2, orig + dir * t2
        return np.inf, None

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

# Scene setup
objects = [
    Sphere(np.array([0, -1, 3]), 1, np.array([0.1, 0.9, 0.5])),
    Sphere(np.array([2, 0, 4]), 1, np.array([0.9, 0.2, 0.3])),
    Sphere(np.array([-2, 0, 4]), 1, np.array([0.2, 0.3, 0.9])),
    Sphere(np.array([0, -5001, 0]), 5000, np.array([0.8, 0.8, 0.0])),  # ground plane
]

light_sources = [
    {"pos": np.array([5, 5, -10]), "color": np.array([1, 0, 0])},
    {"pos": np.array([-5, 5, -10]), "color": np.array([0, 1, 0])},
    {"pos": np.array([0, 20, -5]), "color": np.array([0.5, 0.5, 1])},
]

def trace(orig, dir, depth):
    nearest_t = np.inf
    nearest_obj = None
    nearest_pt = None

    for obj in objects:
        t, pt = obj.intersect(orig, dir)
        if t < nearest_t:
            nearest_t = t
            nearest_obj = obj
            nearest_pt = pt

    if nearest_obj is None:
        return BACKGROUND_COLOR

    normal = normalize(nearest_pt - nearest_obj.center)
    view = -dir
    color = np.zeros(3)

    # Ambient term
    color += 0.1 * nearest_obj.color

    # Lights
    for light in light_sources:
        to_light = light["pos"] - nearest_pt
        light_dist = np.linalg.norm(to_light)
        to_light_dir = normalize(to_light)

        # Shadow check
        shadow_orig = nearest_pt + normal * 1e-5
        shadow_t, _ = closest_intersection(shadow_orig, to_light_dir, light_dist)
        if shadow_t < light_dist:
            continue  # in shadow

        # Diffuse
        diff_intensity = max(0, np.dot(normal, to_light_dir))
        color += diff_intensity * nearest_obj.color * light["color"]

        # Specular
        reflect_dir = normalize(2 * normal * np.dot(normal, to_light_dir) - to_light_dir)
        spec_intensity = max(0, np.dot(reflect_dir, view)) ** (nearest_obj.specular)
        color += spec_intensity * light["color"]

    # Clamp colors
    return np.clip(color, 0, 1)

def closest_intersection(orig, dir, max_dist=np.inf):
    nearest_t = np.inf
    nearest_pt = None
    for obj in objects:
        t, pt = obj.intersect(orig, dir)
        if 1e-5 < t < nearest_t and t < max_dist:
            nearest_t = t
            nearest_pt = pt
    return nearest_t, nearest_pt

def render():
    aspect_ratio = WIDTH / HEIGHT
    fov = np.pi / 3
    img = np.zeros((HEIGHT, WIDTH, 3))

    for y in range(HEIGHT):
        for x in range(WIDTH):
            # Normalized device coords
            px = (2 * (x + 0.5)/WIDTH -1) * np.tan(fov/2) * aspect_ratio
            py = -(2 * (y + 0.5)/HEIGHT -1) * np.tan(fov/2)
            dir = normalize(np.array([px, py, 1]))
            color = trace(np.array([0, 0, 0]), dir, 0)
            img[y,x] = color

    # Gamma correction
    img = np.clip(img, 0, 1)**(1/2.2)
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save("raytraced_scene2.png")

if __name__ == "__main__":
    render()
    print("Saved raytrace.png")
