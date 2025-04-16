import numpy as np
from PIL import Image

# Basic vector functions
def norm(v): return v / np.linalg.norm(v)
def dot(a, b): return np.dot(a, b)
def reflect(I, N): return I - 2 * dot(I, N) * N

# Sphere class
class Sphere:
    def __init__(self, center, radius, color, specular=40, reflective=0.1):
        self.center = np.array(center)
        self.radius = radius
        self.color = np.array(color)
        self.specular = specular
        self.reflective = reflective

    def intersect(self, ray_origin, ray_dir):
        OC = ray_origin - self.center
        a = dot(ray_dir, ray_dir)
        b = 2 * dot(OC, ray_dir)
        c = dot(OC, OC) - self.radius ** 2
        D = b ** 2 - 4 * a * c
        if D < 0:
            return None
        sqrtD = np.sqrt(D)
        t0 = (-b - sqrtD) / (2 * a)
        t1 = (-b + sqrtD) / (2 * a)
        if t0 > 1e-3:
            return t0
        elif t1 > 1e-3:
            return t1
        else:
            return None

# Light class
class Light:
    def __init__(self, pos, color, intensity):
        self.pos = np.array(pos)
        self.color = np.array(color)
        self.intensity = intensity

# Scene
WIDTH = 800
HEIGHT = 600
FOV = np.pi / 3.0
MAX_DEPTH = 2
BACKGROUND_COLOR = np.array([12, 14, 16])

spheres = [
    Sphere([0, -1, 5],        1, [180, 20, 20], 100, 0.6),
    Sphere([2,    0, 7],      1, [30, 144, 255], 500, 0.3), # blue
    Sphere([-2,   0, 7],      1, [34, 177, 76], 10, 0.2),   # green
    Sphere([0,   -501, 8], 500, [255, 234, 175], 1000, 0.0), # big floor
    Sphere([0.5, 0.7, 3.5], 0.3, [250, 0, 250], 80, 0.8),    # magenta
    Sphere([-1.3, 0.3, 3.3], 0.3, [250, 150, 0], 80, 0.8), # yellow/orange
    Sphere([1.5, 0.5, 4], 0.4, [0,255,255], 10, 0.8), # cyan
]

lights = [
    Light([0, 6, 4],     [255, 200, 200], 0.7),
    Light([2, 4, 3],     [200, 200, 255], 0.65),
    Light([-2, 4, 9],    [150, 255, 200], 0.4),
    Light([0.5, 0.5, 2], [255, 255, 120], 0.7),
    Light([3, 1, 12],    [255, 90, 130], 0.4),
    Light([-4, 3, 8],    [70, 250, 255], 0.6),
    Light([0, 8, 16],    [255, 220, 255], 0.4)
]

def trace_ray(origin, direction, depth=0):
    nearest_t = float('inf')
    hit_sphere = None
    for obj in spheres:
        t = obj.intersect(origin, direction)
        if t is not None and t < nearest_t:
            nearest_t = t
            hit_sphere = obj
    if hit_sphere is None:
        return BACKGROUND_COLOR

    # local color
    hit_point = origin + nearest_t * direction
    normal = norm(hit_point - hit_sphere.center)
    local_color = np.zeros(3)
    view = -direction

    # Lighting: ambient, diffuse, specular
    for light in lights:
        to_light = norm(light.pos - hit_point)
        # Shadow check
        shadow = False
        for obj in spheres:
            if obj is not hit_sphere:
                t = obj.intersect(hit_point + normal * 1e-4, to_light)
                if t is not None and t < np.linalg.norm(light.pos - hit_point):
                    shadow = True
                    break
        if shadow:
            continue

        # Diffuse
        diff_int = max(dot(to_light, normal), 0)
        diffuse = diff_int * light.intensity * (light.color/255) * (hit_sphere.color/255)

        # Specular
        half_vec = norm(to_light + view)
        spec_int = max(dot(normal, half_vec), 0) ** hit_sphere.specular
        specular_col = spec_int * light.intensity * (light.color/255)

        local_color += diffuse + specular_col

    local_color = np.clip(local_color * 255, 0, 255)

    # Reflection
    if hit_sphere.reflective > 0 and depth < MAX_DEPTH:
        R = reflect(direction, normal)
        reflected_color = trace_ray(hit_point + normal * 1e-3, R, depth+1)
        return local_color * (1-hit_sphere.reflective) + reflected_color * hit_sphere.reflective

    return local_color

# Main render loop
img = Image.new("RGB", (WIDTH, HEIGHT))
pixels = img.load()

aspect_ratio = WIDTH / HEIGHT
for y in range(HEIGHT):
    for x in range(WIDTH):
        # Convert pixel coordinate to ray from camera
        px = (2 * (x + 0.5) / WIDTH - 1) * np.tan(FOV / 2) * aspect_ratio
        py = (1 - 2 * (y + 0.5) / HEIGHT) * np.tan(FOV / 2)
        direction = norm(np.array([px, py, 1]))
        color = trace_ray(np.array([0, 0, 0]), direction)
        pixels[x, y] = tuple(np.clip(color, 0, 255).astype(np.uint8))
    print(f"\rRendering: {100*y//HEIGHT}%", end="", flush=True)

print("\nDone. Saving image.")
img.save("raytraced_scene4.png")
