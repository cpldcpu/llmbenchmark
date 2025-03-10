import numpy as np
from PIL import Image

# Scene settings
WIDTH, HEIGHT = 800, 600
MAX_DEPTH = 3

# Define colors
BLACK = np.array([0, 0, 0])
WHITE = np.array([1, 1, 1])

# Sphere class
class Sphere:
    def __init__(self, center, radius, color, reflection=0.5):
        self.center = np.array(center)
        self.radius = radius
        self.color = np.array(color)
        self.reflection = reflection

    def intersect(self, ray_origin, ray_direction):
        oc = ray_origin - self.center
        a = np.dot(ray_direction, ray_direction)
        b = 2.0 * np.dot(oc, ray_direction)
        c = np.dot(oc, oc) - self.radius ** 2
        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            return np.inf
        else:
            sqrt_disc = np.sqrt(discriminant)
            t1 = (-b - sqrt_disc) / (2 * a)
            t2 = (-b + sqrt_disc) / (2 * a)
            if t1 > 1e-4:
                return t1
            if t2 > 1e-4:
                return t2
        return np.inf

# Define multiple colorful lightsources
class Light:
    def __init__(self, position, color, intensity):
        self.position = np.array(position)
        self.color = np.array(color)
        self.intensity = intensity

# Scene setup
spheres = [
    Sphere([0.0, -0.5, 3], 0.5, [1.0, 0.2, 0.2], 0.5),  # red sphere
    Sphere([1.0, 0.0, 4], 0.6, [0.2, 1.0, 0.2], 0.6),   # green sphere
    Sphere([-1.2, 0.0, 4.5], 0.8, [0.2, 0.2, 1.0], 0.7), # Blue sphere
    Sphere([0.0, -9999, 0], 9998.5, [0.8, 0.8, 0.8], 0.2), # Gray Floor
]

lights = [
    Light([0, 5, 2], [1, 0.5, 0.5], 1.2),  # reddish-pink overhead
    Light([-4, 3, 1], [0.2, 0.2, 1.0], 0.8), # blue from left side
    Light([4, 2, 3], [0.2, 1.0, 0.2], 0.8), # green from right side
]

# Ray tracing method to compute color from intersection
def trace_ray(ray_origin, ray_direction, depth=0):
    if depth >= MAX_DEPTH:
        return BLACK

    closest_t = np.inf
    closest_sphere = None
    for sphere in spheres:
        t = sphere.intersect(ray_origin, ray_direction)
        if t < closest_t:
            closest_t = t
            closest_sphere = sphere

    if closest_sphere is None:
        return BLACK

    hit_pos = ray_origin + closest_t * ray_direction
    normal = (hit_pos - closest_sphere.center)
    normal /= np.linalg.norm(normal)

    shift_pos = hit_pos + 1e-5 * normal

    # Ambient term
    color = 0.05 * closest_sphere.color

    # Diffuse shading
    for light in lights:
        to_light = light.position - shift_pos
        distance_light = np.linalg.norm(to_light)
        to_light /= distance_light

        # Shadow check
        shadow = False
        for sphere in spheres:
            shadow_t = sphere.intersect(shift_pos, to_light)
            if shadow_t < distance_light:
                shadow = True
                break

        if not shadow:
            # Diffuse shading (Lambertian)
            diffuse_intensity = max(np.dot(normal, to_light), 0) * light.intensity
            color += closest_sphere.color * diffuse_intensity * light.color

    # Reflection
    reflected_direction = ray_direction - 2 * np.dot(ray_direction, normal) * normal
    reflected_color = trace_ray(shift_pos, reflected_direction, depth + 1)
    color = color * (1 - closest_sphere.reflection) + reflected_color * closest_sphere.reflection

    return np.clip(color, 0, 1)

# Render loop
def render():
    aspect_ratio = WIDTH / HEIGHT
    img_data = np.zeros((HEIGHT, WIDTH, 3))

    camera = np.array([0, 0, -1])

    for y in range(HEIGHT):
        for x in range(WIDTH):
            # Normalized device coordinates
            px = (2 * (x + 0.5) / WIDTH - 1) * aspect_ratio
            py = 1 - 2 * (y + 0.5) / HEIGHT
            ray_direction = np.array([px, py, 1])
            ray_direction /= np.linalg.norm(ray_direction)
            color = trace_ray(camera, ray_direction)
            img_data[y, x, :] = color

        print(f"Rendering row {y+1}/{HEIGHT}")

    img = (np.clip(img_data, 0, 1) * 255).astype(np.uint8)
    image = Image.fromarray(img, "RGB")
    image.save("raytracer_scene3.png")
    print("Image saved as 'raytracer_output.png'")

if __name__ == "__main__":
    render()