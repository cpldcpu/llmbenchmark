import numpy as np
from PIL import Image

WIDTH, HEIGHT = 800, 600

# Scene objects
class Sphere:
    def __init__(self, center, radius, color, specular=50, reflective=0.2):
        self.center = np.array(center)
        self.radius = radius
        self.color = np.array(color)
        self.specular = specular
        self.reflective = reflective

    def intersect(self, origin, direction):
        OC = origin - self.center
        b = 2 * np.dot(OC, direction)
        c = np.dot(OC, OC) - self.radius ** 2
        discriminant = b * b - 4 * c
        if discriminant < 0:
            return np.inf, None
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / 2
        t2 = (-b + sqrt_disc) / 2
        t = min(t1, t2)
        if t < 0:
            t = max(t1, t2)
        if t < 0:
            return np.inf, None
        hit_point = origin + direction * t
        normal = (hit_point - self.center) / self.radius
        return t, normal

# Lights
class Light:
    def __init__(self, position, color, intensity=1.0):
        self.position = np.array(position)
        self.color = np.array(color)
        self.intensity = intensity

# Ray tracing function
def trace_ray(origin, direction, spheres, lights, depth=0):
    color = np.zeros(3)
    nearest_t = np.inf
    nearest_sphere = None
    for sphere in spheres:
        t, normal = sphere.intersect(origin, direction)
        if t < nearest_t:
            nearest_t, hit_normal = t, normal
            nearest_sphere = sphere
    if nearest_sphere is None:
        return np.array([0, 0, 0])  # background
    hit_point = origin + direction * nearest_t
    base_color = nearest_sphere.color
    to_cam = -direction
    # Lighting calculation
    local_color = np.zeros(3)
    for light in lights:
        to_light = light.position - hit_point
        dist_to_light = np.linalg.norm(to_light)
        to_light_dir = to_light / dist_to_light

        # Shadow check
        shadow = False
        for sphere in spheres:
            if sphere == nearest_sphere:
                continue
            t_shadow, _ = sphere.intersect(hit_point + 1e-4 * hit_normal, to_light_dir)
            if t_shadow < dist_to_light:
                shadow = True
                break
        if shadow:
            continue

        # Diffuse
        diff_intensity = max(0, np.dot(hit_normal, to_light_dir)) * light.intensity
        diffuse = diff_intensity * light.color * base_color / 255

        # Specular
        reflect_dir = 2 * hit_normal * np.dot(hit_normal, to_light_dir) - to_light_dir
        spec_intensity = max(0, np.dot(reflect_dir, to_cam)) ** nearest_sphere.specular * light.intensity
        specular = spec_intensity * light.color

        local_color += diffuse + specular

    # Reflection
    if depth < 2 and nearest_sphere.reflective > 0:
        reflect_dir = direction - 2 * np.dot(direction, hit_normal) * hit_normal
        reflect_color = trace_ray(hit_point + 1e-4 * hit_normal, reflect_dir, spheres, lights, depth + 1)
        local_color = (1 - nearest_sphere.reflective) * local_color + nearest_sphere.reflective * reflect_color

    return np.clip(local_color, 0, 1)

def render():
    camera = np.array([0, 0, -1])
    screen = (-1, 1, 1, -1)  # left, top, right, bottom
    image = Image.new("RGB", (WIDTH, HEIGHT))
    pixels = image.load()

    spheres = [
        Sphere([0, -0.5, 3], 0.5, [255, 0, 0], reflective=0.3),
        Sphere([-1, 0, 4], 1, [0, 255, 0], reflective=0.3),
        Sphere([1, 0, 4], 1, [0, 0, 255], reflective=0.3),
        Sphere([0, -5001, 0], 5000, [255, 255, 0], reflective=0.2),  # ground
    ]

    lights = [
        Light([2, 5, -3], [255, 0, 0], 0.8),
        Light([-2, 5, -3], [0, 255, 0], 0.8),
        Light([0, 5, -3], [0, 0, 255], 0.8),
        Light([0, 5, 5], [255, 255, 255], 0.3)
    ]

    for i in range(WIDTH):
        for j in range(HEIGHT):
            # Normalize pixel coordinates to screen space
            x = screen[0] + (screen[2] - screen[0]) * (i + 0.5) / WIDTH
            y = screen[1] + (screen[3] - screen[1]) * (j + 0.5) / HEIGHT
            pixel_dir = np.array([x, y, 1])
            pixel_dir = pixel_dir / np.linalg.norm(pixel_dir)

            color = trace_ray(camera, pixel_dir, spheres, lights)
            r, g, b = (np.clip(color * 255, 0, 255)).astype(np.uint8)
            pixels[i, j] = (r, g, b)

    image.save("raytraced_scene3.png")

if __name__ == "__main__":
    render()
