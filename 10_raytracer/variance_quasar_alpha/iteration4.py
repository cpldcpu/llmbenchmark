import numpy as np
from PIL import Image

WIDTH = 800
HEIGHT = 600
FOV = np.pi/3
MAX_DEPTH = 3

BACKGROUND_COLOR = np.array([0.05, 0.05, 0.1])

class Sphere:
    def __init__(self, center, radius, color, specular=50, reflective=0.2):
        self.center = np.array(center)
        self.radius = radius
        self.color = np.array(color)
        self.specular = specular
        self.reflective = reflective

class Light:
    def __init__(self, position, color, intensity=1.0):
        self.position = np.array(position)
        self.color = np.array(color)
        self.intensity = intensity

def normalize(v):
    return v / np.linalg.norm(v)

def intersect_sphere(origin, direction, sphere):
    oc = origin - sphere.center
    b = 2 * np.dot(oc, direction)
    c = np.dot(oc, oc) - sphere.radius**2
    disc = b**2 - 4 * c
    if disc < 0:
        return np.inf, None
    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / 2
    t2 = (-b + sqrt_disc) / 2
    t = min(t1, t2)
    if t < 0:
        t = max(t1, t2)
    if t < 0:
        return np.inf, None
    hit_point = origin + t * direction
    hit_normal = normalize(hit_point - sphere.center)
    return t, hit_normal

def trace_ray(origin, direction, spheres, lights, depth=0):
    color = BACKGROUND_COLOR
    min_t = np.inf
    hit_sphere = None
    hit_normal = None

    for sphere in spheres:
        t, n = intersect_sphere(origin, direction, sphere)
        if t < min_t:
            min_t = t
            hit_sphere = sphere
            hit_normal = n
    
    if hit_sphere is None:
        return color
    
    hit_point = origin + direction * min_t
    surface_color = hit_sphere.color * 0.1  # Ambient

    for light in lights:
        light_dir = normalize(light.position - hit_point)
        # Shadow test
        shadow_orig = hit_point + 1e-5 * hit_normal
        shadow_t, _ = intersect_scene(shadow_orig, light_dir, spheres)
        light_dist = np.linalg.norm(light.position - hit_point)
        if shadow_t < light_dist:
            continue  # In shadow
        
        intensity = light.intensity / (light_dist**2)
        diff = max(np.dot(hit_normal, light_dir), 0)
        diffuse = diff * hit_sphere.color * light.color * intensity

        view_dir = normalize(-direction)
        half_dir = normalize(light_dir + view_dir)
        spec = max(np.dot(hit_normal, half_dir), 0) ** hit_sphere.specular
        specular = spec * light.color * intensity

        surface_color += diffuse + specular

    surface_color = np.clip(surface_color, 0, 1)

    # Reflection
    if depth < MAX_DEPTH and hit_sphere.reflective > 0:
        reflect_dir = normalize(direction - 2 * np.dot(direction, hit_normal) * hit_normal)
        reflect_orig = hit_point + 1e-5 * hit_normal
        reflect_color = trace_ray(reflect_orig, reflect_dir, spheres, lights, depth + 1)
        surface_color = (1 - hit_sphere.reflective) * surface_color + hit_sphere.reflective * reflect_color

    return np.clip(surface_color, 0, 1)

def intersect_scene(origin, direction, spheres):
    min_t = np.inf
    n = None
    for sphere in spheres:
        t, normal = intersect_sphere(origin, direction, sphere)
        if t < min_t:
            min_t = t
            n = normal
    return min_t, n

def main():
    aspect_ratio = WIDTH / HEIGHT
    camera_pos = np.array([0, 0, -1])
    
    spheres = [
        Sphere([0, -10004, 20], 10000, [0.2, 0.2, 0.2], specular=100, reflective=0.3),  # Ground
        Sphere([0, 0, 20], 4, [1, 0, 0], specular=100, reflective=0.5),
        Sphere([-5, -1, 15], 2, [0, 1, 0], specular=50, reflective=0.3),
        Sphere([5, 0, 25], 3, [0, 0, 1], specular=500, reflective=0.1),
        Sphere([-3, 2, 18], 2, [1,1,0], specular=100, reflective=0.4),
    ]

    lights = [
        Light([10, 20, -10], [1, 0, 0], intensity=3000),
        Light([-20, 20, -10], [0, 1, 0], intensity=3000),
        Light([0, 50, 0], [1, 1, 1], intensity=5000),
        Light([0, 5, 10], [0, 0, 1], intensity=1000),
    ]

    image = np.zeros((HEIGHT, WIDTH, 3))

    for y in range(HEIGHT):
        for x in range(WIDTH):
            # Camera space coordinates
            px = (2 * (x + 0.5) / WIDTH - 1) * np.tan(FOV/2) * aspect_ratio
            py = (1 - 2 * (y + 0.5) / HEIGHT) * np.tan(FOV/2)
            direction = normalize(np.array([px, py, 1]))
            color = trace_ray(camera_pos, direction, spheres, lights)
            image[y, x] = np.clip(color, 0, 1)
        if y % 50 == 0:
            print(f"Rendered {y} lines...")

    img = Image.fromarray((image * 255).astype('uint8'))
    img.save('raytraced_scene4.png')
    print("Saved raytraced_scene.png")

if __name__ == '__main__':
    main()
