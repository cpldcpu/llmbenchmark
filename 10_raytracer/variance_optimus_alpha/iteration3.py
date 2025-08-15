import numpy as np
from PIL import Image

# Image dimensions
WIDTH = 800
HEIGHT = 600

# Camera
FOV = np.pi / 3  # 60 degrees

# Define Scene Objects
class Sphere:
    def __init__(self, center, radius, color, specular=50, reflective=0.5):
        self.center = np.array(center)
        self.radius = radius
        self.color = np.array(color)
        self.specular = specular
        self.reflective = reflective

# Lights
class Light:
    def __init__(self, position, color, intensity=1.0):
        self.position = np.array(position)
        self.color = np.array(color)
        self.intensity = intensity

def normalize(v):
    return v / np.linalg.norm(v)

def reflect(ray, normal):
    return ray - 2 * np.dot(ray, normal) * normal

def intersect_sphere(ray_origin, ray_dir, sphere):
    oc = ray_origin - sphere.center
    a = np.dot(ray_dir, ray_dir)
    b = 2 * np.dot(oc, ray_dir)
    c = np.dot(oc, oc) - sphere.radius*sphere.radius
    disc = b*b - 4*a*c
    if disc < 0:
        return np.inf, None
    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)
    t = min(t1, t2)
    if t < 0:
        t = max(t1, t2)
    if t < 0:
        return np.inf, None
    hit_point = ray_origin + ray_dir * t
    normal = normalize(hit_point - sphere.center)
    return t, normal

def trace_ray(ray_origin, ray_dir, scene, lights, depth=0):
    nearest_t = np.inf
    nearest_obj = None
    hit_normal = None
    for obj in scene:
        t, normal = intersect_sphere(ray_origin, ray_dir, obj)
        if t < nearest_t:
            nearest_t = t
            nearest_obj = obj
            hit_normal = normal
    if nearest_obj is None:
        return np.array([30, 30, 50])  # background color, dark blue

    hit_point = ray_origin + ray_dir * nearest_t
    local_color = np.zeros(3)

    # Ambient
    ambient = 0.15
    local_color += nearest_obj.color * ambient

    # Lighting
    for light in lights:
        to_light = light.position - hit_point
        dist_to_light = np.linalg.norm(to_light)
        to_light = normalize(to_light)
        
        # Shadow check
        shadow_orig = hit_point + hit_normal * 1e-5
        shadow_t, _ = intersect_sphere(shadow_orig, to_light, nearest_obj)
        shadow = False
        for obj in scene:
            t, _ = intersect_sphere(shadow_orig, to_light, obj)
            if t < dist_to_light and t > 0.01:
                shadow = True
                break

        if not shadow:
            # Diffuse
            diff = max(0, np.dot(hit_normal, to_light))
            local_color += nearest_obj.color * light.color * light.intensity * diff
            # Specular
            view_dir = normalize(-ray_dir)
            half_dir = normalize(to_light + view_dir)
            spec = max(0, np.dot(hit_normal, half_dir)) ** nearest_obj.specular
            local_color += light.color * light.intensity * spec

    # Reflection
    if depth < 2 and nearest_obj.reflective > 0:
        reflect_dir = reflect(ray_dir, hit_normal)
        reflect_color = trace_ray(hit_point + hit_normal * 1e-4, reflect_dir, scene, lights, depth + 1)
        local_color = local_color * (1 - nearest_obj.reflective) + reflect_color * nearest_obj.reflective

    return np.clip(local_color, 0, 255)

def render(scene, lights):
    aspect_ratio = WIDTH / HEIGHT
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    camera_origin = np.array([0, 0, -1])

    for y in range(HEIGHT):
        for x in range(WIDTH):
            # Screen space coordinates in [-1,1] with aspect ratio
            px = (2 * (x + 0.5) / WIDTH - 1) * np.tan(FOV / 2) * aspect_ratio
            py = (1 - 2 * (y + 0.5) / HEIGHT) * np.tan(FOV / 2)
            direction = normalize(np.array([px, py, 1]))
            color = trace_ray(camera_origin, direction, scene, lights)
            img[y, x] = np.clip(color, 0, 255)
    return img

if __name__ == '__main__':
    # Scene: colorful spheres
    scene = [
        Sphere([0, -1.1, 3], 1, [200, 30, 30], specular=100, reflective=0.25),  # Red
        Sphere([2, 0, 4], 1, [30, 200, 30], specular=50, reflective=0.4),       # Green
        Sphere([-2, 0, 4], 1, [30, 30, 200], specular=50, reflective=0.3),      # Blue
        Sphere([0, -5001, 0], 5000, [230, 230, 230], specular=10, reflective=0.1), # Ground
        Sphere([0.7, 1.30, 2.2], 0.4, [230, 200, 50], specular=100, reflective=0.3), # Top yellow
        Sphere([-1.1, 0.7, 2.5], 0.6, [130, 0, 130], specular=80, reflective=0.5), # purple
    ]

    lights = [
        Light([3, 5, -4], [1.0, 0.85, 0.7], 1.4),       # Warm key light
        Light([-1.5, 8, -6], [0.4, 0.5, 1.0], 1.0),     # Blue fill above
        Light([0, 0, -6], [0.3, 1.0, 0.4], 0.8),        # Green front light
        Light([0, 7, 7], [1.0, 0.21, 0.90], 1.2),       # Pink top light
    ]

    img = render(scene, lights)
    im = Image.fromarray(img, 'RGB')
    im.save('raytraced_scene3.png')
    print("Image saved to raytrace_colorful_lights.png")
