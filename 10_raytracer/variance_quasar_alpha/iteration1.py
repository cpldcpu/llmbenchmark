import numpy as np
from PIL import Image

# Image size
WIDTH, HEIGHT = 800, 600

# Camera setup
fov = np.pi / 3  # 60 degrees
camera_pos = np.array([0, 0, -1])

# Scene objects: list of spheres defined by center, radius, surface color, and specular coefficient
spheres = [
    {'center': np.array([0.0, -1.0, 3.0]), 'radius': 1,   'color': np.array([255, 0, 0]), 'specular': 500},
    {'center': np.array([2.0, 0.0, 4.0]),  'radius': 1,   'color': np.array([0, 0, 255]), 'specular': 500},
    {'center': np.array([-2.0, 0.0, 4.0]), 'radius': 1,   'color': np.array([0, 255, 0]), 'specular': 10},
    {'center': np.array([0.0, -5001, 0]),  'radius': 5000,'color': np.array([255, 255, 0]),'specular': 1000},
]

# Light sources: each has position, intensity, and color
lights = [
    {'position': np.array([5, 5, -10]), 'color': np.array([255, 100, 100]), 'intensity': 1.5},
    {'position': np.array([-5, 5, -10]), 'color': np.array([100, 255, 100]), 'intensity': 1.5},
    {'position': np.array([0, 20, -20]), 'color': np.array([100, 100, 255]), 'intensity': 2.0},
]

# Background color
background_color = np.array([30,30,30])

def normalize(v):
    return v / np.linalg.norm(v)

def reflect(I, N):
    return I - 2 * np.dot(I, N) * N

def intersect_sphere(origin, direction, sphere):
    CO = origin - sphere['center']
    a = np.dot(direction, direction)
    b = 2 * np.dot(CO, direction)
    c = np.dot(CO, CO) - sphere['radius'] **2
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return np.inf
    sqrtdisc = np.sqrt(discriminant)
    t1 = (-b + sqrtdisc) / (2 * a)
    t2 = (-b - sqrtdisc) / (2 * a)
    t_min = min(t1, t2)
    t_max = max(t1, t2)
    if t_min > 0:
        return t_min
    elif t_max > 0:
        return t_max
    else:
        return np.inf

def trace_ray(origin, direction, depth=0):
    nearest_t = np.inf
    nearest_sphere = None
    for sphere in spheres:
        t = intersect_sphere(origin, direction, sphere)
        if t < nearest_t:
            nearest_t = t
            nearest_sphere = sphere
    if nearest_sphere is None:
        return background_color
    point = origin + nearest_t * direction
    normal = normalize(point - nearest_sphere['center'])
    view = -direction

    color = np.zeros(3)
    for light in lights:
        to_light = light['position'] - point
        distance_to_light = np.linalg.norm(to_light)
        to_light_dir = normalize(to_light)

        # Shadow check:
        shadow_t = np.inf
        for sphere in spheres:
            if sphere is nearest_sphere:
                continue
            t = intersect_sphere(point + normal * 1e-5, to_light_dir, sphere)
            if t < distance_to_light:
                shadow_t = t
                break
        if shadow_t < np.inf:
            continue  # in shadow, no light contribution

        # Diffuse Shading
        diffuse_intensity = max(0, np.dot(normal, to_light_dir)) * light['intensity']
        # Specular Shading
        reflect_dir = reflect(-to_light_dir, normal)
        specular_intensity = pow(max(0, np.dot(view, reflect_dir)), nearest_sphere['specular']) * light['intensity']
        color += nearest_sphere['color'] * diffuse_intensity * light['color'] / 255 / 255
        color += specular_intensity * light['color']
    # Clamp color
    color = np.clip(color, 0, 255)
    return color

# Rendering loop
img = Image.new('RGB', (WIDTH, HEIGHT))
pixels = img.load()

for y in range(HEIGHT):
    for x in range(WIDTH):
        # Normalized screen coordinates
        Px = (2 * (x + 0.5) / WIDTH - 1) * np.tan(fov/2) * WIDTH / HEIGHT
        Py = (1 - 2 * (y + 0.5) / HEIGHT) * np.tan(fov/2)
        direction = normalize(np.array([Px, Py, 1]))
        color = trace_ray(camera_pos, direction)
        pixels[x, y] = tuple(color.astype(np.uint8))
    if y % 50 == 0:
        print(f"Rendered {y} / {HEIGHT}")

img.save('raytraced_scene1.png')
print("Saved raytrace.png")
