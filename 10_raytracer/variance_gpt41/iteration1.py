import numpy as np
from PIL import Image

# Image dimensions
width = 800
height = 600

# Camera
camera_pos = np.array([0, 1, -5])
viewport_height = 2
viewport_width = viewport_height * (width / height)
focal_length = 1

# Scene definition
spheres = [
    {"center": np.array([0, 1, 3]), "radius": 1, "color": np.array([1, 1, 1])*0.8, "specular": 500, "reflective": 0.7},
    {"center": np.array([-1.8, 1, 4]), "radius": 1, "color": np.array([1, 0, 0]), "specular": 100, "reflective": 0.3},
    {"center": np.array([2.2, 1, 5]), "radius": 1, "color": np.array([0, 0, 1]), "specular": 100, "reflective": 0.5},
    {"center": np.array([0.2, 0.3, 2]), "radius": 0.3, "color": np.array([1, 1, 0]), "specular": 200, "reflective": 0.7}
]
plane = {"point": np.array([0, 0, 0]), "normal": np.array([0, 1, 0]), "color": np.array([0.5, 0.7, 0.5]), "specular": 10, "reflective": 0.2}

lights = [
    {"pos": np.array([3, 6, -1]), "color": np.array([0.9, 0.8, 0.4]), "intensity": 0.8},
    {"pos": np.array([-8, 9, 5]), "color": np.array([0.1, 0.8, 1.0]), "intensity": 1.5},
    {"pos": np.array([0, 9, 2]), "color": np.array([1.0, 0.3, 0.7]), "intensity": 1.2}
]

background_color = np.array([0.05, 0.1, 0.16])
MAX_DEPTH = 3
EPSILON = 1e-4

def normalize(v):
    return v / np.linalg.norm(v)

def intersect_sphere(ray_origin, ray_dir, sphere):
    oc = ray_origin - sphere["center"]
    a = np.dot(ray_dir, ray_dir)
    b = 2.0 * np.dot(oc, ray_dir)
    c = np.dot(oc, oc) - sphere["radius"] ** 2
    discriminant = b*b - 4*a*c
    if discriminant < 0: return None
    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2*a)
    t2 = (-b + sqrt_disc) / (2*a)
    t = min(t1, t2)
    if t < EPSILON:
        t = max(t1, t2)
        if t < EPSILON:
            return None
    hit = ray_origin + t*ray_dir
    normal = normalize(hit - sphere["center"])
    return (t, hit, normal, sphere)

def intersect_plane(ray_origin, ray_dir, plane):
    denom = np.dot(plane["normal"], ray_dir)
    if abs(denom) < 1e-6: return None
    t = np.dot(plane["point"] - ray_origin, plane["normal"]) / denom
    if t < EPSILON: return None
    hit = ray_origin + t*ray_dir
    normal = plane["normal"]
    return (t, hit, normal, plane)

def closest_intersection(ray_origin, ray_dir):
    min_t, hit_data = float("inf"), None
    # Spheres
    for s in spheres:
        result = intersect_sphere(ray_origin, ray_dir, s)
        if result is not None and result[0] < min_t:
            min_t, hit_data = result[0], result
    # Plane
    result = intersect_plane(ray_origin, ray_dir, plane)
    if result is not None and result[0] < min_t:
        min_t, hit_data = result[0], result
    return hit_data

def reflect(ray_dir, normal):
    return ray_dir - 2 * np.dot(ray_dir, normal) * normal

def compute_lighting(hit_point, normal, view_dir, specular):
    color = np.zeros(3)
    for l in lights:
        light_dir = l["pos"] - hit_point
        dist_to_light = np.linalg.norm(light_dir)
        light_dir = normalize(light_dir)
        # Shadow check
        shadow_origin = hit_point + normal * EPSILON
        shadow_result = closest_intersection(shadow_origin, light_dir)
        if shadow_result is not None:
            shadow_t = shadow_result[0]
            if shadow_t < dist_to_light - EPSILON:
                continue
        intensity = l["intensity"] / (dist_to_light**2)
        color += l["color"] * intensity * max(0, np.dot(light_dir, normal))
        # Specular
        if specular > 0:
            R = reflect(-light_dir, normal)
            spec = max(0, np.dot(R, view_dir)) ** (specular/4)
            color += l["color"] * intensity * spec
    return np.clip(color, 0, 6)

def trace_ray(ray_origin, ray_dir, depth):
    hit = closest_intersection(ray_origin, ray_dir)
    if hit is None:
        return background_color
    _, hit_point, normal, obj = hit
    local_color = obj["color"] * compute_lighting(hit_point, normal, -ray_dir, obj["specular"])
    # Reflection
    r = obj.get("reflective", 0)
    if r > 0 and depth > 0:
        reflect_dir = normalize(reflect(ray_dir, normal))
        reflect_origin = hit_point + normal * EPSILON
        reflected_color = trace_ray(reflect_origin, reflect_dir, depth-1)
        local_color = (1 - r) * local_color + r * reflected_color
    return np.clip(local_color, 0, 1)

# Main render loop
img = np.zeros((height, width, 3), dtype=np.uint8)
for y in range(height):
    sy = 1 - 2 * (y + 0.5) / height
    for x in range(width):
        sx = 2 * (x + 0.5) / width - 1
        dir = normalize(np.array([sx * viewport_width/2, sy * viewport_height/2, focal_length]))
        color = trace_ray(camera_pos, dir, MAX_DEPTH)
        img[y, x] = np.clip(color*255, 0, 255).astype(np.uint8)
    if y % 50 == 0:
        print(f"Rendering line {y}/{height}")

Image.fromarray(img, 'RGB').save("raytraced_scene1.png")
print("Saved as output.png")
