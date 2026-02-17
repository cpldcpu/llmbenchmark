import numpy as np
from PIL import Image

W, H = 800, 600
MAX_DEPTH = 4
AMBIENT = 0.05

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v

def reflect(d, n):
    return d - 2 * np.dot(d, n) * n

# Scene objects: spheres
# Each sphere: center, radius, color, shininess, reflectivity
spheres = [
    # Floor (large sphere below)
    {"c": np.array([0, -10004, 0], float), "r": 10000, "col": np.array([0.15, 0.15, 0.15]), "shin": 10, "refl": 0.3},
    # Back wall
    {"c": np.array([0, 0, -10020], float), "r": 10000, "col": np.array([0.08, 0.08, 0.08]), "shin": 5, "refl": 0.1},
    # Main spheres
    {"c": np.array([-3.5, 0, -10], float),  "r": 1.5, "col": np.array([0.9, 0.1, 0.1]), "shin": 120, "refl": 0.6},
    {"c": np.array([0, 0, -10], float),     "r": 1.5, "col": np.array([0.1, 0.9, 0.1]), "shin": 120, "refl": 0.6},
    {"c": np.array([3.5, 0, -10], float),   "r": 1.5, "col": np.array([0.1, 0.1, 0.9]), "shin": 120, "refl": 0.6},
    {"c": np.array([-1.75, -0.8, -7.5], float), "r": 0.7, "col": np.array([0.9, 0.9, 0.1]), "shin": 200, "refl": 0.8},
    {"c": np.array([1.75, -0.8, -7.5], float),  "r": 0.7, "col": np.array([0.9, 0.1, 0.9]), "shin": 200, "refl": 0.8},
    {"c": np.array([0, -0.8, -7.0], float),     "r": 0.7, "col": np.array([0.1, 0.9, 0.9]), "shin": 200, "refl": 0.8},
    # Small decorative spheres
    {"c": np.array([-5.5, -2.5, -9], float), "r": 0.5, "col": np.array([1.0, 0.5, 0.0]), "shin": 60, "refl": 0.4},
    {"c": np.array([5.5, -2.5, -9], float),  "r": 0.5, "col": np.array([0.5, 0.0, 1.0]), "shin": 60, "refl": 0.4},
    {"c": np.array([0, 3.5, -12], float),    "r": 1.0, "col": np.array([1.0, 0.8, 0.3]), "shin": 80, "refl": 0.5},
]

# Colored point lights: position, color, intensity
lights = [
    {"pos": np.array([-6, 8, -5], float),  "col": np.array([1.0, 0.2, 0.2]), "intensity": 120},
    {"pos": np.array([6, 8, -5], float),   "col": np.array([0.2, 0.4, 1.0]), "intensity": 120},
    {"pos": np.array([0, 10, -8], float),  "col": np.array([0.2, 1.0, 0.2]), "intensity": 100},
    {"pos": np.array([-4, 3, -6], float),  "col": np.array([1.0, 0.6, 0.0]), "intensity": 60},
    {"pos": np.array([4, 3, -6], float),   "col": np.array([0.8, 0.0, 1.0]), "intensity": 60},
    {"pos": np.array([0, 2, -5], float),   "col": np.array([0.0, 1.0, 1.0]), "intensity": 50},
    {"pos": np.array([0, 15, -12], float), "col": np.array([1.0, 1.0, 0.8]), "intensity": 200},
    {"pos": np.array([-8, 1, -12], float), "col": np.array([1.0, 0.3, 0.8]), "intensity": 70},
    {"pos": np.array([8, 1, -12], float),  "col": np.array([0.3, 1.0, 0.5]), "intensity": 70},
]

def intersect_sphere(ray_o, ray_d, sphere):
    oc = ray_o - sphere["c"]
    a = np.dot(ray_d, ray_d)
    b = 2 * np.dot(oc, ray_d)
    c = np.dot(oc, oc) - sphere["r"]**2
    disc = b*b - 4*a*c
    if disc < 0:
        return None
    sq = np.sqrt(disc)
    t1 = (-b - sq) / (2*a)
    t2 = (-b + sq) / (2*a)
    if t1 > 1e-4: return t1
    if t2 > 1e-4: return t2
    return None

def scene_intersect(ray_o, ray_d):
    t_min = np.inf
    hit_sphere = None
    for s in spheres:
        t = intersect_sphere(ray_o, ray_d, s)
        if t is not None and t < t_min:
            t_min = t
            hit_sphere = s
    return (t_min, hit_sphere) if hit_sphere else (None, None)

def shade(ray_o, ray_d, depth):
    t, sphere = scene_intersect(ray_o, ray_d)
    if sphere is None:
        # Sky gradient
        d = normalize(ray_d)
        tt = 0.5 * (d[1] + 1.0)
        return (1 - tt) * np.array([0.02, 0.02, 0.04]) + tt * np.array([0.05, 0.05, 0.15])

    hit = ray_o + t * ray_d
    normal = normalize(hit - sphere["c"])
    if np.dot(ray_d, normal) > 0:
        normal = -normal

    color = sphere["col"] * AMBIENT

    for light in lights:
        to_light = light["pos"] - hit
        dist2 = np.dot(to_light, to_light)
        to_light_n = to_light / np.sqrt(dist2)

        # Shadow check
        t_sh, _ = scene_intersect(hit + normal * 1e-3, to_light_n)
        if t_sh is not None and t_sh**2 < dist2:
            continue

        # Diffuse
        diff = max(np.dot(normal, to_light_n), 0)
        atten = light["intensity"] / dist2
        color += sphere["col"] * light["col"] * diff * atten

        # Specular
        half = normalize(to_light_n - normalize(ray_d))
        spec = max(np.dot(normal, half), 0) ** sphere["shin"]
        color += light["col"] * spec * atten * 0.5

    # Reflection
    if depth < MAX_DEPTH and sphere["refl"] > 0:
        refl_dir = reflect(normalize(ray_d), normal)
        refl_col = shade(hit + normal * 1e-3, refl_dir, depth + 1)
        color = color * (1 - sphere["refl"]) + refl_col * sphere["refl"]

    return np.clip(color, 0, 1)

# Camera setup
fov = np.radians(55)
aspect = W / H
cam_o = np.array([0, 1.0, 0], float)

print("Rendering...")
pixels = np.zeros((H, W, 3), float)

for j in range(H):
    if j % 50 == 0:
        print(f"  Row {j}/{H}")
    for i in range(W):
        x = (2 * (i + 0.5) / W - 1) * np.tan(fov/2) * aspect
        y = (1 - 2 * (j + 0.5) / H) * np.tan(fov/2)
        ray_d = normalize(np.array([x, y, -1], float))
        pixels[j, i] = shade(cam_o, ray_d, 0)

# Tone mapping: Reinhard
pixels = pixels / (pixels + 1)
# Gamma correction
pixels = np.power(pixels, 1/2.2)
img_data = (np.clip(pixels, 0, 1) * 255).astype(np.uint8)
img = Image.fromarray(img_data, 'RGB')
img.save("scene.png")
print("Done! Saved to scene.png")