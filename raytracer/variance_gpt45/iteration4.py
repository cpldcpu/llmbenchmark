import numpy as np
from PIL import Image

# Basic Vector math helpers
def normalize(v):
    return v / np.linalg.norm(v)

def reflected(ray, normal):
    return ray - 2 * np.dot(ray, normal) * normal

# Ray-sphere intersection calculation
def intersect_sphere(O, D, sphere):
    C, r = sphere['center'], sphere['radius']
    OC = O - C
    b = 2 * np.dot(OC, D)
    c = np.dot(OC, OC) - r**2
    discriminant = b**2 - 4*c
    if discriminant < 0:
        return np.inf
    sqdisc = np.sqrt(discriminant)
    t1 = (-b - sqdisc) / 2
    t2 = (-b + sqdisc) / 2
    return t1 if 0 < t1 < t2 else (t2 if (0 < t2 < t1) else np.inf)

# Scene configuration
scene = [
    {'center': np.array([0, 0, 3]), 'radius': 1, 'color': np.array([1, 0, 0]), 'reflection': 0.5}, # Red sphere
    {'center': np.array([2, 0, 4]), 'radius': 1, 'color': np.array([0, 1, 0]), 'reflection': 0.5}, # Green sphere
    {'center': np.array([-2, 0, 4]), 'radius': 1, 'color': np.array([0, 0, 1]), 'reflection': 0.5}, # Blue sphere
    {'center': np.array([0, -1001, 0]), 'radius': 1000, 'color': np.array([1, 1, 1]), 'reflection': 0.25}, # Ground plane
]

# Multiple colorful lightsources
lights = [
    {'position': np.array([5, 5, -5]), 'color': np.array([1, 0, 0])},  # Red light
    {'position': np.array([-5, 5, -5]), 'color': np.array([0, 1, 0])}, # Green light
    {'position': np.array([0, 5, 0]), 'color': np.array([0, 0, 1])},   # Blue light
    {'position': np.array([0, 0, -5]), 'color': np.array([1, 1, 0])},  # Yellow light
]

ambient = 0.05
image_width, image_height = 800, 600
fov = np.pi / 3  # Field of view

def ray_trace(O, D, depth=0):
    min_dist = np.inf
    hit_object = None
    for obj in scene:
        dist = intersect_sphere(O, D, obj)
        if dist < min_dist:
            min_dist, hit_object = dist, obj
    if hit_object is None:
        return np.array([0, 0, 0])  # Black background

    # Point of intersection and normal at that point
    M = O + D * min_dist
    N = normalize(M - hit_object['center'])

    col = ambient * hit_object['color']

    for light in lights:
        L = normalize(light['position'] - M)
        shadow = any(intersect_sphere(M + N * 1e-5, L, obj) < np.linalg.norm(light['position'] - M) for obj in scene)
        if not shadow:
            col += hit_object['color'] * np.clip(np.dot(N, L), 0, 1) * light['color']

    # Reflections
    if depth < 3:
        reflect_dir = normalize(reflected(D, N))
        reflect_col = ray_trace(M + N * 1e-5, reflect_dir, depth + 1)
        col = (1 - hit_object['reflection']) * col + hit_object['reflection'] * reflect_col

    return np.clip(col, 0, 1)

img = np.zeros((image_height, image_width, 3))

for i, y in enumerate(np.linspace(1, -1, image_height)):
    for j, x in enumerate(np.linspace(-image_width/image_height, image_width/image_height, image_width)):
        D = normalize(np.array([x * np.tan(fov/2), y * np.tan(fov/2), 1]))
        img[i, j] = ray_trace(np.array([0, 0, -1]), D)

# Save image
img = (255*np.clip(img,0,1)).astype(np.uint8)
Image.fromarray(img).save('raytraced_scene4.png')

print("Rendering complete. Saved as output.png")