import numpy as np
from PIL import Image

W, H = 800, 600
MAX_DEPTH = 4

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def reflected(d, n):
    return d - 2 * np.dot(d, n) * n

# Scene objects
class Sphere:
    def __init__(self, center, radius, color, specular=50, reflective=0.0, emission=None):
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)
        self.color = np.array(color, dtype=float)
        self.specular = specular
        self.reflective = reflective
        self.emission = np.array(emission, dtype=float) if emission else None

    def intersect(self, ro, rd):
        oc = ro - self.center
        b = np.dot(oc, rd)
        c = np.dot(oc, oc) - self.radius**2
        disc = b*b - c
        if disc < 0:
            return None
        sq = np.sqrt(disc)
        t1 = -b - sq
        t2 = -b + sq
        if t1 > 1e-4:
            return t1
        if t2 > 1e-4:
            return t2
        return None

    def normal_at(self, p):
        return normalize(p - self.center)

class Plane:
    def __init__(self, point, normal, color, specular=10, reflective=0.1, checker=False):
        self.point = np.array(point, dtype=float)
        self.normal = normalize(np.array(normal, dtype=float))
        self.color = np.array(color, dtype=float)
        self.specular = specular
        self.reflective = reflective
        self.checker = checker

    def intersect(self, ro, rd):
        denom = np.dot(self.normal, rd)
        if abs(denom) < 1e-6:
            return None
        t = np.dot(self.point - ro, self.normal) / denom
        return t if t > 1e-4 else None

    def normal_at(self, p):
        return self.normal

    def get_color(self, p):
        if self.checker:
            x, z = int(np.floor(p[0] / 1.5)), int(np.floor(p[2] / 1.5))
            if (x + z) % 2 == 0:
                return np.array([0.9, 0.9, 0.95])
            else:
                return np.array([0.15, 0.15, 0.2])
        return self.color

# Lights: position, color, intensity
lights = [
    {'pos': np.array([ 3.0, 5.0, -2.0]), 'color': np.array([1.0, 0.2, 0.1]), 'intensity': 1.8},
    {'pos': np.array([-4.0, 4.0, -1.0]), 'color': np.array([0.1, 0.4, 1.0]), 'intensity': 1.6},
    {'pos': np.array([ 0.0, 6.0,  3.0]), 'color': np.array([0.2, 1.0, 0.3]), 'intensity': 1.4},
    {'pos': np.array([-2.0, 2.0, -5.0]), 'color': np.array([1.0, 0.6, 0.0]), 'intensity': 1.2},
    {'pos': np.array([ 5.0, 3.0,  1.0]), 'color': np.array([0.8, 0.0, 1.0]), 'intensity': 1.0},
    {'pos': np.array([ 0.0, 8.0, -1.0]), 'color': np.array([1.0, 1.0, 0.8]), 'intensity': 0.6},
]

# Emissive spheres matching lights
objects = [
    # Floor
    Plane([0, -2, 0], [0, 1, 0], [0.7, 0.7, 0.7], specular=20, reflective=0.3, checker=True),
    # Back wall
    Plane([0, 0, -10], [0, 0, 1], [0.15, 0.12, 0.2], specular=5, reflective=0.05),

    # Main scene spheres
    Sphere([ 0.0, 0.0, -5.0], 1.2,  [1.0, 0.9, 0.85], specular=200, reflective=0.6),  # Chrome center
    Sphere([-2.5, -0.5,-4.5], 0.8,  [0.95, 0.3, 0.1], specular=80,  reflective=0.15),  # Red glass-ish
    Sphere([ 2.5, -0.5,-4.5], 0.8,  [0.2, 0.5, 1.0],  specular=150, reflective=0.4),   # Blue mirror
    Sphere([ 0.0, -1.0,-3.5], 0.6,  [0.3, 1.0, 0.4],  specular=100, reflective=0.2),   # Green
    Sphere([-1.2,  0.8,-4.0], 0.5,  [1.0, 0.8, 0.1],  specular=60,  reflective=0.1),   # Gold
    Sphere([ 1.2,  1.0,-4.2], 0.55, [0.8, 0.1, 0.9],  specular=90,  reflective=0.25),  # Purple

    # Small accent spheres
    Sphere([-3.5, -1.3,-5.5], 0.5,  [0.1, 0.9, 0.9],  specular=120, reflective=0.35),
    Sphere([ 3.5, -1.3,-5.5], 0.5,  [1.0, 0.4, 0.7],  specular=120, reflective=0.35),
    Sphere([-0.8, -1.5,-3.2], 0.35, [1.0, 1.0, 0.2],  specular=200, reflective=0.5),
    Sphere([ 0.9, -1.5,-3.0], 0.3,  [0.2, 0.8, 1.0],  specular=200, reflective=0.5),

    # Emissive light spheres (small glowing bulbs)
    Sphere([ 3.0, 5.0, -2.0], 0.22, [1.0, 0.3, 0.2], emission=[3.0, 0.6, 0.3]),
    Sphere([-4.0, 4.0, -1.0], 0.22, [0.3, 0.6, 1.0], emission=[0.3, 0.8, 3.0]),
    Sphere([ 0.0, 6.0,  3.0], 0.22, [0.3, 1.0, 0.4], emission=[0.6, 3.0, 0.9]),
    Sphere([-2.0, 2.0, -5.0], 0.18, [1.0, 0.7, 0.1], emission=[3.0, 1.8, 0.3]),
    Sphere([ 5.0, 3.0,  1.0], 0.18, [0.9, 0.2, 1.0], emission=[2.4, 0.3, 3.0]),
]

AMBIENT = 0.05

def get_color(obj, p):
    if isinstance(obj, Plane):
        return obj.get_color(p)
    return obj.color

def cast_ray(ro, rd, depth=0):
    if depth > MAX_DEPTH:
        return np.zeros(3)

    nearest_t = np.inf
    nearest_obj = None
    for obj in objects:
        t = obj.intersect(ro, rd)
        if t and t < nearest_t:
            nearest_t = t
            nearest_obj = obj

    if nearest_obj is None:
        # Sky gradient
        t = 0.5 * (rd[1] + 1.0)
        sky = (1 - t) * np.array([0.05, 0.05, 0.1]) + t * np.array([0.02, 0.02, 0.08])
        return sky

    p = ro + nearest_t * rd
    n = nearest_obj.normal_at(p)
    obj_color = get_color(nearest_obj, p)

    # Emissive?
    if hasattr(nearest_obj, 'emission') and nearest_obj.emission is not None:
        return nearest_obj.emission

    color = obj_color * AMBIENT

    for light in lights:
        lpos = light['pos']
        lcolor = light['color'] * light['intensity']
        ldir = lpos - p
        ldist = np.linalg.norm(ldir)
        ldir = ldir / ldist

        # Shadow
        shadow = False
        for obj in objects:
            if obj is nearest_obj:
                continue
            t = obj.intersect(p + n * 1e-4, ldir)
            if t and t < ldist:
                shadow = True
                break

        if not shadow:
            # Diffuse
            diff = max(np.dot(n, ldir), 0)
            color += obj_color * lcolor * diff / (1 + 0.04 * ldist)

            # Specular
            if nearest_obj.specular > 0:
                rv = normalize(reflected(-ldir, n))
                spec = max(np.dot(rv, -rd), 0) ** nearest_obj.specular
                color += lcolor * spec * 0.6 / (1 + 0.04 * ldist)

    # Reflection
    if nearest_obj.reflective > 0 and depth < MAX_DEPTH:
        refl_dir = normalize(reflected(rd, n))
        refl_color = cast_ray(p + n * 1e-4, refl_dir, depth + 1)
        color = color * (1 - nearest_obj.reflective) + refl_color * nearest_obj.reflective

    return np.clip(color, 0, 1)

# Camera
cam_pos = np.array([0.0, 1.5, 3.0])
look_at  = np.array([0.0, 0.0, -4.5])
up       = np.array([0.0, 1.0, 0.0])
fov      = 60.0

fwd  = normalize(look_at - cam_pos)
right = normalize(np.cross(fwd, up))
up2  = np.cross(right, fwd)
half_h = np.tan(np.radians(fov / 2))
half_w = half_h * W / H

print("Rendering...")
img = np.zeros((H, W, 3))

for j in range(H):
    if j % 50 == 0:
        print(f"  Row {j}/{H}")
    for i in range(W):
        # 2x2 supersampling
        color = np.zeros(3)
        for si in range(2):
            for sj in range(2):
                u = (i + (si + 0.5) / 2) / W
                v = (j + (sj + 0.5) / 2) / H
                x =  (2 * u - 1) * half_w
                y = -(2 * v - 1) * half_h
                rd = normalize(x * right + y * up2 + fwd)
                color += cast_ray(cam_pos, rd)
        img[j, i] = color / 4.0

# Tone mapping (Reinhard) + gamma
img = img / (img + 1.0)
img = np.power(img, 1.0 / 2.2)
img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

out = Image.fromarray(img)
out.save('render.png')
print("Saved to render.png")
