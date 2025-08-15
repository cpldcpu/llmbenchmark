import numpy as np
from PIL import Image

# Image settings
WIDTH, HEIGHT = 800, 600
CAMERA_POS = np.array([0, 1, -5])
FOV = np.pi / 3

# Scene definition

class Sphere:
    def __init__(self, center, radius, color, spec=50, refl=0.5):
        self.center = np.array(center)
        self.radius = radius
        self.color = np.array(color)
        self.spec = spec
        self.refl = refl

    def intersect(self, orig, dir):
        # |o + td - c|^2 = r^2
        L = self.center - orig
        tca = np.dot(L, dir)
        d2 = np.dot(L, L) - tca * tca
        if d2 > self.radius ** 2:
            return None
        thc = np.sqrt(self.radius ** 2 - d2)
        t0 = tca - thc
        t1 = tca + thc
        if t0 < 1e-3:
            t0 = t1
        if t0 < 1e-3:
            return None
        hit = orig + dir * t0
        normal = (hit - self.center) / self.radius
        return t0, hit, normal

class Plane:
    def __init__(self, p, n, color, spec=10, refl=0.2):
        self.p = np.array(p)
        self.n = np.array(n) / np.linalg.norm(n)
        self.color = np.array(color)
        self.spec = spec
        self.refl = refl

    def intersect(self, orig, dir):
        denom = np.dot(self.n, dir)
        if abs(denom) < 1e-6:
            return None
        d = np.dot(self.p - orig, self.n) / denom
        if d < 1e-3:
            return None
        hit = orig + dir * d
        normal = self.n
        return d, hit, normal

class Light:
    def __init__(self, pos, color, intensity):
        self.pos = np.array(pos)
        self.color = np.array(color)
        self.intensity = intensity

# Build scene: Spheres, Plane, Lots of Colorful Lights
objects = [
    Plane([0, -1, 0], [0, 1, 0], [0.95, 0.95, 1.0], refl=0.0, spec=5),
    Sphere([0, 0, 4], 1, [0.2, 0.7, 0.8], spec=120, refl=0.5),
    Sphere([-2, 0, 5], 1, [0.9, 0.4, 0.4], spec=60, refl=0.3),
    Sphere([2, 0, 5.5], 1, [0.5, 0.9, 0.5], spec=60, refl=0.4),
    Sphere([0.75, -0.2, 2.25], 0.3, [1, 0.8, 0.3], spec=40, refl=0.2),
]

np.random.seed(42)
lights = [
    Light([np.cos(a)*5, 5+np.sin(a*2)*2, 3+np.sin(a)*5],
          [abs(np.cos(a)), abs(np.sin(a*3)), abs(np.cos(a*2))],
          2.5)
    for a in np.linspace(0, 2*np.pi, 12, endpoint=False)
]
# Add some fixed strong colorful lights
lights += [
    Light([2, 5, 3], [1, 0, 0], 5),
    Light([-3, 6, 8], [0, 0, 1], 8),
    Light([0, 8, 6], [1, 0.6, 0.1], 7)
]

MAX_DEPTH = 4

def normalize(v):
    return v/np.linalg.norm(v)

def trace(orig, dir, depth):
    color = np.zeros(3)
    reflection = 1.0
    for _ in range(MAX_DEPTH):
        nearest_t = 1e9
        nearest_obj = None
        for obj in objects:
            res = obj.intersect(orig, dir)
            if res and res[0] < nearest_t:
                nearest_t, hit, normal = res
                nearest_obj = obj
        if nearest_obj is None:
            break
        # Local color (Phong)
        to_cam = -dir
        local_color = np.zeros(3)
        for light in lights:
            to_light = light.pos - hit
            light_dist = np.linalg.norm(to_light)
            to_light_dir = to_light / light_dist

            # Shadow check (soft shadow ray sampling)
            shadow_orig = hit + normal*1e-4
            in_shadow = False
            for obj in objects:
                res = obj.intersect(shadow_orig, to_light_dir)
                if res and res[0] < light_dist:
                    in_shadow = True
                    break
            if in_shadow: continue

            # Diffuse
            diff = max(0, np.dot(normal, to_light_dir))
            # Specular
            half_vec = normalize(to_light_dir + to_cam)
            spec = max(0.0, np.dot(normal, half_vec)) ** nearest_obj.spec
            intensity = light.intensity / (0.5 + light_dist**2)
            local_color += (
                nearest_obj.color * diff * intensity * light.color +
                spec * intensity * light.color
            )
        # Reflection ray
        refl = nearest_obj.refl
        color += reflection * (local_color)
        if depth >= MAX_DEPTH or refl <= 1e-3:
            break
        reflection *= refl
        dir = dir - 2 * np.dot(dir, normal) * normal
        orig = hit + normal*1e-4
        depth += 1
    return np.clip(color, 0, 1)

# Main render
pixels = np.zeros((HEIGHT, WIDTH, 3))
aspect = WIDTH/HEIGHT
screen = (-1, 1/aspect, 1, -1/aspect)

print("Rendering...")
for i, y in enumerate(np.linspace(screen[1], screen[3], HEIGHT)):
    for j, x in enumerate(np.linspace(screen[0], screen[2], WIDTH)):
        # Camera ray
        pixel = np.array([x, y, 0])
        dir = normalize(pixel - CAMERA_POS)
        color = trace(CAMERA_POS, dir, 0)
        pixels[i, j] = color ** (1/2.2)  # Gamma correction
    if i % 32 == 0:
        print(f"{100*i/HEIGHT:.1f}% complete")

# Save image
img = (pixels * 255).astype(np.uint8)
Image.fromarray(img).save("raytrace_multilight.png")
print("Saved raytrace_multilight.png")
