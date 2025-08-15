import numpy as np
from PIL import Image

WIDTH = 800
HEIGHT = 600
FOV = np.pi / 3  # 60 degree field of view
MAX_DEPTH = 3
BACKGROUND = np.array([0.05, 0.05, 0.08])

def normalize(v): return v / np.linalg.norm(v)

def reflect(I, N): return I - 2 * np.dot(I, N) * N

class Sphere:
    def __init__(self, center, radius, color, specular=150, reflect=0.25):
        self.center = np.array(center)
        self.radius = radius
        self.color = np.array(color)
        self.specular = specular
        self.reflect = reflect

    def intersect(self, orig, dir):
        OC = orig - self.center
        b = 2 * np.dot(dir, OC)
        c = np.dot(OC, OC) - self.radius**2
        disc = b*b - 4*c
        if disc < 0: return None
        sqrtd = np.sqrt(disc)
        t1 = (-b - sqrtd) / 2
        t2 = (-b + sqrtd) / 2
        if t1 > 0.001: return t1
        if t2 > 0.001: return t2
        return None

    def normal(self, point):
        return normalize(point - self.center)

class Light:
    def __init__(self, position, color, intensity=1):
        self.position = np.array(position)
        self.color = np.array(color)
        self.intensity = intensity

def clamp(x): return np.clip(x, 0, 1)

def trace(orig, dir, spheres, lights, depth):
    color = BACKGROUND
    min_t = float('inf')
    hit_obj = hit_pt = norm = None

    for obj in spheres:
        t = obj.intersect(orig, dir)
        if t and t < min_t:
            min_t = t
            hit_obj = obj
            hit_pt = orig + dir * t
            norm = obj.normal(hit_pt)

    if not hit_obj:
        return BACKGROUND

    surface_color = hit_obj.color * 0.15  # ambient

    # Phong lighting
    for light in lights:
        to_light = normalize(light.position - hit_pt)
        # Shadow
        shadow = False
        for obj in spheres:
            if obj is hit_obj: continue
            t = obj.intersect(hit_pt + norm*0.01, to_light)
            if t: shadow = True; break

        if not shadow:
            # Diffuse
            diff = max(np.dot(norm, to_light), 0)
            surface_color += hit_obj.color * light.color * light.intensity * diff
            # Specular
            view_dir = normalize(orig - hit_pt)
            half = normalize(to_light + view_dir)
            spec = max(np.dot(norm, half), 0) ** hit_obj.specular
            surface_color += light.color * light.intensity * spec * 0.6

    # Reflection
    if depth < MAX_DEPTH and hit_obj.reflect > 0:
        refl_dir = normalize(reflect(dir, norm))
        refl_col = trace(hit_pt + norm*0.01, refl_dir, spheres, lights, depth+1)
        surface_color = surface_color*(1-hit_obj.reflect) + refl_col*hit_obj.reflect

    return clamp(surface_color)

# Scene setup
spheres = [
    Sphere([0.0, -1.5, 4], 1.5, [0.2, 0.8, 0.8], 300, 0.5),
    Sphere([2, 0, 5], 1, [0.9, 0.4, 0.3], 100, 0.3),
    Sphere([-2, 0, 5], 1, [0.4, 0.9, 0.3], 100, 0.3),
    Sphere([0, -5001, 0], 5000, [0.85, 0.86, 0.6], 1000, 0.1),  # floor
]

# Many colored lights
np.random.seed(41)
lights = [
    Light([np.cos(a)*5, np.sin(a)*4+2, 3 + np.sin(a*2)*3], 
          [np.sin(a+1)*0.5+0.5, np.cos(a)*0.5+0.5, np.cos(a*2)*0.5+0.5],
          intensity=1.2 if i%2==0 else 0.5)
    for i,a in enumerate(np.linspace(0, 2*np.pi, 15, endpoint=False))
]
# Add a main white light
lights.append(Light([0, 10, -2], [1, 1, 1], 1.0))

# Camera origin
camera = np.array([0, 0, -1])

# Prepare image buffer
img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)

aspect = WIDTH / HEIGHT
screen_y = np.tan(FOV/2)*2
screen_x = screen_y*aspect

print("Raytracing...")
for y in range(HEIGHT):
    for x in range(WIDTH):
        # Normalized device coordinates (left,bottom=-1, right,top=+1)
        u = (x + 0.5)/WIDTH*2 - 1
        v = 1 - (y + 0.5)/HEIGHT*2
        pix_dir = normalize(np.array([
            u*screen_x/2, v*screen_y/2, 1
        ]))
        color = trace(camera, pix_dir, spheres, lights, 0)
        img[y,x] = color

    if y % 40 == 0: print(f"{y*100//HEIGHT}% done...")

print("Saving image...")
img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
Image.fromarray(img, 'RGB').save('raytraced_scene4.png')
print("Done! Saved as raytracer.png")
