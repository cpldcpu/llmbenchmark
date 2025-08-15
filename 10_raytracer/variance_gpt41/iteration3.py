import math
import numpy as np
from PIL import Image

WIDTH = 800
HEIGHT = 600
MAX_DEPTH = 4

def normalize(v):
    return v / np.linalg.norm(v)

def reflect(I, N):
    return I - 2*np.dot(I, N)*N

def sphere_intersect(origin, direction, center, radius):
    # Returns distance from origin to intersection or None
    oc = origin - center
    b = 2 * np.dot(direction, oc)
    c = np.dot(oc, oc) - radius*radius
    disc = b*b - 4*c
    if disc < 0:
        return None
    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / 2
    t2 = (-b + sqrt_disc) / 2
    if t1 > 0.001:
        return t1
    if t2 > 0.001:
        return t2
    return None

# Scene definition
spheres = [
    # center,   radius,     color,      reflection
    (np.array([0,   -1,  3]), 1, np.array([220,  30,  30]), 0.5),
    (np.array([2,    0,  4]), 1, np.array([ 20, 210,  20]), 0.35),
    (np.array([-2,   0,  4]), 1, np.array([  0,  60, 220]), 0.5),
    (np.array([0,   -5001, 0]), 5000, np.array([255, 255, 255]), 0.00), # ground
]

# Many colourful lightsources
lights = [
    # position,          color,       intensity
    (np.array([ 10,  10, -10]), np.array([255,255,210]), 1.3),
    (np.array([-10,  10,  10]), np.array([ 46,119,255]), 0.8),
    (np.array([ 10, -10,  10]), np.array([255, 55, 60]), 0.7),
    (np.array([ -5,  10,  1 ]), np.array([100,255,180]), 1.1),
]

background_color = np.array([15, 18, 40])

def trace_ray(origin, direction, depth):
    if depth > MAX_DEPTH:
        return background_color

    min_dist = float('inf')
    hit_obj = None
    for sphere in spheres:
        center, radius, color, refl = sphere
        dist = sphere_intersect(origin, direction, center, radius)
        if dist is not None and dist < min_dist:
            min_dist = dist
            hit_obj = sphere

    if hit_obj is None:
        return background_color

    center, radius, color, refl = hit_obj
    point = origin + direction * min_dist
    normal = normalize(point - center)
    view = -direction
    col = np.zeros(3)
    # Ambient
    col += 0.05 * color

    for lpos, lcolor, lintensity in lights:
        # Shadow check
        light_dir = normalize(lpos - point)
        shadow_orig = point + normal * 0.0005
        shadow_dist = np.linalg.norm(lpos - point)
        shadow = False
        for sphere2 in spheres:
            c2,r2,_,_ = sphere2
            if sphere2 is hit_obj:
                continue
            hit = sphere_intersect(shadow_orig, light_dir, c2, r2)
            if hit is not None and hit < shadow_dist:
                shadow = True
                break
        if shadow:
            continue

        # Diffuse
        diffuse = max(np.dot(normal, light_dir), 0)
        # Specular
        reflect_dir = reflect(-light_dir, normal)
        spec = pow(max(np.dot(view, reflect_dir),0), 50)
        col += diffuse * lintensity * color * lcolor / 255 / 255
        col += spec * lintensity * lcolor
    # Reflection
    if refl > 0.01:
        rdir = reflect(direction, normal)
        rorig = point + normal * 0.0005
        rcol = trace_ray(rorig, normalize(rdir), depth+1)
        col = col * (1-refl) + rcol * refl
    # Clamp
    col = np.clip(col, 0, 255)
    return col

def render():
    camera = np.array([0, 0, -1])
    img = Image.new("RGB", (WIDTH, HEIGHT))
    pixels = img.load()
    aspect = WIDTH/HEIGHT
    fov = math.pi/3
    for y in range(HEIGHT):
        for x in range(WIDTH):
            # NDC
            px =  (2 * (x + 0.5) / WIDTH - 1) * math.tan(fov/2) * aspect
            py = -(2 * (y + 0.5) / HEIGHT - 1) * math.tan(fov/2)
            direction = normalize(np.array([px, py, 1]))
            color = trace_ray(camera, direction, 0)
            pixels[x,y] = tuple(color.astype(int))
        if y % 40 == 0:
            print(f"Rendered {y}/{HEIGHT}")
    img.save("raytraced_scene3.png")
    print("Image saved as colorful_raytraced_scene.png")

if __name__ == "__main__":
    render()
