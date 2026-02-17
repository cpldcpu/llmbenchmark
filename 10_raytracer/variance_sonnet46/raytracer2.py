import numpy as np
from PIL import Image

W, H = 800, 600

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def reflect(v, n):
    return v - 2 * np.dot(v, n) * n

# Scene objects: spheres
spheres = [
    # (center, radius, color, shininess, reflectivity)
    (np.array([0.0, -0.5, 4.0]),   0.8,  np.array([0.9, 0.2, 0.2]), 80,  0.3),   # red
    (np.array([-2.0, 0.0, 5.5]),   0.9,  np.array([0.2, 0.5, 0.9]), 120, 0.5),   # blue
    (np.array([2.0, 0.0, 5.0]),    0.8,  np.array([0.2, 0.9, 0.3]), 60,  0.2),   # green
    (np.array([-1.0, 1.2, 3.5]),   0.5,  np.array([0.9, 0.8, 0.1]), 200, 0.6),   # gold
    (np.array([1.0, 1.0, 3.0]),    0.45, np.array([0.8, 0.2, 0.9]), 150, 0.5),   # purple
    (np.array([0.0, 2.0, 5.0]),    0.6,  np.array([0.1, 0.9, 0.9]), 100, 0.4),   # cyan
    (np.array([-3.0, -0.5, 7.0]),  1.2,  np.array([0.95,0.6, 0.1]), 40,  0.1),   # orange
    (np.array([3.2, 0.5, 6.5]),    1.0,  np.array([0.6, 0.1, 0.5]), 90,  0.3),   # magenta
    # Floor (big sphere trick)
    (np.array([0.0, -101.3, 5.0]), 100.0,np.array([0.8, 0.8, 0.8]), 20,  0.15),  # floor
    # Back wall
    (np.array([0.0, 0.0, 20.0]),   14.0, np.array([0.3, 0.3, 0.4]), 10,  0.05),  # back
]

# Colorful light sources: (position, color, intensity)
lights = [
    (np.array([-4.0, 5.0,  2.0]), np.array([1.0, 0.2, 0.2]), 1.5),  # red
    (np.array([ 4.0, 5.0,  2.0]), np.array([0.2, 0.4, 1.0]), 1.5),  # blue
    (np.array([ 0.0, 6.0,  6.0]), np.array([1.0, 1.0, 0.4]), 1.2),  # warm yellow
    (np.array([-2.0, 2.0,  1.0]), np.array([0.2, 1.0, 0.5]), 1.0),  # green
    (np.array([ 2.0, 1.5,  1.0]), np.array([1.0, 0.3, 1.0]), 1.0),  # purple
    (np.array([ 0.0, 8.0,  3.0]), np.array([1.0, 1.0, 1.0]), 0.8),  # white overhead
    (np.array([ 0.0, 1.0, -1.0]), np.array([0.3, 0.9, 1.0]), 0.6),  # cyan front
]

AMBIENT = 0.08
BG_TOP    = np.array([0.03, 0.03, 0.12])
BG_BOTTOM = np.array([0.08, 0.04, 0.08])

def ray_sphere_intersect(ro, rd, center, radius):
    oc = ro - center
    b = np.dot(oc, rd)
    c = np.dot(oc, oc) - radius * radius
    disc = b * b - c
    if disc < 0:
        return None
    sq = np.sqrt(disc)
    t = -b - sq
    if t > 1e-4:
        return t
    t = -b + sq
    if t > 1e-4:
        return t
    return None

def scene_intersect(ro, rd):
    best_t, best_i = np.inf, -1
    for i, (center, radius, *_) in enumerate(spheres):
        t = ray_sphere_intersect(ro, rd, center, radius)
        if t and t < best_t:
            best_t, best_i = t, i
    return best_t, best_i

def trace(ro, rd, depth=0):
    t, idx = scene_intersect(ro, rd)
    if idx < 0:
        # Background gradient
        u = max(0, rd[1] * 0.5 + 0.5)
        return BG_TOP * (1 - u) + BG_BOTTOM * u

    center, radius, color, shininess, reflectivity = spheres[idx]
    hit = ro + rd * t
    normal = normalize(hit - center)

    color_acc = color * AMBIENT

    for lpos, lcol, lintensity in lights:
        to_light = lpos - hit
        dist_light = np.linalg.norm(to_light)
        ld = to_light / dist_light

        # Shadow check
        sh_t, sh_i = scene_intersect(hit + normal * 1e-4, ld)
        if sh_i >= 0 and sh_t < dist_light:
            continue

        diff = max(0.0, np.dot(normal, ld))
        half = normalize(ld - rd)
        spec = max(0.0, np.dot(normal, half)) ** shininess

        attenuation = lintensity / (1.0 + 0.04 * dist_light * dist_light)
        color_acc = color_acc + lcol * attenuation * (color * diff * 0.9 + spec * 0.6)

    if depth < 3 and reflectivity > 0.05:
        ref_dir = normalize(reflect(rd, normal))
        ref_col = trace(hit + normal * 1e-4, ref_dir, depth + 1)
        color_acc = color_acc * (1 - reflectivity) + ref_col * reflectivity

    return color_acc

print("Rendering 800x600...")
img = np.zeros((H, W, 3))

fov = np.pi / 3  # 60 deg
aspect = W / H
cam = np.array([0.0, 0.5, -1.0])

for j in range(H):
    if j % 50 == 0:
        print(f"  Row {j}/{H}")
    for i in range(W):
        x = (2 * (i + 0.5) / W - 1) * np.tan(fov / 2) * aspect
        y = (1 - 2 * (j + 0.5) / H) * np.tan(fov / 2)
        rd = normalize(np.array([x, y, 1.0]))
        color = trace(cam, rd)
        img[j, i] = color

# Tone map + gamma
img = img / (img + 1.0)          # Reinhard
img = np.clip(img, 0, 1)
img = img ** (1 / 2.2)           # Gamma
img = (img * 255).astype(np.uint8)

out = Image.fromarray(img)
out.save("/mnt/user-data/outputs/scene.png")
print("Done! Saved to scene.png")
