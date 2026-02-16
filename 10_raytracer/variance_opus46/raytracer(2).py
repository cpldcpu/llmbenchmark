import numpy as np
from PIL import Image
import time

W, H = 800, 600
MAX_DEPTH = 4
EPSILON = 1e-6

def normalize(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n < EPSILON, 1.0, n)

# ── Scene objects ────────────────────────────────────────────────────────────

class Sphere:
    def __init__(self, center, radius, color, specular=0.5, reflectivity=0.3, shininess=50):
        self.center = np.array(center, dtype=np.float64)
        self.radius = radius
        self.color = np.array(color, dtype=np.float64)
        self.specular = specular
        self.reflectivity = reflectivity
        self.shininess = shininess

class Plane:
    def __init__(self, point, normal, color, color2=None, specular=0.2, reflectivity=0.15, shininess=20, checker_scale=1.0):
        self.point = np.array(point, dtype=np.float64)
        self.normal = normalize(np.array(normal, dtype=np.float64).reshape(1,3)).flatten()
        self.color = np.array(color, dtype=np.float64)
        self.color2 = np.array(color2, dtype=np.float64) if color2 is not None else None
        self.specular = specular
        self.reflectivity = reflectivity
        self.shininess = shininess
        self.checker_scale = checker_scale

class Light:
    def __init__(self, position, color, intensity=1.0):
        self.position = np.array(position, dtype=np.float64)
        self.color = np.array(color, dtype=np.float64) * intensity

# ── Intersection ─────────────────────────────────────────────────────────────

def intersect_sphere(origins, dirs, sphere):
    oc = origins - sphere.center
    b = 2.0 * np.sum(dirs * oc, axis=-1)
    c = np.sum(oc * oc, axis=-1) - sphere.radius ** 2
    disc = b * b - 4 * c
    mask = disc > 0
    t = np.full(origins.shape[0], np.inf)
    sq = np.sqrt(np.maximum(disc, 0))
    t1 = (-b - sq) / 2.0
    t2 = (-b + sq) / 2.0
    t_candidate = np.where(t1 > EPSILON, t1, t2)
    valid = mask & (t_candidate > EPSILON)
    t[valid] = t_candidate[valid]
    return t

def intersect_plane(origins, dirs, plane):
    denom = np.sum(dirs * plane.normal, axis=-1)
    t = np.sum((plane.point - origins) * plane.normal, axis=-1) / np.where(np.abs(denom) < EPSILON, 1.0, denom)
    valid = (np.abs(denom) > EPSILON) & (t > EPSILON)
    result = np.full(origins.shape[0], np.inf)
    result[valid] = t[valid]
    return result

# ── Scene setup ──────────────────────────────────────────────────────────────

spheres = [
    # Large central reflective sphere
    Sphere([0, 1.2, 8], 1.5, [0.95, 0.95, 0.95], specular=0.9, reflectivity=0.7, shininess=200),
    # Colored spheres arranged around it
    Sphere([-3.0, 0.6, 6.5], 0.8, [0.9, 0.15, 0.15], specular=0.6, reflectivity=0.25, shininess=80),
    Sphere([3.0, 0.6, 6.5], 0.8, [0.15, 0.15, 0.9], specular=0.6, reflectivity=0.25, shininess=80),
    Sphere([-1.8, 0.5, 5.0], 0.65, [0.1, 0.85, 0.2], specular=0.5, reflectivity=0.2, shininess=60),
    Sphere([1.8, 0.5, 5.0], 0.65, [0.9, 0.7, 0.1], specular=0.5, reflectivity=0.2, shininess=60),
    Sphere([0, 0.45, 4.2], 0.55, [0.85, 0.2, 0.85], specular=0.5, reflectivity=0.2, shininess=60),
    # Background spheres
    Sphere([-4.5, 1.0, 12], 1.2, [0.2, 0.7, 0.8], specular=0.4, reflectivity=0.15, shininess=40),
    Sphere([4.5, 1.0, 12], 1.2, [0.8, 0.4, 0.2], specular=0.4, reflectivity=0.15, shininess=40),
    Sphere([0, 2.8, 14], 1.8, [0.6, 0.3, 0.7], specular=0.5, reflectivity=0.3, shininess=60),
    # Small accent spheres
    Sphere([-1.0, 0.25, 3.5], 0.3, [1.0, 0.5, 0.0], specular=0.7, reflectivity=0.1, shininess=100),
    Sphere([1.0, 0.25, 3.5], 0.3, [0.0, 0.8, 0.8], specular=0.7, reflectivity=0.1, shininess=100),
    Sphere([-2.5, 0.2, 4.5], 0.25, [1.0, 1.0, 0.3], specular=0.6, reflectivity=0.1, shininess=80),
    Sphere([2.5, 0.2, 4.5], 0.25, [0.3, 1.0, 0.5], specular=0.6, reflectivity=0.1, shininess=80),
]

planes = [
    # Floor with checkerboard
    Plane([0, -0.15, 0], [0, 1, 0], [0.85, 0.85, 0.85], color2=[0.2, 0.2, 0.25],
          specular=0.3, reflectivity=0.25, shininess=30, checker_scale=1.5),
]

# Many colorful lights!
lights = [
    # Key lights
    Light([-4, 8, 2], [1.0, 0.4, 0.4], intensity=1.4),
    Light([4, 8, 2], [0.4, 0.4, 1.0], intensity=1.4),
    Light([0, 10, 8], [1.0, 1.0, 0.9], intensity=1.0),
    # Colored accent lights close to the scene
    Light([-3, 2, 4], [1.0, 0.1, 0.3], intensity=0.8),
    Light([3, 2, 4], [0.1, 0.3, 1.0], intensity=0.8),
    Light([0, 3, 3], [0.2, 1.0, 0.3], intensity=0.7),
    Light([-2, 1, 7], [1.0, 0.8, 0.1], intensity=0.6),
    Light([2, 1, 7], [0.1, 0.8, 1.0], intensity=0.6),
    # Rear / rim lights
    Light([0, 5, 15], [0.9, 0.5, 1.0], intensity=1.2),
    Light([-5, 3, 10], [1.0, 0.6, 0.2], intensity=0.7),
    Light([5, 3, 10], [0.2, 1.0, 0.6], intensity=0.7),
    # Low colored fills
    Light([-6, 0.5, 6], [0.8, 0.2, 0.6], intensity=0.5),
    Light([6, 0.5, 6], [0.6, 0.2, 0.8], intensity=0.5),
]

objects = spheres + planes
ambient = np.array([0.04, 0.04, 0.06])

# ── Ray tracing core ────────────────────────────────────────────────────────

def find_nearest(origins, dirs, n):
    best_t = np.full(n, np.inf)
    best_idx = np.full(n, -1, dtype=np.int32)
    for i, obj in enumerate(objects):
        if isinstance(obj, Sphere):
            t = intersect_sphere(origins, dirs, obj)
        else:
            t = intersect_plane(origins, dirs, obj)
        closer = t < best_t
        best_t[closer] = t[closer]
        best_idx[closer] = i
    return best_t, best_idx

def get_surface_props(hit_points, obj_idx, n):
    colors = np.zeros((n, 3))
    normals = np.zeros((n, 3))
    spec = np.zeros(n)
    refl = np.zeros(n)
    shin = np.zeros(n)

    for i, obj in enumerate(objects):
        mask = obj_idx == i
        if not np.any(mask):
            continue
        if isinstance(obj, Sphere):
            normals[mask] = normalize(hit_points[mask] - obj.center)
            colors[mask] = obj.color
        else:
            normals[mask] = obj.normal
            if obj.color2 is not None:
                pts = hit_points[mask]
                cx = np.floor(pts[:, 0] / obj.checker_scale).astype(int)
                cz = np.floor(pts[:, 2] / obj.checker_scale).astype(int)
                checker = ((cx + cz) % 2 == 0).astype(float)
                colors[mask] = checker[:, None] * obj.color + (1 - checker[:, None]) * obj.color2
            else:
                colors[mask] = obj.color
        spec[mask] = obj.specular
        refl[mask] = obj.reflectivity
        shin[mask] = obj.shininess
    return colors, normals, spec, refl, shin

def shade(origins, dirs, depth=0):
    n = origins.shape[0]
    result = np.zeros((n, 3))
    if depth > MAX_DEPTH or n == 0:
        return result

    best_t, best_idx = find_nearest(origins, dirs, n)
    hit_mask = best_idx >= 0
    if not np.any(hit_mask):
        t_sky = 0.5 * (dirs[:, 1] + 1.0)
        result[:] = (1 - t_sky)[:, None] * np.array([0.05, 0.05, 0.1]) + t_sky[:, None] * np.array([0.02, 0.02, 0.08])
        return result

    # Sky for missed rays
    miss = ~hit_mask
    t_sky = 0.5 * (dirs[miss, 1] + 1.0)
    result[miss] = (1 - t_sky)[:, None] * np.array([0.05, 0.05, 0.1]) + t_sky[:, None] * np.array([0.02, 0.02, 0.08])

    # Hit points
    idx = np.where(hit_mask)[0]
    hp = origins[idx] + best_t[idx, None] * dirs[idx]
    oi = best_idx[idx]
    m = idx.shape[0]

    colors, normals, spec, refl, shin = get_surface_props(hp, oi, m)

    # Lighting
    color_accum = np.tile(ambient, (m, 1)) * colors
    view_dirs = normalize(-dirs[idx])

    for light in lights:
        to_light = light.position - hp
        dist2 = np.sum(to_light * to_light, axis=-1, keepdims=True)
        dist = np.sqrt(dist2)
        L = to_light / dist
        ndotl = np.sum(normals * L, axis=-1)

        # Shadow check
        shadow_orig = hp + normals * EPSILON
        shadow_t, _ = find_nearest(shadow_orig, L, m)
        in_shadow = shadow_t < dist.flatten()

        # Attenuation
        atten = 1.0 / (1.0 + 0.05 * dist.flatten() + 0.01 * dist2.flatten())

        lit = (~in_shadow) & (ndotl > 0)
        lit_f = lit.astype(float)

        # Diffuse
        diff = np.maximum(ndotl, 0) * lit_f * atten
        color_accum += colors * light.color * diff[:, None]

        # Specular (Blinn-Phong)
        H = normalize(L + view_dirs)
        ndoth = np.maximum(np.sum(normals * H, axis=-1), 0)
        spec_comp = (ndoth ** shin) * spec * lit_f * atten
        color_accum += light.color * spec_comp[:, None]

    color_accum = np.clip(color_accum, 0, 1)

    # Reflections
    refl_mask = refl > 0.01
    if np.any(refl_mask) and depth < MAX_DEPTH:
        ri = np.where(refl_mask)[0]
        refl_dirs = dirs[idx[ri]] - 2 * np.sum(dirs[idx[ri]] * normals[ri], axis=-1, keepdims=True) * normals[ri]
        refl_dirs = normalize(refl_dirs)
        refl_orig = hp[ri] + normals[ri] * EPSILON
        refl_color = shade(refl_orig, refl_dirs, depth + 1)
        color_accum[ri] = color_accum[ri] * (1 - refl[ri, None]) + refl_color * refl[ri, None]

    result[idx] = color_accum
    return result

# ── Render ───────────────────────────────────────────────────────────────────

def render():
    print(f"Rendering {W}x{H} with {len(lights)} lights, {len(objects)} objects, max depth {MAX_DEPTH}")
    t0 = time.time()

    fov = 60
    aspect = W / H
    scale = np.tan(np.radians(fov / 2))

    cam_pos = np.array([0.0, 2.5, -2.0])
    cam_target = np.array([0.0, 1.0, 8.0])
    cam_fwd = normalize((cam_target - cam_pos).reshape(1, 3)).flatten()
    cam_right = normalize(np.cross(cam_fwd, np.array([0, 1, 0])).reshape(1, 3)).flatten()
    cam_up = np.cross(cam_right, cam_fwd)

    jj, ii = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    px = (2 * (ii + 0.5) / W - 1) * aspect * scale
    py = (1 - 2 * (jj + 0.5) / H) * scale

    flat_px = px.flatten()
    flat_py = py.flatten()
    n = flat_px.shape[0]

    dirs = flat_px[:, None] * cam_right + flat_py[:, None] * cam_up + cam_fwd
    dirs = normalize(dirs)
    origins = np.tile(cam_pos, (n, 1))

    chunk = 48000
    img_flat = np.zeros((n, 3))
    total_chunks = (n + chunk - 1) // chunk
    for ci, start in enumerate(range(0, n, chunk)):
        end = min(start + chunk, n)
        print(f"  chunk {ci+1}/{total_chunks}...")
        img_flat[start:end] = shade(origins[start:end], dirs[start:end])

    # Reinhard tone mapping + gamma
    img_flat = np.clip(img_flat, 0, None)
    img_flat = img_flat / (1.0 + img_flat)
    img_flat = np.power(img_flat, 0.85)
    img_flat = np.clip(img_flat * 255, 0, 255).astype(np.uint8)

    img = img_flat.reshape(H, W, 3)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")
    return Image.fromarray(img)

if __name__ == "__main__":
    img = render()
    img.save("/home/claude/scene.png")
    print("Saved scene.png")
