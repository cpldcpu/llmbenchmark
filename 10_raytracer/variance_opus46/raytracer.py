import numpy as np
from PIL import Image
import time

W, H = 800, 600
MAX_DEPTH = 4
AMBIENT = 0.05
EPS = 1e-4

def normalize(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1, n)

# ── Scene objects ──────────────────────────────────────────────

class Sphere:
    def __init__(self, center, radius, color, specular=50, reflectivity=0.3, checkered=False):
        self.c = np.array(center, dtype=np.float64)
        self.r = radius
        self.color = np.array(color, dtype=np.float64)
        self.spec = specular
        self.refl = reflectivity
        self.checkered = checkered

    def intersect(self, origins, dirs):
        oc = origins - self.c
        b = 2 * np.sum(dirs * oc, axis=-1)
        c = np.sum(oc * oc, axis=-1) - self.r ** 2
        disc = b * b - 4 * c
        mask = disc >= 0
        sqrt_disc = np.sqrt(np.where(mask, disc, 0))
        t1 = (-b - sqrt_disc) / 2
        t2 = (-b + sqrt_disc) / 2
        t = np.where(t1 > EPS, t1, np.where(t2 > EPS, t2, np.inf))
        t = np.where(mask, t, np.inf)
        return t

    def normal(self, points):
        return normalize(points - self.c)

    def get_color(self, points):
        if not self.checkered:
            return self.color
        # Checkerboard pattern
        x = np.floor(points[..., 0] * 2).astype(int)
        z = np.floor(points[..., 2] * 2).astype(int)
        check = ((x + z) % 2 == 0).astype(np.float64)
        return self.color * (0.3 + 0.7 * check[..., np.newaxis])


class Plane:
    def __init__(self, point, normal_vec, color, specular=10, reflectivity=0.2, checkered=False):
        self.p = np.array(point, dtype=np.float64)
        self.n = normalize(np.array(normal_vec, dtype=np.float64).reshape(1, 3)).flatten()
        self.color = np.array(color, dtype=np.float64)
        self.spec = specular
        self.refl = reflectivity
        self.checkered = checkered

    def intersect(self, origins, dirs):
        denom = np.sum(dirs * self.n, axis=-1)
        t = np.sum((self.p - origins) * self.n, axis=-1) / np.where(np.abs(denom) < EPS, EPS, denom)
        return np.where((t > EPS) & (np.abs(denom) > EPS), t, np.inf)

    def normal(self, points):
        return np.broadcast_to(self.n, points.shape).copy()

    def get_color(self, points):
        if not self.checkered:
            return self.color
        x = np.floor(points[..., 0]).astype(int)
        z = np.floor(points[..., 2]).astype(int)
        check = ((x + z) % 2 == 0).astype(np.float64)
        c2 = np.array([0.95, 0.95, 0.95])
        return self.color * (1 - check[..., np.newaxis]) + c2 * check[..., np.newaxis]


# ── Scene setup ────────────────────────────────────────────────

objects = [
    # Floor
    Plane([0, -1, 0], [0, 1, 0], [0.15, 0.15, 0.2], specular=80, reflectivity=0.35, checkered=True),

    # Central large mirrored sphere
    Sphere([0, 0.5, 5], 1.5, [0.9, 0.9, 0.95], specular=300, reflectivity=0.75),

    # Surrounding smaller spheres in a ring
    Sphere([-2.8, -0.3, 4.0], 0.7, [0.95, 0.15, 0.15], specular=100, reflectivity=0.3),
    Sphere([-1.8, -0.5, 7.0], 0.5, [0.15, 0.95, 0.15], specular=100, reflectivity=0.25),
    Sphere([2.5, -0.3, 3.5],  0.7, [0.15, 0.15, 0.95], specular=100, reflectivity=0.3),
    Sphere([2.0, -0.5, 7.0],  0.5, [0.95, 0.95, 0.15], specular=100, reflectivity=0.25),
    Sphere([-1.2, -0.6, 2.5], 0.4, [0.95, 0.5, 0.1],  specular=80,  reflectivity=0.2),
    Sphere([1.3, -0.6, 2.3],  0.4, [0.1, 0.95, 0.95],  specular=80,  reflectivity=0.2),
    Sphere([0, -0.5, 8.0],    0.5, [0.8, 0.15, 0.9],   specular=100, reflectivity=0.3),

    # Floating gems
    Sphere([-3.5, 1.5, 6.5], 0.35, [1.0, 0.4, 0.7], specular=200, reflectivity=0.5),
    Sphere([3.5, 1.8, 6.0],  0.35, [0.4, 1.0, 0.7], specular=200, reflectivity=0.5),
    Sphere([0, 3.0, 7.0],    0.4,  [0.7, 0.7, 1.0], specular=200, reflectivity=0.6),
]

# Many colourful light sources!
lights = [
    # Warm key light (upper right)
    {"pos": np.array([5.0, 8.0, -2.0]),  "color": np.array([1.0, 0.9, 0.75]),  "intensity": 1.2},
    # Cool fill (upper left)
    {"pos": np.array([-6.0, 6.0, 0.0]),  "color": np.array([0.5, 0.6, 1.0]),   "intensity": 0.7},

    # Coloured point lights scattered around the scene
    {"pos": np.array([-3.0, 2.0, 3.0]),  "color": np.array([1.0, 0.1, 0.1]),   "intensity": 0.9},   # Red
    {"pos": np.array([3.0, 2.0, 3.0]),   "color": np.array([0.1, 0.1, 1.0]),   "intensity": 0.9},   # Blue
    {"pos": np.array([0.0, 1.5, 2.0]),   "color": np.array([0.1, 1.0, 0.1]),   "intensity": 0.6},   # Green
    {"pos": np.array([-2.0, 0.5, 7.0]),  "color": np.array([1.0, 0.6, 0.0]),   "intensity": 0.7},   # Orange
    {"pos": np.array([2.0, 0.5, 7.0]),   "color": np.array([0.8, 0.0, 1.0]),   "intensity": 0.7},   # Purple
    {"pos": np.array([0.0, 4.0, 5.0]),   "color": np.array([1.0, 1.0, 0.2]),   "intensity": 0.6},   # Yellow
    {"pos": np.array([-4.0, 0.3, 5.5]),  "color": np.array([0.0, 1.0, 1.0]),   "intensity": 0.5},   # Cyan
    {"pos": np.array([4.0, 0.3, 5.5]),   "color": np.array([1.0, 0.2, 0.6]),   "intensity": 0.5},   # Pink
    {"pos": np.array([0.0, 0.0, 9.0]),   "color": np.array([0.6, 1.0, 0.6]),   "intensity": 0.5},   # Lime
    {"pos": np.array([0.0, -0.8, 0.5]),  "color": np.array([1.0, 0.5, 0.3]),   "intensity": 0.3},   # Warm ground bounce
]

# ── Ray tracing engine ─────────────────────────────────────────

def find_nearest(origins, dirs):
    """Returns (t_min, obj_indices) for closest intersection per ray."""
    N = origins.shape[0]
    t_min = np.full(N, np.inf)
    obj_idx = np.full(N, -1, dtype=int)
    for i, obj in enumerate(objects):
        t = obj.intersect(origins, dirs)
        closer = t < t_min
        t_min = np.where(closer, t, t_min)
        obj_idx = np.where(closer, i, obj_idx)
    return t_min, obj_idx

def shade(hit_points, normals, view_dirs, obj_indices):
    """Compute direct illumination with shadows for all hit points."""
    N = hit_points.shape[0]
    color = np.zeros((N, 3))

    for light in lights:
        to_light = light["pos"] - hit_points
        dist_light = np.linalg.norm(to_light, axis=-1, keepdims=True)
        light_dir = to_light / dist_light
        dist_light = dist_light.flatten()

        # Shadow test
        shadow_orig = hit_points + normals * EPS
        t_shadow, _ = find_nearest(shadow_orig, light_dir)
        in_shadow = t_shadow < dist_light

        # Attenuation (quadratic with min clamp)
        atten = light["intensity"] / (1.0 + 0.05 * dist_light + 0.01 * dist_light ** 2)

        # Diffuse
        ndotl = np.sum(normals * light_dir, axis=-1)
        diffuse = np.maximum(ndotl, 0) * atten

        # Specular (Blinn-Phong)
        half_vec = normalize(light_dir + view_dirs)
        ndoth = np.sum(normals * half_vec, axis=-1)
        # Gather specular exponents per hit
        spec_exp = np.zeros(N)
        for i, obj in enumerate(objects):
            mask = obj_indices == i
            spec_exp[mask] = obj.spec
        specular = np.power(np.maximum(ndoth, 0), spec_exp) * atten

        # Combine, masked by shadow
        shadow_mask = (~in_shadow).astype(np.float64)
        color += (diffuse[..., np.newaxis] + specular[..., np.newaxis]) * light["color"] * shadow_mask[..., np.newaxis]

    return color

def trace_rays(origins, dirs, depth=0):
    """Recursive ray tracer. Returns (N,3) color array."""
    N = origins.shape[0]
    result = np.zeros((N, 3))
    if depth > MAX_DEPTH or N == 0:
        return result

    t_min, obj_idx = find_nearest(origins, dirs)
    hit_mask = t_min < np.inf
    hit_indices = np.where(hit_mask)[0]

    if len(hit_indices) == 0:
        # Sky gradient
        t_sky = 0.5 * (dirs[:, 1] + 1.0)
        sky = (1 - t_sky)[..., np.newaxis] * np.array([0.05, 0.02, 0.1]) + \
              t_sky[..., np.newaxis] * np.array([0.02, 0.0, 0.08])
        return sky

    # Sky for missed rays
    t_sky = 0.5 * (dirs[~hit_mask, 1] + 1.0)
    sky_miss = (1 - t_sky)[..., np.newaxis] * np.array([0.05, 0.02, 0.1]) + \
               t_sky[..., np.newaxis] * np.array([0.02, 0.0, 0.08])
    result[~hit_mask] = sky_miss

    # Hit computations
    h_t = t_min[hit_indices]
    h_dirs = dirs[hit_indices]
    h_origins = origins[hit_indices]
    h_obj = obj_idx[hit_indices]

    hit_points = h_origins + h_dirs * h_t[..., np.newaxis]

    # Gather normals and colors
    Nh = len(hit_indices)
    normals = np.zeros((Nh, 3))
    base_colors = np.zeros((Nh, 3))
    reflectivities = np.zeros(Nh)

    for i, obj in enumerate(objects):
        mask = h_obj == i
        if not np.any(mask):
            continue
        pts = hit_points[mask]
        normals[mask] = obj.normal(pts)
        base_colors[mask] = obj.get_color(pts)
        reflectivities[mask] = obj.refl

    view_dirs = normalize(-h_dirs)

    # Direct illumination
    direct = shade(hit_points, normals, view_dirs, h_obj)

    # Object color modulated by lighting + ambient
    local_color = base_colors * (direct + AMBIENT)

    # Reflections
    refl_mask = reflectivities > 0.01
    if np.any(refl_mask) and depth < MAX_DEPTH:
        refl_dirs = h_dirs - 2 * np.sum(h_dirs * normals, axis=-1, keepdims=True) * normals
        refl_dirs = normalize(refl_dirs)
        refl_origins = hit_points + normals * EPS

        refl_indices = np.where(refl_mask)[0]
        refl_color = trace_rays(refl_origins[refl_indices], refl_dirs[refl_indices], depth + 1)

        r = reflectivities[refl_indices, np.newaxis]
        local_color[refl_indices] = local_color[refl_indices] * (1 - r) + refl_color * r

    result[hit_indices] = local_color
    return result

# ── Camera & render ────────────────────────────────────────────

def render():
    print(f"Rendering {W}x{H} with {len(lights)} coloured lights...")
    t0 = time.time()

    cam_pos = np.array([0.0, 1.5, -3.0])
    cam_target = np.array([0.0, 0.5, 5.0])
    cam_up = np.array([0.0, 1.0, 0.0])

    forward = normalize((cam_target - cam_pos).reshape(1, 3)).flatten()
    right = normalize(np.cross(forward, cam_up).reshape(1, 3)).flatten()
    up = np.cross(right, forward)

    fov = 60
    aspect = W / H
    scale = np.tan(np.radians(fov / 2))

    # Generate all pixel coordinates
    j, i = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    px = (2 * (i + 0.5) / W - 1) * aspect * scale
    py = (1 - 2 * (j + 0.5) / H) * scale

    dirs = px[..., np.newaxis] * right + py[..., np.newaxis] * up + forward
    dirs = normalize(dirs.reshape(-1, 3))
    origins = np.broadcast_to(cam_pos, dirs.shape).copy()

    # Process in chunks to manage memory
    chunk_size = 80000
    N = dirs.shape[0]
    pixels = np.zeros((N, 3))

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        pct = start / N * 100
        print(f"  {pct:.0f}% ...", end='\r')
        pixels[start:end] = trace_rays(origins[start:end], dirs[start:end])

    # Tone mapping (simple Reinhard) and gamma
    pixels = pixels / (1 + pixels)
    pixels = np.power(np.clip(pixels, 0, 1), 1 / 2.2)

    img_array = (pixels.reshape(H, W, 3) * 255).astype(np.uint8)
    img = Image.fromarray(img_array)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    return img

if __name__ == "__main__":
    img = render()
    img.save("/home/claude/scene.png")
    print("Saved scene.png")
