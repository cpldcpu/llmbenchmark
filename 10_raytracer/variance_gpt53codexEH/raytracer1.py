import colorsys
import math
import time

import numpy as np
from PIL import Image


WIDTH = 800
HEIGHT = 600
FOV_DEG = 55.0
MAX_BOUNCES = 1
EPS = 1e-4
OUTPUT_FILE = "raytrace_scene_800x600.png"


def normalize(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, 1e-8)


def dot(a, b):
    return np.sum(a * b, axis=1)


def reflect(v, n):
    return v - 2.0 * dot(v, n)[:, None] * n


def hsv_rgb(h, s, v):
    return np.array(colorsys.hsv_to_rgb(h, s, v), dtype=np.float32)


def build_scene():
    spheres = []
    lights = []

    def add_sphere(center, radius, albedo, specular, shininess, reflectivity, emission):
        spheres.append(
            {
                "center": np.array(center, dtype=np.float32),
                "radius": float(radius),
                "albedo": np.array(albedo, dtype=np.float32),
                "specular": float(specular),
                "shininess": float(shininess),
                "reflectivity": float(reflectivity),
                "emission": np.array(emission, dtype=np.float32),
            }
        )

    # Main geometry
    add_sphere((0.0, 0.3, 0.8), 1.3, (0.94, 0.95, 0.98), 1.0, 180.0, 0.7, (0.0, 0.0, 0.0))
    add_sphere((-2.4, -0.2, -1.6), 0.9, (0.92, 0.28, 0.20), 0.5, 48.0, 0.12, (0.0, 0.0, 0.0))
    add_sphere((2.25, -0.25, -1.2), 0.95, (0.15, 0.85, 0.72), 0.9, 110.0, 0.35, (0.0, 0.0, 0.0))
    add_sphere((1.0, -0.5, 2.8), 0.6, (0.58, 0.3, 0.92), 0.3, 36.0, 0.08, (0.0, 0.0, 0.0))
    add_sphere((-1.3, -0.55, 2.2), 0.45, (0.95, 0.86, 0.22), 0.4, 30.0, 0.06, (0.0, 0.0, 0.0))
    add_sphere((3.15, 0.7, 1.7), 0.5, (0.25, 0.46, 0.95), 0.7, 90.0, 0.24, (0.0, 0.0, 0.0))
    add_sphere((-3.1, 0.55, 1.55), 0.48, (0.88, 0.25, 0.84), 0.6, 80.0, 0.2, (0.0, 0.0, 0.0))

    # Colorful ring of point lights with visible emissive bulbs
    ring_count = 14
    for i in range(ring_count):
        a = (2.0 * math.pi * i) / ring_count
        pos = np.array(
            [
                5.8 * math.cos(a),
                1.8 + 0.7 * math.sin(2.0 * a),
                2.0 + 5.8 * math.sin(a),
            ],
            dtype=np.float32,
        )
        color = hsv_rgb(i / ring_count, 0.95, 1.0)
        lights.append(
            {
                "pos": pos,
                "color": color,
                "intensity": 34.0,
                "shadow": (i % 2 == 0),
            }
        )
        add_sphere(pos, 0.14, color * 0.8, 0.0, 1.0, 0.0, color * 8.0)

    # Additional overhead lights
    top_count = 6
    for i in range(top_count):
        a = (2.0 * math.pi * i) / top_count + 0.35
        pos = np.array(
            [
                3.0 * math.cos(a),
                4.3 + 0.3 * math.sin(3.0 * a),
                1.0 + 3.0 * math.sin(a),
            ],
            dtype=np.float32,
        )
        color = hsv_rgb((i / top_count + 0.12) % 1.0, 0.8, 1.0)
        lights.append(
            {
                "pos": pos,
                "color": color,
                "intensity": 26.0,
                "shadow": (i % 3 == 0),
            }
        )
        add_sphere(pos, 0.12, color * 0.7, 0.0, 1.0, 0.0, color * 7.0)

    centers = np.array([s["center"] for s in spheres], dtype=np.float32)
    radii = np.array([s["radius"] for s in spheres], dtype=np.float32)
    albedos = np.array([s["albedo"] for s in spheres], dtype=np.float32)
    speculars = np.array([s["specular"] for s in spheres], dtype=np.float32)
    shininesses = np.array([s["shininess"] for s in spheres], dtype=np.float32)
    reflectivities = np.array([s["reflectivity"] for s in spheres], dtype=np.float32)
    emissions = np.array([s["emission"] for s in spheres], dtype=np.float32)
    emissive_mask = np.linalg.norm(emissions, axis=1) > 1e-6
    occluder_indices = np.where(~emissive_mask)[0]

    return {
        "centers": centers,
        "radii": radii,
        "albedos": albedos,
        "speculars": speculars,
        "shininesses": shininesses,
        "reflectivities": reflectivities,
        "emissions": emissions,
        "lights": lights,
        "occluder_indices": occluder_indices,
        "plane_y": -1.0,
    }


def intersect_scene(origins, dirs, scene):
    n = origins.shape[0]
    centers = scene["centers"]
    radii = scene["radii"]
    plane_y = scene["plane_y"]

    nearest_t = np.full(n, np.inf, dtype=np.float32)
    hit_type = np.full(n, -1, dtype=np.int32)  # -1 miss, -2 plane, >=0 sphere index

    # Sphere intersections
    for i in range(centers.shape[0]):
        oc = origins - centers[i]
        b = dot(oc, dirs)
        c = dot(oc, oc) - radii[i] * radii[i]
        disc = b * b - c
        valid = disc > 0.0
        if not np.any(valid):
            continue

        idx = np.where(valid)[0]
        sqrt_disc = np.sqrt(disc[idx])
        t0 = -b[idx] - sqrt_disc
        t1 = -b[idx] + sqrt_disc
        t = np.where(t0 > EPS, t0, np.where(t1 > EPS, t1, np.inf))

        better = t < nearest_t[idx]
        if np.any(better):
            upd = idx[better]
            nearest_t[upd] = t[better]
            hit_type[upd] = i

    # Floor plane
    dy = dirs[:, 1]
    valid_plane = np.abs(dy) > 1e-7
    t_plane = np.where(valid_plane, (plane_y - origins[:, 1]) / dy, np.inf)
    valid_plane &= t_plane > EPS
    idx = np.where(valid_plane)[0]
    if idx.size:
        better = t_plane[idx] < nearest_t[idx]
        if np.any(better):
            upd = idx[better]
            nearest_t[upd] = t_plane[upd]
            hit_type[upd] = -2

    hit_mask = hit_type != -1
    hit_pos = origins + dirs * nearest_t[:, None]

    normals = np.zeros((n, 3), dtype=np.float32)
    albedo = np.zeros((n, 3), dtype=np.float32)
    specular = np.zeros(n, dtype=np.float32)
    shininess = np.zeros(n, dtype=np.float32)
    reflectivity = np.zeros(n, dtype=np.float32)
    emission = np.zeros((n, 3), dtype=np.float32)

    sphere_mask = hit_type >= 0
    if np.any(sphere_mask):
        s_idx = hit_type[sphere_mask]
        p = hit_pos[sphere_mask]
        normals[sphere_mask] = (p - centers[s_idx]) / radii[s_idx][:, None]
        albedo[sphere_mask] = scene["albedos"][s_idx]
        specular[sphere_mask] = scene["speculars"][s_idx]
        shininess[sphere_mask] = scene["shininesses"][s_idx]
        reflectivity[sphere_mask] = scene["reflectivities"][s_idx]
        emission[sphere_mask] = scene["emissions"][s_idx]

    plane_mask = hit_type == -2
    if np.any(plane_mask):
        normals[plane_mask] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        px = hit_pos[plane_mask, 0]
        pz = hit_pos[plane_mask, 2]

        checker = (np.floor(px * 0.7) + np.floor(pz * 0.7)).astype(np.int32) & 1
        even = np.array([0.89, 0.90, 0.93], dtype=np.float32)
        odd = np.array([0.11, 0.12, 0.14], dtype=np.float32)
        base = np.where(checker[:, None] == 0, even, odd)

        tint = 0.11 * np.stack(
            [
                np.sin(px * 0.8) + 1.0,
                np.sin(pz * 0.9 + 2.0) + 1.0,
                np.sin((px + pz) * 0.6 + 4.0) + 1.0,
            ],
            axis=1,
        ).astype(np.float32)

        albedo[plane_mask] = np.clip(base * (0.84 + tint), 0.03, 1.0)
        specular[plane_mask] = 0.18
        shininess[plane_mask] = 28.0
        reflectivity[plane_mask] = 0.06

    return {
        "hit_mask": hit_mask,
        "hit_pos": hit_pos,
        "normals": normals,
        "albedo": albedo,
        "specular": specular,
        "shininess": shininess,
        "reflectivity": reflectivity,
        "emission": emission,
    }


def is_occluded(origins, dirs, max_dist, scene):
    n = origins.shape[0]
    blocked = np.zeros(n, dtype=bool)

    centers = scene["centers"]
    radii = scene["radii"]
    plane_y = scene["plane_y"]

    for i in scene["occluder_indices"]:
        active = ~blocked
        if not np.any(active):
            break

        idx = np.where(active)[0]
        o = origins[idx] - centers[i]
        d = dirs[idx]

        b = dot(o, d)
        c = dot(o, o) - radii[i] * radii[i]
        disc = b * b - c
        valid = disc > 0.0
        if not np.any(valid):
            continue

        idv = idx[valid]
        sqrt_disc = np.sqrt(disc[valid])
        t0 = -b[valid] - sqrt_disc
        t1 = -b[valid] + sqrt_disc
        t = np.where(t0 > EPS, t0, np.where(t1 > EPS, t1, np.inf))
        hit = (t < (max_dist[idv] - EPS)) & np.isfinite(t)
        if np.any(hit):
            blocked[idv[hit]] = True

    # Plane as occluder
    active = ~blocked
    if np.any(active):
        idx = np.where(active)[0]
        dy = dirs[idx, 1]
        valid = np.abs(dy) > 1e-7
        if np.any(valid):
            idv = idx[valid]
            t = (plane_y - origins[idv, 1]) / dy[valid]
            hit = (t > EPS) & (t < (max_dist[idv] - EPS))
            if np.any(hit):
                blocked[idv[hit]] = True

    return blocked


def background(dirs):
    t = np.clip(0.5 * (dirs[:, 1] + 1.0), 0.0, 1.0)
    c0 = np.array([0.02, 0.015, 0.05], dtype=np.float32)
    c1 = np.array([0.02, 0.16, 0.30], dtype=np.float32)
    base = c0[None, :] * (1.0 - t[:, None]) + c1[None, :] * t[:, None]

    az = np.arctan2(dirs[:, 2], dirs[:, 0])
    swirl = np.stack(
        [
            0.05 * (np.sin(2.0 * az + 0.4) + 1.0),
            0.03 * (np.sin(3.0 * az + 2.0) + 1.0),
            0.05 * (np.sin(4.0 * az + 4.0) + 1.0),
        ],
        axis=1,
    ).astype(np.float32)
    return base + swirl


def trace_rays(origins, dirs, scene, depth):
    hit = intersect_scene(origins, dirs, scene)
    hit_mask = hit["hit_mask"]
    color = background(dirs)

    if not np.any(hit_mask):
        return color

    h_idx = np.where(hit_mask)[0]
    hp = hit["hit_pos"][hit_mask]
    nrm = hit["normals"][hit_mask]
    alb = hit["albedo"][hit_mask]
    spe = hit["specular"][hit_mask]
    shi = hit["shininess"][hit_mask]
    ref = hit["reflectivity"][hit_mask]
    emi = hit["emission"][hit_mask]
    view = normalize(-dirs[hit_mask])

    surf = alb * 0.03 + emi
    non_emissive = np.linalg.norm(emi, axis=1) < 1e-5

    if np.any(non_emissive):
        l_idx = np.where(non_emissive)[0]
        l_hp = hp[l_idx]
        l_n = nrm[l_idx]
        l_a = alb[l_idx]
        l_s = spe[l_idx]
        l_sh = shi[l_idx]
        l_v = view[l_idx]

        for light in scene["lights"]:
            lvec = light["pos"][None, :] - l_hp
            dist2 = dot(lvec, lvec) + 1e-6
            inv_dist = 1.0 / np.sqrt(dist2)
            ldir = lvec * inv_dist[:, None]
            ndotl = dot(l_n, ldir)

            lit = ndotl > 0.02
            if not np.any(lit):
                continue

            lit_idx = np.where(lit)[0]
            vis_idx = lit_idx

            if light["shadow"]:
                so = l_hp[lit_idx] + l_n[lit_idx] * EPS
                sd = ldir[lit_idx]
                md = 1.0 / inv_dist[lit_idx]
                occluded = is_occluded(so, sd, md, scene)
                visible = ~occluded
                if not np.any(visible):
                    continue
                vis_idx = lit_idx[visible]

            nd = ndotl[vis_idx]
            att = light["intensity"] / (1.0 + 0.09 * dist2[vis_idx])
            lcol = light["color"][None, :] * att[:, None]

            surf[l_idx[vis_idx]] += l_a[vis_idx] * nd[:, None] * lcol

            h = ldir[vis_idx] + l_v[vis_idx]
            h = h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-8)
            ndoth = np.clip(dot(l_n[vis_idx], h), 0.0, 1.0)
            spec = (ndoth ** l_sh[vis_idx]) * l_s[vis_idx]
            surf[l_idx[vis_idx]] += spec[:, None] * lcol

    color[h_idx] = surf

    if depth > 0:
        refl_local = np.where(ref > 1e-3)[0]
        if refl_local.size:
            w_idx = h_idx[refl_local]
            r_dir = reflect(dirs[w_idx], nrm[refl_local])
            r_dir = normalize(r_dir)
            r_org = hp[refl_local] + nrm[refl_local] * EPS
            r_col = trace_rays(r_org, r_dir, scene, depth - 1)
            k = ref[refl_local][:, None]
            color[w_idx] = color[w_idx] * (1.0 - k) + r_col * k

    return color


def build_camera_rays(width, height, fov_deg):
    cam_pos = np.array([0.0, 1.0, -8.0], dtype=np.float32)
    cam_target = np.array([0.0, 0.15, 1.2], dtype=np.float32)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    forward = cam_target - cam_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    aspect = width / float(height)
    scale = math.tan(math.radians(fov_deg) * 0.5)

    xs = (np.arange(width, dtype=np.float32) + 0.5) / width
    ys = (np.arange(height, dtype=np.float32) + 0.5) / height
    px = (2.0 * xs - 1.0) * aspect * scale
    py = (1.0 - 2.0 * ys) * scale
    gx, gy = np.meshgrid(px, py)

    dirs = (
        forward[None, None, :]
        + gx[:, :, None] * right[None, None, :]
        + gy[:, :, None] * up[None, None, :]
    )
    dirs = dirs.reshape(-1, 3).astype(np.float32)
    dirs = normalize(dirs)

    origins = np.repeat(cam_pos[None, :], width * height, axis=0).astype(np.float32)
    return origins, dirs


def main():
    start = time.time()
    scene = build_scene()
    origins, dirs = build_camera_rays(WIDTH, HEIGHT, FOV_DEG)

    print(f"Rendering {WIDTH}x{HEIGHT} with {len(scene['lights'])} colorful lights...")
    linear = trace_rays(origins, dirs, scene, MAX_BOUNCES)

    # Tonemap and gamma
    linear = np.clip(linear, 0.0, None)
    tonemapped = linear / (1.0 + linear)
    srgb = np.clip(tonemapped ** (1.0 / 2.2), 0.0, 1.0)
    img = (srgb.reshape(HEIGHT, WIDTH, 3) * 255.0 + 0.5).astype(np.uint8)

    Image.fromarray(img, mode="RGB").save(OUTPUT_FILE)
    elapsed = time.time() - start
    print(f"Saved {OUTPUT_FILE}")
    print(f"Render time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
