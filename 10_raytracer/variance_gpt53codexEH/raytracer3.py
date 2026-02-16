from __future__ import annotations

import colorsys
import math
import struct
import time
import zlib
from dataclasses import dataclass

Vec3 = tuple[float, float, float]
EPSILON = 1e-4
MAX_BOUNCES = 2


def v_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v_mul(a: Vec3, scalar: float) -> Vec3:
    return (a[0] * scalar, a[1] * scalar, a[2] * scalar)


def v_hadamard(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] * b[0], a[1] * b[1], a[2] * b[2])


def v_dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_length(a: Vec3) -> float:
    return math.sqrt(v_dot(a, a))


def v_normalize(a: Vec3) -> Vec3:
    length = v_length(a)
    if length == 0.0:
        return (0.0, 0.0, 0.0)
    inv = 1.0 / length
    return (a[0] * inv, a[1] * inv, a[2] * inv)


def v_cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def v_reflect(direction: Vec3, normal: Vec3) -> Vec3:
    return v_sub(direction, v_mul(normal, 2.0 * v_dot(direction, normal)))


def v_mix(a: Vec3, b: Vec3, t: float) -> Vec3:
    s = 1.0 - t
    return (a[0] * s + b[0] * t, a[1] * s + b[1] * t, a[2] * s + b[2] * t)


def v_clamp01(a: Vec3) -> Vec3:
    return (
        min(1.0, max(0.0, a[0])),
        min(1.0, max(0.0, a[1])),
        min(1.0, max(0.0, a[2])),
    )


@dataclass(frozen=True)
class Material:
    color_a: Vec3
    color_b: Vec3 = (0.0, 0.0, 0.0)
    checker_scale: float = 0.0
    ambient: float = 0.03
    diffuse: float = 0.9
    specular: float = 0.35
    shininess: float = 64.0
    reflectivity: float = 0.0

    def color_at(self, point: Vec3) -> Vec3:
        if self.checker_scale <= 0.0:
            return self.color_a
        tile = int(math.floor(point[0] * self.checker_scale) + math.floor(point[2] * self.checker_scale))
        if tile & 1:
            return self.color_b
        return self.color_a


@dataclass(frozen=True)
class Hit:
    t: float
    point: Vec3
    normal: Vec3
    material: Material


@dataclass(frozen=True)
class Sphere:
    center: Vec3
    radius: float
    material: Material

    def intersect(self, origin: Vec3, direction: Vec3) -> Hit | None:
        oc = v_sub(origin, self.center)
        a = v_dot(direction, direction)
        b = 2.0 * v_dot(oc, direction)
        c = v_dot(oc, oc) - self.radius * self.radius
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return None

        sqrtd = math.sqrt(disc)
        inv = 0.5 / a
        t0 = (-b - sqrtd) * inv
        t1 = (-b + sqrtd) * inv
        t = t0 if t0 > EPSILON else t1
        if t <= EPSILON:
            return None

        point = v_add(origin, v_mul(direction, t))
        outward = v_normalize(v_sub(point, self.center))
        normal = outward if v_dot(outward, direction) < 0.0 else v_mul(outward, -1.0)
        return Hit(t=t, point=point, normal=normal, material=self.material)


@dataclass(frozen=True)
class Plane:
    point: Vec3
    normal: Vec3
    material: Material

    def intersect(self, origin: Vec3, direction: Vec3) -> Hit | None:
        denom = v_dot(self.normal, direction)
        if abs(denom) < 1e-6:
            return None
        t = v_dot(v_sub(self.point, origin), self.normal) / denom
        if t <= EPSILON:
            return None
        point = v_add(origin, v_mul(direction, t))
        normal = self.normal if denom < 0.0 else v_mul(self.normal, -1.0)
        return Hit(t=t, point=point, normal=normal, material=self.material)


@dataclass(frozen=True)
class PointLight:
    position: Vec3
    color: Vec3
    intensity: float


def closest_hit(origin: Vec3, direction: Vec3, objects: list[Sphere | Plane], t_limit: float = float("inf")) -> Hit | None:
    nearest: Hit | None = None
    nearest_t = t_limit
    for obj in objects:
        hit = obj.intersect(origin, direction)
        if hit is not None and hit.t < nearest_t:
            nearest = hit
            nearest_t = hit.t
    return nearest


def is_in_shadow(point: Vec3, normal: Vec3, light_dir: Vec3, light_dist: float, objects: list[Sphere | Plane]) -> bool:
    origin = v_add(point, v_mul(normal, EPSILON * 6.0))
    blocker = closest_hit(origin, light_dir, objects, light_dist - EPSILON)
    return blocker is not None


def background(direction: Vec3) -> Vec3:
    t = 0.5 * (direction[1] + 1.0)
    sky_low = (0.02, 0.02, 0.06)
    sky_high = (0.12, 0.09, 0.17)
    base = v_mix(sky_low, sky_high, t)
    haze = math.exp(-16.0 * abs(direction[1] - 0.08))
    tint = 0.07 * haze
    return (base[0] + tint * 0.7, base[1] + tint * 0.3, base[2] + tint)


def trace_ray(origin: Vec3, direction: Vec3, objects: list[Sphere | Plane], lights: list[PointLight], bounce: int) -> Vec3:
    hit = closest_hit(origin, direction, objects)
    if hit is None:
        return background(direction)

    material = hit.material
    base_color = material.color_at(hit.point)
    view_dir = v_mul(direction, -1.0)
    color = v_mul(base_color, material.ambient)

    for light in lights:
        to_light = v_sub(light.position, hit.point)
        light_dist_sq = v_dot(to_light, to_light)
        if light_dist_sq <= 0.0:
            continue
        light_dist = math.sqrt(light_dist_sq)
        light_dir = v_mul(to_light, 1.0 / light_dist)

        ndotl = v_dot(hit.normal, light_dir)
        if ndotl <= 0.0:
            continue
        if is_in_shadow(hit.point, hit.normal, light_dir, light_dist, objects):
            continue

        attenuation = light.intensity / (1.0 + 0.14 * light_dist + 0.06 * light_dist_sq)
        diff = material.diffuse * ndotl * attenuation
        diffuse = v_mul(v_hadamard(base_color, light.color), diff)
        color = v_add(color, diffuse)

        if material.specular > 0.0:
            reflect_dir = v_reflect(v_mul(light_dir, -1.0), hit.normal)
            spec_angle = max(0.0, v_dot(view_dir, reflect_dir))
            if spec_angle > 0.0:
                spec = material.specular * (spec_angle ** material.shininess) * attenuation
                color = v_add(color, v_mul(light.color, spec))

    if material.reflectivity > 0.0 and bounce < MAX_BOUNCES:
        reflect_dir = v_normalize(v_reflect(direction, hit.normal))
        reflect_origin = v_add(hit.point, v_mul(hit.normal, EPSILON * 8.0))
        reflected = trace_ray(reflect_origin, reflect_dir, objects, lights, bounce + 1)
        color = v_mix(color, reflected, material.reflectivity)

    return color


def tone_map(color: Vec3) -> Vec3:
    # Simple Reinhard tone map to keep highlights bright but bounded.
    mapped = (
        color[0] / (1.0 + color[0]),
        color[1] / (1.0 + color[1]),
        color[2] / (1.0 + color[2]),
    )
    return v_clamp01(mapped)


def to_srgb8(color: Vec3) -> tuple[int, int, int]:
    clamped = v_clamp01(color)
    return (
        int((clamped[0] ** (1.0 / 2.2)) * 255.0 + 0.5),
        int((clamped[1] ** (1.0 / 2.2)) * 255.0 + 0.5),
        int((clamped[2] ** (1.0 / 2.2)) * 255.0 + 0.5),
    )


def write_png(path: str, width: int, height: int, rgb_data: bytes) -> None:
    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    scanlines = bytearray()
    stride = width * 3
    for y in range(height):
        scanlines.append(0)  # filter type 0 (None)
        start = y * stride
        scanlines.extend(rgb_data[start : start + stride])

    compressed = zlib.compress(bytes(scanlines), level=9)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", compressed))
        f.write(chunk(b"IEND", b""))


def build_scene() -> tuple[list[Sphere | Plane], list[PointLight]]:
    floor = Plane(
        point=(0.0, -1.0, 0.0),
        normal=(0.0, 1.0, 0.0),
        material=Material(
            color_a=(0.86, 0.86, 0.88),
            color_b=(0.14, 0.14, 0.16),
            checker_scale=1.0,
            ambient=0.04,
            diffuse=0.95,
            specular=0.12,
            shininess=24.0,
            reflectivity=0.08,
        ),
    )

    objects: list[Sphere | Plane] = [
        floor,
        Sphere(
            center=(0.0, 0.55, 4.2),
            radius=1.35,
            material=Material(
                color_a=(0.95, 0.97, 1.0),
                ambient=0.02,
                diffuse=0.3,
                specular=0.95,
                shininess=180.0,
                reflectivity=0.72,
            ),
        ),
        Sphere(
            center=(-2.45, -0.15, 3.0),
            radius=0.85,
            material=Material(
                color_a=(1.0, 0.36, 0.25),
                ambient=0.03,
                diffuse=0.92,
                specular=0.28,
                shininess=42.0,
                reflectivity=0.1,
            ),
        ),
        Sphere(
            center=(2.1, -0.1, 3.05),
            radius=0.9,
            material=Material(
                color_a=(0.2, 0.4, 1.0),
                ambient=0.03,
                diffuse=0.9,
                specular=0.32,
                shininess=48.0,
                reflectivity=0.12,
            ),
        ),
        Sphere(
            center=(-0.75, -0.55, 2.1),
            radius=0.42,
            material=Material(
                color_a=(0.3, 1.0, 0.52),
                ambient=0.03,
                diffuse=0.9,
                specular=0.26,
                shininess=40.0,
                reflectivity=0.08,
            ),
        ),
        Sphere(
            center=(1.12, -0.58, 2.2),
            radius=0.38,
            material=Material(
                color_a=(0.95, 0.3, 1.0),
                ambient=0.03,
                diffuse=0.9,
                specular=0.3,
                shininess=52.0,
                reflectivity=0.08,
            ),
        ),
        Sphere(
            center=(0.03, -0.67, 1.45),
            radius=0.29,
            material=Material(
                color_a=(1.0, 0.78, 0.22),
                ambient=0.03,
                diffuse=0.88,
                specular=0.22,
                shininess=28.0,
                reflectivity=0.06,
            ),
        ),
        Sphere(
            center=(0.0, 1.85, 7.2),
            radius=1.2,
            material=Material(
                color_a=(0.4, 0.92, 1.0),
                ambient=0.03,
                diffuse=0.85,
                specular=0.4,
                shininess=72.0,
                reflectivity=0.15,
            ),
        ),
    ]

    lights: list[PointLight] = []
    upper_count = 10
    lower_count = 6
    for i in range(upper_count):
        angle = 2.0 * math.pi * i / upper_count
        hue = i / upper_count
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        lights.append(
            PointLight(
                position=(
                    math.cos(angle) * 5.6,
                    3.0 + 0.35 * math.sin(angle * 2.0),
                    4.3 + math.sin(angle) * 2.8,
                ),
                color=(r, g, b),
                intensity=2.2,
            )
        )

    for i in range(lower_count):
        angle = 2.0 * math.pi * i / lower_count
        hue = (i / lower_count + 0.08) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.92, 1.0)
        lights.append(
            PointLight(
                position=(
                    math.cos(angle) * 3.1,
                    0.7 + 0.25 * math.cos(angle * 3.0),
                    3.8 + math.sin(angle) * 1.8,
                ),
                color=(r, g, b),
                intensity=1.5,
            )
        )

    return objects, lights


def render(width: int, height: int) -> bytes:
    objects, lights = build_scene()
    camera_pos = (0.0, 1.1, -8.6)
    camera_target = (0.0, 0.2, 4.0)
    world_up = (0.0, 1.0, 0.0)

    forward = v_normalize(v_sub(camera_target, camera_pos))
    right = v_normalize(v_cross(forward, world_up))
    up = v_cross(right, forward)

    fov = math.radians(56.0)
    aspect = width / height
    scale = math.tan(fov * 0.5)

    pixels = bytearray(width * height * 3)
    offset = 0
    for y in range(height):
        py = (1.0 - (2.0 * (y + 0.5) / height)) * scale
        for x in range(width):
            px = ((2.0 * (x + 0.5) / width) - 1.0) * aspect * scale
            ray_dir = v_normalize(
                v_add(
                    forward,
                    v_add(v_mul(right, px), v_mul(up, py)),
                )
            )
            color = trace_ray(camera_pos, ray_dir, objects, lights, 0)
            mapped = tone_map(color)
            r, g, b = to_srgb8(mapped)
            pixels[offset] = r
            pixels[offset + 1] = g
            pixels[offset + 2] = b
            offset += 3

        if y % 25 == 0:
            print(f"Rendered row {y + 1}/{height}")

    return bytes(pixels)


def main() -> None:
    width, height = 800, 600
    output_path = "raytrace.png"
    print(f"Rendering {width}x{height} scene with many colorful lights...")
    start = time.time()
    rgb = render(width, height)
    write_png(output_path, width, height, rgb)
    elapsed = time.time() - start
    print(f"Wrote {output_path} in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
