import math
import struct
import time
import zlib
from dataclasses import dataclass


EPSILON = 1e-4
MAX_DEPTH = 3


class Vec3:
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, value: float) -> "Vec3":
        return Vec3(self.x * value, self.y * value, self.z * value)

    def __rmul__(self, value: float) -> "Vec3":
        return self * value

    def __truediv__(self, value: float) -> "Vec3":
        inv = 1.0 / value
        return Vec3(self.x * inv, self.y * inv, self.z * inv)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length_squared(self) -> float:
        return self.dot(self)

    def length(self) -> float:
        return math.sqrt(self.length_squared())

    def normalized(self) -> "Vec3":
        length = self.length()
        if length == 0.0:
            return Vec3(0.0, 0.0, 0.0)
        return self / length

    def hadamard(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x * other.x, self.y * other.y, self.z * other.z)

    def clamp(self, low: float, high: float) -> "Vec3":
        return Vec3(
            max(low, min(high, self.x)),
            max(low, min(high, self.y)),
            max(low, min(high, self.z)),
        )


@dataclass(slots=True)
class Material:
    color: Vec3
    specular: float
    shininess: float
    reflectivity: float
    emission: Vec3


@dataclass(slots=True)
class Sphere:
    center: Vec3
    radius: float
    material: Material

    def intersect(self, ray_origin: Vec3, ray_dir: Vec3):
        oc = ray_origin - self.center
        b = oc.dot(ray_dir)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = b * b - c
        if discriminant < 0.0:
            return None
        sqrt_disc = math.sqrt(discriminant)
        t = -b - sqrt_disc
        if t <= EPSILON:
            t = -b + sqrt_disc
            if t <= EPSILON:
                return None
        point = ray_origin + ray_dir * t
        normal = (point - self.center) / self.radius
        return t, point, normal, self.material


@dataclass(slots=True)
class Plane:
    point: Vec3
    normal: Vec3
    material_a: Material
    material_b: Material
    checker_scale: float

    def intersect(self, ray_origin: Vec3, ray_dir: Vec3):
        denom = self.normal.dot(ray_dir)
        if abs(denom) < 1e-6:
            return None
        t = (self.point - ray_origin).dot(self.normal) / denom
        if t <= EPSILON:
            return None
        hit_point = ray_origin + ray_dir * t
        grid_x = math.floor(hit_point.x * self.checker_scale)
        grid_z = math.floor(hit_point.z * self.checker_scale)
        material = self.material_a if (grid_x + grid_z) % 2 == 0 else self.material_b
        return t, hit_point, self.normal, material


@dataclass(slots=True)
class PointLight:
    position: Vec3
    color: Vec3
    intensity: float


def reflect(direction: Vec3, normal: Vec3) -> Vec3:
    return direction - normal * (2.0 * direction.dot(normal))


def hsv_to_rgb(h: float, s: float, v: float) -> Vec3:
    h = h % 1.0
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i %= 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return Vec3(r, g, b)


def tonemap(color: Vec3) -> Vec3:
    mapped = Vec3(
        color.x / (1.0 + color.x),
        color.y / (1.0 + color.y),
        color.z / (1.0 + color.z),
    )
    gamma = 1.0 / 2.2
    return Vec3(
        mapped.x ** gamma,
        mapped.y ** gamma,
        mapped.z ** gamma,
    )


def write_png_rgb8(path: str, width: int, height: int, rgb_data: bytes):
    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        return length + chunk_type + data + crc

    png_sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)

    row_size = width * 3
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        start = y * row_size
        raw.extend(rgb_data[start : start + row_size])

    compressed = zlib.compress(bytes(raw), level=9)
    png = (
        png_sig
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )

    with open(path, "wb") as f:
        f.write(png)


class Scene:
    def __init__(self):
        self.objects = []
        self.lights = []
        self._build_scene()

    def _build_scene(self):
        matte_dark = Material(Vec3(0.18, 0.18, 0.2), 0.15, 24.0, 0.02, Vec3(0.0, 0.0, 0.0))
        matte_light = Material(Vec3(0.78, 0.78, 0.82), 0.2, 28.0, 0.04, Vec3(0.0, 0.0, 0.0))
        self.objects.append(
            Plane(
                point=Vec3(0.0, -1.0, 0.0),
                normal=Vec3(0.0, 1.0, 0.0),
                material_a=matte_dark,
                material_b=matte_light,
                checker_scale=0.7,
            )
        )

        self.objects.extend(
            [
                Sphere(
                    Vec3(0.0, 0.15, 5.8),
                    1.15,
                    Material(Vec3(0.96, 0.96, 1.0), 0.95, 140.0, 0.55, Vec3(0.0, 0.0, 0.0)),
                ),
                Sphere(
                    Vec3(-2.7, -0.15, 4.4),
                    0.85,
                    Material(Vec3(1.0, 0.35, 0.28), 0.65, 80.0, 0.25, Vec3(0.0, 0.0, 0.0)),
                ),
                Sphere(
                    Vec3(2.4, -0.2, 4.8),
                    0.8,
                    Material(Vec3(0.3, 0.55, 1.0), 0.75, 100.0, 0.3, Vec3(0.0, 0.0, 0.0)),
                ),
                Sphere(
                    Vec3(-0.75, -0.45, 3.7),
                    0.52,
                    Material(Vec3(0.42, 1.0, 0.65), 0.55, 64.0, 0.14, Vec3(0.0, 0.0, 0.0)),
                ),
                Sphere(
                    Vec3(1.05, -0.55, 3.9),
                    0.45,
                    Material(Vec3(1.0, 0.88, 0.35), 0.5, 52.0, 0.12, Vec3(0.0, 0.0, 0.0)),
                ),
            ]
        )

        arc_count = 18
        for i in range(arc_count):
            a = i * 0.46
            radius = 3.6 + 0.65 * math.sin(i * 1.1)
            x = radius * math.cos(a)
            z = 6.1 + radius * math.sin(a)
            y = -0.35 + 0.6 * math.sin(i * 0.8)
            color = hsv_to_rgb(i / arc_count, 0.78, 0.95)
            self.objects.append(
                Sphere(
                    Vec3(x, y, z),
                    0.25,
                    Material(color, 0.55, 54.0, 0.1, Vec3(0.0, 0.0, 0.0)),
                )
            )

        light_count = 22
        for i in range(light_count):
            angle = (2.0 * math.pi * i) / light_count
            ring_radius = 7.0 + 1.2 * math.sin(i * 0.5)
            pos = Vec3(
                ring_radius * math.cos(angle),
                3.6 + 1.35 * math.sin(i * 1.27),
                6.0 + ring_radius * math.sin(angle),
            )
            color = hsv_to_rgb(i / light_count, 0.95, 1.0)
            self.lights.append(PointLight(pos, color, 26.0))
            self.objects.append(
                Sphere(
                    pos,
                    0.18,
                    Material(Vec3(1.0, 1.0, 1.0), 0.0, 1.0, 0.0, color * 10.0),
                )
            )

        self.lights.append(PointLight(Vec3(0.0, 5.8, -3.0), Vec3(1.0, 0.96, 0.9), 18.0))
        self.objects.append(
            Sphere(
                Vec3(0.0, 5.8, -3.0),
                0.32,
                Material(Vec3(1.0, 1.0, 1.0), 0.0, 1.0, 0.0, Vec3(1.0, 0.96, 0.9) * 8.0),
            )
        )

    def intersect(self, ray_origin: Vec3, ray_dir: Vec3):
        closest = None
        min_t = float("inf")
        for obj in self.objects:
            hit = obj.intersect(ray_origin, ray_dir)
            if hit is not None and hit[0] < min_t:
                min_t = hit[0]
                closest = hit
        return closest

    def is_shadowed(self, point: Vec3, light_dir: Vec3, max_dist: float) -> bool:
        shadow_origin = point + light_dir * EPSILON
        for obj in self.objects:
            hit = obj.intersect(shadow_origin, light_dir)
            if hit is not None and hit[0] < max_dist:
                return True
        return False


def sky_color(direction: Vec3) -> Vec3:
    t = 0.5 * (direction.y + 1.0)
    high = Vec3(0.08, 0.12, 0.25)
    low = Vec3(0.01, 0.01, 0.03)
    return low * (1.0 - t) + high * t


def trace(scene: Scene, ray_origin: Vec3, ray_dir: Vec3, depth: int) -> Vec3:
    hit = scene.intersect(ray_origin, ray_dir)
    if hit is None:
        return sky_color(ray_dir)

    _, point, normal, material = hit
    view_dir = (ray_dir * -1.0).normalized()

    color = material.emission + material.color * 0.02
    for light in scene.lights:
        to_light = light.position - point
        dist_sq = to_light.length_squared()
        dist = math.sqrt(dist_sq)
        light_dir = to_light / dist

        ndotl = normal.dot(light_dir)
        if ndotl <= 0.0:
            continue

        if scene.is_shadowed(point + normal * EPSILON, light_dir, dist - 0.25):
            continue

        attenuation = light.intensity / (1.0 + 0.18 * dist + 0.05 * dist_sq)
        light_col = light.color * attenuation

        diffuse = material.color.hadamard(light_col) * ndotl
        reflection = reflect(light_dir * -1.0, normal).normalized()
        specular_term = max(0.0, view_dir.dot(reflection)) ** material.shininess
        specular = light_col * (material.specular * specular_term)
        color = color + diffuse + specular

    if depth < MAX_DEPTH and material.reflectivity > 0.0:
        reflected_dir = reflect(ray_dir, normal).normalized()
        reflected_origin = point + normal * EPSILON
        reflected_color = trace(scene, reflected_origin, reflected_dir, depth + 1)
        color = color * (1.0 - material.reflectivity) + reflected_color * material.reflectivity

    return color


def render(width: int = 800, height: int = 600, output_path: str = "raytrace_800x600.png"):
    scene = Scene()

    camera_pos = Vec3(0.0, 1.0, -8.5)
    look_at = Vec3(0.0, 0.25, 5.5)
    world_up = Vec3(0.0, 1.0, 0.0)

    forward = (look_at - camera_pos).normalized()
    right = forward.cross(world_up).normalized()
    up = right.cross(forward).normalized()

    fov = math.radians(57.0)
    scale = math.tan(fov * 0.5)
    aspect = width / height

    pixels = bytearray(width * height * 3)

    start = time.time()
    for y in range(height):
        py = (1.0 - 2.0 * ((y + 0.5) / height)) * scale
        for x in range(width):
            px = (2.0 * ((x + 0.5) / width) - 1.0) * aspect * scale
            ray_dir = (forward + right * px + up * py).normalized()
            color = trace(scene, camera_pos, ray_dir, 0)
            color = tonemap(color).clamp(0.0, 1.0)

            idx = (y * width + x) * 3
            pixels[idx] = int(color.x * 255.0 + 0.5)
            pixels[idx + 1] = int(color.y * 255.0 + 0.5)
            pixels[idx + 2] = int(color.z * 255.0 + 0.5)

        if y % 40 == 0 or y == height - 1:
            print(f"Scanline {y + 1}/{height}")

    write_png_rgb8(output_path, width, height, bytes(pixels))
    elapsed = time.time() - start
    print(f"Rendered {width}x{height} to {output_path} in {elapsed:.2f}s")


if __name__ == "__main__":
    render()
