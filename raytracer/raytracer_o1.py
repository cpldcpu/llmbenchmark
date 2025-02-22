# ...existing code...

import math
from PIL import Image

class Vec3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def norm(self):
        return math.sqrt(self.dot(self))

    def normalize(self):
        n = self.norm()
        if n == 0:
            return Vec3(0, 0, 0)
        return Vec3(self.x / n, self.y / n, self.z / n)

class Sphere:
    def __init__(self, center, radius, color):
        self.center = center
        self.radius = radius
        self.color = color

    def intersect(self, origin, direction):
        oc = origin - self.center
        b = 2.0 * oc.dot(direction)
        c = oc.dot(oc) - self.radius * self.radius
        disc = b * b - 4 * c
        if disc < 0:
            return None
        dist = (-b - math.sqrt(disc)) / 2
        if dist < 0:
            dist = (-b + math.sqrt(disc)) / 2
        if dist < 0:
            return None
        return dist

# Scene setup
width, height = 800, 600
camera = Vec3(0, 0, -1)
objects = [
    Sphere(Vec3(-1.5, 0, 3), 1, (255, 0, 0)),
    Sphere(Vec3(1.0, 0.5, 4), 1.5, (0, 255, 0)),
    Sphere(Vec3(0.5, -1.0, 2.5), 0.5, (0, 0, 255)),
]
lights = [
    (Vec3(-10, 10, -5), (255, 255, 150)),
    (Vec3(10, -10, -5), (150, 255, 255)),
    (Vec3(0, 15, 0), (255, 150, 255)),
]

# Render
image = Image.new("RGB", (width, height))
pixels = image.load()
fov = math.pi / 3

for y in range(height):
    for x in range(width):
        # Normalized device coords
        i = (2 * (x + 0.5) / width - 1) * math.tan(fov / 2) * width / height
        j = (1 - 2 * (y + 0.5) / height) * math.tan(fov / 2)
        ray_dir = Vec3(i, j, 1).normalize()
        color = (0, 0, 0)
        closest = float('inf')
        hit_obj = None

        # Find nearest intersection
        for obj in objects:
            dist = obj.intersect(camera, ray_dir)
            if dist and dist < closest:
                closest = dist
                hit_obj = obj

        # Compute shading
        if hit_obj:
            hit_point = camera + ray_dir * closest
            normal = (hit_point - hit_obj.center).normalize()
            r, g, b = 0, 0, 0
            for light_pos, light_col in lights:
                to_light = (light_pos - hit_point).normalize()
                diff = max(normal.dot(to_light), 0)
                r += int(hit_obj.color[0] * diff * (light_col[0] / 255.0) / 3)
                g += int(hit_obj.color[1] * diff * (light_col[1] / 255.0) / 3)
                b += int(hit_obj.color[2] * diff * (light_col[2] / 255.0) / 3)
            color = (min(r, 255), min(g, 255), min(b, 255))

        pixels[x, y] = color

image.save("raytrace_result.png", "PNG")

# ...existing code...