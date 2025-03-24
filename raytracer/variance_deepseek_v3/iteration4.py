from PIL import Image
import math

# Constants
WIDTH = 800
HEIGHT = 600
ASPECT_RATIO = WIDTH / HEIGHT
FOV = 45
INF = float('inf')
MAX_DEPTH = 3

# Vec3 class for vector operations
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

    def length(self):
        return math.sqrt(self.dot(self))

    def normalize(self):
        length = self.length()
        return Vec3(self.x / length, self.y / length, self.z / length)

# Ray class
class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

# Sphere class
class Sphere:
    def __init__(self, center, radius, color):
        self.center = center
        self.radius = radius
        self.color = color

    def intersect(self, ray):
        oc = ray.origin - self.center
        a = ray.direction.dot(ray.direction)
        b = 2.0 * oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return INF
        else:
            return (-b - math.sqrt(discriminant)) / (2.0 * a)

# Function to calculate pixel color
def trace(ray, objects, depth):
    if depth >= MAX_DEPTH:
        return Vec3(0, 0, 0)  # Black

    min_distance = INF
    hit_sphere = None

    for sphere in objects:
        distance = sphere.intersect(ray)
        if distance < min_distance:
            min_distance = distance
            hit_sphere = sphere

    if hit_sphere is None:
        return Vec3(0, 0, 0)  # Background color (black)

    hit_point = ray.origin + ray.direction * min_distance
    normal = (hit_point - hit_sphere.center).normalize()

    # Simple diffuse shading
    light_dir = Vec3(1, -1, -1).normalize()
    light_intensity = max(0, normal.dot(light_dir))

    # Add some ambient light
    ambient = Vec3(0.1, 0.1, 0.1)
    color = hit_sphere.color * light_intensity + ambient

    return color

# Main rendering function
def render(objects):
    image = Image.new("RGB", (WIDTH, HEIGHT))
    pixels = image.load()

    for y in range(HEIGHT):
        for x in range(WIDTH):
            # Normalize pixel coordinates to [-1, 1]
            nx = (x / WIDTH) * 2 - 1
            ny = 1 - (y / HEIGHT) * 2
            nx *= ASPECT_RATIO * math.tan(math.radians(FOV / 2))

            direction = Vec3(nx, ny, -1).normalize()
            ray = Ray(Vec3(0, 0, 0), direction)
            color = trace(ray, objects, 0)

            # Clamp and convert to RGB
            r = min(255, int(color.x * 255))
            g = min(255, int(color.y * 255))
            b = min(255, int(color.z * 255))
            pixels[x, y] = (r, g, b)

    return image

# Scene setup
objects = [
    Sphere(Vec3(0, 0, -5), 1, Vec3(1, 0, 0)),  # Red sphere
    Sphere(Vec3(2, 1, -7), 1.5, Vec3(0, 1, 0)),  # Green sphere
    Sphere(Vec3(-2, 0, -6), 1, Vec3(0, 0, 1)),  # Blue sphere
]

# Render and save image
image = render(objects)
image.save("raytraced_scene4.png")
print("Image saved as raytrace.png")
