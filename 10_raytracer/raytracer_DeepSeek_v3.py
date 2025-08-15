from PIL import Image
import math

# Vector math utilities
class Vector:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar, self.z * scalar)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def norm(self):
        return math.sqrt(self.dot(self))

    def normalize(self):
        norm = self.norm()
        return Vector(self.x / norm, self.y / norm, self.z / norm)

# Ray class
class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction.normalize()

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
            return None
        else:
            t = (-b - math.sqrt(discriminant)) / (2.0 * a)
            if t > 0:
                return t
            else:
                return None

# Light source class
class Light:
    def __init__(self, position, color):
        self.position = position
        self.color = color

# Scene setup
width = 800
height = 600
aspect_ratio = width / height
fov = 90
camera = Vector(0, 0, -5)
scene = [
    Sphere(Vector(0, 0, 0), 1, (255, 0, 0)),  # Red sphere
    Sphere(Vector(-2, 0, 0), 1, (0, 255, 0)), # Green sphere
    Sphere(Vector(2, 0, 0), 1, (0, 0, 255)),  # Blue sphere
]
lights = [
    Light(Vector(-5, 5, -5), (255, 255, 0)),  # Yellow light
    Light(Vector(5, 5, -5), (0, 255, 255)),   # Cyan light
]

# Raytracing function
def trace(ray):
    closest_t = float('inf')
    closest_sphere = None

    for sphere in scene:
        t = sphere.intersect(ray)
        if t and t < closest_t:
            closest_t = t
            closest_sphere = sphere

    if closest_sphere:
        hit_point = ray.origin + ray.direction * closest_t
        normal = (hit_point - closest_sphere.center).normalize()
        color = [0, 0, 0]

        for light in lights:
            light_dir = (light.position - hit_point).normalize()
            intensity = max(0, normal.dot(light_dir))
            color[0] += closest_sphere.color[0] * light.color[0] * intensity / 255
            color[1] += closest_sphere.color[1] * light.color[1] * intensity / 255
            color[2] += closest_sphere.color[2] * light.color[2] * intensity / 255

        return tuple(min(int(c), 255) for c in color)
    else:
        return (0, 0, 0)  # Background color

# Render the scene
image = Image.new("RGB", (width, height))
pixels = image.load()

for y in range(height):
    for x in range(width):
        # Convert pixel coordinates to normalized device coordinates
        ndc_x = (x + 0.5) / width * 2 - 1
        ndc_y = 1 - (y + 0.5) / height * 2
        ndc_x *= aspect_ratio * math.tan(math.radians(fov / 2))
        ndc_y *= math.tan(math.radians(fov / 2))

        # Create ray
        ray = Ray(camera, Vector(ndc_x, ndc_y, 1) - camera)
        color = trace(ray)
        pixels[x, y] = color

# Save the image
image.save("raytraced_scene.png")
image.show()