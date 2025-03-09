from PIL import Image
import math

# Define the scene
class Scene:
    def __init__(self):
        self.objects = []
        self.lights = []

    def add_object(self, obj):
        self.objects.append(obj)

    def add_light(self, light):
        self.lights.append(light)

# Define a sphere
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
        t1 = (-b - math.sqrt(discriminant)) / (2.0 * a)
        t2 = (-b + math.sqrt(discriminant)) / (2.0 * a)
        if t1 < t2 and t1 > 0:
            return t1
        elif t2 > 0:
            return t2
        return None

# Define a light
class Light:
    def __init__(self, position, color):
        self.position = position
        self.color = color

# Define a ray
class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

# Define a vector
class Vector:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def normalize(self):
        length = math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
        return Vector(self.x / length, self.y / length, self.z / length)

    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar, self.z * scalar)

# Render the scene
def render(scene, width, height):
    image = Image.new('RGB', (width, height))
    pixels = image.load()

    camera = Vector(0, 0, -5)
    screen = (-1, -1, 1)

    for y in range(height):
        for x in range(width):
            u = (x / width) * 2 - 1
            v = (y / height) * 2 - 1
            ray = Ray(camera, Vector(u, v, screen[2]).normalize())
            color = trace(ray, scene)
            pixels[x, y] = (int(color.x * 255), int(color.y * 255), int(color.z * 255))

    image.save('output.png')

# Trace the ray
def trace(ray, scene):
    t_min = float('inf')
    hit_object = None
    for obj in scene.objects:
        t = obj.intersect(ray)
        if t is not None and t < t_min:
            t_min = t
            hit_object = obj

    if hit_object is None:
        return Vector(0, 0, 0)

    hit_point = ray.origin + ray.direction * t_min
    normal = (hit_point - hit_object.center).normalize()
    color = Vector(0, 0, 0)

    for light in scene.lights:
        light_dir = (light.position - hit_point).normalize()
        shadow_ray = Ray(hit_point, light_dir)
        in_shadow = False
        for obj in scene.objects:
            if obj.intersect(shadow_ray) is not None:
                in_shadow = True
                break
        if not in_shadow:
            lambert = max(normal.dot(light_dir), 0)
            color += light.color * lambert

    return color

# Create the scene
scene = Scene()
scene.add_object(Sphere(Vector(0, 0, 0), 1, Vector(1, 0, 0)))
scene.add_object(Sphere(Vector(1, 1, 1), 0.5, Vector(0, 1, 0)))
scene.add_object(Sphere(Vector(-1, -1, 1), 0.5, Vector(0, 0, 1)))

scene.add_light(Light(Vector(2, 2, -2), Vector(1, 1, 1)))
scene.add_light(Light(Vector(-2, 2, -2), Vector(1, 1, 0)))
scene.add_light(Light(Vector(2, -2, -2), Vector(1, 0, 1)))
scene.add_light(Light(Vector(-2, -2, -2), Vector(0, 1, 1)))

# Render the scene
render(scene, 800, 600)
