import numpy as np
from PIL import Image
import math

# Define a 3D vector class
class Vec3:
    def __init__(self, x=0, y=0, z=0):
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

    def magnitude(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalize(self):
        magnitude = self.magnitude()
        return Vec3(self.x / magnitude, self.y / magnitude, self.z / magnitude)

# Define a ray class
class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

# Define a light source class
class Light:
    def __init__(self, position, color):
        self.position = position
        self.color = color

# Define a material class
class Material:
    def __init__(self, color, reflectivity):
        self.color = color
        self.reflectivity = reflectivity

# Define a sphere class
class Sphere:
    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.material = material

    def intersect(self, ray):
        L = self.center - ray.origin
        tca = L.dot(ray.direction)
        if tca < 0:
            return None
        d2 = L.dot(L) - tca * tca
        if d2 > self.radius * self.radius:
            return None
        thc = math.sqrt(self.radius * self.radius - d2)
        t0 = tca - thc
        t1 = tca + thc
        return min(t0, t1)

# Define the scene
def create_scene():
    # Create light sources
    lights = [
        Light(Vec3(-5, 5, -5), Vec3(1, 0, 0)),
        Light(Vec3(0, 5, 0), Vec3(0, 1, 0)),
        Light(Vec3(5, 5, 5), Vec3(0, 0, 1)),
    ]

    # Create spheres
    spheres = [
        Sphere(Vec3(0, 0, -5), 1, Material(Vec3(1, 0, 0), 0.5)),
        Sphere(Vec3(-3, 0, -5), 1, Material(Vec3(0, 1, 0), 0.5)),
        Sphere(Vec3(3, 0, -5), 1, Material(Vec3(0, 0, 1), 0.5)),
    ]

    return lights, spheres

# Ray tracing function
def ray_trace(ray, lights, spheres, max_bounces=5):
    closest_t = float('inf')
    closest_sphere = None
    for sphere in spheres:
        t = sphere.intersect(ray)
        if t is not None and t < closest_t:
            closest_t = t
            closest_sphere = sphere

    if closest_sphere is None:
        return Vec3(0, 0, 0)

    intersection_point = ray.origin + ray.direction * closest_t
    normal = (intersection_point - closest_sphere.center).normalize()

    color = Vec3(0, 0, 0)
    for light in lights:
        L = light.position - intersection_point
        L.normalize()
        diffuse = max(0, normal.dot(L))
        color += light.color * diffuse

    return color

# Render the scene
def render(width, height, lights, spheres):
    image = np.zeros((height, width, 3))
    for y in range(height):
        for x in range(width):
            ray = Ray(Vec3(0, 0, 0), Vec3((x - width // 2) / width * 2, (y - height // 2) / height * 2, -1).normalize())
            color = ray_trace(ray, lights, spheres)
            image[y, x] = [color.x, color.y, color.z]

    return image

# Save the image
def save_image(image, filename):
    img = Image.fromarray((image * 255).astype(np.uint8))
    img.save(filename)

# Main function
def main():
    lights, spheres = create_scene()
    image = render(800, 600, lights, spheres)
    save_image(image, 'raytraced_scene1.png')

if __name__ == "__main__":
    main()
