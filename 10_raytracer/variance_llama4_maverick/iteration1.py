import numpy as np
from PIL import Image

# Define a 3D vector class
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

    def __truediv__(self, scalar):
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def norm(self):
        return np.sqrt(self.dot(self))

    def normalize(self):
        return self / self.norm()

# Define a ray class
class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

# Define a material class
class Material:
    def __init__(self, color, albedo, specular):
        self.color = color
        self.albedo = albedo
        self.specular = specular

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
        thc = np.sqrt(self.radius * self.radius - d2)
        t0 = tca - thc
        t1 = tca + thc
        return min(t0, t1)

# Define a light source class
class Light:
    def __init__(self, position, color):
        self.position = position
        self.color = color

# Define the scene
def create_scene():
    spheres = [
        Sphere(Vec3(0, 0, -5), 1, Material(Vec3(1, 0, 0), 0.5, 0.5)),
        Sphere(Vec3(-2, 0, -5), 1, Material(Vec3(0, 1, 0), 0.5, 0.5)),
        Sphere(Vec3(2, 0, -5), 1, Material(Vec3(0, 0, 1), 0.5, 0.5)),
    ]

    lights = [
        Light(Vec3(-5, 5, 0), Vec3(1, 0, 0)),
        Light(Vec3(5, 5, 0), Vec3(0, 1, 0)),
        Light(Vec3(0, -5, 0), Vec3(0, 0, 1)),
        Light(Vec3(-3, 3, 3), Vec3(1, 1, 0)),
        Light(Vec3(3, 3, 3), Vec3(0, 1, 1)),
    ]

    return spheres, lights

# Ray tracing function
def ray_trace(ray, spheres, lights):
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
        light_dir = (light.position - intersection_point).normalize()
        diffuse = max(0, normal.dot(light_dir)) * closest_sphere.material.albedo
        color = color + light.color * diffuse

    return color

# Main function
def main():
    width, height = 800, 600
    spheres, lights = create_scene()

    image = Image.new('RGB', (width, height))
    pixels = image.load()

    for x in range(width):
        for y in range(height):
            ray_dir = Vec3((x - width / 2) / width * 2, (y - height / 2) / height * 2, -1).normalize()
            ray = Ray(Vec3(0, 0, 0), ray_dir)
            color = ray_trace(ray, spheres, lights)

            # Clamp color values to [0, 255]
            r = int(min(max(color.x * 255, 0), 255))
            g = int(min(max(color.y * 255, 0), 255))
            b = int(min(max(color.z * 255, 0), 255))

            pixels[x, y] = (r, g, b)

    image.save('ray_traced_image.png')

if __name__ == "__main__":
    main()