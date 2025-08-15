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
    def __init__(self, color, albedo, diffuse, specular):
        self.color = color
        self.albedo = albedo
        self.diffuse = diffuse
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
def render(width, height):
    # Define the camera
    camera_pos = Vec3(0, 0, -5)
    camera_dir = Vec3(0, 0, 1)

    # Define the scene objects
    sphere = Sphere(Vec3(0, 0, 0), 1, Material(Vec3(1, 1, 1), 0.5, 0.7, 0.3))

    # Define the light sources
    lights = [
        Light(Vec3(-2, 2, 2), Vec3(1, 0, 0)),
        Light(Vec3(2, 2, 2), Vec3(0, 1, 0)),
        Light(Vec3(0, -2, 2), Vec3(0, 0, 1)),
        Light(Vec3(-2, -2, 2), Vec3(1, 1, 0)),
        Light(Vec3(2, -2, 2), Vec3(0, 1, 1)),
    ]

    # Render the image
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            # Calculate the ray direction
            x = (2 * j - width + 1) / width
            y = (2 * i - height + 1) / height
            ray_dir = Vec3(x, y, 1).normalize()
            ray = Ray(camera_pos, ray_dir)

            # Check for intersection with the sphere
            t = sphere.intersect(ray)
            if t is not None:
                # Calculate the intersection point
                intersection = ray.origin + ray.direction * t

                # Calculate the normal at the intersection point
                normal = (intersection - sphere.center).normalize()

                # Calculate the color using the Phong reflection model
                color = Vec3(0, 0, 0)
                for light in lights:
                    light_dir = (light.position - intersection).normalize()
                    diffuse = max(0, normal.dot(light_dir)) * sphere.material.diffuse
                    specular = max(0, (ray.direction * -1).dot(light_dir)) ** 32 * sphere.material.specular
                    color = color + light.color * (diffuse + specular)

                # Clamp the color values
                color = Vec3(min(color.x, 1), min(color.y, 1), min(color.z, 1))

                # Convert the color to uint8
                image[i, j] = (color.x * 255, color.y * 255, color.z * 255)

    return image

# Render the image
image = render(800, 600)

# Save the image to a PNG file
img = Image.fromarray(image)
img.save('raytraced_scene2.png')