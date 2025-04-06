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
    def __init__(self, color, albedo, specular_exponent):
        self.color = color
        self.albedo = albedo
        self.specular_exponent = specular_exponent

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
    camera_position = Vec3(0, 0, -5)
    camera_direction = Vec3(0, 0, 1)

    # Define the scene objects
    sphere = Sphere(Vec3(0, 0, 0), 1, Material(Vec3(1, 1, 1), 0.5, 10))
    lights = [
        Light(Vec3(-2, 2, 0), Vec3(1, 0, 0)),
        Light(Vec3(2, 2, 0), Vec3(0, 1, 0)),
        Light(Vec3(0, -2, 0), Vec3(0, 0, 1)),
        Light(Vec3(-1, -1, 2), Vec3(1, 1, 0)),
        Light(Vec3(1, -1, 2), Vec3(0, 1, 1)),
    ]

    # Render the image
    image = np.zeros((height, width, 3))
    for i in range(height):
        for j in range(width):
            # Generate a ray
            x = (2 * j - width + 1) / width
            y = (2 * i - height + 1) / height
            ray_direction = Vec3(x, y, 1).normalize()
            ray = Ray(camera_position, ray_direction)

            # Check if the ray intersects with the sphere
            t = sphere.intersect(ray)
            if t is not None:
                # Compute the intersection point and normal
                intersection_point = ray.origin + ray.direction * t
                normal = (intersection_point - sphere.center).normalize()

                # Compute the color
                color = Vec3(0, 0, 0)
                for light in lights:
                    light_direction = (light.position - intersection_point).normalize()
                    diffuse = max(0, normal.dot(light_direction))
                    specular = max(0, (ray.direction * -1).normalize().dot(light_direction)) ** sphere.material.specular_exponent
                    color = color + light.color * (diffuse * sphere.material.albedo + specular * (1 - sphere.material.albedo))

                # Store the color in the image
                image[i, j] = np.clip([color.x, color.y, color.z], 0, 1)

    return image

# Render the image
image = render(800, 600)

# Save the image to a PNG file
img = Image.fromarray((image * 255).astype(np.uint8))
img.save('raytraced_scene4.png')