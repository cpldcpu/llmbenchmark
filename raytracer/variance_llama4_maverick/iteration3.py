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
def render(width, height):
    # Create the image
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Define the camera
    camera_pos = Vec3(0, 0, -5)
    camera_dir = Vec3(0, 0, 1)

    # Define the sphere
    sphere_center = Vec3(0, 0, 0)
    sphere_radius = 1.0
    sphere_material = Material(Vec3(1, 0, 0), 0.8, 0.2)
    sphere = Sphere(sphere_center, sphere_radius, sphere_material)

    # Define the light sources
    lights = [
        Light(Vec3(-2, 2, -2), Vec3(1, 0, 0)),
        Light(Vec3(2, 2, -2), Vec3(0, 1, 0)),
        Light(Vec3(0, -2, -2), Vec3(0, 0, 1)),
        Light(Vec3(-2, -2, -2), Vec3(1, 1, 0)),
        Light(Vec3(2, -2, -2), Vec3(1, 0, 1)),
    ]

    # Render the scene
    for i in range(height):
        for j in range(width):
            # Calculate the ray direction
            x = (2 * j - width + 1) / width
            y = (2 * i - height + 1) / height
            ray_dir = Vec3(x, y, 1).normalize()
            ray = Ray(camera_pos, ray_dir)

            # Check if the ray intersects the sphere
            t = sphere.intersect(ray)
            if t is not None:
                # Calculate the intersection point
                intersection_point = ray.origin + ray.direction * t

                # Calculate the normal at the intersection point
                normal = (intersection_point - sphere_center).normalize()

                # Calculate the color
                color = Vec3(0, 0, 0)
                for light in lights:
                    # Calculate the light direction
                    light_dir = (light.position - intersection_point).normalize()

                    # Check if the light is visible
                    light_ray = Ray(intersection_point, light_dir)
                    if sphere.intersect(light_ray) is None:
                        # Calculate the diffuse component
                        diffuse = max(0, normal.dot(light_dir)) * sphere.material.albedo

                        # Calculate the specular component
                        view_dir = (camera_pos - intersection_point).normalize()
                        specular = max(0, (light_dir + view_dir).normalize().dot(normal)) ** 32 * sphere.material.specular

                        # Add the light contribution
                        color = color + Vec3(diffuse * light.color.x, diffuse * light.color.y, diffuse * light.color.z) + Vec3(specular * light.color.x, specular * light.color.y, specular * light.color.z)

                # Clamp the color values
                color.x = min(max(color.x, 0), 1)
                color.y = min(max(color.y, 0), 1)
                color.z = min(max(color.z, 0), 1)

                # Set the pixel color
                img[i, j] = (int(color.x * 255), int(color.y * 255), int(color.z * 255))

    return img

# Render the image
img = render(800, 600)

# Save the image
Image.fromarray(img).save('raytraced_scene3.png')