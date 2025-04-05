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

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def magnitude(self):
        return np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

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
            return False, None
        d2 = L.dot(L) - tca * tca
        if d2 > self.radius * self.radius:
            return False, None
        thc = np.sqrt(self.radius * self.radius - d2)
        t0 = tca - thc
        t1 = tca + thc
        return True, min(t0, t1)

# Define the scene
def create_scene():
    # Create light sources
    lights = [
        Light(Vec3(-5, 5, -5), Vec3(1, 0, 0)),  # Red light
        Light(Vec3(0, 5, -5), Vec3(0, 1, 0)),  # Green light
        Light(Vec3(5, 5, -5), Vec3(0, 0, 1)),  # Blue light
    ]

    # Create spheres
    spheres = [
        Sphere(Vec3(0, 0, -10), 2, Material(Vec3(1, 1, 1), 0.5)),  # White sphere
        Sphere(Vec3(-3, 0, -10), 1, Material(Vec3(1, 0, 0), 0.3)),  # Red sphere
        Sphere(Vec3(3, 0, -10), 1, Material(Vec3(0, 1, 0), 0.3)),  # Green sphere
    ]

    return lights, spheres

# Ray tracing function
def ray_trace(ray, lights, spheres, max_bounces=5):
    closest_sphere = None
    closest_distance = float('inf')

    for sphere in spheres:
        intersect, distance = sphere.intersect(ray)
        if intersect and distance < closest_distance:
            closest_sphere = sphere
            closest_distance = distance

    if closest_sphere is None:
        return Vec3(0, 0, 0)

    intersection_point = ray.origin + ray.direction * closest_distance
    normal = (intersection_point - closest_sphere.center).normalize()

    color = closest_sphere.material.color

    # Calculate illumination
    illumination = Vec3(0, 0, 0)
    for light in lights:
        light_vector = (light.position - intersection_point).normalize()
        light_distance = (light.position - intersection_point).magnitude()
        shadow_ray = Ray(intersection_point, light_vector)
        shadow_sphere = None
        shadow_distance = float('inf')

        for sphere in spheres:
            intersect, distance = sphere.intersect(shadow_ray)
            if intersect and distance < shadow_distance:
                shadow_sphere = sphere
                shadow_distance = distance

        if shadow_sphere is None and light_distance < 10:
            illumination += light.color * max(0, normal.dot(light_vector))

    color = Vec3(
        min(color.x * illumination.x, 1),
        min(color.y * illumination.y, 1),
        min(color.z * illumination.z, 1),
    )

    # Recursive reflection
    if closest_sphere.material.reflectivity > 0 and max_bounces > 0:
        reflected_ray = Ray(intersection_point, (ray.direction - normal * 2 * normal.dot(ray.direction)).normalize())
        reflected_color = ray_trace(reflected_ray, lights, spheres, max_bounces - 1)
        color = Vec3(
            color.x + reflected_color.x * closest_sphere.material.reflectivity,
            color.y + reflected_color.y * closest_sphere.material.reflectivity,
            color.z + reflected_color.z * closest_sphere.material.reflectivity,
        )

    return color

# Render the scene
def render_scene(width, height):
    lights, spheres = create_scene()
    image = np.zeros((height, width, 3))

    for y in range(height):
        for x in range(width):
            # Camera settings
            origin = Vec3(0, 0, 0)
            direction = Vec3(
                (x - width // 2) / width * 2,
                (y - height // 2) / height * 2,
                -5,
            ).normalize()

            ray = Ray(origin, direction)
            color = ray_trace(ray, lights, spheres)

            image[y, x] = [color.x, color.y, color.z]

    return image

# Save the rendered image
def main():
    width, height = 800, 600
    image = render_scene(width, height)
    image = (image * 255).astype(np.uint8)
    Image.fromarray(image).save('raytraced_scene3.png')

if __name__ == '__main__':
    main()
