import numpy as np
from PIL import Image

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

    def length(self):
        return np.sqrt(self.dot(self))

    def normalize(self):
        length = self.length()
        return Vec3(self.x / length, self.y / length, self.z / length)

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
        thc = np.sqrt(self.radius * self.radius - d2)
        t0 = tca - thc
        t1 = tca + thc
        return min(t0, t1)

# Define the scene
def create_scene():
    # Create materials
    red_material = Material(Vec3(1, 0, 0), 0.5)
    blue_material = Material(Vec3(0, 0, 1), 0.5)
    green_material = Material(Vec3(0, 1, 0), 0.5)

    # Create spheres
    spheres = [
        Sphere(Vec3(0, 0, -5), 1, red_material),
        Sphere(Vec3(2, 0, -5), 1, blue_material),
        Sphere(Vec3(-2, 0, -5), 1, green_material),
    ]

    # Create light sources
    lights = [
        Light(Vec3(5, 5, -5), Vec3(1, 1, 1)),
        Light(Vec3(-5, 5, -5), Vec3(1, 0, 0)),
        Light(Vec3(0, -5, -5), Vec3(0, 1, 0)),
    ]

    return spheres, lights

# Ray tracing function
def ray_trace(ray, spheres, lights, max_bounces=5):
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

    color = closest_sphere.material.color

    for light in lights:
        light_vector = (light.position - intersection_point).normalize()
        light_distance = (light.position - intersection_point).length()
        shadow_ray = Ray(intersection_point, light_vector)
        shadow_t = float('inf')
        for sphere in spheres:
            t = sphere.intersect(shadow_ray)
            if t is not None and t < shadow_t and t < light_distance:
                shadow_t = t
        if shadow_t == float('inf'):
            color = color + light.color * max(0, normal.dot(light_vector))

    if closest_sphere.material.reflectivity > 0 and max_bounces > 0:
        reflected_ray = Ray(intersection_point, ray.direction - normal * 2 * normal.dot(ray.direction))
        reflected_color = ray_trace(reflected_ray, spheres, lights, max_bounces - 1)
        color = color + reflected_color * closest_sphere.material.reflectivity

    return color

# Render the scene
def render_scene(spheres, lights, width=800, height=600):
    image = np.zeros((height, width, 3))

    for y in range(height):
        for x in range(width):
            ray_origin = Vec3(0, 0, 0)
            ray_direction = Vec3((x - width // 2) / width * 2, (y - height // 2) / height * 2, -1).normalize()
            ray = Ray(ray_origin, ray_direction)

            color = ray_trace(ray, spheres, lights)

            # Clamp color values to [0, 1]
            color = Vec3(min(max(color.x, 0), 1), min(max(color.y, 0), 1), min(max(color.z, 0), 1))

            # Convert to [0, 255] and store in image
            image[y, x] = [int(color.x * 255), int(color.y * 255), int(color.z * 255)]

    return image

# Main function
def main():
    spheres, lights = create_scene()
    image = render_scene(spheres, lights)
    img = Image.fromarray(image.astype(np.uint8))
    img.save('raytraced_scene4.png')

if __name__ == "__main__":
    main()
