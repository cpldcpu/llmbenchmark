import numpy as np
from PIL import Image

# Constants
WIDTH, HEIGHT = 800, 600
MAX_DEPTH = 3
BACKGROUND_COLOR = np.array([0.0, 0.0, 0.0])

# Vector operations
def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def reflect(incoming, normal):
    return incoming - 2 * np.dot(incoming, normal) * normal

# Sphere class
class Sphere:
    def __init__(self, center, radius, color, specular=50, reflective=0.5):
        self.center = np.array(center)
        self.radius = radius
        self.color = np.array(color)
        self.specular = specular
        self.reflective = reflective

    def intersect(self, origin, direction):
        oc = origin - self.center
        a = np.dot(direction, direction)
        b = 2 * np.dot(oc, direction)
        c = np.dot(oc, oc) - self.radius**2
        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            return None
        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
        t2 = (-b - np.sqrt(discriminant)) / (2 * a)
        t = min(t1, t2)
        if t > 0:
            return t
        return None

# Light class
class Light:
    def __init__(self, position, color, intensity):
        self.position = np.array(position)
        self.color = np.array(color)
        self.intensity = intensity

# Scene setup
objects = [
    Sphere([0, -1, 3], 1, [1, 0, 0]),  # Red sphere
    Sphere([2, 0, 4], 1, [0, 1, 0]),   # Green sphere
    Sphere([-2, 0, 4], 1, [0, 0, 1]),  # Blue sphere
    Sphere([0, -5001, 0], 5000, [1, 1, 1])  # Large white sphere (floor)
]

lights = [
    Light([5, 5, -10], [1, 0, 0], 1.0),  # Red light
    Light([-5, 5, -10], [0, 1, 0], 1.0), # Green light
    Light([0, 5, -10], [0, 0, 1], 1.0)   # Blue light
]

# Ray tracing function
def trace_ray(origin, direction, depth):
    if depth > MAX_DEPTH:
        return BACKGROUND_COLOR

    closest_t = float('inf')
    closest_object = None

    # Find the closest intersection
    for obj in objects:
        t = obj.intersect(origin, direction)
        if t and t < closest_t:
            closest_t = t
            closest_object = obj

    if closest_object is None:
        return BACKGROUND_COLOR

    # Compute intersection point and normal
    point = origin + closest_t * direction
    normal = normalize(point - closest_object.center)

    # Compute lighting
    color = np.zeros(3)
    for light in lights:
        to_light = normalize(light.position - point)
        light_distance = np.linalg.norm(light.position - point)

        # Shadow check
        shadowed = False
        for obj in objects:
            t = obj.intersect(point + 1e-5 * to_light, to_light)
            if t and t < light_distance:
                shadowed = True
                break

        if not shadowed:
            diffuse = max(np.dot(normal, to_light), 0)
            specular = max(np.dot(reflect(-to_light, normal), -direction), 0) ** closest_object.specular
            color += closest_object.color * diffuse * light.color * light.intensity
            color += specular * light.color * light.intensity

    # Reflection
    reflected_color = np.zeros(3)
    if closest_object.reflective > 0:
        reflected_direction = reflect(direction, normal)
        reflected_color = trace_ray(point + 1e-5 * reflected_direction, reflected_direction, depth + 1)
        color = color * (1 - closest_object.reflective) + reflected_color * closest_object.reflective

    return np.clip(color, 0, 1)

# Render the scene
def render():
    camera = np.array([0, 0, -1])
    ratio = WIDTH / HEIGHT
    screen = (-1, 1 / ratio, 1, -1 / ratio)  # Left, Top, Right, Bottom

    image = np.zeros((HEIGHT, WIDTH, 3))
    for i, y in enumerate(np.linspace(screen[1], screen[3], HEIGHT)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], WIDTH)):
            pixel = np.array([x, y, 1])
            direction = normalize(pixel - camera)
            color = trace_ray(camera, direction, 0)
            image[i, j] = color

    return image

# Save the image
def save_image(image, filename):
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(image)
    img.save(filename)

# Main function
if __name__ == "__main__":
    print("Rendering scene...")
    image = render()
    save_image(image, "output.png")
    print("Image saved as 'output.png'")