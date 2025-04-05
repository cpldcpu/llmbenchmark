import numpy as np
from PIL import Image

# Define constants
WIDTH, HEIGHT = 800, 600
BACKGROUND_COLOR = (0, 0, 0)

# Define a sphere in 3D space
class Sphere:
    def __init__(self, center, radius, color):
        self.center = np.array(center)
        self.radius = radius
        self.color = color

# Define a light source
class Light:
    def __init__(self, position, color):
        self.position = np.array(position)
        self.color = color

# Define the scene
spheres = [
    Sphere((0, 0, -5), 1, (255, 0, 0)),  # Red
    Sphere((2, 0, -5), 1, (0, 255, 0)),  # Green
    Sphere((-2, 0, -5), 1, (0, 0, 255)),  # Blue
]

lights = [
    Light((-2, 2, -5), (255, 255, 0)),  # Yellow
    Light((2, 2, -5), (255, 0, 255)),   # Magenta
    Light((0, -2, -5), (0, 255, 255)),  # Cyan
]

# Ray tracing function
def ray_trace(origin, direction):
    for sphere in spheres:
        L = sphere.center - origin
        tca = np.dot(L, direction)
        if tca > 0:
            d2 = np.dot(L, L) - tca * tca
            if d2 <= sphere.radius ** 2:
                thc = np.sqrt(sphere.radius ** 2 - d2)
                t0 = tca - thc
                t1 = tca + thc
                # Simple closest hit logic, can be improved for performance
                if 0 < t0 < t1:
                    return sphere.color
                elif 0 < t1 < t0:
                    return sphere.color
    return BACKGROUND_COLOR

def render_pixel(x, y):
    # Perspective projection
    # This is a very basic implementation and doesn't account for many factors like FOV, etc.
    direction = np.array([(x - WIDTH / 2) / (WIDTH / 2), (y - HEIGHT / 2) / (HEIGHT / 2), -1])
    direction = direction / np.linalg.norm(direction)
    return ray_trace(np.array([0, 0, 0]), direction)

# Generate the image
img = Image.new('RGB', (WIDTH, HEIGHT))
pixels = img.load()

for x in range(WIDTH):
    for y in range(HEIGHT):
        pixels[x, y] = render_pixel(x, y)

# Save the image
img.save('output.png')
