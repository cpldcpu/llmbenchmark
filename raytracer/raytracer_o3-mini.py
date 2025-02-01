import math
from PIL import Image

# ---------------------------
# Vector class for 3D math
# ---------------------------
class Vector:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    # Vector addition
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    # Vector subtraction
    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    # Scalar multiplication
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Vector(self.x * other, self.y * other, self.z * other)
        raise NotImplementedError("Multiplication only supports scalars.")

    __rmul__ = __mul__

    # Element-wise multiplication (for colors)
    def multiply(self, other):
        return Vector(self.x * other.x, self.y * other.y, self.z * other.z)

    # Dot product
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    # Return vector magnitude
    def norm(self):
        return math.sqrt(self.dot(self))

    # Return normalized vector
    def normalize(self):
        n = self.norm()
        if n == 0:
            return self
        return self * (1/n)

    # Clamp each component between 0 and 1
    def clamp(self):
        return Vector(min(1, max(0, self.x)),
                      min(1, max(0, self.y)),
                      min(1, max(0, self.z)))

    # Convert color to integer RGB tuple (0-255)
    def to_rgb(self):
        c = self.clamp()
        return (int(c.x * 255), int(c.y * 255), int(c.z * 255))

# ---------------------------
# Sphere class
# ---------------------------
class Sphere:
    def __init__(self, center, radius, color):
        self.center = center  # Center position (Vector)
        self.radius = radius  # Radius (float)
        self.color = color    # Surface color (Vector, each component 0-1)

    def intersect(self, ray_origin, ray_direction):
        # Ray-sphere intersection using quadratic formula
        oc = ray_origin - self.center
        a = ray_direction.dot(ray_direction)
        b = 2.0 * oc.dot(ray_direction)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return None  # No intersection
        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)
        if t1 > 1e-4:
            return t1
        if t2 > 1e-4:
            return t2
        return None

# ---------------------------
# Light class
# ---------------------------
class Light:
    def __init__(self, position, color, intensity):
        self.position = position  # Position of light (Vector)
        self.color = color        # Color of light (Vector, each component 0-1)
        self.intensity = intensity  # Scalar intensity

# ---------------------------
# Raytracing function
# ---------------------------
def trace_ray(ray_origin, ray_direction, spheres, lights):
    # Find closest sphere intersection
    t_min = float('inf')
    hit_sphere = None
    for sphere in spheres:
        t = sphere.intersect(ray_origin, ray_direction)
        if t is not None and t < t_min:
            t_min = t
            hit_sphere = sphere

    if hit_sphere is None:
        return Vector(0, 0, 0)  # Background color (black)

    # Compute intersection point and surface normal
    hit_point = ray_origin + ray_direction * t_min
    normal = (hit_point - hit_sphere.center).normalize()

    # Start with a small ambient light component
    ambient = 0.1
    final_color = hit_sphere.color * ambient

    # For each light, add diffuse shading if not in shadow
    for light in lights:
        light_dir = (light.position - hit_point).normalize()

        # Shadow check: offset hit point a bit along the normal
        shadow_origin = hit_point + normal * 1e-4
        in_shadow = False
        for sphere in spheres:
            if sphere is hit_sphere:
                continue
            if sphere.intersect(shadow_origin, light_dir):
                in_shadow = True
                break
        if in_shadow:
            continue

        # Diffuse shading (Lambertian)
        diffuse_intensity = max(normal.dot(light_dir), 0) * light.intensity
        # Multiply the sphere's color with the light color and the diffuse intensity
        final_color += hit_sphere.color.multiply(light.color) * diffuse_intensity

    return final_color

# ---------------------------
# Main render function
# ---------------------------
def render():
    width, height = 800, 600
    camera = Vector(0, 0, -1)  # Camera positioned a bit back along z
    viewport_distance = 1
    aspect_ratio = width / height

    # Define spheres in the scene
    spheres = [
        Sphere(Vector(0, -0.25, 3), 0.5, Vector(1, 0.2, 0.2)),  # Red-ish sphere
        Sphere(Vector(1, 0, 4), 0.5, Vector(0.2, 1, 0.2)),        # Green-ish sphere
        Sphere(Vector(-1, 0, 4), 0.5, Vector(0.2, 0.2, 1)),       # Blue-ish sphere
        Sphere(Vector(0, -5001, 0), 5000, Vector(1, 1, 0.8))        # Large ground plane
    ]

    # Define several colorful lights at different positions
    lights = [
        Light(Vector(5, 5, -10), Vector(1, 0, 0), 0.6),   # Red light
        Light(Vector(-5, 5, -10), Vector(0, 1, 0), 0.6),  # Green light
        Light(Vector(0, 5, -10), Vector(0, 0, 1), 0.6),   # Blue light
        Light(Vector(0, -5, -10), Vector(1, 1, 0), 0.4),  # Yellow light
        Light(Vector(5, -5, -10), Vector(1, 0, 1), 0.4),  # Magenta light
        Light(Vector(-5, -5, -10), Vector(0, 1, 1), 0.4)  # Cyan light
    ]

    # Create an image
    image = Image.new("RGB", (width, height))
    
    # Loop over each pixel and cast a ray through the viewport
    for j in range(height):
        for i in range(width):
            # Convert pixel coordinate to viewport coordinate
            # The viewport is centered at (0, 0) in the image plane.
            x = (2 * (i + 0.5) / width - 1) * aspect_ratio
            y = 1 - 2 * (j + 0.5) / height  # Invert y so that the image is not upside-down

            # Create a direction from the camera to the pixel on the viewport
            direction = Vector(x, y, viewport_distance).normalize()

            # Trace the ray from the camera
            color = trace_ray(camera, direction, spheres, lights)
            image.putpixel((i, j), color.to_rgb())

    # Save the rendered image
    image.save("output.png")
    print("Rendering complete. Image saved as output.png")

if __name__ == "__main__":
    render()
