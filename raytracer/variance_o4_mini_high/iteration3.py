#!/usr/bin/env python3
import math
from PIL import Image

# -----------------------------------
# Vector utilities
# -----------------------------------
def vec_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

def vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

def vec_mul(a, k):
    return (a[0] * k, a[1] * k, a[2] * k)

def vec_dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def vec_length(v):
    return math.sqrt(vec_dot(v, v))

def vec_norm(v):
    l = vec_length(v)
    if l == 0:
        return (0.0,0.0,0.0)
    return (v[0]/l, v[1]/l, v[2]/l)

def vec_reflect(I, N):
    # R = I - 2*(I·N)*N
    dn = vec_dot(I, N)
    return vec_sub(I, vec_mul(N, 2*dn))

def clamp(x, minv=0.0, maxv=1.0):
    return max(minv, min(maxv, x))

# -----------------------------------
# Scene primitives
# -----------------------------------
class Sphere:
    def __init__(self, center, radius, color, specular=50, reflect=0.5):
        self.center = center
        self.radius = radius
        self.color = color
        self.specular = specular
        self.reflect = reflect

    def intersect(self, orig, dir):
        # Solve (orig + t dir - C)² = R²
        CO = vec_sub(orig, self.center)
        b = 2 * vec_dot(dir, CO)
        c = vec_dot(CO, CO) - self.radius * self.radius
        disc = b*b - 4*c
        if disc < 0:
            return None
        sqrt_disc = math.sqrt(disc)
        t1 = (-b + sqrt_disc) / 2
        t2 = (-b - sqrt_disc) / 2
        t = min(t1, t2)
        if t < 0:
            t = max(t1, t2)
            if t < 0:
                return None
        return t

class Light:
    def __init__(self, position, color, intensity=1.0):
        self.position = position
        self.color = color
        self.intensity = intensity

# -----------------------------------
# Ray tracing
# -----------------------------------
MAX_DEPTH = 3
EPSILON = 1e-4

def trace_ray(orig, dir, spheres, lights, depth=0):
    # Find nearest intersection
    nearest_t = float('inf')
    nearest_obj = None
    for obj in spheres:
        t = obj.intersect(orig, dir)
        if t and t < nearest_t:
            nearest_t = t
            nearest_obj = obj

    # Intersect with plane y = -1 (checkerboard)
    plane_t = None
    if abs(dir[1]) > 1e-6:
        t_plane = - (orig[1] + 1) / dir[1]
        if t_plane > 0 and t_plane < nearest_t:
            plane_t = t_plane
            nearest_obj = None
            nearest_t = t_plane

    if nearest_obj is None and plane_t is None:
        # No hit -> background
        return (0.2, 0.7, 0.8)  # sky color

    # Compute intersection point and normal
    if plane_t is not None and nearest_obj is None:
        # Checkerboard plane
        point = vec_add(orig, vec_mul(dir, plane_t))
        normal = (0, 1, 0)
        # checker pattern
        checker = (int(math.floor(point[0])) + int(math.floor(point[2]))) & 1
        base_color = (1,1,1) if checker else (0,0,0)
        mat_spec = 10
        mat_reflect = 0.2
    else:
        point = vec_add(orig, vec_mul(dir, nearest_t))
        normal = vec_sub(point, nearest_obj.center)
        normal = vec_norm(normal)
        base_color = nearest_obj.color
        mat_spec = nearest_obj.specular
        mat_reflect = nearest_obj.reflect

    # Start lighting with ambient
    ambient = 0.1
    color = vec_mul(base_color, ambient)

    # For each light: diffuse + specular + shadows
    for light in lights:
        to_light = vec_sub(light.position, point)
        dist_to_light = vec_length(to_light)
        L = vec_norm(to_light)

        # Shadow check
        shadow_orig = vec_add(point, vec_mul(L, EPSILON))
        in_shadow = False
        for obj in spheres:
            t_sh = obj.intersect(shadow_orig, L)
            if t_sh and t_sh < dist_to_light:
                in_shadow = True
                break
        if in_shadow:
            continue

        # Diffuse
        diff_intensity = max(0.0, vec_dot(normal, L)) * light.intensity
        diff = vec_mul(
                (base_color[0]*light.color[0],
                 base_color[1]*light.color[1],
                 base_color[2]*light.color[2]),
                diff_intensity
        )
        color = vec_add(color, diff)

        # Specular
        if mat_spec > 0:
            R = vec_reflect(vec_mul(L, -1), normal)
            spec_intensity = (max(0.0, vec_dot(R, vec_mul(dir, -1))) ** mat_spec) * light.intensity
            spec = vec_mul(light.color, spec_intensity)
            color = vec_add(color, spec)

    # Reflection
    if depth < MAX_DEPTH and mat_reflect > 0:
        refl_dir = vec_reflect(dir, normal)
        refl_dir = vec_norm(refl_dir)
        refl_orig = vec_add(point, vec_mul(refl_dir, EPSILON))
        refl_color = trace_ray(refl_orig, refl_dir, spheres, lights, depth+1)
        color = vec_add(color, vec_mul(refl_color, mat_reflect))

    # Clamp
    return (clamp(color[0]), clamp(color[1]), clamp(color[2]))


# -----------------------------------
# Build scene and render
# -----------------------------------
def render(width, height):
    # Camera
    eye = (0, 0, 0)
    fov = math.pi/3  # 60 degrees

    # Scene objects
    spheres = [
        Sphere((0.0, 0.0,  5), 1.0, (1, 0, 0), specular=100, reflect=0.5),
        Sphere((2.0, 0.0,  7), 1.0, (0, 1, 0), specular=50,  reflect=0.3),
        Sphere((-2.0, 0.0, 7), 1.0, (0, 0, 1), specular=10,  reflect=0.8),
        Sphere((0.0, -5001, 0), 5000, (1,1,1), specular=0,   reflect=0.2)  # big sphere for ground
    ]

    # Colored lights
    lights = [
        Light(( 5, 5, -5), (1, 0, 0), 0.6),  # red
        Light((-5, 5, -5), (0, 1, 0), 0.6),  # green
        Light(( 0, 5, -5), (0, 0, 1), 0.6),  # blue
        Light(( 5, 5,  5), (1, 1, 0), 0.6),  # yellow
        Light((-5, 5,  5), (1, 0, 1), 0.6),  # magenta
        Light(( 0, 5, 10), (0, 1, 1), 0.6),  # cyan
    ]

    image = Image.new("RGB", (width, height))
    pixels = image.load()

    for y in range(height):
        for x in range(width):
            # Convert pixel coordinate to viewport ray
            px = (2*(x + 0.5)/width - 1) * math.tan(fov/2) * width/height
            py = (1 - 2*(y + 0.5)/height) * math.tan(fov/2)
            ray_dir = vec_norm((px, py, 1))
            col = trace_ray(eye, ray_dir, spheres, lights)
            # to 0–255
            ir = int(col[0]*255)
            ig = int(col[1]*255)
            ib = int(col[2]*255)
            pixels[x,y] = (ir, ig, ib)

    return image

if __name__ == "__main__":
    WIDTH = 800
    HEIGHT = 600
    img = render(WIDTH, HEIGHT)
    img.save("raytraced_scnene3.png")
    print("Saved render.png")
