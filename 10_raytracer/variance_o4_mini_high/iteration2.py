import math
from PIL import Image

# --------------- Vector utilities ---------------

def vec_add(u, v):
    return (u[0]+v[0], u[1]+v[1], u[2]+v[2])

def vec_sub(u, v):
    return (u[0]-v[0], u[1]-v[1], u[2]-v[2])

def vec_mul(u, s):
    return (u[0]*s, u[1]*s, u[2]*s)

def vec_dot(u, v):
    return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]

def vec_length(u):
    return math.sqrt(vec_dot(u,u))

def vec_normalize(u):
    l = vec_length(u)
    if l == 0: return (0.0,0.0,0.0)
    return (u[0]/l, u[1]/l, u[2]/l)

def vec_reflect(I, N):
    # Reflect I around normal N (both assumed normalized)
    # R = I - 2*(I·N)*N
    dot = vec_dot(I, N)
    return vec_sub(I, vec_mul(N, 2*dot))

def clamp(x, minimum=0.0, maximum=1.0):
    return max(minimum, min(maximum, x))

# --------------- Scene primitives ---------------

class Sphere:
    def __init__(self, center, radius, color,
                 specular=50, reflective=0.5):
        self.center = center
        self.radius = radius
        self.color = color        # (r,g,b) in 0..1
        self.specular = specular  # shininess exponent
        self.reflective = reflective

    def intersect(self, ray_o, ray_d):
        # returns (t1, t2) distances where ray hits sphere, or None
        L = vec_sub(self.center, ray_o)
        tca = vec_dot(L, ray_d)
        d2 = vec_dot(L,L) - tca*tca
        r2 = self.radius*self.radius
        if d2 > r2: 
            return None
        thc = math.sqrt(r2 - d2)
        return (tca - thc, tca + thc)

class Plane:
    def __init__(self, point, normal, color=(1,1,1),
                 specular=50, reflective=0.2):
        self.point = point
        self.normal = vec_normalize(normal)
        self.color = color
        self.specular = specular
        self.reflective = reflective

    def intersect(self, ray_o, ray_d):
        # returns t distance where ray hits plane, or None
        denom = vec_dot(self.normal, ray_d)
        if abs(denom) < 1e-6:
            return None
        t = vec_dot(vec_sub(self.point, ray_o), self.normal) / denom
        return (t,)

# --------------- Lights ---------------

class Light:
    def __init__(self, position, color, intensity=1.0):
        self.position = position
        self.color = color    # (r,g,b) 0..1
        self.intensity = intensity

# --------------- Ray tracing ---------------

MAX_DEPTH = 3
BACKGROUND_COLOR = (0.1, 0.1, 0.1)
AMBIENT_LIGHT = 0.1

def trace_ray(ray_o, ray_d, objects, lights, depth=0):
    if depth > MAX_DEPTH:
        return BACKGROUND_COLOR

    # Find closest intersection
    nearest_t = float('inf')
    nearest_obj = None
    for obj in objects:
        ts = obj.intersect(ray_o, ray_d)
        if not ts: continue
        for t in ts:
            if t and t > 1e-4 and t < nearest_t:
                nearest_t = t
                nearest_obj = obj

    if not nearest_obj:
        return BACKGROUND_COLOR

    # Compute intersection info
    hit_point = vec_add(ray_o, vec_mul(ray_d, nearest_t))
    if isinstance(nearest_obj, Sphere):
        normal = vec_sub(hit_point, nearest_obj.center)
    else:
        normal = nearest_obj.normal
    normal = vec_normalize(normal)

    view_dir = vec_mul(ray_d, -1.0)
    base_color = nearest_obj.color

    # Phong illumination
    color = [AMBIENT_LIGHT * base_color[i] for i in range(3)]

    for light in lights:
        # Shadow check
        to_light = vec_sub(light.position, hit_point)
        dist_to_light = vec_length(to_light)
        shadow_dir = vec_normalize(to_light)
        shadow_orig = vec_add(hit_point, vec_mul(normal, 1e-4))
        in_shadow = False
        for obj in objects:
            ts = obj.intersect(shadow_orig, shadow_dir)
            if not ts: continue
            t_closest = min([t for t in ts if t>1e-4], default=None)
            if t_closest and t_closest < dist_to_light:
                in_shadow = True
                break
        if in_shadow:
            continue

        # Diffuse
        LdotN = vec_dot(shadow_dir, normal)
        if LdotN > 0:
            diff = light.intensity * LdotN
            for i in range(3):
                color[i] += diff * base_color[i] * light.color[i]

        # Specular
        reflect_dir = vec_reflect(vec_mul(shadow_dir, -1.0), normal)
        RdotV = vec_dot(reflect_dir, view_dir)
        if RdotV > 0:
            spec = light.intensity * (RdotV ** nearest_obj.specular)
            for i in range(3):
                color[i] += spec * light.color[i]

    # Reflection
    if nearest_obj.reflective > 0:
        reflect_dir = vec_reflect(ray_d, normal)
        reflect_orig = vec_add(hit_point, vec_mul(normal, 1e-4))
        reflected = trace_ray(reflect_orig, reflect_dir,
                              objects, lights, depth+1)
        for i in range(3):
            color[i] = (1-nearest_obj.reflective)*color[i] + \
                       nearest_obj.reflective * reflected[i]

    return (clamp(color[0]), clamp(color[1]), clamp(color[2]))


# --------------- Scene setup ---------------

def make_scene():
    objects = [
        # Spheres
        Sphere(center=( 0.0, -1.0, -3.5), radius=1.0,
               color=(1.0, 0.32, 0.36), specular=50, reflective=0.3),
        Sphere(center=( 2.0,  0.0, -4.5), radius=1.0,
               color=(0.9, 0.76, 0.46), specular=10, reflective=0.5),
        Sphere(center=(-2.0,  0.0, -4.0), radius=1.0,
               color=(0.65, 0.77, 0.97), specular=125, reflective=0.2),
        Sphere(center=( 0.0, -5001, -20), radius=5000,
               color=(0.5,0.5,0.5), specular=1000, reflective=0.0),  # ground

        # A small extra sphere up high
        Sphere(center=( 0.0,  2.0, -6.0), radius=1.0,
               color=(0.3,0.9,0.3), specular=50, reflective=0.4),
    ]

    lights = [
        Light(position=(  5,  5, -10), color=(1,0,0), intensity=0.6),
        Light(position=( -3,  5, -2 ), color=(0,1,0), intensity=0.7),
        Light(position=(  2, -3, -3 ), color=(0,0,1), intensity=0.8),
        Light(position=(  0,  5, -5 ), color=(1,1,0), intensity=0.5),
        Light(position=(  5, -5, -5 ), color=(1,0,1), intensity=0.3),
        Light(position=( -5, -5, -5 ), color=(0,1,1), intensity=0.3),
    ]

    return objects, lights

# --------------- Main rendering ---------------

def render(width, height, fov_deg=60):
    aspect_ratio = width / float(height)
    fov = math.radians(fov_deg)
    scale = math.tan(fov / 2)

    objects, lights = make_scene()
    img = Image.new("RGB", (width, height))
    pixels = img.load()

    for j in range(height):
        for i in range(width):
            # Normalized device coordinates (in [-1,1])
            x = (2 * (i + 0.5) / width  - 1) * aspect_ratio * scale
            y = (1 - 2 * (j + 0.5) / height) * scale
            dir = vec_normalize((x, y, -1))
            color = trace_ray((0,0,0), dir, objects, lights, 0)
            r = int(clamp(color[0]) * 255)
            g = int(clamp(color[1]) * 255)
            b = int(clamp(color[2]) * 255)
            pixels[i, j] = (r, g, b)

        if j % 50 == 0:
            print(f"Rendered {j}/{height}")

    img.save("raytraced_scene2.png")
    print("Done. Saved to output.png")

if __name__ == "__main__":
    render(800, 600)
