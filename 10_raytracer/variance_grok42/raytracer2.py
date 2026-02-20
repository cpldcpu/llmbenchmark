import math

class Vec3:
    def __init__(self, x=0, y=0, z=0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __add__(self, v):
        return Vec3(self.x + v.x, self.y + v.y, self.z + v.z)

    def __sub__(self, v):
        return Vec3(self.x - v.x, self.y - v.y, self.z - v.z)

    def __mul__(self, s):
        if isinstance(s, Vec3):
            return Vec3(self.x * s.x, self.y * s.y, self.z * s.z)
        return Vec3(self.x * s, self.y * s, self.z * s)

    def __rmul__(self, s):
        return self * s

    def __truediv__(self, s):
        return Vec3(self.x / s, self.y / s, self.z / s)

    def dot(self, v):
        return self.x * v.x + self.y * v.y + self.z * v.z

    def length(self):
        return math.sqrt(self.dot(self))

    def normalize(self):
        l = self.length()
        return self / l if l > 0 else self

class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction.normalize()

class Light:
    def __init__(self, position, color, intensity=2.0):
        self.position = position
        self.color = color
        self.intensity = intensity

class Material:
    def __init__(self, color, diffuse=0.7, specular=0.3, shininess=20):
        self.color = color
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess

class Sphere:
    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.material = material

    def intersect(self, ray):
        oc = ray.origin - self.center
        a = ray.direction.dot(ray.direction)
        b = 2.0 * oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius**2
        disc = b*b - 4*a*c
        if disc < 0:
            return float('inf')
        sqrt_disc = math.sqrt(disc)
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)
        if t1 > 0.0001:
            return t1
        if t2 > 0.0001:
            return t2
        return float('inf')

    def get_normal(self, p):
        return (p - self.center).normalize()

class Plane:
    def __init__(self, point, normal, material):
        self.point = point
        self.normal = normal.normalize()
        self.material = material

    def intersect(self, ray):
        denom = self.normal.dot(ray.direction)
        if abs(denom) < 0.0001:
            return float('inf')
        t = (self.point - ray.origin).dot(self.normal) / denom
        if t > 0.0001:
            return t
        return float('inf')

    def get_normal(self, p):
        return self.normal

# Scene setup
WIDTH = 800
HEIGHT = 600

cam_pos = Vec3(0, 5, -18)

lights = []
light_colors = [
    Vec3(1.0, 0.3, 0.3),
    Vec3(0.3, 1.0, 0.3),
    Vec3(0.3, 0.3, 1.0),
    Vec3(1.0, 1.0, 0.3),
    Vec3(1.0, 0.3, 1.0),
    Vec3(0.3, 1.0, 1.0),
    Vec3(1.0, 0.6, 0.2),
    Vec3(0.6, 0.2, 1.0),
    Vec3(0.8, 1.0, 0.4),
    Vec3(0.4, 0.8, 1.0),
]

for i in range(16):
    angle = i * (math.pi / 8) - math.pi / 2
    radius = 10 + (i % 3) * 2
    x = math.cos(angle) * radius
    z = 6 + math.sin(angle) * 5
    y = 7 + math.sin(i * 1.3) * 3
    col = light_colors[i % len(light_colors)]
    lights.append(Light(Vec3(x, y, z), col, intensity=1.8))

objects = []

# Floor
floor_mat = Material(Vec3(0.7, 0.7, 0.75), diffuse=0.6, specular=0.2, shininess=8)
objects.append(Plane(Vec3(0, 0, 0), Vec3(0, 1, 0), floor_mat))

# Colorful spheres
sphere_data = [
    (Vec3(-5, 2, 4), 2.0, Vec3(0.9, 0.2, 0.2), 0.8, 40),
    (Vec3(4, 1.8, 7), 1.8, Vec3(0.2, 0.9, 0.3), 0.6, 15),
    (Vec3(-2, 3.5, 9), 1.5, Vec3(0.2, 0.4, 1.0), 0.9, 80),
    (Vec3(5, 2.5, 3), 1.2, Vec3(1.0, 0.8, 0.1), 0.5, 25),
    (Vec3(-6, 4, 12), 1.6, Vec3(0.8, 0.3, 1.0), 0.7, 35),
    (Vec3(3, 4.2, 11), 2.1, Vec3(0.1, 0.8, 0.9), 0.75, 50),
    (Vec3(0, 1.5, 5), 1.0, Vec3(1.0, 0.5, 0.9), 0.85, 60),
    (Vec3(-3.5, 2.8, 6.5), 1.4, Vec3(0.9, 0.9, 0.2), 0.65, 20),
]

for center, radius, col, spec, shin in sphere_data:
    mat = Material(col, diffuse=0.65, specular=spec, shininess=shin)
    objects.append(Sphere(center, radius, mat))

# Render
fov = 55 * math.pi / 180
aspect = WIDTH / HEIGHT
scale = math.tan(fov / 2)
half_width = aspect * scale
half_height = scale

print("Rendering... (this may take 10-40 seconds depending on machine)")

with open("raytraced_scene.ppm", "wb") as f:
    f.write(b"P6\n")
    f.write(f"{WIDTH} {HEIGHT}\n".encode())
    f.write(b"255\n")

    for py in range(HEIGHT):
        for px in range(WIDTH):
            screen_x = (2 * (px + 0.5) / WIDTH - 1) * half_width
            screen_y = (1 - 2 * (py + 0.5) / HEIGHT) * half_height

            dir_vec = Vec3(screen_x, screen_y, 1).normalize()
            ray = Ray(cam_pos, dir_vec)

            # Find closest intersection
            closest_t = float('inf')
            hit_obj = None
            hit_point = None

            for obj in objects:
                t = obj.intersect(ray)
                if t < closest_t:
                    closest_t = t
                    hit_obj = obj
                    hit_point = ray.origin + ray.direction * t

            if hit_obj is None:
                # Sky gradient
                t = (ray.direction.y + 1) * 0.5
                bg = Vec3(0.5, 0.6, 0.9) * (1 - t) + Vec3(0.9, 0.95, 1.0) * t
                r = int(min(255, bg.x * 255))
                g = int(min(255, bg.y * 255))
                b = int(min(255, bg.z * 255))
            else:
                normal = hit_obj.get_normal(hit_point)
                mat = hit_obj.material
                view_dir = (cam_pos - hit_point).normalize()

                # Ambient
                color = mat.color * 0.08

                for light in lights:
                    to_light = light.position - hit_point
                    light_dist = to_light.length()
                    light_dir = to_light / light_dist

                    # Shadow
                    shadow = False
                    shadow_ray = Ray(hit_point + normal * 0.002, light_dir)
                    for other in objects:
                        st = other.intersect(shadow_ray)
                        if 0.001 < st < light_dist - 0.001:
                            shadow = True
                            break

                    if not shadow:
                        # Diffuse
                        ndotl = max(0, normal.dot(light_dir))
                        diffuse = mat.color * light.color * ndotl * mat.diffuse * light.intensity

                        # Specular (Phong)
                        reflect_dir = normal * (2 * normal.dot(light_dir)) - light_dir
                        rdotv = max(0, reflect_dir.dot(view_dir))
                        specular = Vec3(1,1,1) * light.color * (rdotv ** mat.shininess) * mat.specular * light.intensity * 0.8

                        color = color + diffuse + specular

                # Clamp
                r = int(min(255, color.x * 255))
                g = int(min(255, color.y * 255))
                b = int(min(255, color.z * 255))

            f.write(bytes([r, g, b]))

print("Done! Image saved as raytraced_scene.ppm")
print("To convert to PNG: use any tool like ImageMagick `convert raytraced_scene.ppm raytraced_scene.png`")