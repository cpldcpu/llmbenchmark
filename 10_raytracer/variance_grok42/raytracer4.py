import numpy as np
import matplotlib.pyplot as plt
import math

class Vec3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        if isinstance(other, Vec3):
            return Vec3(self.x * other.x, self.y * other.y, self.z * other.z)
        return Vec3(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return Vec3(-self.x, -self.y, -self.z)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def length(self):
        return math.sqrt(self.dot(self))

    def normalize(self):
        l = self.length()
        if l > 0:
            return self / l
        return Vec3(0,0,1)

    def __truediv__(self, scalar):
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)

class Sphere:
    def __init__(self, center, radius, color):
        self.center = center
        self.radius = radius
        self.color = color

class Plane:
    def __init__(self, y):
        self.y = y

    def intersect(self, orig, dir):
        if abs(dir.y) < 1e-6:
            return None
        t = (self.y - orig.y) / dir.y
        if t > 0.001:
            return t
        return None

# Scene
world = [
    Plane(-1.5),
    Sphere(Vec3(-2, -0.5, -1), 1, Vec3(1.0, 0.3, 0.3)),
    Sphere(Vec3(2, -0.8, -2), 0.8, Vec3(0.3, 1.0, 0.3)),
    Sphere(Vec3(0, 0.5, -3), 1.2, Vec3(0.3, 0.3, 1.0)),
    Sphere(Vec3(-1.5, -1.0, 1), 0.6, Vec3(1.0, 1.0, 0.3)),
    Sphere(Vec3(1.5, -0.5, 2), 0.7, Vec3(0.8, 0.3, 0.8)),
    Sphere(Vec3(-3, -1.2, -4), 0.5, Vec3(0.3, 0.8, 0.9)),
]

lights = []
for i in range(12):
    ang = i * 2 * math.pi / 12
    x = math.cos(ang) * 6
    z = math.sin(ang) * 6 - 1
    y = 3.5 + math.sin(i) * 1.5
    col = Vec3(
        0.4 + 0.6 * math.sin(i * 0.8),
        0.4 + 0.6 * math.cos(i * 1.2),
        0.4 + 0.6 * math.sin(i * 2.1)
    )
    lights.append( (Vec3(x, y, z), col) )

# Render
WIDTH = 800
HEIGHT = 600
cam_pos = Vec3(0, 2, -8)
aspect = WIDTH / HEIGHT
fov_scale = math.tan(math.radians(35))
image = np.zeros((HEIGHT, WIDTH, 3))

print("Starting render...")

for py in range(HEIGHT):
    for px in range(WIDTH):
        sx = (2 * (px + 0.5) / WIDTH - 1) * aspect * fov_scale
        sy = (1 - 2 * (py + 0.5) / HEIGHT ) * fov_scale
        dir = Vec3(sx, sy, 1).normalize()
        # find hit
        t_min = float('inf')
        hit_p = None
        normal = None
        obj_color = None
        is_plane = False
        for obj in world:
            if isinstance(obj, Plane):
                t = obj.intersect(cam_pos, dir)
                if t and t < t_min:
                    t_min = t
                    hit_p = cam_pos + dir * t
                    normal = Vec3(0,1,0)
                    is_plane = True
                    # checker
                    cx = int(hit_p.x) % 2
                    cz = int(hit_p.z) % 2
                    if (cx + cz) % 2 == 0:
                        obj_color = Vec3(0.9, 0.9, 0.9)
                    else:
                        obj_color = Vec3(0.4, 0.4, 0.4)
            else:
                # sphere intersect
                oc = cam_pos - obj.center
                a = dir.dot(dir)
                b = 2 * oc.dot(dir)
                c = oc.dot(oc) - obj.radius**2
                disc = b*b - 4*a*c
                if disc > 0:
                    sqrt_d = math.sqrt(disc)
                    t1 = (-b - sqrt_d) / (2*a)
                    t2 = (-b + sqrt_d) / (2*a)
                    t = t1 if t1 > 0.001 else t2
                    if t > 0.001 and t < t_min:
                        t_min = t
                        hit_p = cam_pos + dir * t
                        normal = (hit_p - obj.center).normalize()
                        obj_color = obj.color
                        is_plane = False
        if t_min == float('inf'):
            # sky
            t = 0.5 * (dir.y + 1.0)
            col = Vec3(0.5, 0.7, 1.0) * (1.0 - t) + Vec3(1.0, 1.0, 1.0) * t
        else:
            col = Vec3(0.05,0.05,0.05) * obj_color  # ambient
            view_dir = - dir
            for lpos, lcol in lights:
                ldir = (lpos - hit_p).normalize()
                dist = (lpos - hit_p).length()
                # no shadow
                ndl = max(0.0, normal.dot(ldir))
                diff = obj_color * lcol * ndl
                att = 1.0 / (1 + 0.1*dist**2)
                col = col + diff * att
                # specular
                if not is_plane:
                    reflect_d = normal * (2 * normal.dot(ldir)) - ldir
                    rdv = max(0.0, reflect_d.dot(view_dir))
                    spec = rdv ** 30
                    col = col + lcol * spec * 0.6
        image[py, px] = [min(1.0, col.x), min(1.0, col.y), min(1.0, col.z)]

# gamma
image = np.power(image, 1.0 / 2.2)
plt.imsave("raytracer_scene.png", image)
print("Done! Image saved as raytracer_scene.png")