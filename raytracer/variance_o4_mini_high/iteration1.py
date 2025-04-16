import math
from PIL import Image

# ---------- Vector helpers ----------
def vector_add(a, b):
    return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]

def vector_sub(a, b):
    return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]

def vector_mul(a, k):
    return [a[0]*k,   a[1]*k,   a[2]*k]

def vector_dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def vector_length(a):
    return math.sqrt(vector_dot(a, a))

def vector_norm(a):
    L = vector_length(a)
    if L == 0: return [0,0,0]
    return [a[0]/L, a[1]/L, a[2]/L]

def vector_reflect(v, n):
    # R = v - 2*(v·n)*n
    dn = 2 * vector_dot(v, n)
    return [v[0] - dn*n[0], v[1] - dn*n[1], v[2] - dn*n[2]]

# ---------- Scene primitives ----------
class Sphere:
    def __init__(self, center, radius, color, specular, reflective):
        self.center     = center
        self.radius     = radius
        # normalize color to [0,1]
        self.color      = [c/255.0 for c in color]
        self.specular   = specular      # shininess exponent
        self.reflective = reflective    # 0=matte, 1=mirror

class Plane:
    def __init__(self, normal, d, color1, color2, specular, reflective):
        # plane: normal·P + d = 0
        self.normal     = vector_norm(normal)
        self.d          = d
        self.color1     = color1  # two colors for checkerboard
        self.color2     = color2
        self.specular   = specular
        self.reflective = reflective

class Light:
    def __init__(self, position, color):
        self.position = position
        self.color    = color  # RGB intensities in [0,1]

# ---------- Scene setup ----------
objects = [
    Sphere([ 0, -1,  3], 1, [255,   0,   0], specular= 50, reflective=0.2),
    Sphere([ 2,  0,  4], 1, [  0,   0, 255], specular=500, reflective=0.3),
    Sphere([-2,  0,  4], 1, [  0, 255,   0], specular= 10, reflective=0.4),
    # infinite checkerboard plane at y = -1
    Plane([0, 1, 0], 1, [1,1,1], [0,0,0], specular=1000, reflective=0.5),
]

lights = [
    Light([ 5,  5, -10], [1, 0, 0]),  # red
    Light([-5,  5, -10], [0, 1, 0]),  # green
    Light([ 5, -5, -10], [0, 0, 1]),  # blue
    Light([-5, -5, -10], [1, 1, 0]),  # yellow
    Light([ 5,  0,  10], [0, 1, 1]),  # cyan
    Light([-5,  0,  10], [1, 0, 1]),  # magenta
]

ambient_light = [0.2, 0.2, 0.2]

WIDTH, HEIGHT = 800, 600
FOV = math.pi/3
MAX_DEPTH = 3

# ---------- Ray‐object intersections ----------
def intersect_ray_sphere(orig, dir, sph):
    # solve t^2*(dir·dir) + 2 t (oc·dir) + (oc·oc - R^2) = 0
    oc = vector_sub(orig, sph.center)
    k1 = vector_dot(dir, dir)
    k2 = 2 * vector_dot(oc, dir)
    k3 = vector_dot(oc, oc) - sph.radius*sph.radius
    disc = k2*k2 - 4*k1*k3
    if disc < 0:
        return math.inf, math.inf
    sq = math.sqrt(disc)
    return ((-k2 + sq)/(2*k1), (-k2 - sq)/(2*k1))

def intersect_ray_plane(orig, dir, pl):
    denom = vector_dot(dir, pl.normal)
    if abs(denom) < 1e-6:
        return math.inf
    t = -(vector_dot(orig, pl.normal) + pl.d) / denom
    return t if t >= 0 else math.inf

def closest_intersection(orig, dir, t_min, t_max):
    closest_t = math.inf
    hit_obj   = None
    for obj in objects:
        if isinstance(obj, Sphere):
            t1, t2 = intersect_ray_sphere(orig, dir, obj)
            if t_min < t1 < t_max and t1 < closest_t:
                closest_t, hit_obj = t1, obj
            if t_min < t2 < t_max and t2 < closest_t:
                closest_t, hit_obj = t2, obj
        else:  # plane
            t = intersect_ray_plane(orig, dir, obj)
            if t_min < t < t_max and t < closest_t:
                closest_t, hit_obj = t, obj
    return closest_t, hit_obj

# ---------- Lighting & shading ----------
def compute_lighting(P, N, V, specular, obj_color):
    # P=point, N=normal, V=view vector, specular=exponent, obj_color=[r,g,b]
    diffuse = [ambient_light[i]*obj_color[i] for i in range(3)]
    spec_col = [0.0,0.0,0.0]
    for light in lights:
        # direction to light
        L = vector_sub(light.position, P)
        dist = vector_length(L)
        L = vector_norm(L)
        # shadow?
        t_sh, sh_obj = closest_intersection(P, L, 1e-3, dist)
        if sh_obj is not None:
            continue
        # diffuse
        n_dot_l = vector_dot(N, L)
        if n_dot_l > 0:
            for i in range(3):
                diffuse[i] += light.color[i] * obj_color[i] * n_dot_l
        # specular
        if specular > 0:
            R = vector_reflect(vector_mul(L, -1), N)
            r_dot_v = vector_dot(R, V)
            if r_dot_v > 0:
                factor = r_dot_v ** specular
                for i in range(3):
                    spec_col[i] += light.color[i] * factor
    return diffuse, spec_col

def trace_ray(orig, dir, depth):
    t, obj = closest_intersection(orig, dir, 1e-3, math.inf)
    if obj is None:
        return [0,0,0]
    # compute intersection P, normal N
    P = vector_add(orig, vector_mul(dir, t))
    if isinstance(obj, Sphere):
        N = vector_norm(vector_sub(P, obj.center))
        col = obj.color
    else:  # Plane
        N = obj.normal
        # checkerboard
        if (math.floor(P[0]) + math.floor(P[2])) % 2 == 0:
            col = obj.color1
        else:
            col = obj.color2
    V = vector_mul(dir, -1)
    diff, spec = compute_lighting(P, N, V, obj.specular, col)
    local_color = [diff[i] + spec[i] for i in range(3)]
    # reflection
    refl = obj.reflective
    if depth <= 0 or refl <= 0:
        return local_color
    R = vector_reflect(dir, N)
    refl_color = trace_ray(vector_add(P, vector_mul(N, 1e-3)), R, depth-1)
    return [(1-refl)*local_color[i] + refl*refl_color[i] for i in range(3)]

# ---------- Render ----------
def render():
    img = Image.new("RGB", (WIDTH, HEIGHT))
    pixels = []
    aspect = WIDTH/HEIGHT
    screen_h = 2 * math.tan(FOV/2)
    screen_w = screen_h * aspect

    for y in range(HEIGHT):
        for x in range(WIDTH):
            # map pixel -> viewport
            px = (x+0.5)/WIDTH  * screen_w  - screen_w/2
            py = -(y+0.5)/HEIGHT* screen_h  + screen_h/2
            D  = vector_norm([px, py, 1])
            color = trace_ray([0,0,0], D, MAX_DEPTH)
            # clamp & to 0–255
            r = max(0, min(255, int(color[0]*255)))
            g = max(0, min(255, int(color[1]*255)))
            b = max(0, min(255, int(color[2]*255)))
            pixels.append((r,g,b))
    img.putdata(pixels)
    return img

if __name__ == "__main__":
    result = render()
    result.save("raytraced_scene1.png")
    print("Rendered 800×600 image to output.png")
