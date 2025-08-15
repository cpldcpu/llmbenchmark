import math
from PIL import Image

# simple 3D vector class
class Vec:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z
    def __add__(self, o):   return Vec(self.x+o.x, self.y+o.y, self.z+o.z)
    def __sub__(self, o):   return Vec(self.x-o.x, self.y-o.y, self.z-o.z)
    def __mul__(self, o):
        if isinstance(o, Vec):
            return Vec(self.x*o.x, self.y*o.y, self.z*o.z)
        else:
            return Vec(self.x*o,   self.y*o,   self.z*o)
    __rmul__ = __mul__
    def __neg__(self):      return Vec(-self.x, -self.y, -self.z)
    def dot(self, o):       return self.x*o.x + self.y*o.y + self.z*o.z
    def norm(self):         return math.sqrt(self.dot(self))
    def normalize(self):    
        n = self.norm()
        return self*(1/n) if n>0 else self
    def reflect(self, n):   # reflect self around normal n
        return self - n*(2*self.dot(n))

# clamp to byte
def clamp(x):  
    return int(max(0, min(255, x)))

# sphere‐ray intersection
def intersect_sphere(orig, dir, sph):
    CO = orig - sph['center']
    a = dir.dot(dir)
    b = 2 * CO.dot(dir)
    c = CO.dot(CO) - sph['radius']*sph['radius']
    disc = b*b - 4*a*c
    if disc < 0:
        return float('inf'), None
    s = math.sqrt(disc)
    t1 = (-b + s)/(2*a)
    t2 = (-b - s)/(2*a)
    t = min(t1, t2)
    if t <= 0:
        t = max(t1, t2)
    if t <= 0:
        return float('inf'), None
    return t, sph

# plane‑ray intersection (y = -1)
def intersect_plane(orig, dir):
    if abs(dir.y) < 1e-6:
        return float('inf')
    t = -(orig.y + 1)/dir.y
    return t if t>0 else float('inf')

# trace a ray into the scene
def trace_ray(orig, dir, spheres, lights, depth=3):
    # find nearest intersection
    nearest_t = float('inf')
    nearest_obj = None
    for sph in spheres:
        t, obj = intersect_sphere(orig, dir, sph)
        if t < nearest_t:
            nearest_t, nearest_obj = t, obj
    t_plane = intersect_plane(orig, dir)
    if t_plane < nearest_t:
        nearest_t, nearest_obj = t_plane, 'plane'
    # no hit -> background
    if nearest_obj is None:
        return Vec(0.2, 0.7, 0.8)  # sky colour

    hit = orig + dir*nearest_t
    if nearest_obj == 'plane':
        normal = Vec(0,1,0)
        # checkerboard
        checker = (int(math.floor(hit.x)) + int(math.floor(hit.z))) & 1
        base_color = Vec(1,1,1) if checker else Vec(0.3,0.3,0.3)
        material = {'color': base_color, 'specular': 0,   'reflective': 0.2}
    else:
        normal = (hit - nearest_obj['center']).normalize()
        base_color = nearest_obj['color']
        material = nearest_obj

    view_dir = -dir
    color = Vec(0,0,0)

    # accumulate each light
    for light in lights:
        if light['type'] == 'ambient':
            color += base_color * light['intensity']
        else:
            # point light
            light_dir = (light['position'] - hit).normalize()
            # shadow check
            shadow_orig = hit + normal*1e-3
            shadow_t = float('inf')
            for sph in spheres:
                t, _ = intersect_sphere(shadow_orig, light_dir, sph)
                if t < (light['position'] - hit).norm():
                    shadow_t = t
                    break
            if intersect_plane(shadow_orig, light_dir) < (light['position'] - hit).norm():
                shadow_t = 1
            if shadow_t < float('inf'):
                continue
            # diffuse
            diff = max(0, normal.dot(light_dir))
            color += base_color * light['color'] * light['intensity'] * diff
            # specular
            if material['specular'] > 0:
                refl = ( -light_dir ).reflect(normal)
                spec = max(0, refl.dot(view_dir)) ** (material['specular'])
                color += light['color'] * light['intensity'] * spec

    # reflection
    r = material['reflective']
    if r>0 and depth>0:
        refl_dir = dir.reflect(normal).normalize()
        refl_color = trace_ray(hit + normal*1e-3, refl_dir, spheres, lights, depth-1)
        color = color*(1-r) + refl_color*r

    return color

def render():
    width, height = 800, 600
    fov = math.pi/3
    image = Image.new('RGB', (width, height))
    pixels = image.load()

    # define scene
    spheres = [
        {'center': Vec(-3, 1, -16), 'radius': 2, 'color': Vec(1,0,0), 'specular': 500,  'reflective': 0.2},
        {'center': Vec( 1.5,-0.5,-12), 'radius': 2, 'color': Vec(0,1,0), 'specular': 500,  'reflective': 0.3},
        {'center': Vec( 7,   5, -18), 'radius': 3, 'color': Vec(0,0,1), 'specular': 10,   'reflective': 0.4},
        {'center': Vec(-7,   3, -15), 'radius': 3, 'color': Vec(1,1,0), 'specular': 1000, 'reflective': 0.5},
        {'center': Vec( 4,  -1,  -8), 'radius': 1, 'color': Vec(1,0,1), 'specular': 500,  'reflective': 0.3},
    ]
    lights = [
        {'type': 'ambient', 'intensity': 0.2},
        {'type': 'point',   'intensity': 0.6, 'position': Vec( 2,  1,  0), 'color': Vec(1,1,1)},
        {'type': 'point',   'intensity': 0.3, 'position': Vec(-3,  5, -2), 'color': Vec(1,0,0)},
        {'type': 'point',   'intensity': 0.3, 'position': Vec( 3,  5, -2), 'color': Vec(0,1,1)},
        {'type': 'point',   'intensity': 0.2, 'position': Vec( 0,  5, -5), 'color': Vec(1,1,0)},
    ]

    for y in range(height):
        for x in range(width):
            # camera rays
            px = (2*(x+0.5)/width - 1) * math.tan(fov/2) * width/height
            py = -(2*(y+0.5)/height - 1) * math.tan(fov/2)
            dir = Vec(px, py, -1).normalize()
            col = trace_ray(Vec(0,0,0), dir, spheres, lights)
            pixels[x,y] = (clamp(col.x*255), clamp(col.y*255), clamp(col.z*255))

    image.save('raytraced_scene4.png')
    print("Rendered and saved to output.png")

if __name__=='__main__':
    render()
