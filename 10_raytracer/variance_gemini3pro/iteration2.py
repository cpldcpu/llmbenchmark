import numpy as np
from PIL import Image
import time

def normalize(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (norm + 1e-9)

def sphere_intersect(centers, radii, origin, direction):
    # Vector from ray origin to sphere centers
    b = centers - origin
    # Project b onto ray direction
    epsilon = 1e-5
    t_projection = np.einsum('ijk,ijk->ij', b, direction)
    
    # Distance from center to ray
    d_sq = np.sum(b * b, axis=-1) - t_projection**2
    
    # Check if ray misses sphere (distance > radius)
    radius_sq = radii * radii
    miss_mask = d_sq > radius_sq
    
    # Calculate intersection depth
    thickness = np.sqrt(np.maximum(0, radius_sq - d_sq))
    t0 = t_projection - thickness
    t1 = t_projection + thickness
    
    # We only care about the closest intersection in front of the camera
    t = np.where(t0 > epsilon, t0, np.where(t1 > epsilon, t1, np.inf))
    t[miss_mask] = np.inf
    
    return t

def raytrace(width, height):
    # --- Scene Setup ---
    # Camera setup
    aspect_ratio = width / height
    x = np.linspace(-1, 1, width) * aspect_ratio
    y = np.linspace(1, -1, height) # +y is up
    
    # Create a grid of rays
    # origin is (0,0,0)
    origin = np.zeros((height, width, 3)) 
    
    # Direction vectors (screen is at z = -1)
    X, Y = np.meshgrid(x, y)
    direction = np.stack((X, Y, np.full_like(X, -1)), axis=-1)
    direction = normalize(direction)

    # Objects: [x, y, z]
    # Format: {'center': vec3, 'radius': float, 'color': vec3, 'specular': float, 'reflect': float}
    spheres = [
        # Floor (Giant sphere)
        {'center': np.array([0, -10001, -5]), 'radius': 10000, 'color': np.array([0.2, 0.2, 0.2]), 'specular': 0.0, 'reflect': 0.3},
        # Center Mirror Sphere
        {'center': np.array([0, 0.5, -3]), 'radius': 1.0, 'color': np.array([0.1, 0.1, 0.1]), 'specular': 1.0, 'reflect': 0.9},
        # Left Matte Red
        {'center': np.array([-1.8, 0, -4]), 'radius': 0.8, 'color': np.array([0.6, 0.1, 0.1]), 'specular': 0.2, 'reflect': 0.1},
        # Right Shiny Blue
        {'center': np.array([1.8, 0, -4]), 'radius': 0.8, 'color': np.array([0.1, 0.1, 0.6]), 'specular': 0.8, 'reflect': 0.5},
        # Small Gold (Front)
        {'center': np.array([0.5, -0.6, -2.0]), 'radius': 0.3, 'color': np.array([0.8, 0.6, 0.1]), 'specular': 0.9, 'reflect': 0.4},
    ]

    # Lights: [position, color, intensity]
    lights = [
        {'pos': np.array([-5, 5, -2]), 'color': np.array([1, 0.4, 0.4])},  # Warm Red Light
        {'pos': np.array([5, 5, -2]),  'color': np.array([0.4, 0.4, 1])},  # Cool Blue Light
        {'pos': np.array([0, 5, 0]),   'color': np.array([0.8, 0.8, 0.8])},# Top White Light
        {'pos': np.array([-2, 1, -1]), 'color': np.array([0.2, 1.0, 0.2])} # Small Green Fill
    ]

    # Initialize Image Buffer
    image = np.zeros((height, width, 3))
    reflection_strength = np.ones((height, width, 1)) # Tracks how much light is left after bounces

    # --- Main Tracing Loop (Recursive reflections via iteration) ---
    max_depth = 3
    
    for depth in range(max_depth):
        # 1. Find nearest intersection for every pixel
        nearest_t = np.full((height, width), np.inf)
        obj_indices = np.full((height, width), -1, dtype=int)
        
        for i, sphere in enumerate(spheres):
            t = sphere_intersect(sphere['center'], sphere['radius'], origin, direction)
            mask = t < nearest_t
            nearest_t[mask] = t[mask]
            obj_indices[mask] = i

        # Mask for pixels that hit something
        hit_mask = obj_indices != -1
        if not np.any(hit_mask):
            break

        # 2. Compute Intersection Points and Normals
        # We only compute math for pixels that actually hit something to save time
        hit_points = origin + direction * nearest_t[..., np.newaxis]
        
        # Prepare arrays for surface properties
        normals = np.zeros((height, width, 3))
        diffuse_colors = np.zeros((height, width, 3))
        specular_factors = np.zeros((height, width))
        reflectivity = np.zeros((height, width))

        # Fill properties based on which object was hit
        for i, sphere in enumerate(spheres):
            mask = (obj_indices == i)
            # Normal = (HitPoint - Center) / Radius
            normals[mask] = (hit_points[mask] - sphere['center']) / sphere['radius']
            diffuse_colors[mask] = sphere['color']
            specular_factors[mask] = sphere['specular']
            reflectivity[mask] = sphere['reflect']

        normals = normalize(normals)

        # Nudge hit points slightly along normal to prevent self-intersection acne
        hit_points = hit_points + normals * 1e-4

        # 3. Compute Lighting (Phong Model)
        pixel_color = np.zeros((height, width, 3))
        
        for light in lights:
            # Vector from intersection to light
            to_light = light['pos'] - hit_points
            dist_to_light = np.linalg.norm(to_light, axis=-1, keepdims=True)
            to_light = normalize(to_light)
            
            # Diffuse Shading (Lambert)
            # Dot product of Normal and Light Direction
            nl_dot = np.sum(normals * to_light, axis=-1)
            diffuse_intensity = np.maximum(0, nl_dot)[..., np.newaxis]
            
            # Specular Shading (Blinn-Phong)
            # Reflection of light vector
            # Vector to camera (view vector) is -direction
            view_dir = -direction
            half_vector = normalize(to_light + view_dir)
            nh_dot = np.maximum(0, np.sum(normals * half_vector, axis=-1))
            specular_intensity = (nh_dot ** 50)[..., np.newaxis] * specular_factors[..., np.newaxis]
            
            # Add light contribution
            # Simple shadows could be added here by raymarching to light, but skipping for speed/simplicity
            light_contribution = (diffuse_colors * diffuse_intensity * 0.7) + (specular_intensity * 1.0)
            pixel_color += light_contribution * light['color']

        # Ambient light
        pixel_color += diffuse_colors * 0.05

        # Accumulate color into final image
        # If this is a reflection pass, we multiply by the reflection strength of previous bounces
        image += pixel_color * reflection_strength * (1 - reflectivity[..., np.newaxis])
        
        # 4. Prepare for next bounce
        reflection_strength *= reflectivity[..., np.newaxis]
        
        # Update rays for reflection
        # R = D - 2(D . N)N
        d_dot_n = np.sum(direction * normals, axis=-1, keepdims=True)
        reflection_dir = direction - 2 * d_dot_n * normals
        
        origin = hit_points
        direction = normalize(reflection_dir)
        
        # Optimization: If reflection strength is negligible everywhere, stop
        if np.max(reflection_strength) < 0.01:
            break

    # Clip values to 0-1 range
    image = np.clip(image, 0, 1)
    return image

if __name__ == "__main__":
    w, h = 800, 600
    print(f"Rendering {w}x{h} scene with NumPy...")
    start_time = time.time()
    
    img_data = raytrace(w, h)
    
    # Convert to 8-bit integer format
    img_data = (img_data * 255).astype(np.uint8)
    img = Image.fromarray(img_data)
    
    filename = "raytraced_scene.png"
    img.save(filename)
    
    print(f"Done in {time.time() - start_time:.2f} seconds.")
    print(f"Saved to {filename}")