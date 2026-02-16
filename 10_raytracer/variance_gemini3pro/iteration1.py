import numpy as np
from PIL import Image
import time

def normalize(v):
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return v / (norm + 1e-10) # Avoid division by zero

def sphere_intersect(center, radius, rays_o, rays_d):
    """
    Vectorized ray-sphere intersection.
    Returns distance to intersection (t) or infinity if no hit.
    """
    b = 2 * np.einsum('ij,ij->i', rays_d, rays_o - center)
    c = np.linalg.norm(rays_o - center, axis=1) ** 2 - radius ** 2
    delta = b ** 2 - 4 * c
    
    # Initialize t as infinity
    t = np.full(rays_o.shape[0], np.inf)
    
    # Find valid intersections (delta > 0)
    cond = delta > 0
    d_sqrt = np.sqrt(delta[cond])
    
    # Calculate both solutions
    t1 = (-b[cond] - d_sqrt) / 2
    t2 = (-b[cond] + d_sqrt) / 2
    
    # We only care about positive t (in front of camera)
    # Since t1 < t2, if t1 > 0, it is the first hit. 
    # If t1 < 0 and t2 > 0, we are inside the sphere (t2 is hit).
    
    # Update t for the valid indices
    # Filter for positive t1
    mask_t1 = t1 > 0
    t[np.where(cond)[0][mask_t1]] = t1[mask_t1]
    
    return t

def trace_rays(rays_o, rays_d, spheres, lights, ambient=0.05, bounces=0):
    """
    Main raytracing function.
    """
    # Find nearest intersection
    nearest_t = np.full(rays_o.shape[0], np.inf)
    nearest_obj = np.full(rays_o.shape[0], -1, dtype=int)
    
    for i, sphere in enumerate(spheres):
        t = sphere_intersect(sphere['center'], sphere['radius'], rays_o, rays_d)
        mask = t < nearest_t
        nearest_t[mask] = t[mask]
        nearest_obj[mask] = i
        
    # Initialize color array (black background)
    color = np.zeros(rays_o.shape)
    
    # Indices where a ray hit an object
    hit_mask = nearest_t != np.inf
    if not np.any(hit_mask):
        return color
        
    # Intersection points and normals
    P = rays_o[hit_mask] + nearest_t[hit_mask][:, np.newaxis] * rays_d[hit_mask]
    
    # Calculate normals
    N = np.zeros_like(P)
    
    # Material properties arrays for vectorized calculation
    mat_colors = np.zeros_like(P)
    mat_specular = np.zeros(P.shape[0])
    mat_reflect = np.zeros(P.shape[0])
    
    # Fill in material properties based on which object was hit
    unique_hits = np.unique(nearest_obj[hit_mask])
    for idx in unique_hits:
        obj_mask = nearest_obj[hit_mask] == idx
        sphere = spheres[idx]
        
        # Normal = (Point - Center) / Radius
        N[obj_mask] = (P[obj_mask] - sphere['center']) / sphere['radius']
        
        mat_colors[obj_mask] = sphere['color']
        mat_specular[obj_mask] = sphere['specular']
        mat_reflect[obj_mask] = sphere['reflect']

    N = normalize(N)
    
    # Nudge point slightly along normal to avoid self-intersection artifacts
    P = P + N * 1e-5
    
    # --- Lighting Calculation ---
    
    # Start with Ambient
    pixel_colors = np.zeros_like(P)
    
    for light in lights:
        L = light['position'] - P
        dist_light = np.linalg.norm(L, axis=1, keepdims=True)
        L = normalize(L)
        
        # Diffuse (Lambert)
        diffuse_intensity = np.maximum(0, np.einsum('ij,ij->i', N, L))
        
        # Specular (Blinn-Phong)
        V = -normalize(rays_d[hit_mask])
        H = normalize(L + V)
        specular_intensity = np.maximum(0, np.einsum('ij,ij->i', N, H)) ** 50
        
        # Shadows (simple check)
        # Cast ray from P towards Light. If it hits anything closer than the light distance, it's shadowed.
        in_shadow = np.zeros(P.shape[0], dtype=bool)
        
        for sphere in spheres:
            t_shadow = sphere_intersect(sphere['center'], sphere['radius'], P, L)
            # If t is positive and less than distance to light
            mask_shadow = (t_shadow > 0) & (t_shadow < dist_light.flatten())
            in_shadow |= mask_shadow
            
        # Combine
        light_color = np.array(light['color'])
        
        # Apply Diffuse (if not shadowed)
        diffuse = (~in_shadow)[:, np.newaxis] * diffuse_intensity[:, np.newaxis] * light_color
        
        # Apply Specular (if not shadowed)
        specular = (~in_shadow)[:, np.newaxis] * specular_intensity[:, np.newaxis] * light_color * mat_specular[:, np.newaxis]
        
        pixel_colors += (mat_colors * diffuse) + specular

    # Add ambient
    pixel_colors += mat_colors * ambient
    
    # --- Reflection (Recursive) ---
    if bounces > 0:
        # Reflect ray: R = D - 2(D.N)N
        D_reflect = rays_d[hit_mask] - 2 * np.einsum('ij,ij->i', rays_d[hit_mask], N)[:, np.newaxis] * N
        D_reflect = normalize(D_reflect)
        
        # Identify pixels that have reflectivity
        reflect_mask = mat_reflect > 0
        
        if np.any(reflect_mask):
            # Recursively trace only the reflective rays
            # We need to map the subset of P (reflective ones) to new rays
            # This is complex in pure vectorized form without full recursion depth logic.
            # Simplified: We calculate reflection for ALL hit pixels, but weight by reflectivity.
            
            reflected_colors = trace_rays(P, D_reflect, spheres, lights, ambient, bounces - 1)
            
            # Blend based on reflectivity
            pixel_colors += reflected_colors * mat_reflect[:, np.newaxis]

    color[hit_mask] = np.clip(pixel_colors, 0, 1)
    return color

def main():
    width = 800
    height = 600
    
    # --- Scene Setup ---
    
    # Spheres: {center, radius, color, specular_strength, reflectivity}
    spheres = [
        # Large floor sphere
        {'center': np.array([0, -10001, 0]), 'radius': 10000, 'color': np.array([0.2, 0.2, 0.2]), 'specular': 0.1, 'reflect': 0.2},
        # Central Main Sphere (highly reflective chrome)
        {'center': np.array([0, 0, 0]), 'radius': 1, 'color': np.array([0.1, 0.1, 0.1]), 'specular': 1.0, 'reflect': 0.8},
        # Left Sphere (Matte Red)
        {'center': np.array([-2.2, 0, -1]), 'radius': 1, 'color': np.array([0.7, 0.1, 0.1]), 'specular': 0.2, 'reflect': 0.1},
        # Right Sphere (Glassy Blue)
        {'center': np.array([2.2, 0, -1]), 'radius': 1, 'color': np.array([0.1, 0.1, 0.7]), 'specular': 0.8, 'reflect': 0.4},
        # Back Sphere (Purple)
        {'center': np.array([0, 1.5, -3]), 'radius': 1.5, 'color': np.array([0.4, 0.1, 0.4]), 'specular': 0.5, 'reflect': 0.3},
    ]

    # Lights: {position, color} - Many colorful lights
    lights = [
        {'position': np.array([5, 5, 5]), 'color': np.array([1, 1, 1])},       # White Main
        {'position': np.array([-5, 5, 5]), 'color': np.array([1, 0, 0])},      # Red
        {'position': np.array([0, 5, -5]), 'color': np.array([0, 1, 0])},      # Green
        {'position': np.array([0, 2, 3]), 'color': np.array([0, 0.5, 1])},     # Blue filler
        {'position': np.array([-3, 1, 2]), 'color': np.array([1, 0.5, 0])},    # Orange low
        {'position': np.array([3, 1, 2]), 'color': np.array([1, 0, 1])},       # Magenta low
    ]

    # --- Camera Setup ---
    camera = np.array([0, 1, 4])
    ratio = float(width) / height
    screen = (-1, 1 / ratio, 1, -1 / ratio) # left, top, right, bottom

    print(f"Rendering {width}x{height} image with {len(lights)} lights and {len(spheres)} objects...")
    start_time = time.time()

    # Generate ray directions
    Y, X = np.meshgrid(np.linspace(screen[1], screen[3], height), np.linspace(screen[0], screen[2], width), indexing='ij')
    # Directions: vectors from camera to screen pixels (Z-axis points into screen at -1)
    rays_d = np.stack([X, Y, np.full_like(X, -1)], axis=-1)
    # Normalize
    rays_d = rays_d / np.linalg.norm(rays_d, axis=2, keepdims=True)
    
    # Flatten for vectorized processing
    rays_o = np.full_like(rays_d, camera)
    
    rays_o_flat = rays_o.reshape(-1, 3)
    rays_d_flat = rays_d.reshape(-1, 3)
    
    # --- Render ---
    # We do 1 bounce of reflection for performance/simplicity
    col_flat = trace_rays(rays_o_flat, rays_d_flat, spheres, lights, bounces=1)
    
    # Reshape and Save
    img_data = col_flat.reshape(height, width, 3)
    # Convert to 8-bit integer
    img_data = (np.clip(img_data, 0, 1) * 255).astype(np.uint8)
    
    img = Image.fromarray(img_data)
    img.save("raytracer_output.png")
    
    print(f"Done! Saved to 'raytracer_output.png'. Time taken: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()