import torch
import numpy as np
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def normalize(vector):
    return vector / torch.norm(vector, dim=-1, keepdim=True)


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin.to(device)
        self.direction = normalize(direction.to(device))


class Sphere:
    def __init__(self, center, radius, color, material):
        self.center = center.to(device)
        self.radius = radius
        self.color = color.to(device)
        self.material = material

    def intersect(self, ray_origins, ray_directions):
        oc = ray_origins - self.center
        a = torch.sum(ray_directions * ray_directions, dim=-1)
        b = 2.0 * torch.sum(oc * ray_directions, dim=-1)
        c = torch.sum(oc * oc, dim=-1) - self.radius * self.radius
        discriminant = b * b - 4 * a * c

        mask = discriminant > 0
        t = torch.where(mask, (-b - torch.sqrt(discriminant)) / (2.0 * a), torch.tensor(float('inf'), device=device))
        hit_mask = (t > 0) & mask
        points = ray_origins + t.unsqueeze(-1) * ray_directions
        normals = normalize(points - self.center)
        return t, points, normals, hit_mask


class Material:
    def __init__(self, albedo, specular, reflection):
        self.albedo = albedo.to(device)
        self.specular = specular
        self.reflection = reflection


class Light:
    def __init__(self, position, intensity):
        self.position = position.to(device)
        self.intensity = intensity


def reflect(v, n):
    return v - 2 * torch.sum(v * n, dim=-1, keepdim=True) * n


def ray_color(ray_origins, ray_directions, world, lights, depth=0):
    if depth > 3:
        return torch.zeros_like(ray_origins)

    orig_shape = ray_origins.shape[:-1]
    if len(orig_shape) == 1:
        ray_origins = ray_origins.unsqueeze(0)
        ray_directions = ray_directions.unsqueeze(0)

    color = torch.zeros_like(ray_origins)
    hit_anything = torch.zeros(ray_origins.shape[:-1], dtype=torch.bool, device=device)
    closest_t = torch.ones(ray_origins.shape[:-1], device=device) * float('inf')
    closest_sphere_idx = torch.zeros(ray_origins.shape[:-1], dtype=torch.long, device=device)

    for i, sphere in enumerate(world):
        t, points, normals, hit_mask = sphere.intersect(ray_origins, ray_directions)
        closer_hit = (t < closest_t) & hit_mask
        closest_t = torch.where(closer_hit, t, closest_t)
        closest_sphere_idx = torch.where(closer_hit, torch.full_like(closest_sphere_idx, i), closest_sphere_idx)
        hit_anything |= hit_mask

    sky_color = torch.tensor([0.5, 0.7, 1.0], device=device).expand_as(color)
    color = torch.where(hit_anything.unsqueeze(-1), color, sky_color)

    for i, sphere in enumerate(world):
        sphere_mask = (closest_sphere_idx == i) & hit_anything
        if not sphere_mask.any():
            continue

        points = ray_origins[sphere_mask] + closest_t[sphere_mask].unsqueeze(-1) * ray_directions[sphere_mask]
        normals = normalize(points - sphere.center)

        material = sphere.material
        sphere_color = torch.zeros_like(points)

        for light in lights:
            light_dir = normalize(light.position - points)
            diffuse = torch.clamp(torch.sum(normals * light_dir, dim=-1), min=0)
            sphere_color += material.albedo * sphere.color * diffuse.unsqueeze(-1) * light.intensity

            reflect_dir = reflect(-light_dir, normals)
            spec = torch.clamp(torch.sum(reflect_dir * -ray_directions[sphere_mask], dim=-1), min=0)
            specular = torch.pow(spec, material.specular)
            sphere_color += torch.tensor([1., 1., 1.], device=device) * specular.unsqueeze(-1) * light.intensity

        if material.reflection > 0 and depth < 3:
            reflect_dir = reflect(ray_directions[sphere_mask], normals)
            reflect_origins = points + normals * 1e-4
            reflect_color = ray_color(reflect_origins, reflect_dir, world, lights, depth + 1)
            sphere_color = sphere_color * (1 - material.reflection) + reflect_color * material.reflection

        color[sphere_mask] = sphere_color

    if len(orig_shape) == 1:
        color = color.squeeze(0)

    return torch.clamp(color, 0, 1)


def render(world, lights, width, height):
    aspect_ratio = float(width) / float(height)

    x = torch.linspace(0, 1, width, device=device)
    y = torch.linspace(0, 1, height, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    ray_directions = normalize(torch.stack(
        [
            -2 + 4 * xx * aspect_ratio,
            -2 + 4 * yy,
            -torch.ones_like(xx)
        ], dim=-1
    ))

    ray_origins = torch.zeros(height, width, 3, device=device)

    image = ray_color(ray_origins, ray_directions, world, lights)
    return image


# Set up the world (unchanged)
world = [
    Sphere(torch.tensor([0., -0.5, -3.]), 0.5, torch.tensor([1., 0., 0.]), Material(torch.tensor([0.7, 0.3, 0.3]), 50., 0.2)),
    Sphere(torch.tensor([-1., 0., -4.]), 0.7, torch.tensor([0., 1., 0.]), Material(torch.tensor([0.3, 0.7, 0.3]), 10., 0.4)),
    Sphere(torch.tensor([1., 0., -4.]), 0.7, torch.tensor([0., 0., 1.]), Material(torch.tensor([0.3, 0.3, 0.7]), 100., 0.6)),
    Sphere(torch.tensor([0., -1000.5, -1.]), 1000., torch.tensor([0.5, 0.5, 0.5]), Material(torch.tensor([0.5, 0.5, 0.5]), 1., 0.)),
]

lights = [
    Light(torch.tensor([-5., 5., -5.]), 0.7),
    Light(torch.tensor([5., 5., -5.]), 0.3),
]

# Render
print("Starting render...")
image = render(world, lights, 800, 600)
print("Render complete!")

# Convert to numpy and save
print("Saving image...")
img_np = (image.cpu().numpy() * 255).astype(np.uint8)
Image.fromarray(img_np).save("optimized_cuda_raytraced_image.png")
print("Image saved as 'optimized_cuda_raytraced_image.png'")
