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
    def __init__(self, center, radius, material):
        self.center = center.to(device)
        self.radius = radius
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
    def __init__(self, color, albedo, specular, reflection, refraction, refractive_index):
        self.color = color.to(device)
        self.albedo = albedo
        self.specular = specular
        self.reflection = reflection
        self.refraction = refraction
        self.refractive_index = refractive_index


class Light:
    def __init__(self, position, intensity, radius):
        self.position = position.to(device)
        self.intensity = intensity
        self.radius = radius


def reflect(v, n):
    return v - 2 * torch.sum(v * n, dim=-1, keepdim=True) * n


def refract(v, n, ni_over_nt):
    dt = torch.sum(v * n, dim=-1, keepdim=True)
    discriminant = 1.0 - ni_over_nt * ni_over_nt * (1 - dt * dt)
    return torch.where(
        discriminant > 0,
        ni_over_nt * (v - n * dt) - n * torch.sqrt(discriminant),
        reflect(v, n)
    )


def fresnel(v, n, ni_over_nt):
    cosine = torch.clamp(torch.sum(-v * n, dim=-1), min=0.0)
    sine = torch.sqrt(1 - cosine * cosine)
    r0 = ((1 - ni_over_nt) / (1 + ni_over_nt)) ** 2
    return r0 + (1 - r0) * ((1 - cosine) ** 5)


def ray_color(ray_origins, ray_directions, world, lights, depth=0):
    if depth > 5:
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
            shadow_origins = points + normals * 1e-4

            # Soft shadows
            shadow_samples = 10
            shadow_rays = shadow_samples * points.shape[0]
            random_offsets = torch.randn(shadow_rays, 3, device=device) * light.radius
            shadow_targets = light.position.unsqueeze(0) + random_offsets
            shadow_directions = normalize(shadow_targets - shadow_origins.repeat_interleave(shadow_samples, dim=0))

            in_shadow = torch.zeros(shadow_rays, dtype=torch.bool, device=device)
            for other_sphere in world:
                if other_sphere != sphere:
                    _, _, _, shadow_hit = other_sphere.intersect(shadow_origins.repeat_interleave(shadow_samples, dim=0), shadow_directions)
                    in_shadow |= shadow_hit

            shadow_intensity = 1 - in_shadow.float().view(-1, shadow_samples).mean(dim=1)

            diffuse = torch.clamp(torch.sum(normals * light_dir, dim=-1), min=0)
            sphere_color += material.color * material.albedo * diffuse.unsqueeze(-1) * light.intensity * shadow_intensity.unsqueeze(-1)

            reflect_dir = reflect(-light_dir, normals)
            spec = torch.clamp(torch.sum(reflect_dir * -ray_directions[sphere_mask], dim=-1), min=0)
            specular = torch.pow(spec, material.specular)
            sphere_color += torch.tensor([1., 1., 1.], device=device) * specular.unsqueeze(-1) * light.intensity * shadow_intensity.unsqueeze(-1)

        if material.reflection > 0:
            reflect_dir = reflect(ray_directions[sphere_mask], normals)
            reflect_origins = points + normals * 1e-4
            reflect_color = ray_color(reflect_origins, reflect_dir, world, lights, depth + 1)
            sphere_color = sphere_color * (1 - material.reflection) + reflect_color * material.reflection

        if material.refraction > 0:
            refract_dir = refract(ray_directions[sphere_mask], normals, 1.0 / material.refractive_index)
            refract_origins = points - normals * 1e-4
            refract_color = ray_color(refract_origins, refract_dir, world, lights, depth + 1)

            fresnel_factor = fresnel(ray_directions[sphere_mask], normals, 1.0 / material.refractive_index)
            sphere_color = sphere_color * fresnel_factor.unsqueeze(-1) + refract_color * (1 - fresnel_factor.unsqueeze(-1))

        color[sphere_mask] = sphere_color

    if len(orig_shape) == 1:
        color = color.squeeze(0)

    return torch.clamp(color, 0, 1)


def render(world, lights, width, height, samples_per_pixel=4):
    aspect_ratio = float(width) / float(height)

    image = torch.zeros((height, width, 3), device=device)

    for s in range(samples_per_pixel):
        print(f"Rendering sample {s + 1}/{samples_per_pixel}")
        x = torch.linspace(0, 1, width, device=device) + torch.rand(width, device=device) / width
        y = torch.linspace(0, 1, height, device=device) + torch.rand(height, device=device) / height
        yy, xx = torch.meshgrid(y, x, indexing='ij')

        ray_directions = normalize(torch.stack(
            [
                -1 + 2 * xx * aspect_ratio,
                -1 + 2 * yy,
                -torch.ones_like(xx)
            ], dim=-1
        ))

        ray_origins = torch.tensor([0, 0, 1], device=device).expand(height, width, 3)

        image += ray_color(ray_origins, ray_directions, world, lights)

    image /= samples_per_pixel
    return image


world = [
    # Ground (move to bottom)
    Sphere(torch.tensor([0, -1000, -1]), 999.5, Material(torch.tensor([0.2, 0.2, 0.2]), 0.9, 10, 0.0, 0.0, 1.0)),
    # Large center sphere
    Sphere(torch.tensor([0, -0.5, -1]), 0.5, Material(torch.tensor([0.7, 0.7, 0.7]), 0.1, 50, 0.8, 0.0, 1.0)),
    # Small left sphere
    Sphere(torch.tensor([-1, 0.25, -0.5]), 0.25, Material(torch.tensor([0.9, 0.2, 0.2]), 0.9, 10, 0.0, 0.0, 1.0)),
    # Small right sphere (glass)
    Sphere(torch.tensor([1, 0.25, -0.5]), 0.25, Material(torch.tensor([1.0, 1.0, 1.0]), 0.1, 100, 0.1, 0.9, 1.5)),
    # Back-left sphere
    Sphere(torch.tensor([-0.75, -0.25, -2]), 0.5, Material(torch.tensor([0.2, 0.8, 0.2]), 0.1, 50, 0.7, 0.0, 1.0)),
    # Back-right sphere
    Sphere(torch.tensor([0.75, -0.25, -2]), 0.5, Material(torch.tensor([0.2, 0.2, 0.8]), 0.9, 10, 0.0, 0.0, 1.0)),
]

lights = [
    Light(torch.tensor([-5, 5, 5]), 0.7, 0.5),
    Light(torch.tensor([5, 5, 5]), 0.3, 0.5),
]

# Render
print("Starting render...")
image = render(world, lights, 1024, 768, samples_per_pixel=64)
print("Render complete!")

# Convert to numpy and save
print("Saving image...")
img_np = (image.cpu().numpy() * 255).astype(np.uint8)
Image.fromarray(img_np).save("complex_scene_raytraced_image.png")
print("Image saved as 'complex_scene_raytraced_image.png'")
