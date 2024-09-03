import torch
import numpy as np
from PIL import Image
from tqdm import tqdm


def normalize(vector):
    return vector / torch.norm(vector)


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = normalize(direction)


class Sphere:
    def __init__(self, center, radius, color, material):
        self.center = center
        self.radius = radius
        self.color = color
        self.material = material

    def intersect(self, ray):
        oc = ray.origin - self.center
        a = torch.dot(ray.direction, ray.direction)
        b = 2.0 * torch.dot(oc, ray.direction)
        c = torch.dot(oc, oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c

        if discriminant > 0:
            t = (-b - torch.sqrt(discriminant)) / (2.0 * a)
            if t > 0:
                point = ray.origin + t * ray.direction
                normal = normalize(point - self.center)
                return t, point, normal
        return None


class Material:
    def __init__(self, albedo, specular, reflection):
        self.albedo = albedo
        self.specular = specular
        self.reflection = reflection


class Light:
    def __init__(self, position, intensity):
        self.position = position
        self.intensity = intensity


def reflect(v, n):
    return v - 2 * torch.dot(v, n) * n


def ray_color(ray, world, lights, depth=0):
    if depth > 3:
        return torch.tensor([0., 0., 0.])

    closest_hit = None
    closest_obj = None

    for obj in world:
        hit = obj.intersect(ray)
        if hit:
            t, point, normal = hit
            if closest_hit is None or t < closest_hit[0]:
                closest_hit = hit
                closest_obj = obj

    if closest_hit is None:
        return torch.tensor([0.5, 0.7, 1.0])  # Sky color

    t, point, normal = closest_hit
    material = closest_obj.material
    color = torch.tensor([0., 0., 0.])

    for light in lights:
        light_dir = normalize(light.position - point)
        shadow_ray = Ray(point + normal * 1e-4, light_dir)
        shadow_hit = any(obj.intersect(shadow_ray) for obj in world if obj != closest_obj)

        if not shadow_hit:
            diffuse = torch.max(torch.dot(normal, light_dir), torch.tensor(0.0))
            reflect_dir = reflect(-light_dir, normal)
            spec = torch.max(torch.dot(reflect_dir, -ray.direction), torch.tensor(0.0))
            specular = torch.pow(spec, torch.tensor(material.specular))
            color += material.albedo * closest_obj.color * diffuse * light.intensity
            color += torch.tensor([1., 1., 1.]) * specular * light.intensity

    if material.reflection > 0:
        reflect_dir = reflect(ray.direction, normal)
        reflect_ray = Ray(point + normal * 1e-4, reflect_dir)
        reflect_color = ray_color(reflect_ray, world, lights, depth + 1)
        color = color * (1 - material.reflection) + reflect_color * material.reflection

    return torch.clamp(color, 0, 1)


def render(world, lights, width, height):
    aspect_ratio = float(width) / float(height)
    image = torch.zeros((height, width, 3))

    total_pixels = width * height
    with tqdm(total=total_pixels, desc="Rendering", unit="pixel") as pbar:
        for j in range(height):
            for i in range(width):
                u = float(i) / float(width)
                v = float(j) / float(height)

                ray = Ray(
                    origin=torch.tensor([0., 0., 0.]),
                    direction=normalize(torch.tensor([-2. + 4. * u * aspect_ratio, -2. + 4. * v, -1.]))
                )

                color = ray_color(ray, world, lights)
                image[j, i] = color
                pbar.update(1)

    return image


# Set up the world
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
img_np = (image.numpy() * 255).astype(np.uint8)
Image.fromarray(img_np).save("improved_raytraced_image.png")
print("Image saved as 'improved_raytraced_image.png'")
