import os
import glob

image_dir = "data/images"
image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))

print(f"Nombre d'images détectées : {len(image_paths)}")
