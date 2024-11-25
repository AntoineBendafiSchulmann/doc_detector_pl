import os
import glob
import cv2
import numpy as np

def generate_document_masks(image_dir, mask_dir):
    os.makedirs(mask_dir, exist_ok=True)
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
    if not image_paths:
        print(f"Aucune image trouvée dans {image_dir}.")
        return

    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Impossible de lire {image_path}. Ignoré.")
            continue

        blurred = cv2.GaussianBlur(image, (5, 5), 0)

        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros_like(image)

        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)


            if len(approx) == 4 and cv2.contourArea(approx) > 1000:

                cv2.drawContours(mask, [approx], -1, (255), thickness=cv2.FILLED)

        mask_path = os.path.join(mask_dir, os.path.basename(image_path))
        cv2.imwrite(mask_path, mask)
        print(f"Masque généré pour {image_path} -> {mask_path}")

if __name__ == "__main__":
    image_dir = "data/images"  
    mask_dir = "data/masks"    

    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Le dossier {image_dir} n'existe pas.")
    
    generate_document_masks(image_dir, mask_dir)
    print(f"Tous les masques ont été générés dans {mask_dir}.")
