import os
import glob
import cv2
import numpy as np

def generate_document_masks(image_dir, mask_dir, debug_dir=None):
    os.makedirs(mask_dir, exist_ok=True)
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

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

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(blurred)

        _, thresh = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(image)

        for contour in contours:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 4 and 1000 < cv2.contourArea(approx) < 1e6:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = h / w
                if 0.5 < aspect_ratio < 2.0: 
                    cv2.drawContours(mask, [approx], -1, (255), thickness=cv2.FILLED)

        mask_path = os.path.join(mask_dir, os.path.basename(image_path))
        cv2.imwrite(mask_path, mask)
        print(f"Masque généré pour {image_path} -> {mask_path}")

        if debug_dir:
            overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            mask_colored = cv2.merge([mask, mask, mask])
            overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
            debug_path = os.path.join(debug_dir, os.path.basename(image_path))
            cv2.imwrite(debug_path, overlay)
            print(f"Image avec masque superposé sauvegardée dans {debug_path}")

if __name__ == "__main__":
    image_dir = "data/images"
    mask_dir = "data/masks"
    debug_dir = "data/debug"

    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Le dossier {image_dir} n'existe pas.")
    
    generate_document_masks(image_dir, mask_dir, debug_dir)
    print(f"Tous les masques ont été générés dans {mask_dir}.")
