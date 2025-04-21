import argparse
import os
import cv2
import numpy as np
from PIL import Image, ExifTags
from segment_anything import SamPredictor, sam_model_registry

def parse_args():
    parser = argparse.ArgumentParser(description="Segment Anything Model (SAM) for image segmentation.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images to segment.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save segmented images.")
    parser.add_argument("--sam_checkpoint", type=str, required=True, help="Path to SAM checkpoint file.")
    parser.add_argument("--model_type", type=str, default="vit_h", choices=["vit_h", "vit_l", "vit_b"], help="Type of SAM model to use.")
    return parser.parse_args()

args = parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# === Initialize SAM ===
sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint).cuda()
predictor = SamPredictor(sam)

# === interactive function ===
box = []
def draw_rectangle(event, x, y, flags, param):
    global box, temp_img
    if event == cv2.EVENT_LBUTTONDOWN:
        box = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        box.append((x, y))
        temp_img = img.copy()
        cv2.rectangle(temp_img, box[0], box[1], (0, 255, 0), 2)
        cv2.imshow("Select Bounding Box", temp_img)

# === main process ===
image_files = [f for f in sorted(os.listdir(args.image_dir)) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for filename in image_files:
    filepath = os.path.join(args.image_dir, filename)
    print(f"\nProcessing {filename}")

    # load image
    img = cv2.imread(filepath)
    temp_img = img.copy()

    # Set interactive window
    cv2.namedWindow("Select Bounding Box")
    cv2.setMouseCallback("Select Bounding Box", draw_rectangle)
    while True:
        cv2.imshow("Select Bounding Box", temp_img)
        key = cv2.waitKey(1)
        if key == ord('q') and len(box) == 2:
            break
    cv2.destroyAllWindows()
    # === Segment Anything Inference ===
    predictor.set_image(img)
    x0, y0 = box[0]
    x1, y1 = box[1]
    input_box = np.array([[min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]])
    masks, scores, logits = predictor.predict(box=input_box, multimask_output=False)
    mask = masks[0]
    # === Save mask ===
    mask = mask.astype(np.uint8) * 255
    out_path = os.path.join(args.output_dir, filename.replace(".jpg", ".png").replace(".jpeg", ".png"))
    cv2.imwrite(out_path, mask)
    print(f"Saved mask to {out_path}")

print("\n All images processed and saved.")
