import cv2
import mediapipe as mp
import numpy as np
import math
import argparse
import json
import torch
from PIL import Image
import pycocotools.mask as mask_util

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

VISIBILITY_THRESHOLD = 0.9
REQUIRED_FRAMES = 30
HEIGHT_OVERRIDE_THRESHOLD = 0.05  # 5%

TEXT_PROMPT = "card . person."
CARD_WIDTH = 8.6 #Cm

GROUNDING_MODEL = "IDEA-Research/grounding-dino-tiny"
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE = "cpu"

# Pose landmarks
LANDMARKS_FRONT = {
    'Right Shoulder': 11,
    'Left Shoulder': 12,
    'Right Hip': 23,
    'Left Hip': 24,
    'Right Knee': 25,
    'Left Knee': 26,
    'Right Foot': 27,
    'Left Foot': 28,
    'Right Elbow': 13,
    'Left Elbow': 14
}

LANDMARKS_LEFT = {
    'Left Shoulder': 12,
    'Left Knee': 26,
    'Left Foot': 28,
    'Left Elbow': 14
}

LANDMARKS_RIGHT = {
    'Right Shoulder': 11,
    'Right Knee': 25,
    'Right Foot': 27,
    'Right Elbow': 13
}

SCALING_CONSTANTS = {
    'TORSO_SCALE': 0.95,
    'LEG_SCALE': 1.2,
}

# Models
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

processor = AutoProcessor.from_pretrained(GROUNDING_MODEL)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
    GROUNDING_MODEL
).to(DEVICE)

# Utility functions
def calculate_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def extract_landmarks(landmarks, frame_w, frame_h, required):
    coords = {}
    for name, idx in required.items():
        lm = landmarks[idx]
        if lm.visibility < VISIBILITY_THRESHOLD:
            return None
        coords[name] = (int(lm.x * frame_w), int(lm.y * frame_h))
    return coords

def wait_for_valid_frame(video_path, required_landmarks):
    cap = cv2.VideoCapture(video_path)
    count = 0
    final_coords = None
    final_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if not results.pose_landmarks:
            count = 0
            continue

        coords = extract_landmarks(
            results.pose_landmarks.landmark, w, h, required_landmarks
        )

        if coords is None:
            count = 0
            continue

        count += 1
        if count == REQUIRED_FRAMES:
            final_coords = coords
            final_frame = frame
            break

    cap.release()

    if final_coords is None:
        raise RuntimeError("No valid frame sequence found.")

    return final_frame, final_coords

def grounding_and_sam(image):
    sam2_predictor.set_image(np.array(image))
    inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.1,
        target_sizes=[image.size[::-1]]
    )

    boxes = results[0]["boxes"].cpu().numpy()
    labels = results[0]["labels"]

    masks, _, _ = sam2_predictor.predict(
        box=boxes, point_coords=None, point_labels=None, multimask_output=False
    )

    return labels, masks

def rle_to_mask(mask):
    return mask.astype(np.uint8)

def longest_edge(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    e1 = np.linalg.norm(box[0] - box[1])
    e2 = np.linalg.norm(box[1] - box[2])
    return max(e1, e2)

def bbox_dims(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return h, w

def run_method(front_video, side_video, height_cli):
    # Front view
    front_frame, front_coords = wait_for_valid_frame(front_video, LANDMARKS_FRONT)
    image = Image.fromarray(cv2.cvtColor(front_frame, cv2.COLOR_BGR2RGB))

    labels, masks = grounding_and_sam(image)
    if 'card' not in labels or 'person' not in labels:
        return 'Invalid input sequence. Card or person not present!'

    card_idx = labels.index("card")
    person_idx = labels.index("person")

    card_mask = rle_to_mask(masks[card_idx][0])
    person_mask = rle_to_mask(masks[person_idx][0])

    card_px = longest_edge(card_mask)
    pixel_to_inch = card_px / CARD_WIDTH

    person_h_px, person_w_px = bbox_dims(person_mask)
    height_est = person_h_px / pixel_to_inch

    if abs(height_est - height_cli) / height_cli > HEIGHT_OVERRIDE_THRESHOLD:
        final_height = height_cli
    else:
        final_height = height_est

    scaling_factor = person_h_px / final_height

    shoulder = calculate_distance(front_coords['Right Shoulder'], front_coords['Left Shoulder']) / scaling_factor
    hip = calculate_distance(front_coords['Right Hip'], front_coords['Left Hip']) / scaling_factor
    torso = calculate_distance(front_coords['Right Shoulder'], front_coords['Right Hip']) * SCALING_CONSTANTS['TORSO_SCALE'] / scaling_factor
    leg = calculate_distance(front_coords['Right Hip'], front_coords['Right Foot']) * SCALING_CONSTANTS['LEG_SCALE'] / scaling_factor

    # Side view
    side_frame, _ = wait_for_valid_frame(side_video, LANDMARKS_RIGHT)
    side_img = Image.fromarray(cv2.cvtColor(side_frame, cv2.COLOR_BGR2RGB))
    labels_s, masks_s = grounding_and_sam(side_img)

    if 'person' not in labels:
        return 'Invalid Side view! Person not clearly visible'
    person_idx = labels_s.index("person")
    side_mask = rle_to_mask(masks_s[person_idx][0])

    h_px, w_px = bbox_dims(side_mask)
    side_width = w_px / scaling_factor

    waist = math.pi * (3*((hip/2) + (side_width/2)) - math.sqrt((3*(hip/2) + (side_width/2))*((hip/2) + 3*(side_width/2))))

    return {
        "shoulder_width": round(shoulder  / 2.54, 2),
        "torso_height": round(torso, 2) / 2.54,
        "leg_height": round(leg, 2) / 2.54,
        "waist": round(waist, 2) / 2.54,
        "height_used": round(final_height, 2) / 2.54
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object-calibrated")
    parser.add_argument("--front_video", required=True)
    parser.add_argument("--side_video", required=True)
    parser.add_argument("--height", type=float, required=True)
    parser.add_argument("--output", default="object_calibrated_output.json")

    args = parser.parse_args()

    results = run_method(args.front_video, args.side_video, args.height)
    if type(results) == 'str':
        print('Run failed!')

    else:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

        print("Output Saved")