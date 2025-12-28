import cv2
import mediapipe as mp
import math
import argparse
import json
import pickle
import pandas as pd

VISIBILITY_THRESHOLD = 0.9
VISIBLE_FRAMES = 15

LANDMARKS_TO_TRACK = {
    'Right Shoulder': 11,
    'Left Shoulder': 12,
    'Right Hip': 23,
    'Left Hip': 24,
    'Right Knee': 25,
    'Left Knee': 26,
    'Right Foot': 27,
    'Left Foot': 28,
    'Right Elbow': 13,
    'Left Elbow': 14,
    'Right Wrist': 15,
    'Left Wrist': 16
}

SCALING_CONSTANTS = {
    'TORSO_SCALE': 0.95,
    'LEG_SCALE': 1.1
}

def calculate_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

def extract_visible_landmarks(landmarks, frame_w, frame_h):
    coords = {}

    for name, idx in LANDMARKS_TO_TRACK.items():
        lm = landmarks[idx]
        if lm.visibility < VISIBILITY_THRESHOLD:
            return False, None

        coords[name] = (
            int(lm.x * frame_w),
            int(lm.y * frame_h)
        )

    return True, coords

def run_method2(video_path, shoulder_width_inches, gender_input, chest_model, waist_model):
    cap = cv2.VideoCapture(video_path)

    pose = mp.solutions.pose.Pose()

    valid_counter = 0
    final_coords = None
    scaling_factor = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if not results.pose_landmarks:
            valid_counter = 0
            continue

        valid, coords = extract_visible_landmarks(
            results.pose_landmarks.landmark, w, h
        )

        if not valid:
            valid_counter = 0
            continue

        valid_counter += 1

        if valid_counter == VISIBLE_FRAMES:
            shoulder_px = calculate_distance(
                coords['Right Shoulder'],
                coords['Left Shoulder']
            )

            scaling_factor = shoulder_width_inches / shoulder_px
            final_coords = coords
            break

    cap.release()

    if final_coords is None:
        raise RuntimeError("No valid frame sequence found.")

    if gender_input.lower() == 'male' or gender_input.lower() == 'm':
        gender = 0
    else:
        gender = 1
    
    thigh = calculate_distance(final_coords['Right Knee'], final_coords['Right Hip']) * scaling_factor
    calf = thigh * 0.65
    forearm = calculate_distance(final_coords['Right Elbow'], final_coords['Right Wrist']) * scaling_factor
    shoulder = shoulder_width_inches
    torso_height = calculate_distance(final_coords['Right Shoulder'],final_coords['Right Hip']) * scaling_factor * SCALING_CONSTANTS['TORSO_SCALE']
    leg_height = calculate_distance(final_coords['Right Hip'],final_coords['Right Foot']) * scaling_factor * SCALING_CONSTANTS['LEG_SCALE']

    X = pd.DataFrame([[
        gender,
        calf,
        forearm,
        shoulder,
        thigh
    ]], columns=['gender', 'calf', 'forearm', 'shoulder-breadth', 'thigh'])

    chest = float(chest_model.predict(X)[0])

    X['torso-to-leg-length'] = torso_height / leg_height
    waist = float(waist_model.predict(X)[0])

    return {
        "shoulder_width": round(shoulder, 2),
        "calf": round(calf, 2),
        "forearm": round(forearm, 2),
        "thigh": round(thigh, 2),
        "torso_height": round(torso_height, 2),
        "leg_height": round(leg_height, 2),
        "chest": round(chest, 2),
        "waist": round(waist, 2),
        "gender": gender
    }

# -----------------------------
# CLI Entry Point
# -----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regression Method")
    parser.add_argument("--video", required=True)
    parser.add_argument("--shoulder_width", type=float, required=True)
    parser.add_argument("--gender", type=str, required=True)
    parser.add_argument("--chest_model", default="models/chest_model.pkl")
    parser.add_argument("--waist_model", default="models/waist_model.pkl")
    parser.add_argument("--output", default="outputs/regression_output.json")

    args = parser.parse_args()

    with open(args.chest_model, "rb") as f:
        chest_model = pickle.load(f)

    with open(args.waist_model, "rb") as f:
        waist_model = pickle.load(f)

    results = run_method2(
        args.video,
        args.shoulder_width,
        args.gender,
        chest_model,
        waist_model
    )

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print('Output Saved!')
