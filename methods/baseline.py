import cv2
import mediapipe as mp
import math
import argparse
import json

VISIBILITY_THRESHOLD = 0.9
VISIBLE_FRAMES = 15

# Mediapipe landmark IDs for the necessary landmarks
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
    'Left Elbow': 14
}

SCALING_CONSTANTS = {
    'WAIST_SCALE': 3.7,
    'LEG_SCALE': 1,
    'TORSO_SCALE': 0.95,
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

# Core functionality

def run_baseline(video_path, shoulder_width_inches):
    cap = cv2.VideoCapture(video_path)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    valid_counter = 0
    scaling_factor = None
    final_coords = None

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
        raise RuntimeError("No valid frame sequence found for baseline method.")

    measurements = {
        "shoulder_width": shoulder_width_inches,
        "waist": round(calculate_distance(final_coords['Right Hip'], final_coords['Left Hip']) * scaling_factor * SCALING_CONSTANTS['WAIST_SCALE'], 2),
        "torso_height": round(calculate_distance(final_coords['Right Shoulder'], final_coords['Right Hip']) * scaling_factor * SCALING_CONSTANTS['TORSO_SCALE'], 2),
        "leg_height": round(calculate_distance(final_coords['Right Hip'], final_coords['Right Foot']) * scaling_factor * SCALING_CONSTANTS['LEG_SCALE'], 2),
    }

    return measurements

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Version")
    parser.add_argument("--video", required=True, help="video path")
    parser.add_argument("--shoulder_width", type=float, required=True, help="Calibration reference measurement")
    parser.add_argument("--output", default="outputs/baseline_output.json")

    args = parser.parse_args()

    results = run_baseline(args.video, args.shoulder_width)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print('Output saved')