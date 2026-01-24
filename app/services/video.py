"""
Video analysis service for CasperPrep backend.
Handles eye contact detection using dlib facial landmarks.
"""

import cv2
import dlib
import numpy as np
from pathlib import Path
from typing import List, Tuple

# Lazy-loaded dlib models
_face_detector = None
_landmark_predictor = None

# Path to the facial landmark model
LANDMARK_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "shape_predictor_68_face_landmarks.dat"


def get_face_detector():
    """Lazy load dlib face detector."""
    global _face_detector
    if _face_detector is None:
        _face_detector = dlib.get_frontal_face_detector()
    return _face_detector


def get_landmark_predictor():
    """Lazy load dlib facial landmark predictor."""
    global _landmark_predictor
    if _landmark_predictor is None:
        if not LANDMARK_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Facial landmark model not found at {LANDMARK_MODEL_PATH}. "
                "Download from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            )
        print(f"Loading facial landmark model...")
        _landmark_predictor = dlib.shape_predictor(str(LANDMARK_MODEL_PATH))
    return _landmark_predictor


def analyze_eye_contact(video_path: Path, sample_rate: int = 5) -> dict:
    """
    Analyze eye contact throughout video using dlib facial landmarks.

    Args:
        video_path: Path to video file
        sample_rate: analyze every Nth frame (5 = every 5th frame)

    Returns:
        {
            "eye_contact_percentage": float,
            "issues": [{"start": float, "end": float}, ...]  # periods of lost eye contact > 2s
            "timeline": [{"time": float, "hasContact": bool}, ...]
        }
    """
    face_detector = get_face_detector()
    landmark_predictor = get_landmark_predictor()

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0  # fallback

    frame_count = 0
    analyzed_frames: List[Tuple[float, bool]] = []  # (timestamp, has_eye_contact)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only analyze every Nth frame
        if frame_count % sample_rate != 0:
            frame_count += 1
            continue

        timestamp = frame_count / fps
        eye_contact = _detect_eye_contact_dlib(frame, face_detector, landmark_predictor)
        analyzed_frames.append((timestamp, eye_contact))

        frame_count += 1

    cap.release()

    # Calculate percentage
    if not analyzed_frames:
        return {
            "eye_contact_percentage": 0.0,
            "issues": [],
            "timeline": []
        }

    # Normalize/smooth the data to remove rapid flickers (< 0.3s state changes)
    # 0.3s threshold from benchmark testing
    smoothed_frames = _smooth_eye_contact_data(analyzed_frames, min_duration=0.3)

    eye_contact_count = sum(1 for _, ec in smoothed_frames if ec)
    eye_contact_percentage = (eye_contact_count / len(smoothed_frames)) * 100

    # Find periods of lost eye contact > 2s (using smoothed data)
    issues = _find_eye_contact_issues(smoothed_frames, threshold_seconds=2.0)

    # Build timeline for graphing (convert to native Python types for JSON serialization)
    timeline = [{"time": float(t), "hasContact": bool(ec)} for t, ec in smoothed_frames]

    return {
        "eye_contact_percentage": round(eye_contact_percentage, 1),
        "issues": issues,
        "timeline": timeline
    }


def _detect_eye_contact_dlib(frame: np.ndarray, face_detector, landmark_predictor) -> bool:
    """
    Detect eye contact using dlib 68-point facial landmarks.

    Uses iris position relative to eye corners to determine gaze direction.
    Returns True if looking approximately at camera.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector(gray, 0)  # 0 = no upsampling for speed
    if len(faces) == 0:
        return False

    # Use largest face
    face = max(faces, key=lambda f: f.width() * f.height())

    # Get facial landmarks
    landmarks = landmark_predictor(gray, face)

    # Eye landmark indices (dlib 68-point model):
    # Left eye: 36-41
    # Right eye: 42-47

    left_eye_ratio = _get_gaze_ratio(gray, landmarks, [36, 37, 38, 39, 40, 41])
    right_eye_ratio = _get_gaze_ratio(gray, landmarks, [42, 43, 44, 45, 46, 47])

    if left_eye_ratio is None or right_eye_ratio is None:
        return False

    # Average gaze ratio
    avg_ratio = (left_eye_ratio + right_eye_ratio) / 2

    # Eye contact if gaze is centered (ratio between 0.40 and 0.60)
    # Tighter range from benchmark testing - less false positives
    return 0.40 < avg_ratio < 0.60


def _get_gaze_ratio(gray: np.ndarray, landmarks, eye_indices: List[int]) -> float | None:
    """
    Calculate gaze ratio for one eye using pupil position.

    Returns ratio from 0 (looking left) to 1 (looking right).
    Center gaze is around 0.5.
    """
    # Get eye region points
    eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in eye_indices])

    # Create mask for eye region
    height, width = gray.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [eye_points], 255)

    # Get bounding box of eye
    min_x = np.min(eye_points[:, 0])
    max_x = np.max(eye_points[:, 0])
    min_y = np.min(eye_points[:, 1])
    max_y = np.max(eye_points[:, 1])

    # Ensure bounds are valid
    min_x = max(0, min_x)
    max_x = min(width, max_x)
    min_y = max(0, min_y)
    max_y = min(height, max_y)

    if max_x <= min_x or max_y <= min_y:
        return None

    # Extract eye region
    eye_region = gray[min_y:max_y, min_x:max_x]
    eye_mask = mask[min_y:max_y, min_x:max_x]

    if eye_region.size == 0:
        return None

    # Apply mask
    eye_gray = cv2.bitwise_and(eye_region, eye_region, mask=eye_mask)

    # Threshold to find pupil (dark area) using percentile-based threshold
    # Percentile 35 performed best in benchmarks vs fixed threshold
    thresh_value = np.percentile(eye_gray[eye_mask > 0], 35) if np.any(eye_mask > 0) else 50
    _, threshold = cv2.threshold(eye_gray, thresh_value, 255, cv2.THRESH_BINARY_INV)
    threshold = cv2.bitwise_and(threshold, threshold, mask=eye_mask)

    # Find contours (pupil)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Find largest contour (assumed to be pupil)
    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)

    if M["m00"] == 0:
        return None

    # Centroid of pupil relative to eye width
    cx = int(M["m10"] / M["m00"])
    eye_width = max_x - min_x

    if eye_width == 0:
        return None

    return cx / eye_width


def _smooth_eye_contact_data(
    analyzed_frames: List[Tuple[float, bool]],
    min_duration: float = 0.5
) -> List[Tuple[float, bool]]:
    """
    Smooth eye contact data by removing rapid state changes.

    If a state (contact or no contact) lasts less than min_duration seconds,
    it's considered noise and merged with the surrounding state.

    Args:
        analyzed_frames: List of (timestamp, has_eye_contact) tuples
        min_duration: Minimum duration for a state change to be considered valid

    Returns:
        Smoothed list of (timestamp, has_eye_contact) tuples
    """
    if len(analyzed_frames) < 3:
        return analyzed_frames

    # First pass: identify segments and their durations
    segments = []  # [(start_idx, end_idx, state, duration)]

    current_state = analyzed_frames[0][1]
    segment_start_idx = 0

    for i in range(1, len(analyzed_frames)):
        if analyzed_frames[i][1] != current_state:
            # State changed
            duration = analyzed_frames[i][0] - analyzed_frames[segment_start_idx][0]
            segments.append((segment_start_idx, i - 1, current_state, duration))
            segment_start_idx = i
            current_state = analyzed_frames[i][1]

    # Add final segment
    duration = analyzed_frames[-1][0] - analyzed_frames[segment_start_idx][0]
    segments.append((segment_start_idx, len(analyzed_frames) - 1, current_state, duration))

    # Second pass: merge short segments with neighbors
    smoothed_states = [frame[1] for frame in analyzed_frames]

    for i, (start_idx, end_idx, state, duration) in enumerate(segments):
        if duration < min_duration and i > 0 and i < len(segments) - 1:
            # This segment is too short - merge with previous state
            prev_state = segments[i - 1][2]
            for j in range(start_idx, end_idx + 1):
                smoothed_states[j] = prev_state

    # Build smoothed frames
    return [(analyzed_frames[i][0], smoothed_states[i]) for i in range(len(analyzed_frames))]


def _find_eye_contact_issues(
    analyzed_frames: List[Tuple[float, bool]],
    threshold_seconds: float
) -> List[dict]:
    """
    Find periods where eye contact was lost for more than threshold_seconds.

    Returns list of issues with start and end timestamps.
    """
    issues = []
    lost_start = None

    for timestamp, has_eye_contact in analyzed_frames:
        if not has_eye_contact and lost_start is None:
            # Eye contact lost, start tracking
            lost_start = timestamp
        elif has_eye_contact and lost_start is not None:
            # Eye contact regained
            duration = timestamp - lost_start
            if duration > threshold_seconds:
                issues.append({
                    "start": round(lost_start, 1),
                    "end": round(timestamp, 1)
                })
            lost_start = None

    # Handle case where eye contact is still lost at end of video
    if lost_start is not None and analyzed_frames:
        final_timestamp = analyzed_frames[-1][0]
        duration = final_timestamp - lost_start
        if duration > threshold_seconds:
            issues.append({
                "start": round(lost_start, 1),
                "end": round(final_timestamp, 1)
            })

    return issues
