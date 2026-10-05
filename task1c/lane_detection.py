'''
*****************************************************************************************
*
*  ===============================================
*     Niti Vahan (NV) Theme of eYRC 2026-27
*  ===============================================
*
*  This script is intended for implementation of Task 1C of Niti Vahan (NV) Theme.
*
*  Filename:         lane_detection.py
*  Created:          2026
*  Last Modified:
*  Author:           e-Yantra Team
*
*  You are ONLY allowed to write your code inside the block marked
*  "ADD YOUR IMPLEMENTATION HERE". Do not change anything outside it - the
*  evaluation script relies on the rest of this file staying as it is.
*
*****************************************************************************************
'''

# Team ID:          < Team-ID >
# Author List:      < Names of the team members who worked on this file, comma separated >
# Filename:         lane_detection.py
# Functions:        detect_lane
# Global variables: < List any global variables you add, "None" if you add none >


####################### IMPORT MODULES #######################
import argparse
import json
import os

import cv2
import numpy as np
##############################################################

# The only three values "lane" is allowed to take.
LANE_LEFT = "left"
LANE_RIGHT = "right"
LANE_UNKNOWN = "unknown"
VALID_LANES = (LANE_LEFT, LANE_RIGHT, LANE_UNKNOWN)


##############################################################
############### ADD YOUR IMPLEMENTATION HERE #################
##############################################################

# Team ID: 5974
_PREV_CENTER_X = None
_PREV_LANE = LANE_UNKNOWN
_FRAMES_LOST = 0

def detect_lane(frame):
    global _PREV_CENTER_X, _PREV_LANE, _FRAMES_LOST
    if frame is None or frame.shape[0] != 480 or frame.shape[1] != 640:
        return {"center_x": -1, "lane": LANE_UNKNOWN}

    h, w = frame.shape[:2]
    vehicle_x = w // 2

    # HSV conversion & color masks
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array([0, 0, 175], dtype=np.uint8), np.array([180, 50, 255], dtype=np.uint8))
    yellow = cv2.inRange(hsv, np.array([14, 50, 90], dtype=np.uint8), np.array([36, 255, 255], dtype=np.uint8))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 130)

    combined = cv2.bitwise_or(cv2.bitwise_or(white, yellow), edges)

    # Perspective ROI focusing on the immediate drivable lane ahead
    poly = np.array([[(20, 470), (220, 260), (420, 260), (620, 470)]], dtype=np.int32)
    roi = np.zeros_like(gray)
    cv2.fillPoly(roi, poly, 255)
    binary = cv2.bitwise_and(combined, roi)

    lines = cv2.HoughLinesP(binary, 1, np.pi / 180, 20, minLineLength=25, maxLineGap=45)
    eval_y = 430

    left_lines = []
    right_lines = []

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0:
                    x_at_eval = x1
                    slope = 999.0
                else:
                    slope = dy / float(dx)
                    if abs(slope) < 0.25:
                        continue
                    x_at_eval = int(x1 + (eval_y - y1) / slope)

                if -80 <= x_at_eval <= 720:
                    if x_at_eval < vehicle_x:
                        left_lines.append((x_at_eval, slope))
                    else:
                        right_lines.append((x_at_eval, slope))

    nominal_half_width = 155
    center_x = -1
    lane = LANE_UNKNOWN

    # Filter out outlier line coordinates
    valid_left = [x for x, s in left_lines if x > 10]
    valid_right = [x for x, s in right_lines if x < 630]

    best_left = max(valid_left) if valid_left else None
    best_right = min(valid_right) if valid_right else None

    if best_left is not None and best_right is not None:
        measured_width = best_right - best_left
        if 200 <= measured_width <= 440:
            center_x = (best_left + best_right) // 2
            # Lane assignment based on divider location
            if center_x < 280:
                lane = LANE_LEFT
            elif center_x > 360:
                lane = LANE_RIGHT
            else:
                # If centered in camera, distinguish by boundary distance from image edges
                lane = LANE_RIGHT if best_left > 120 else LANE_LEFT
        elif best_left > 140:
            # Strong divider on left -> vehicle in right lane
            center_x = best_left + nominal_half_width
            lane = LANE_RIGHT
        elif best_right < 500:
            # Strong divider on right -> vehicle in left lane
            center_x = best_right - nominal_half_width
            lane = LANE_LEFT
        else:
            center_x = (best_left + best_right) // 2
            lane = LANE_RIGHT if center_x >= vehicle_x else LANE_LEFT
    elif best_left is not None:
        center_x = best_left + nominal_half_width
        lane = LANE_RIGHT
    elif best_right is not None:
        center_x = best_right - nominal_half_width
        lane = LANE_LEFT

    if center_x != -1 and lane != LANE_UNKNOWN:
        center_x = int(max(40, min(600, center_x)))
        if _PREV_CENTER_X is not None:
            center_x = int(0.75 * center_x + 0.25 * _PREV_CENTER_X)
        _PREV_CENTER_X = center_x
        _PREV_LANE = lane
        _FRAMES_LOST = 0
    elif _PREV_CENTER_X is not None and _FRAMES_LOST < 8:
        center_x = _PREV_CENTER_X
        lane = _PREV_LANE
        _FRAMES_LOST += 1
    else:
        center_x = 320
        lane = _PREV_LANE if _PREV_LANE != LANE_UNKNOWN else LANE_RIGHT

    return {"center_x": int(center_x), "lane": lane}

##############################################################
################ END OF YOUR IMPLEMENTATION ##################
##############################################################


#################### DO NOT EDIT BELOW THIS LINE ####################

def validate_result(result, frame_index):
    '''
    Purpose:
    ---
    Check that detect_lane() returned the expected structure and normalise it,
    so that a malformed return is reported here instead of silently scoring
    zero during evaluation.

    Input Arguments:
    ---
    `result` :          [ object ]      whatever detect_lane() returned
    `frame_index` :     [ int ]         index of the frame, used in error messages

    Returns:
    ---
    `clean` :           [ dict ]        {"center_x": int, "lane": str}
    '''
    where = "detect_lane() on frame {}".format(frame_index)

    if not isinstance(result, dict):
        raise TypeError("{} must return a dict, got {}".format(where, type(result).__name__))

    missing = {"center_x", "lane"} - set(result.keys())
    if missing:
        raise ValueError("{} is missing the key(s): {}".format(where, ", ".join(sorted(missing))))

    center_x = result["center_x"]
    if isinstance(center_x, bool) or not isinstance(center_x, (int, float, np.integer, np.floating)):
        raise TypeError("{} returned center_x of type {}, expected a number".format(
            where, type(center_x).__name__))
    center_x = int(round(float(center_x)))

    lane = result["lane"]
    if not isinstance(lane, str):
        raise TypeError("{} returned lane of type {}, expected a string".format(
            where, type(lane).__name__))
    lane = lane.strip().lower()
    if lane not in VALID_LANES:
        raise ValueError("{} returned lane = '{}', expected one of {}".format(
            where, result["lane"], ", ".join(VALID_LANES)))

    return {"center_x": center_x, "lane": lane}


def draw_overlay(frame, result):
    '''
    Purpose:
    ---
    Draw the detected lane centre and lane label on a copy of the frame.
    This is where display code belongs - never inside detect_lane().

    Input Arguments:
    ---
    `frame` :   [ numpy.ndarray ]   the frame that was passed to detect_lane()
    `result` :  [ dict ]            the validated result for that frame

    Returns:
    ---
    `canvas` :  [ numpy.ndarray ]   a copy of the frame with the overlay drawn
    '''
    canvas = frame.copy()
    height, width = canvas.shape[:2]

    # frame centre, for reference - roughly where the vehicle is pointing
    cv2.line(canvas, (width // 2, height), (width // 2, height - 40), (128, 128, 128), 1)

    center_x = result["center_x"]
    if 0 <= center_x < width:
        cv2.line(canvas, (center_x, height), (center_x, height // 2), (0, 0, 255), 2)
        cv2.circle(canvas, (center_x, height - 10), 5, (0, 0, 255), -1)

    label = "lane: {}   center_x: {}".format(result["lane"], center_x)
    cv2.putText(canvas, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return canvas


def process_video(video_path, show=False):
    '''
    Purpose:
    ---
    Read a video frame by frame, hand each frame to detect_lane() and collect
    the results.

    Input Arguments:
    ---
    `video_path` :  [ str ]     path to the video file
    `show` :        [ bool ]    if True, display the overlay while processing

    Returns:
    ---
    `results` :     [ list ]    one dict per frame:
                                {"frame": int, "center_x": int, "lane": str}
    '''
    if not os.path.isfile(video_path):
        raise FileNotFoundError("no such video file: {}".format(video_path))

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise IOError("OpenCV could not open the video: {}".format(video_path))

    window = "Task 1C - {}".format(os.path.basename(video_path))
    results = []
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            # A copy is passed in, so anything drawn inside detect_lane() cannot
            # corrupt the frame used for display.
            result = validate_result(detect_lane(frame.copy()), frame_index)
            results.append({"frame": frame_index, **result})

            if show:
                cv2.imshow(window, draw_overlay(frame, result))
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break

            frame_index += 1
    finally:
        capture.release()
        if show:
            cv2.destroyAllWindows()

    if not results:
        raise IOError("no frames could be read from: {}".format(video_path))

    return results


def summarise(video_path, results):
    '''
    Purpose:
    ---
    Print a one-line-per-video summary, so you can see at a glance whether the
    detector is returning anything sensible.

    Input Arguments:
    ---
    `video_path` :  [ str ]     path to the video that was processed
    `results` :     [ list ]    output of process_video()

    Returns:
    ---
    None
    '''
    total = len(results)
    counts = {lane: 0 for lane in VALID_LANES}
    for entry in results:
        counts[entry["lane"]] += 1
    not_found = sum(1 for entry in results if entry["center_x"] < 0)

    print("{:<28} {:>5} frames | left {:>5} | right {:>5} | unknown {:>5} | no centre {:>5}".format(
        os.path.basename(video_path), total,
        counts[LANE_LEFT], counts[LANE_RIGHT], counts[LANE_UNKNOWN], not_found))


def expand_videos(paths):
    '''
    Purpose:
    ---
    Turn the command-line arguments into a list of video files, accepting a
    FOLDER as well as individual files.

    A folder is the portable way to say "all the clips": Windows shells do not
    expand `public/*.mp4` the way bash does - cmd and PowerShell hand the
    pattern through verbatim and the script would look for a file literally
    named "*.mp4". `python lane_detection.py public` behaves the same on every
    platform.

    Input Arguments:
    ---
    `paths` :   [ list ]    the raw command-line arguments

    Returns:
    ---
    `videos` :  [ list ]    paths to individual video files, folders expanded
    '''
    videos = []
    for raw in paths:
        if os.path.isdir(raw):
            found = sorted(f for f in os.listdir(raw) if f.lower().endswith(".mp4"))
            if not found:
                raise FileNotFoundError("no .mp4 files in the folder: {}".format(raw))
            videos.extend(os.path.join(raw, f) for f in found)
        else:
            videos.append(raw)
    return videos


def main():
    parser = argparse.ArgumentParser(
        description="Task 1C - run your lane detector over one or more videos.")
    parser.add_argument("videos", nargs="+",
                        help="video file(s), or a folder holding them "
                             "(e.g. 'public')")
    parser.add_argument("--show", action="store_true",
                        help="display the detection overlay while processing (press q to stop)")
    parser.add_argument("--out", metavar="FILE",
                        help="write the per-frame results to this JSON file")
    args = parser.parse_args()

    all_results = {}
    for video_path in expand_videos(args.videos):
        results = process_video(video_path, show=args.show)
        summarise(video_path, results)
        all_results[os.path.basename(video_path)] = results

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(all_results, handle, indent=2)
        print("\nresults written to {}".format(args.out))


if __name__ == "__main__":
    main()
