# import cv2
# import supervision as sv

# from tracker import PolypTracker
# from mqi import compute_mqi_from_mask
# from repair import repair_mask_in_memory
# from severity import compute_severity_from_mask


# def process_video(video_path, model):
#     """
#     Process uploaded video frame-by-frame.
#     Returns annotated frame and latest severity.
#     """

#     cap = cv2.VideoCapture(video_path)

#     tracker = PolypTracker()

#     last_severity = None

#     while cap.isOpened():

#         ret, frame = cap.read()

#         if not ret:
#             break

#         # ---------------- DETECTION ----------------
#         # UPDATED: lower confidence threshold for better recall
#         results = model(frame, conf=0.25, verbose=False)[0]

#         if results.boxes is None or len(results.boxes) == 0:
#             yield frame, last_severity
#             continue

#         detections = sv.Detections.from_ultralytics(results)

#         # ---------------- TRACKING ----------------
#         tracks = tracker.update(detections)

#         if len(tracks) == 0:
#             yield frame, last_severity
#             continue

#         # ---------------- PROCESS EACH DETECTION ----------------
#         for i in range(len(tracks)):

#             track_id = tracks.tracker_id[i]

#             x1, y1, x2, y2 = map(int, tracks.xyxy[i])

#             if results.masks is None:
#                 continue

#             if i >= len(results.masks.data):
#                 continue

#             mask = results.masks.data[i].cpu().numpy()
#             mask = (mask > 0.5).astype("uint8")

#             # MQI
#             mqi_score, mask_quality = compute_mqi_from_mask(mask)

#             # REPAIR
#             if mask_quality == "Needs_Repair":
#                 mask = repair_mask_in_memory(mask)

#             # SEVERITY
#             severity_score, severity_label = compute_severity_from_mask(mask)

#             last_severity = severity_label

#             # DRAW BOX
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#             text = f"ID:{track_id} MQI:{mqi_score:.2f} Sev:{severity_label}"

#             cv2.putText(
#                 frame,
#                 text,
#                 (x1, max(y1 - 10, 10)),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.6,
#                 (0, 255, 0),
#                 2
#             )

#         yield frame, last_severity

#     cap.release()












import cv2
import supervision as sv
from collections import Counter

from tracker import PolypTracker
from mqi import compute_mqi_from_mask
from repair import repair_mask_in_memory
from severity import compute_severity_from_mask


def process_video(video_path, model):
    """
    Process uploaded video frame-by-frame.
    Returns:
        frame,
        final_severity,
        mqi_score,
        mask_quality,
        severity_label
    """

    cap = cv2.VideoCapture(video_path)

    tracker = PolypTracker()

    severity_history = []

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        mqi_score = None
        mask_quality = None
        severity_label = None

        # ---------------- DETECTION ----------------
        results = model(frame, conf=0.25, verbose=False)[0]

        if results.boxes is None or len(results.boxes) == 0:

            final_severity = Counter(severity_history).most_common(1)[0][0] if severity_history else None

            yield frame, final_severity, None, None, None
            continue

        detections = sv.Detections.from_ultralytics(results)

        # ---------------- TRACKING ----------------
        tracks = tracker.update(detections)

        if len(tracks) == 0:

            final_severity = Counter(severity_history).most_common(1)[0][0] if severity_history else None

            yield frame, final_severity, None, None, None
            continue

        # ---------------- PROCESS EACH DETECTION ----------------
        for i in range(len(tracks)):

            track_id = tracks.tracker_id[i]

            x1, y1, x2, y2 = map(int, tracks.xyxy[i])

            if results.masks is None:
                continue

            if i >= len(results.masks.data):
                continue

            mask = results.masks.data[i].cpu().numpy()
            mask = (mask > 0.5).astype("uint8")

            # MQI
            mqi_score, mask_quality = compute_mqi_from_mask(mask)

            # ---------------- REJECT ----------------
            if mask_quality == "Reject":

                severity_label = None

                text = f"ID:{track_id} MQI:{mqi_score:.2f} Mask:Rejected"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                cv2.putText(
                    frame,
                    text,
                    (x1, max(y1 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

                continue

            # ---------------- REPAIR ----------------
            if mask_quality == "Needs_Repair":
                mask = repair_mask_in_memory(mask)

            # ---------------- SEVERITY ----------------
            severity_score, severity_label = compute_severity_from_mask(mask)

            severity_history.append(severity_label)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            text = f"ID:{track_id} MQI:{mqi_score:.2f} Sev:{severity_label}"

            cv2.putText(
                frame,
                text,
                (x1, max(y1 - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        final_severity = Counter(severity_history).most_common(1)[0][0] if severity_history else None

        yield frame, final_severity, mqi_score, mask_quality, severity_label

    cap.release()