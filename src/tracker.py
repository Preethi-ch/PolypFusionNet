import supervision as sv


class PolypTracker:
    """
    ByteTrack-based tracker for polyp tracking
    """

    def __init__(self):
        self.tracker = sv.ByteTrack()

    def update(self, detections: sv.Detections) -> sv.Detections:
        if len(detections) == 0:
            return detections

        tracks = self.tracker.update_with_detections(detections)
        return tracks