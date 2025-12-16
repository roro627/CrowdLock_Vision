import backend.core.trackers.simple_tracker as st
from backend.core.types import Detection


def test_iou_edge_cases():
    assert st.iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0
    assert st.iou((0, 0, 2, 2), (1, 1, 3, 3)) > 0.0
    assert st.iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


def test_tracker_creates_tracks_and_ages_out():
    tracker = st.SimpleTracker(iou_threshold=0.1, max_missed=1)

    det = Detection(bbox=(0.0, 0.0, 10.0, 10.0), confidence=0.9, keypoints=None)
    persons = tracker.update([det])
    assert len(persons) == 1

    # no detections => miss increment; still present
    persons2 = tracker.update([])
    assert len(persons2) == 1

    # another miss => deleted
    persons3 = tracker.update([])
    assert len(persons3) == 0


def test_tracker_greedy_assignment_and_new_track():
    tracker = st.SimpleTracker(iou_threshold=0.1, max_missed=5)

    d1 = Detection(bbox=(0.0, 0.0, 10.0, 10.0), confidence=0.9, keypoints=None)
    d2 = Detection(bbox=(100.0, 100.0, 110.0, 110.0), confidence=0.8, keypoints=None)
    tracker.update([d1])

    # One overlaps existing track, the other creates a new track.
    persons = tracker.update([d1, d2])
    assert len(persons) == 2


def test_tracker_handles_truthy_empty_detections_object():
    tracker = st.SimpleTracker(iou_threshold=0.1, max_missed=5)
    d1 = Detection(bbox=(0.0, 0.0, 10.0, 10.0), confidence=0.9, keypoints=None)
    tracker.update([d1])

    class TruthyEmpty:
        def __bool__(self):
            return True

        def __len__(self):
            return 0

        def __iter__(self):
            return iter(())

    persons = tracker.update(TruthyEmpty())
    assert len(persons) == 1


def test_tracker_deletes_unmatched_after_assignment():
    tracker = st.SimpleTracker(iou_threshold=0.1, max_missed=0)
    d1 = Detection(bbox=(0.0, 0.0, 10.0, 10.0), confidence=0.9, keypoints=None)
    d2 = Detection(bbox=(100.0, 100.0, 110.0, 110.0), confidence=0.9, keypoints=None)
    tracker.update([d1, d2])

    # Only one detection overlaps => the other track is unmatched and should be deleted (max_missed=0).
    persons = tracker.update([d1])
    assert len(persons) == 1
