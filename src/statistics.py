import numpy as np

class Statistics:
    def __init__(self):
        self.total_frames = 0
        self.total_points = 0
        self.lost_points = 0
        self.motion_values = []

    def update(self, old_points, new_points):
        if old_points is None:
            return

        self.total_frames += 1
        current_points = len(new_points)
        self.total_points += current_points

        lost = len(old_points) - current_points
        self.lost_points += max(lost, 0)

        for old, new in zip(old_points, new_points):
            movement = np.linalg.norm(new - old)
            self.motion_values.append(movement)

    def report(self):
        avg_motion = np.mean(self.motion_values) if len(self.motion_values) > 0 else 0
        if self.total_points > 0:
            success = (1 - self.lost_points / self.total_points) * 100
        else:
            success = 0

        return {
            "Frames": self.total_frames,
            "Total Points": self.total_points,
            "Lost Points": self.lost_points,
            "Tracking Success %": round(success, 2),
            "Average Motion": round(avg_motion, 2)
        }