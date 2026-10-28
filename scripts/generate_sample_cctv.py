"""
Synthetic CCTV video generator for StreetGuard AI demonstration.
Generates an MP4 video with CCTV visual artifacts, timestamp watermark,
a person walking across a perimeter zone, and a bag being dropped and abandoned.
"""

from pathlib import Path
import cv2
import numpy as np


def generate_sample_cctv_video(
    output_path: str = "sample_data/cctv_demo.mp4",
    duration_sec: int = 12,
    fps: int = 24,
    width: int = 800,
    height: int = 480,
) -> str:
    """Generate a synthetic surveillance video for testing and viva demonstration."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height))
    total_frames = duration_sec * fps

    # Restricted polygon area in the center-right: [350..700, 180..420]
    # Person trajectory:
    # Frame 0 -> 70: Person walks from left (x=60, y=260) to center (x=380, y=300) carrying a bag
    # Frame 70: Drops bag at (x=390, y=340)
    # Frame 71 -> 150: Person walks into restricted zone (x=450..550, y=280) -> Intrusion!
    # Frame 151 -> 220: Person leaves restricted zone towards top-right and exits frame (x=780, y=200)
    # Frame 220 -> 288: Bag remains completely unattended at (390, 340) with no person nearby -> Unattended alert!

    bag_pos = None

    for i in range(total_frames):
        t = i / float(fps)
        frame = np.full((height, width, 3), (35, 38, 42), dtype=np.uint8)

        # Draw hallway perspective lines & floor tiles
        cv2.line(frame, (0, 140), (width, 140), (60, 65, 70), 2)  # Horizon wall line
        # Wall color upper
        frame[0:140, :] = (25, 27, 30)

        # Grid perspective floor tiles
        for x in range(0, width + 100, 100):
            cv2.line(frame, (x, 140), (int(x * 1.5 - 200), height), (45, 50, 55), 1)
        for y in range(140, height, 50):
            cv2.line(frame, (0, y), (width, y), (45, 50, 55), 1)

        # Draw a marked restricted zone hatching on floor
        hatch_pts = np.array([[360, 200], [680, 200], [750, 420], [330, 420]], dtype=np.int32)
        cv2.polylines(frame, [hatch_pts], isClosed=True, color=(60, 80, 110), thickness=1)

        # Determine person position
        person_x, person_y = None, None
        person_visible = True

        if i < 70:
            # Walking towards drop point
            progress = i / 70.0
            person_x = int(60 + progress * (380 - 60))
            person_y = int(270 + progress * (300 - 270))
        elif i < 150:
            # Person walks into restricted zone
            progress = (i - 70) / 80.0
            person_x = int(380 + progress * (560 - 380))
            person_y = int(300 + progress * (310 - 300))
            if bag_pos is None:
                bag_pos = (380, 340)
        elif i < 220:
            # Person walks out towards right
            progress = (i - 150) / 70.0
            person_x = int(560 + progress * (820 - 560))
            person_y = int(310 + progress * (250 - 310))
        else:
            # Person has exited the camera FOV
            person_visible = False

        # Render Bag
        if bag_pos is not None:
            bx, by = bag_pos
            # Realistic backpack visual: Dark blue/brown rounded rectangle
            cv2.rectangle(frame, (bx - 18, by - 26), (bx + 18, by + 12), (120, 70, 30), -1)
            cv2.rectangle(frame, (bx - 18, by - 26), (bx + 18, by + 12), (180, 120, 60), 2)
            # Bag straps and zipper
            cv2.line(frame, (bx - 8, by - 26), (bx - 8, by + 10), (160, 100, 50), 2)
            cv2.line(frame, (bx + 8, by - 26), (bx + 8, by + 10), (160, 100, 50), 2)
            cv2.circle(frame, (bx, by - 28), 6, (140, 90, 40), 2)

        # Render Person
        if person_visible and person_x is not None:
            px, py = person_x, person_y
            # Head
            cv2.circle(frame, (px, py - 60), 14, (190, 170, 150), -1)
            cv2.circle(frame, (px, py - 60), 14, (120, 100, 80), 1)
            # Torso (Dark jacket)
            cv2.rectangle(frame, (px - 18, py - 46), (px + 18, py), (70, 60, 50), -1)
            # Legs
            leg_phase = np.sin(i * 0.4) * 8
            cv2.line(frame, (px - 8, py), (int(px - 10 + leg_phase), py + 35), (40, 40, 45), 6)
            cv2.line(frame, (px + 8, py), (int(px + 10 - leg_phase), py + 35), (40, 40, 45), 6)

            # If before drop point, person carries bag on side
            if bag_pos is None:
                cv2.rectangle(frame, (px + 14, py - 20), (px + 32, py + 8), (120, 70, 30), -1)

        # CCTV timestamp and camera watermark
        cctv_time = f"2026-10-28 14:{int(t//60):02d}:{int(t%60):02d}.{int((t*100)%100):02d}"
        cv2.putText(frame, "CAM-02 NORTH CORRIDOR", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, cctv_time, (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, "REC [SECURE]", (width - 130, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 240), 2, cv2.LINE_AA)

        # Add mild surveillance scanline noise
        if i % 3 == 0:
            frame[::6, :, :] = (frame[::6, :, :].astype(np.float32) * 0.94).astype(np.uint8)

        out.write(frame)

    out.release()
    return str(out_file)


if __name__ == "__main__":
    path = generate_sample_cctv_video()
    print(f"Sample CCTV video generated successfully at: {path}")
