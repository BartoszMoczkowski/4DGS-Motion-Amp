"""USB multi-camera recorder.

Design notes (reliability fixes):

- Timebase: the mp4 writer still stamps a fixed TARGET_FPS, but the
  AUTHORITATIVE timebase is the per-camera sidecar CSV
  (``<filename>_timestamps.csv``) recording the wall-clock and monotonic
  timestamp of every captured frame, plus the camera index and frame counter.
  Use the sidecar for any frequency-domain (FFT) analysis.

- Cross-camera alignment: all workers and the controller meet at a
  ``threading.Barrier`` once per capture cycle, so frame k in every camera
  file is captured in the same cycle. A worker that dies breaks the barrier,
  which aborts the recording loudly instead of drifting silently.

- Resolution: the requested resolution is verified against the actual frame
  shape; if the camera rejects it, the VideoWriter is adjusted to the real
  frame size and a warning is printed (frames are never silently dropped).

- Failure handling: if one camera's read() fails, a shared error flag stops
  ALL cameras and reports which camera failed.

- Stop path: read() waits are bounded where the backend supports it
  (CAP_PROP_READ_TIMEOUT_MSEC), the barrier is aborted on stop so waiting
  workers wake up, and joins use a timeout with a loud warning.
"""

import csv
import os
import threading
import time

import cv2

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
TARGET_FPS = 20.0
JOIN_TIMEOUT_S = 5.0
READ_TIMEOUT_MS = 2000


def _flag_error(error_state, camera_index, msg):
    # Record the first failing camera and signal all threads to stop.
    if not error_state["event"].is_set():
        error_state["camera"] = camera_index
        error_state["msg"] = msg
    error_state["event"].set()


def record_camera(camera_index, filename, barrier, stop_event, error_state):
    """Capture frames from one USB camera into an mp4 plus a timestamp sidecar.

    The sidecar CSV (<filename>_timestamps.csv) is the authoritative timebase;
    the mp4 writer's fixed fps stamp is only a convenience preview.
    """
    sidecar_path = os.path.splitext(filename)[0] + "_timestamps.csv"
    out = None
    cap = cv2.VideoCapture(camera_index)
    try:
        if not cap.isOpened():
            raise RuntimeError("camera could not be opened")

        # Request the target resolution, then verify what we actually got.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
        # Bound the read() wait where the backend supports it, so a stalled
        # camera cannot block the stop path forever.
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MS)

        ret, frame = cap.read()
        if not ret or frame is None:
            raise RuntimeError("initial frame read failed")
        actual_h, actual_w = frame.shape[:2]
        reported = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if (actual_w, actual_h) != (TARGET_WIDTH, TARGET_HEIGHT):
            print(
                f"WARNING: camera {camera_index} rejected {TARGET_WIDTH}x{TARGET_HEIGHT} "
                f"(driver reports {reported[0]}x{reported[1]}); actual frame size is "
                f"{actual_w}x{actual_h}. Adjusting VideoWriter to the real size."
            )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(filename, fourcc, TARGET_FPS, (actual_w, actual_h))
        if not out.isOpened():
            raise RuntimeError(f"VideoWriter for {filename} could not be opened")

        with open(sidecar_path, "w", newline="") as sidecar:
            sidecar_writer = csv.writer(sidecar)
            sidecar_writer.writerow(["frame_idx", "camera_index", "wall_time_ns", "monotonic_ns"])

            frame_counter = 0
            # Write the frame already captured during resolution probing.
            out.write(frame)
            sidecar_writer.writerow([frame_counter, camera_index, time.time_ns(), time.monotonic_ns()])
            frame_counter += 1

            while not stop_event.is_set():
                # Start of a capture cycle: all cameras + the controller meet
                # here, so every camera captures the same cycle.
                barrier.wait()
                if stop_event.is_set():
                    break

                ret, frame = cap.read()
                if not ret or frame is None:
                    raise RuntimeError("frame read failed (camera stalled or disconnected)")

                out.write(frame)
                sidecar_writer.writerow([frame_counter, camera_index, time.time_ns(), time.monotonic_ns()])
                frame_counter += 1

    except threading.BrokenBarrierError:
        # The barrier was broken by a peer failure or by the controller's stop
        # path; if neither stop nor error is flagged, a worker missed a cycle —
        # report it loudly.
        if not stop_event.is_set() and not error_state["event"].is_set():
            print(f"ERROR: camera {camera_index} missed a capture cycle (barrier broken).")
            _flag_error(error_state, camera_index, "missed a capture cycle (barrier broken)")
    except Exception as e:
        # One camera failing stops ALL cameras, with the culprit identified.
        _flag_error(error_state, camera_index, str(e))
        print(f"ERROR: camera {camera_index} failed: {e}. Stopping all cameras.")
        barrier.abort()
    finally:
        cap.release()
        if out is not None:
            out.release()


def main(camera_indexes=(0, 1, 2, 3), filename_prefix="camera"):
    """Record from all given cameras until Ctrl+C or a camera failure."""
    barrier = threading.Barrier(len(camera_indexes) + 1)  # workers + controller
    stop_event = threading.Event()
    error_state = {"event": threading.Event(), "camera": None, "msg": None}

    threads = []
    for pos, camera_index in enumerate(camera_indexes):
        t = threading.Thread(
            target=record_camera,
            args=(camera_index, f"{filename_prefix}{pos + 1}.mp4", barrier, stop_event, error_state),
            daemon=True,
        )
        t.start()
        threads.append(t)

    try:
        while True:
            if error_state["event"].is_set():
                print(
                    f"ERROR: camera {error_state['camera']} failed ({error_state['msg']}); "
                    "all cameras stopped."
                )
                stop_event.set()
                barrier.abort()
                break
            time.sleep(1.0 / TARGET_FPS)
            barrier.wait()  # release all workers for one capture cycle
    except KeyboardInterrupt:
        print("Recording stopped by user.")
        stop_event.set()
        barrier.abort()  # wake any workers waiting on the barrier
    except threading.BrokenBarrierError:
        stop_event.set()

    for camera_index, t in zip(camera_indexes, threads):
        t.join(timeout=JOIN_TIMEOUT_S)
        if t.is_alive():
            print(
                f"WARNING: camera {camera_index} thread did not exit within "
                f"{JOIN_TIMEOUT_S}s; likely stalled in cap.read()."
            )

    if error_state["event"].is_set():
        raise SystemExit(f"Recording aborted: camera {error_state['camera']} failed ({error_state['msg']}).")


if __name__ == "__main__":
    # Modify camera_indexes as needed depending on the number of cameras.
    main()
