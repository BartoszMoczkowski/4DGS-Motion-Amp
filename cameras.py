import cv2
import threading
import time

def record_camera(camera_index, filename, sync_event, stop_event : threading.Event):
    # Initialize camera
    cap = cv2.VideoCapture(camera_index)

    cap.set(3,1280)
    cap.set(4,720)
    if not cap.isOpened():
        print(f"Error: Camera {camera_index} could not be opened.")
        return
    
    # Set video codec, resolution, and frame rate
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 20.0, (1280, 720))

    while True:
        sync_event.wait()  # Synchronize frames
        if stop_event.isSet():
            break
        
        
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Frame not captured from camera {camera_index}.")
            break
        
        out.write(frame)  # Write frame to video file
        # cv2.imshow(f"Camera {camera_index}", frame)

        # Stop recording when 'q' is pressed
    cap.release()
    out.release()
    cv2.destroyWindow(f"Camera {camera_index}")

# Synchronization event
sync_event = threading.Event()
stop_event = threading.Event()
stop_event.clear()

# Define threads for both cameras
thread1 = threading.Thread(target=record_camera, args=(0, "camera1.mp4", sync_event, stop_event))
thread2 = threading.Thread(target=record_camera, args=(1, "camera2.mp4", sync_event, stop_event))
thread3 = threading.Thread(target=record_camera, args=(2, "camera3.mp4", sync_event, stop_event))
thread4 = threading.Thread(target=record_camera, args=(3, "camera4.mp4", sync_event, stop_event))

# Start threads
thread1.start()
thread2.start()
thread3.start()
thread4.start()

# Synchronize frames periodically
try:
    while True:
        time.sleep(0.05)  # 50 ms interval (adjust as needed for FPS)
        sync_event.set()  # Release threads to record frames
        sync_event.clear()
except KeyboardInterrupt:
    sync_event.set()  # Release threads to record frames
    stop_event.set()  # Release threads to record frames
    print("Recording stopped.")

# Wait for threads to finish
thread1.join()
thread2.join()
thread3.join()
thread4.join()
