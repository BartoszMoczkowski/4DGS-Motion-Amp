import cv2
import threading
import time
# A basic script used to record the images from USB cameras 
def record_camera(camera_index, filename, sync_event, stop_event : threading.Event):
    cap = cv2.VideoCapture(camera_index)

    cap.set(3,1280)
    cap.set(4,720)
    if not cap.isOpened():
        print(f"Error: Camera {camera_index} could not be opened.")
        return
    
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
        
        out.write(frame) 

    cap.release()
    out.release()
    cv2.destroyWindow(f"Camera {camera_index}")

sync_event = threading.Event()
stop_event = threading.Event()
stop_event.clear()
# Modify the number of threads as needed depending on the number of cameras
thread1 = threading.Thread(target=record_camera, args=(0, "camera1.mp4", sync_event, stop_event))
thread2 = threading.Thread(target=record_camera, args=(1, "camera2.mp4", sync_event, stop_event))
thread3 = threading.Thread(target=record_camera, args=(2, "camera3.mp4", sync_event, stop_event))
thread4 = threading.Thread(target=record_camera, args=(3, "camera4.mp4", sync_event, stop_event))

thread1.start()
thread2.start()
thread3.start()
thread4.start()

try:
    while True:
        # Change sleep time to 1/fps and hope that it works well enough
        time.sleep(0.05)  
        sync_event.set()  
        sync_event.clear()
except KeyboardInterrupt:
    sync_event.set()  
    stop_event.set() 
    print("Recording stopped.")


thread1.join()
thread2.join()
thread3.join()
thread4.join()
