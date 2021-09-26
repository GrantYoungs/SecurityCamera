"""
This script is an implementation of a security camera program that records and saves
videos when faces or bodies are detected in the camera's view.

Utilizes OpenCV for Python, so that must be installed in order for this to run correctly.

@Author: Grant Youngs
@Date: 26 September 2021
"""

import cv2
import time
import datetime

# Determines the accuracy and the speed of the haarcascade algorithm
# Should keep this number between 1.1 and 1.5, but must be > 1.0 (The closer to 1.0, the more accurate but slower the algorithm will run)
SCALE_FACTOR = 1.2

# How many faces do I need to detect in a vicinity for it to be classified as a specific face.
# Should keep this number between 3 and 6
MINIMUM_NUM_NEIGHBORS = 5

# Thickness of the rectangle to be drawn to the screen.
RECTANGLE_THICKNESS = 3

# Frame rate of the video to be saved.
FRAME_RATE = 20.0

# The name of the directory to save the videos.
VIDEO_FOLDER = "videos"

# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Cascades needed to detect faces and bodies
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_fullbody_default.xml")

detection = False
detection_stopped_time = None
timer_started = False
SECONDS_TO_RECORD_AFTER_DETECTION = 5

frame_size = (int(cap.get(3)), int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

while True:
    _, frame = cap.read()

    # Gives us a new image that is grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, SCALE_FACTOR, MINIMUM_NUM_NEIGHBORS)
    bodies = face_cascade.detectMultiScale(gray, SCALE_FACTOR, MINIMUM_NUM_NEIGHBORS)

    # Draw the face on video
    # for (x, y, width, height) in faces:
        # cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), RECTANGLE_THICKNESS)

    # Is there a face or body detected?
    if len(faces) + len(bodies) > 0:

        # Were we just detecting a body or face?
        if detection:
            timer_started = False
        
        # We're detecting something new
        else:
            detection = True
            current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            out = cv2.VideoWriter(f"{VIDEO_FOLDER}/{current_time}.mp4", fourcc, FRAME_RATE, frame_size)
            print("Started Recording!")

    # Were we just detecting something before?
    elif detection:
        if timer_started:
            if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                detection = False
                timer_started = False
                out.release()
                print("Stop Recording!")
        else:
            timer_started = True
            detection_stopped_time = time.time()

    # Are we recording? Then write out the frame
    if detection:
        out.write(frame)

    # This line shows the frame to the screen for live feedback of the video.
    # If you don't wish to see the live stream, simply comment this line out
    # (For example, you were running this on a Raspberry Pi)
    cv2.imshow("Camera", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
out.release()   # Saves the video
cap.release()
cv2.destroyAllWindows()
