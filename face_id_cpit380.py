import face_recognition
import cv2
import numpy as np
import sys

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load sample images and encode them
ali_image = face_recognition.load_image_file("images/Ali1.jpg")
ali_face_encoding = face_recognition.face_encodings(ali_image)[0]

saim_image = face_recognition.load_image_file("images/Dr Saim1.jpg")
saim_face_encoding = face_recognition.face_encodings(saim_image)[0]

mohammad_image = face_recognition.load_image_file("images/Mohammed1.jpg")
mohammad_face_encoding = face_recognition.face_encodings(mohammad_image)[0]

saud_image = face_recognition.load_image_file("images/Saud2.jpg")
saud_face_encoding = face_recognition.face_encodings(saud_image)[0]

# Known face encodings and names
known_face_encodings = [
    ali_face_encoding,
    saim_face_encoding,
    mohammad_face_encoding,
    saud_face_encoding
]
known_face_names = [
    "Ali Zaid",
    "Dr. Saim",
    "Mohammad Serajuldeen",
    "Saud Aljedani"
]

try:
    while True:
        # Capture a single frame
        ret, frame = video_capture.read()
        if not ret:
            break

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Match faces
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the frame
        cv2.imshow('Video', frame)

        # Check if the 'q' key is pressed or the window is closed
        if cv2.waitKey(1) & 0xFF == ord('q') or not cv2.getWindowProperty('Video', cv2.WND_PROP_VISIBLE):
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Ensure all resources are released
    video_capture.release()
    cv2.destroyAllWindows()
    print("Program terminated.")
    sys.exit(0)
