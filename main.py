# import required  libraries
from ultralytics import YOLO
import face_recognition
import os
#custom helpers
from helpers.helpers import *

# initiate the YOLO pre-trained model
model = YOLO('models/yolov8n.pt')

# define the faces that will be encoded
known_faces_dir = 'data'

# empty arrays to store facial landmarks data
known_face_encodings = []
known_face_names = []

# iterate through the file containing the known faces directory
for filename in os.listdir(known_faces_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png', 'webp')):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)

        face_encodings = face_recognition.face_encodings(image)  # define the face encodings var
        # if statement to append face encodings into the empty array
        if face_encodings:
            known_face_encodings.append(face_encodings[0])

            name = os.path.splitext(filename)[0]
            known_face_names.append(name)

cap = cv2.VideoCapture(0)  # initiate the video capture

# infinite loop to process video frames continuously
while True:
    ret, frame = cap.read()  # ret is a boolean that indicates if the frame was captured
    if not ret:
        break

    results = model(frame)  # use the YOLO model to detect desired objects

    for result in results:  # loop through each result from the model
        for box in result.boxes:  # loop through each detected box in result
            if int(box.cls) == 0:  # check if object is a person (0 = person)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_in_frame = frame[y1:y2, x1:x2]  # extract the region of the frame containing person

                rgb_small_frame = np.ascontiguousarray(frame[:, :, ::-1])  # convert frame to RGB frame

                face_locations = face_recognition.face_locations(rgb_small_frame)

                # this entire block of code encodes faces
                if face_locations is not None and face_locations:
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                    face_names = []
                    for face_encoding in face_encodings:  # loop through each face encoding

                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                        name = "unKnown"

                        # calculate the distance between face en conding stored and the ones in frame
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)  # find the index that best matches
                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]

                        face_names.append(name)

                    for (top, right, bottom, left), name in zip(face_locations, face_names):

                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                        font = cv2.FONT_HERSHEY_COMPLEX
                        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = model.names[int(box.cls)]
                confidence = box.conf[0]
                cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('YOLOv8 Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # cap.release()
    # cv2.destroyAllWindows()