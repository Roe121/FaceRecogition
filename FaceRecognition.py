import face_recognition
import cv2
import os
import glob
import numpy as np

class FaceRec:
    # Initialize variables
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        # Resize frame for faster speed if necessary, 1.0 means no resizing (more resizing = more speed)
        self.frame_resizing = 0.75

    # Load encoding images (path to encoding images in our case is "images/")
    def load_encoding_images(self, images_path):
        
        # Load images
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print("{} encoding images found.".format(len(images_path)))

        # Store encoding and names of images
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get only file name from initial path.
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            # Get encoding
            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            # Store file name and encoding of the file
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
        print("Encoding images loaded.")
        
        
    # Detect known faces in the current video frame or image
    def detect_known_faces(self, frame):

        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        
        # Find all faces and face encodings in the current video frame
        # Convert BGR color image (used by OpenCV) to RGB color image (used by face_recognition)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Check if the face matches one or more known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first match.
            # Or use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        # Convert to numpy array to adjust coordinates quickly with frame resizing
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names
