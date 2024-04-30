import cv2
from FaceRecognition import FaceRec
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Create an instance of the FaceRec class
FR = FaceRec()
# Load encoding images
FR.load_encoding_images("images/")

# Maximum size of the displayed image
MAX_WIDTH = 800
MAX_HEIGHT = 600

# Resize the image if it exceeds the maximum size
def resize_image(image):
    height, width, _ = image.shape
    
    if width > MAX_WIDTH or height > MAX_HEIGHT:
        ratio = min(MAX_WIDTH / width, MAX_HEIGHT / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        return cv2.resize(image, (new_width, new_height))
    else:
        return image

# Start real-time detection
def start_realtime_detection():
    # Create a label to display the message "Opening webcam, please wait..."
    webcam_label = tk.Label(root, text="Opening webcam, please wait...", font=("Helvetica", 16), bg="#f0f0f0")
    webcam_label.pack(pady=20)

    def open_webcam():
        # Open webcam
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()

            # Resize the image if necessary
            frame = resize_image(frame)

            # Detect faces
            face_locations, face_names = FR.detect_known_faces(frame)
            
            # Display face names and rectangles around faces
            for face_loc, name in zip(face_locations, face_names):
                y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

                cv2.putText(frame, name,(x1, y1 - 15), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

            cv2.imshow("Frame", frame)

            # Wait for ESC key to quit
            key = cv2.waitKey(1)
            if key == 27:
                break
            
            # Close the window if the user clicks on the cross
            if cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1:
                break

        cap.release()
        cv2.destroyAllWindows()
        # Destroy the webcam label
        webcam_label.destroy()

    # Wait for 3 seconds before opening the webcam
    root.after(3000, open_webcam)

# Upload an image and detect faces
def upload_image():
    # Open a dialog box to select a file
    filepath = filedialog.askopenfilename()
    
    if filepath:
        image = cv2.imread(filepath)
        # Resize the image if necessary
        image = resize_image(image)

        # Detect faces
        face_locations, face_names = FR.detect_known_faces(image)
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

            cv2.putText(image, name,(x1, y1 - 15 ), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 3)

        # Convert the image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        
        # Display the image in the tkinter window
        panel.config(image=image)
        panel.image = image

# Create a tkinter window
root = tk.Tk()
root.title("EasyFaceRec")

# Set window size
root.geometry("1000x800")

# Change background color
root.config(bg="#f0f0f0")

# Create a title
title_label = tk.Label(root, text="EasyFaceRec", font=("Helvetica", 40), bg="#f0f0f0")
title_label.pack(pady=(20, 0))  

# Create a description
description_label = tk.Label(root, text="A Simple Face Recognition Application", font=("Helvetica", 16), bg="#f0f0f0")
description_label.pack(pady=(20, 0))  

# Create a frame for buttons
button_frame = tk.Frame(root, bg="#f0f0f0")
button_frame.pack(pady=20)

# Create buttons for real-time detection and image detection
realtime_button = tk.Button(button_frame, text="Realtime Detection", command=start_realtime_detection, width=20, height=3, font=("Helvetica", 16))
realtime_button.grid(row=0, column=0, padx=10)

upload_button = tk.Button(button_frame, text="Image Detection", command=upload_image, width=20, height=3, font=("Helvetica", 16))
upload_button.grid(row=0, column=1, padx=10)

# Create a frame to display the image
panel = tk.Label(root, bg="#f0f0f0")
panel.pack(pady=20)

root.mainloop()
