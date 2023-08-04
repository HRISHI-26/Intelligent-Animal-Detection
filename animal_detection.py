import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model("animal_detection_model.h5")

# Define the target size for the images
target_size = (224, 224)

# Define the class labels
class_labels = ["Bear", "Cattle", "Cheetah", "Deer", "Crocodile", "Giraffe", "Horse",
                "Elephant", "Lion", "Monkey", "Panda", "Tiger", "Rhinoceros", "Zebra"]

# Define the accuracy threshold
accuracy_threshold = 0.8

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to perform animal detection on a frame
def detect_animal(frame):
    # Preprocess the frame
    image = Image.fromarray(frame)
    image = preprocess_image(image)

    # Make predictions using the model
    predictions = model.predict(image)
    predicted_label_index = np.argmax(predictions)
    predicted_label = class_labels[predicted_label_index]
    predicted_label_accuracy = predictions[0][predicted_label_index]

    # Update animal frequencies if the predicted label's accuracy is above the threshold
    if predicted_label_accuracy > accuracy_threshold:
        # Set label color based on accuracy percentage
        label_color = (0, 0, 255)  # Default color is red
        if predicted_label_accuracy > 0.99:
            label_color = (0, 255, 0)  # Set color to green if accuracy > 90%

        # Draw the predicted label and accuracy on the frame
        label_text = f"{predicted_label} ({predicted_label_accuracy * 100:.2f}% accuracy)"
        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2, cv2.LINE_AA)

    return frame

# Function to capture video from the webcam and perform animal detection
def perform_animal_detection():
    # Open the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        if ret:
            # Perform animal detection on the frame
            processed_frame = detect_animal(frame)

            # Display the processed frame
            cv2.imshow("Animal Detection", processed_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Run the animal detection program
perform_animal_detection()
