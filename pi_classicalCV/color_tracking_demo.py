# Import necessary libraries
import cv2
import numpy as np

# Define the color tracking function
def color_tracking():
    # Open a connection to the default webcam (camera index 0)
    cap = cv2.VideoCapture(0)

    # Check if the webcam is successfully opened
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Infinite loop to process video frames in real-time
    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Convert the frame to the HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the range of green color in HSV
        lower_green = np.array([40, 50, 50])  # Lower boundary of green
        upper_green = np.array([80, 255, 255])  # Upper boundary of green

        # Create a binary mask where green colors are white, and the rest are black
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Loop through each detected contour
        for contour in contours:
            # Skip small contours (less than the specified area threshold)
            if cv2.contourArea(contour) < 500:
                continue

            # Calculate the bounding box for the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Draw a rectangle around the detected green object
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the original frame with bounding boxes
        cv2.imshow('Green Object Tracking', frame)

        # Display the mask showing detected green regions
        cv2.imshow('Mask', mask)

        # Exit the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Run the color tracking function if the script is executed directly
if __name__ == "__main__":
    color_tracking()
