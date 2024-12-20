# Import the OpenCV library
import cv2

# Define the webcam preview function
def webcam_preview():
    # Open a connection to the default webcam (camera index 0)
    cap = cv2.VideoCapture(0)

    # Check if the webcam is successfully opened
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Infinite loop to display webcam feed in real-time
    while True:
        # Capture a single frame from the webcam
        ret, frame = cap.read()

        # Check if the frame was successfully captured
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Display the captured frame in a window named 'Webcam Preview'
        cv2.imshow('Webcam Preview', frame)

        # Break the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Run the webcam preview function if this script is executed directly
if __name__ == "__main__":
    webcam_preview()
