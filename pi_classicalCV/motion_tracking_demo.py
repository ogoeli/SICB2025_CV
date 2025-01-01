# Import the OpenCV library
import cv2

# Define a function for motion tracking
def motion_tracking():
    # Open a connection to the default webcam (camera index 0)
    cap = cv2.VideoCapture(0)

    # Check if the webcam is successfully opened
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Read two initial frames from the webcam for comparison
    _, frame1 = cap.read()
    _, frame2 = cap.read()

    # Infinite loop to process video frames in real-time
    while True:
        # Compute the absolute difference between the two frames
        diff = cv2.absdiff(frame1, frame2)

        # Convert the difference image to grayscale
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise and improve contour detection
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold the blurred image to create a binary image
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

        # Dilate the binary image to fill small holes and connect adjacent areas
        dilated = cv2.dilate(thresh, None, iterations=3)

        # Find contours in the dilated image
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Loop through each contour detected
        for contour in contours:
            # Skip small contours (less than a specified area threshold)
            if cv2.contourArea(contour) < 1000:
                continue

            # Compute the bounding box for the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Draw a rectangle around the detected motion
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the frame with motion tracking
        cv2.imshow('Motion Tracking', frame1)

        # Update frames for the next iteration
        frame1 = frame2
        ret, frame2 = cap.read()

        # Exit the loop if 'q' is pressed or no new frame is read
        if not ret or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Run the motion_tracking function if the script is executed directly
if __name__ == "__main__":
    motion_tracking()
