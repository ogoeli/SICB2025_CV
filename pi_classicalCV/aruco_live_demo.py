# Import necessary libraries
import cv2
import time

def aruco_tracking_demo(video_source=0, aruco_dict_name="DICT_6X6_250", display_fps=True):
    """
    Live ArUco marker tracking demo using a USB webcam.

    Parameters:
        video_source (int or str): Webcam index or video file path.
        aruco_dict_name (str): Name of the ArUco dictionary to use.
        display_fps (bool): Whether to display the FPS on the output window.
    """
    # Load the specified ArUco dictionary and detector parameters
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.__getattribute__(aruco_dict_name))
    parameters = cv2.aruco.DetectorParameters()

    # Open the video source (default is webcam with index 0)
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Display a message to inform the user how to quit
    print("Press 'q' to quit the demo.")

    # Initialize variables for FPS calculation
    prev_time = time.time()
    frame_count = 0

    try:
        # Main loop for processing video frames
        while True:
            # Capture a frame from the video source
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to capture video.")
                break

            # Convert the frame to grayscale (required for ArUco detection)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect ArUco markers in the grayscale image
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            # If markers are detected, draw them on the frame
            if ids is not None:
                # Draw the marker outlines and IDs
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                for i, marker_id in enumerate(ids.flatten()):
                    # Compute the center of the marker
                    center = corners[i][0].mean(axis=0)

                    # Display the marker ID near its center
                    cv2.putText(frame, f"ID: {marker_id}", 
                                (int(center[0]), int(center[1] - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Optionally calculate and display FPS
            if display_fps:
                frame_count += 1
                current_time = time.time()
                fps = frame_count / (current_time - prev_time)
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show the output frame with annotations
            cv2.imshow("ArUco Tracking Demo", frame)

            # Exit the loop if the user presses the 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        # Handle manual interruption (Ctrl+C)
        print("Demo interrupted by user.")

    finally:
        # Clean up: release the video source and close all windows
        cap.release()
        cv2.destroyAllWindows()
        print("Demo terminated.")

# Run the demo if this script is executed directly
if __name__ == "__main__":
    # Use the default webcam (index 0) and a predefined ArUco dictionary
    aruco_tracking_demo(video_source=0, aruco_dict_name="DICT_6X6_250", display_fps=True)
