import cv2
import time

def aruco_tracking_demo(video_source=0, aruco_dict_name="DICT_6X6_250", display_fps=True):
    """
    Live ArUco marker tracking demo using a USB webcam.

    Parameters:
        video_source (int or str): Webcam index or video source path.
        aruco_dict_name (str): Name of the ArUco dictionary to use.
        display_fps (bool): Whether to display the FPS on the output window.
    """
    # Load the specified ArUco dictionary and detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.__getattribute__(aruco_dict_name))
    parameters = cv2.aruco.DetectorParameters()

    # Open the webcam video stream
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    print("Press 'q' to quit the demo.")

    prev_time = time.time()
    frame_count = 0

    try:
        while True:
            # Capture a frame from the webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to capture video.")
                break

            # Convert frame to grayscale for ArUco detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect ArUco markers
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            # Draw detected markers and IDs on the frame
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                for i, marker_id in enumerate(ids.flatten()):
                    center = corners[i][0].mean(axis=0)
                    cv2.putText(frame, f"ID: {marker_id}", 
                                (int(center[0]), int(center[1] - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Optionally display FPS
            if display_fps:
                frame_count += 1
                current_time = time.time()
                fps = frame_count / (current_time - prev_time)
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display the output frame
            cv2.imshow("ArUco Tracking Demo", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Demo interrupted by user.")

    finally:
        # Release the video stream and close all windows
        cap.release()
        cv2.destroyAllWindows()
        print("Demo terminated.")

if __name__ == "__main__":
    # Run the demo with a USB webcam (index 0)
    aruco_tracking_demo(video_source=0, aruco_dict_name="DICT_6X6_250", display_fps=True)
