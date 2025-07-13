import cv2
import numpy as np

# Open the webcam (0 means the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    _, frame = cap.read()

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    ### --- RED COLOR DETECTION ---
    lower_red = np.array([160, 100, 100])
    upper_red = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red, upper_red)

    ### --- BLUE COLOR DETECTION ---
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Combine both masks (optional if you want a full mask)
    mask_combined = cv2.bitwise_or(mask_red, mask_blue)

    # Apply masks separately for visualization
    result_red = cv2.bitwise_and(frame, frame, mask=mask_red)
    result_blue = cv2.bitwise_and(frame, frame, mask=mask_blue)

    # Check if red is detected
    if cv2.countNonZero(mask_red) > 1000:
        cv2.putText(frame, "Red Detected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Check if blue is detected
    if cv2.countNonZero(mask_blue) > 1000:
        cv2.putText(frame, "Blue Detected", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the original frame and the masked results
    cv2.imshow("Camera", frame)
    cv2.imshow("Red Mask", result_red)
    cv2.imshow("Blue Mask", result_blue)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
