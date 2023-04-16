import cv2

# Replace with the RTSP link of your camera
rtsp_link = "rtsp://zephyr.rtsp.stream/pattern?streamKey=f92d2f6d0b459007353b23b86f94b1f7"

# Open the RTSP link with OpenCV
cap = cv2.VideoCapture(rtsp_link)

# Use a while loop to continuously read frames from the stream
while True:
    ret, frame = cap.read()

    # If frame is read correctly, show it in a window
    if ret:
        cv2.imshow('frame', frame)

    # Press Q on keyboard to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy any OpenCV windows
cap.release()
cv2.destroyAllWindows()
