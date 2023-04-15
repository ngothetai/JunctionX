import cv2
import re 

file_gt = '/home/anhalu/anhalu-data/junction_AITrack/Public_Test/groundtruth/scene2cam_03/CAM_1.txt'
cap = cv2.VideoCapture('/home/anhalu/anhalu-data/junction_AITrack/Public_Test/videos/scene2cam_03/CAM_1.mp4')

with open(file_gt, 'r') as f : 
    str = f.readlines() 
    
    x, y = [], []
    
    for i in str : 
        match = re.search(r"\(([0-9,\s]+)\)", i)
        numbers = match.group(1).split(",")
        print(len(numbers))
        xi = [] 
        yi = []
        for index in range(0,int(len(numbers)),2) : 
            # print(index)
            xi.append(int(numbers[index]))
            yi.append(int(numbers[index+1]))
            # print(len(xi), index)
        
        # print(len(xi))
        # exit()
        x.append(xi)
        y.append(yi) 
    # print(len(x[0]))
    if not cap.isOpened():
        print("Không thể mở file")
        exit()


    frame = None
    number_frames = 0 

    while True:

        ret, next_frame = cap.read()

        if not ret:
            last_frame_read = True
            break


        frame = next_frame
        for i in range(0, int(len(x[number_frames]))) : 
            cv2.circle(frame,(x[number_frames][i], y[number_frames][i]), color=(0, 0, 255), thickness=3, radius=1)
        cv2.imshow('Video', frame)

        number_frames += 1
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    while not last_frame_read:
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cv2.imshow('Video', frame)
    cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()
