from imageai.Detection import ObjectDetection
import os
import cv2

exec_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(exec_path, "yolov3/yolov3.pt"))
detector.loadModel()

custom_objects = detector.CustomObjects(car=True, bicycle=True)

cap = cv2.VideoCapture('1234.mp4')
frames_per_second = 20

output_file_path = 'test_video'
output_video_filepath = output_file_path + '.mp4'

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
output_video = cv2.VideoWriter(
    output_video_filepath,
    cv2.VideoWriter_fourcc(*"MP4V"),
    frames_per_second,
    (frame_width, frame_height)
    )

while True:
    success, img = cap.read()
    if not success:
        break

    detections = detector.detectObjectsFromImage(
        input_image=img,
        output_image_path=os.path.join(exec_path, "new_objects.jpg"),
        display_percentage_probability=False
    )

    for eachObject in detections:
        print(eachObject["name"], " : ", eachObject["percentage_probability"],
              " : ", eachObject["box_points"])
        name = eachObject["name"]
        x, y, w, h = eachObject["box_points"]
        if name == 'person':
            cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 1)
            cv2.putText(img, name, (x + 2, y + 8),
                        cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 0, 255), 1)
        else:
            cv2.rectangle(img, (x, y), (w, h), (80, 127, 255), 1)
            cv2.putText(img, name, (x + 2, y + 8),
                        cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 0, 255), 1)

    output_video.write(img)

    cv2.imshow("WebCam", img)
    cv2.waitKey(1)

output_video.release()
