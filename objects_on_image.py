from imageai.Detection import ObjectDetection
import os

exec_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(exec_path, "yolov3/yolov3.pt"))
detector.loadModel()

custom_objects = detector.CustomObjects(car=True, bicycle=True)

detections = detector.detectObjectsFromImage(
    custom_objects=custom_objects,
    input_image=os.path.join(exec_path, "image/2cars_people.jpeg"),
    output_image_path=os.path.join(exec_path, "new_objects.jpg"),
    display_percentage_probability=False
)

for eachObject in detections:
    print(
        eachObject["name"],
        " : ",
        eachObject["percentage_probability"],
        " : ",
        eachObject["box_points"]
    )
    print("--------------------------------")
