import cv2
import numpy as np

config_file = 'yolov3/yolov3.cfg'  # Путь к файлу конфигурации модели
weights_file = 'yolov3/yolov3.weights'  # Путь к файлу весов модели
classes_file = 'yolov3/coco.names'  # Путь к файлу с названиями классов

# Список названий классов
classes = []
with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Загрузка модели
net = cv2.dnn.readNetFromDarknet(config_file, weights_file)

layers_names = net.getLayerNames()
output_layers = [layers_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Загрузка изображения
img = cv2.imread('image/2cars_people.jpeg')

# Преобразование изображения в формат, который может обработать модель
blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(416, 416),
                             mean=(0, 0, 0), swapRB=True, crop=False)

# Запуск распознавания объектов
net.setInput(blob)
outs = net.forward(output_layers)

# Обработка выходных данных
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Координаты ограничивающего прямоугольника объекта
            center_x = int(detection[0] * img.shape[1])
            center_y = int(detection[1] * img.shape[0])
            w = int(detection[2] * img.shape[1])
            h = int(detection[3] * img.shape[0])
            x = center_x - w // 2
            y = center_y - h // 2
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Нанесение ограничивающих прямоугольников на изображение
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(classes), 3))
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        cv2.putText(img, label, (x, y + 10), font, 1, color, 1)

# Отображение изображения с распознанными объектами
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
