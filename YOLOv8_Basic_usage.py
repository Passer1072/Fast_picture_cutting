import cv2
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8x.pt')

# Read an image using OpenCV
source = cv2.imread('img.png')

# # 对源进行推理
# results = model(source)
#
# # 查看结果
# for r in results:
#     boxes = r.boxes.xyxy.cpu().numpy()
#     print(boxes)
#
#     # 遍历每个框并画出它
#     for box in boxes:
#         # 获得整型的边界框坐标
#         x1, y1, x2, y2 = map(int, box)
#         # 在源图像上绘制边界框
#         cv2.rectangle(source, (x1, y1), (x2, y2), (255, 0, 0), 2)
#
# # 显示带有边界框的图像
# cv2.imshow('Detections', source)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


results = model.predict(source, save=True, imgsz=320, conf=0.3)
for r in results:
    boxes = r.boxes.xyxy.cpu().numpy()  # 获取框坐标,参数:.xyxy
    print(boxes)  # 打印包含检测边界框的框对象

# 绘制结果
frame_ = results[0].plot()

# 显示结果
cv2.imshow('frame', frame_)
cv2.waitKey(0)
cv2.destroyAllWindows()