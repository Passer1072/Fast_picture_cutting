import os
from PIL import Image
from ultralytics import YOLO
import cv2
import time

start_time = time.time()
directory_path = 'Person_photos'  # 这里填入需要剪切的图片集文件夹名称或绝对路径
output_dir = 'output'  # 输出目录名称，这里无需修改
model = YOLO('yolov8x.pt')  # 所调用的模型文件，无需修改。
Target_ID = 0  # 在这里设置检测/裁切目标，根据自己的模型进行修改,这里只能设置单个目标，需要识别多个目标需要修改下面的class_ids内容
counter = 1  # 计数器命名,无需修改
counter_images = 0  # 计数器命名,无需修改

# 检查output目录是否存在，如果不存在则创建
if not os.path.exists(output_dir):
    print("未找到output文件夹，创建文件夹")
    os.makedirs(output_dir)

for filename in os.listdir(directory_path):  # 检测文件夹内容
    if filename.endswith(".jpg") or filename.endswith(".png"):  # 读取".jpg"".png"后缀的文件
        counter_images += 1
        print("\n")
        print(f"开始处理第{counter_images}张图片")
        full_filename = os.path.join(directory_path, filename)  # 合并为完整路径
        print("已获取图像路径：", full_filename, "开始打开图像...")
        img = Image.open(full_filename)  # 打开图像
        # 处理图像
        print("当前正在处理图像：", full_filename)
        source = cv2.imread(full_filename)  # 读取图片文件

        # 进行推理
        # 设置只检测类别ID为0的对象，0代表'Person'
        class_ids = [Target_ID]  # 仅检测'Person'(人) ####
        results = model.predict(source, save=True, imgsz=320, conf=0.5, classes=class_ids)  # YOLO官方快捷推理方法

        # 查看结果
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()  # 获取框坐标以xyxy格式

            # 根据边界框坐标裁剪图像
            for box in boxes:
                print("开始获取裁剪坐标...")
                x1, y1, x2, y2 = box
                print("裁剪坐标：", x1, y1, x2, y2)
                print("开始裁剪...")
                crop_img = source[int(y1):int(y2), int(x1):int(x2)]
                print("裁剪完成")

                # 保存裁剪的图像
                cv2.imwrite(f'{output_dir}/cropped_img_{counter}.jpg', crop_img)
                counter += 1

            # print(boxes)  # 打印包含检测边界框的框对象


# -------------------------------------------------------测试部分---------------------------------------------------------
#         # 绘制结果
#         frame_ = results[0].plot()
#
#         cv2.imshow('frame_', frame_)  # 显示图像
#         cv2.waitKey(0)
#
#         time.sleep(0.1)
#
# ----------------------------------------------------------------------------------------------------------------------


end_time = time.time()

print("\n")
print("已处理完所有图片")
# 打印程序花费的总时间
print("程序的总运行时间为: ", end_time - start_time, " seconds")
print("总共处理了: ", counter_images, "张图片")