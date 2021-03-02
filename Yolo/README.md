# Yolo Test

## 訓練自己的模型
<font size="14"> Training: darknet.exe  detector train data/statue.data cfg/yolo4.cfg 
<font size="14"> Testing: darknet.exe  detector test data/statue.data  cfg/yolo4-test.cfg backup/yolo4_last.weights -thresh 0.5

## Yolo模型測試
<font size="14"> darknet.exe detector test cfg/coco.data cfg/yolov4.cfg backup/yolov4.weights -thresh 0.25
