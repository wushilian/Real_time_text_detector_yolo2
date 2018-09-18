# Real_time_text_detector_yolo2
most projects focus on text line detection,this project use yolo2 to detect character which is easy for recognition

## Differences from paper
we use smooth L1 loss  and cross entropy insted of L2 loss in darknet.<br>
With smaller backbone,our speed could reach 300FPS
## Dependency Library
tensorflow 1.8(low version should also be able to run)<br>
opencv<br>
## How to use
1.run train_from_Synthetic.py<br>
2.run test.py

## example result
 ![image](https://github.com/wushilian/Real_time_text_detector_yolo2/raw/master/result/result.jpg)
 ![image](https://github.com/wushilian/Real_time_text_detector_yolo2/raw/master/result/cls.jpg)
 
## wait to update
