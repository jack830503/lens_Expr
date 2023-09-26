# lens_Expr
用yolov5辨識透鏡成像實驗中的透鏡和成像，並判斷其在尺規上位置

##Hardware Enviornment
- GPU : 選用 NVIDIA Geforce RTX 3060 Laptop GPU
***
##Software Enviornment
- OS : Window10

- [Python](https://www.python.org/)** : 3.9.6

- [CUDA](https://developer.nvidia.com/cuda-toolkit-archive)** : 11.1.0 (network版)
- [CUDNN](https://developer.nvidia.com/rdp/cudnn-archive): v8.1.1 for CUDA 11.0,11.1 and 11.2
- [pytorch](https://pytorch.org/): 1.9.0 + cu111
  ![image](https://github.com/jack830503/lens_Expr/blob/main/pictures/Untitled%20(1).png)
`下指令安裝pytorch，須注意其支援的CUDA版本號。`
***
##Install YOLOv5
1. 下載source code
2. 打開PS，進入source code所在路徑
3. 安裝所需python套件
```
pip install -r requirements.txt
``` 
4. 可直接執行detect.py，看是否成功安裝

***
##Training YOLOv5
1. 建立資料夾
  - 要訓練的照片放在yolov5\images資料夾裡
  - 如label為.xml則放在yolov5\Annotations
  - ImageSets/Main，之後會生成train.txt，val.txt，test.txt和trainval.txt四個文件，存放訓練集、驗證集、測試集圖片的名字
2. split_train_val.py，資料劃分
  - 在yolov5\ImageSets\Main資料夾下生成訓練集、驗證集、測試集
  - txt內容為照片檔名
![image](https://github.com/jack830503/lens_Expr/blob/main/pictures/Untitled%20(2).png)
3. voc_label.py，轉檔並將數據集路徑存到txt中
  - 如label的格式是.xml格式，則需要轉檔，轉成yolo_txt格式，一張照片對應到一個txt文件
  - 格式如下: <br>  
![image](https://github.com/jack830503/lens_Expr/blob/main/pictures/Untitled%20(3).png)
  內容為 class,  x_center,  y_center,  width,  height，一行為一個目標訊息 <br>
![image](https://github.com/jack830503/lens_Expr/blob/main/pictures/Untitled.png)

  - 轉檔完成的label放在yolov5\labels資料夾裡，train.txt等txt文件為劃分後圖像所在位置的絕對路徑 <br>
![image](https://github.com/jack830503/lens_Expr/blob/main/pictures/Untitled%20(4).png) <br>

![image](https://github.com/jack830503/lens_Expr/blob/main/pictures/Untitled%20(5).png) <br>

4. 配置文件
  - 在yolov5\data資料夾下新建一個rgbo.yaml文件(名字可以自定義)，用來存放訓練集和驗證集的劃分文件(train.txt和val.txt)
  - rgbo.yaml內容如下 <br>
![image](https://github.com/jack830503/lens_Expr/blob/main/pictures/Untitled%20(6).png)

5. 模型訓練
```
python train.py --img 640 --batch 16 --epoch 300 --data data/ab.yaml   --weights weights/yolov5s.pt --device '0'
```
  - 訓練完成的結果放在yolov5\runs\train裡面

6. 使用模型來辨識圖片
```
python detect.py --weights runs/train/exp10/weights/best.pt --source inference/images/ --device 0 --save-txt
```
  - weights: 剛剛訓練產生的權重
  - source: 需要辨識的照片，0代表即時辨識
7. 結果放在yolov5\runs\detect裡面 <br>
![image](https://github.com/jack830503/lens_Expr/blob/main/pictures/155-380-530-l.png) <br>
  - 每個測試圖片會生成結果圖片和同名的txt
  - txt內容如下: <br>
![image](https://github.com/jack830503/lens_Expr/blob/main/pictures/Untitled%20(7).png) <br>
  內容為: 類別編號、xcenter、ycenter、w、h，後面四個為正規化後的數值

***
## preDetect
1. 尺標上60公分(600mm)。每間格x mm拍1張照片。ex: 在5,10,15...mm處個拍一張照片。照片中只能有一個標籤。

2. 右相機拍到照片放入preDetect/rightPhoto裡; 左相機拍到照片放入preDetect/leftPhoto裡。

3. 執行rename.py

4. 執行pre_detect.py。生成照片的label檔以及標記尺上座標的json檔。rightPhoto和leftPhoto各有一個。

***

## Implement
- 執行 detect.py
`output : 物件'object',位置'position'`

## Add hand gesture to start
- 執行handtest.py
`在鏡頭下比讚啟動辨識，有go表示開始辨識`