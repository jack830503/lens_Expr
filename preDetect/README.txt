物件位置訓練流程:

pre_processing:
1. 尺上60公分(600mm)。每間格x mm拍1張照片。ex: 在5,10,15...mm處個拍一張照片。照片中只能有一個標籤。

2. 右相機拍到照片放入preDetect/rightPhoto裡; 左相機拍到照片放入preDetect/leftPhoto裡。

3. 執行rename.py

4. 執行pre_detect.py。生成照片的label檔以及標記尺上座標的json檔。rightPhoto和leftPhoto各有一個。


detect: 執行detect.py。output : 物件'object',位置'position'








