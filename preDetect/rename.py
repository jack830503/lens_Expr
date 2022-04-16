'''拍到的照片檔名都是日期時間，需重新命名。右相機照片由時間先後依序重新命名為5, 10, 15...(依照照片中標籤位置命名)。左相機則為595, 590, 585...。 '''
import os


pathL = 'C:/Users/Jen/Documents/project_1_lens/preDetect/leftPhoto/'  #左邊相機拍到的照片放在此資料夾
pathR = 'C:/Users/Jen/Documents/project_1_lens/preDetect/rightPhoto/' #右邊相機拍到的照片放在此資料夾

dir = os.listdir(pathL)
i = 595
for file in dir:
    filename = pathL+file
    newname = pathL+ str(i) + '.jpg'
    os.rename(filename,newname)
    i -= 5

dir = os.listdir(pathR)
i = 5
for file in dir:
    filename = pathR+file
    newname = pathR+ str(i) + '.jpg'
    os.rename(filename,newname)
    i += 5


