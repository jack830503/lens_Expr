'''將每個訓練照片位置x座標平均後記錄下來。測試照片x座標靠近紀錄的哪個點就認定在那個位置'''
import os
import time
pathR = 'C:/Users/hscc/Pictures/right/label/'
pathL = 'C:/Users/hscc/Pictures/left/label/'

def avg(path):      #計算每個位置的平均
    folder = os.listdir(path)
    x = []
    tmp = -1
    sum = 0
    count = 100
    
    for file in folder:
        if file.split('_')[0] != tmp:
            avg = sum/count
            x.append(avg)
            sum = 0
            count = 0
        with open(path+file) as f:
            for line in f:
                sum += float(line.split(' ')[1])
                count += 1
        tmp = file.split('_')[0]
    x.append(sum/count)     #加最後一個
    x.pop(0)
    return x

def RL(file):  #判別R or L相機
	if file.split('-')[-1].split('.')[0] == 'l':
		return 1
	elif file.split('-')[-1].split('.')[0] == 'r':
		return 0


def open_data(path):    #開啟檔案，回傳該檔案辨識的物件種類和x座標。以list回傳
	test_label = []
	obj = []
	with open(path) as f:
		data = f.readlines()
		#for label in data:
			#test_label.append(list(map(float,label.split(' ')[1::])))
			
		for i in range(len(data)):
			test_label.append(float(data[i].split(' ')[1]))
			obj.append(int(data[i].split(' ')[0]))
	#print(test_label)
	return test_label, obj

# 輸入該檔案的座標(list型態)，x紀錄每個位置(avg()回傳結果)，int表示左還是右相機
def detect(test_label,x,int):
    length = len(x)
    position = []
    for value in test_label:
        if int == 0:
            if float(value) < x[-1]:
                position.append(None)
            else:
                for i in range(length):
                    if float(value) < x[i]:
                        continue
                    elif float(value) > x[0]:
                        position.append(0) 
                        break
                    elif float(value) > x[i] and (float(value)-x[i] < x[i-1]-float(value)):
                        position.append(i*5)
                        break  
                    elif float(value) > x[i] and (float(value)-x[i] > x[i-1]-float(value)):
                        position.append((i-1)*5) 
                        break             
        else:
            if float(value) > x[-1]:
                position.append(None)
            else:
                for i in range(length):
                    if float(value) > x[i]:
                        continue
                    elif float(value) < x[0]:
                        position.append(600) 
                        break
                    elif float(value) < x[i] and (float(value)-x[i-1] < x[i]-float(value)):
                        position.append(600 -(i-1)*5)
                        break 
                    elif float(value) < x[i] and (float(value)-x[i-1] > x[i]-float(value)):
                        position.append(600 - i*5)
                        break 
    return position


        

x_R = avg(pathR)
x_L = avg(pathL)
test_path = 'C:/Users/hscc/Documents/smalltag/runs/detect/exp22/labels/'
test_Folder = os.listdir(test_path)
start = time.time()
for file in test_Folder:
    #print(file)
    test_label, obj = open_data(test_path+file)
    din = RL(file)
    if din == 0:
        position = detect(test_label,x_R,0)
    else:
        position = detect(test_label,x_L,1)
    #print(obj,position)

end = time.time()
print(end-start)
