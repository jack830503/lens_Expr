
import numpy as np
import os


def datasets(path):
	data_int = []	#存放訓練集(x座標)
	target = []		#存放訓練集的標籤
	count = 0
	detect = os.listdir(path+'label')   #yolo訓練出來的label資料夾
	print(detect)
	album = os.listdir(path+'photo')    #訓練圖集
	print(album)
	#album = sorted(list(map(int,album)))
	for photo in album:
		data = os.listdir(path+'photo/'+str(photo))
		for i in data:
			if i.split('.')[0] == detect[count].split('.')[0]:
				with open(path+'label/'+detect[count]) as f:
					d = f.readlines()
					for j in range(len(d)):
						#data_int.append(list(map(float,d[j].split(' ')[1::])))
						data_int.append(float(d[j].split(' ')[1]))
						target.append(int(photo))
				count += 1


	return data_int, target

def datasets2(path):
	data_x = []
	target = []
	folder = os.listdir(path)
	for file in folder:
		with open(path+file) as f:
			for line in f:
				data_x.append(float(line.split(' ')[1]))
				target.append(int(file.split('_')[0]))
	print(folder)
	print(data_x)
	print(target)
	print(len(data_x),len(target))
	return data_x,target












