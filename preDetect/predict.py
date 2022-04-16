 
# %%
import os
import joblib
import numpy as np
import json
import requests
import joblib

# %%
test_path = 'C:/Users/hscc/Documents/smalltag/runs/detect/exp22/labels/'
model_pathR = 'C:/Users/hscc/Documents/smalltag/pos/model/modelRight2_knn'
model_pathL = 'C:/Users/hscc/Documents/smalltag/pos/model/modelleft2_knn'
result = []
obj_total = []
def open_data(path):
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

def RL(label):  #判別R or L相機
	if label.split('-')[-1].split('.')[0] == 'l':
		return 1
	elif label.split('-')[-1].split('.')[0] == 'r':
		return 0

def switch(result,label):
	rst = []
	if RL(label) == 1:
		for i in range(len(result)):
			pos = 600 - int(result[i])*5
			rst.append(pos)

	elif RL(label) == 0:
		for i in range(len(result)):
			pos = int(result[i])*5
			rst.append(pos)
	return rst



data = os.listdir(test_path)

#model = joblib.load(model_path)

for label in data:
	test_label , obj = open_data(test_path+label)
	test_label = np.array(test_label).reshape(len(test_label),1) #一微陣列要轉為二維陣列
	#print(test_label)
	if RL(label) == 1:
		model = joblib.load(model_pathL)
		rst = model.predict(test_label)

	elif RL(label) == 0:
		model = joblib.load(model_pathR)
		rst = model.predict(test_label)
	
	rst = switch(rst,label)
	#print(rst)
	#result.append(rst)
	#obj_total.append(obj)
	print(label,obj,rst,RL(label))
	#print(label)

#print(result)
#print(len(result))

'''
for i in range(len(result)):
	for j in range(len(result[i])):
		pos = 5 + (result[i][j]-1)*10
		result[i][j] = pos

print(os.listdir('C:/Users/hscc/Documents/smalltag/pic_location_new/test/labels/'))
print(result)
print(obj_total)
'''
'''json傳輸'''


num_total = len(result)
url = 'https://cosci.tw/lens'
#headers = {}
#with open('C:/Users/hscc/Documents/smalltag/pic_location_new/test/exp4/json.txt','a+') as f:
for i in range(num_total):
	num = len(result[i])
	for j in range(num):
		v_obj = int(obj_total[i][j])
		v_rst = int(result[i][j])
		#df = json.dumps({'obj':v_obj,'x':v_rst,'hash':'2XJj/rmRh)~)?E5v'})
		df = {'obj':v_obj,'x':v_rst,'hash':'2XJj/rmRh)~)?E5v'}
		response = requests.post(url,data = df,verify = False)
		print(response.text)
		#f.write(df+'\n')


