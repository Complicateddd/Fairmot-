import os




def getimage_path(track_root="/home/ubuntu/deepsort/deep_sort_pytorch-master/deep_sort_pytorch-master/dataset/A-data/Track1"):
	list_path=[]
	
	filelist=os.listdir(track_root)

	sencondfilelist=os.listdir(track_root)
	sencondfilelist.sort(key=lambda x:int(x[:-4]))
	for j in sencondfilelist:
		path2=track_root+"/"+j
		list_path.append(path2)
		# print(path2)
	return list_path
# getimage_path('/home/ubuntu/deepsort/xianjiaoshuo')