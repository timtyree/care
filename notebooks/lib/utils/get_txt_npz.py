import os, numpy as np
from .chunk_array import chunk_array
from .gdown import download_file_from_google_drive
#install dependencies
# os.system('python3 -m pip install --upgrade pip')
# os.system('python3 -m pip install gdown')
#suppose get_txt.sh has run
# os.system('chmod +x get_txt.sh')
# os.system(f'./get_txt.sh {txt_id}')

def run_downloader(gid):
	txt_ic_fn='ic/ic1800x1800.npz'
	destination=txt_ic_fn
	retval=download_file_from_google_drive(gid, destination)
	return None

def get_gid(txt_id):
	import random
	def decision(probability):
	    return random.random() < probability
	#two gid's per texture lowers the load on google drive servers
	if txt_id==0:#at time, 1210
		if decision(0.5):
			gid='1OYtQNnp5KnGfKMkskk7GeDQSCe3Mo7Gu'
		else:
			gid='1LTQxE9sacdb3BidFYeefqKUzEA_HiSOu'
	if txt_id==1:#at time, 2020
		if decision(0.5):
			gid='1td_6aQHFWzvunt1kU14ViW5DZ69rUhMD'
		else:
			gid='1qf2-Cf5Bbfjos5QDxp2FZtyJoL3FU4zO'
	if txt_id==2:#at time, 2830
		if decision(0.5):
			gid='12dLQ_YFwSAvuuZc1lhNsKPcv4QXZB86u'
		else:
			gid='1MCM6hVxC0Ch73ZnK97PKhjPChHI0PRxx'
	if txt_id==3:#at time, 3640
		if decision(0.5):
			gid='14SipoA-gemvfyuA5v9tAUQRP3Firmu8G'
		else:
			gid='1vmeI5SyyveaZ00qEeiqb04MVDssw9s5p'
	return gid

def download_txt(txt_id,worker_dir):
	'''returns the first gdrive download file found in the directory, worker_dir.'''
	os.chdir(worker_dir)
	if not os.path.exists('ic'):
		os.mkdir('ic')
	gid=get_gid(txt_id)
	run_downloader(gid=gid)
	# cmd=f'gdown https://drive.google.com/uc?id={gid} -O ic/ic1800x1800.npz'
	# os.system(cmd)#at time, 1210
	# if txt_id==0:
	# 	# run_downloader(gid='1OYtQNnp5KnGfKMkskk7GeDQSCe3Mo7Gu')
	# 	os.system('gdown https://drive.google.com/uc?id=1OYtQNnp5KnGfKMkskk7GeDQSCe3Mo7Gu -O ic/ic1800x1800.npz')#at time, 1210
	# if txt_id==1:
	# 	# run_downloader(gid='1td_6aQHFWzvunt1kU14ViW5DZ69rUhMD')
	# 	os.system('gdown https://drive.google.com/uc?id=1td_6aQHFWzvunt1kU14ViW5DZ69rUhMD -O ic/ic1800x1800.npz')#at time, 1210
	# if txt_id==2:
	# 	# run_downloader(gid='12dLQ_YFwSAvuuZc1lhNsKPcv4QXZB86u')
	# 	os.system('gdown https://drive.google.com/uc?id=12dLQ_YFwSAvuuZc1lhNsKPcv4QXZB86u -O ic/ic1800x1800.npz')#at time, 1210
	# if txt_id==3:
	# 	# run_downloader(gid='14SipoA-gemvfyuA5v9tAUQRP3Firmu8G')
	os.chdir(worker_dir)
	txt=load_buffer('ic/ic1800x1800.npz')[0]#,allow_pickle=True)
	# txt=load_buffer('ic/ic1800x1800.npz')[0]#,allow_pickle=True)
	return txt

def get_txt_lst(txt_id1,width,height,worker_dir):
	txt_in=download_txt(txt_id1,worker_dir)
	array_lst = chunk_array(txt_in, width, height, typeout='float64')
	return array_lst

def get_txt(txt_id1,txt_id2,width,height,worker_dir):
	array_lst=get_txt_lst(txt_id1,width,height,worker_dir)
	# N=len(array_lst)
	try:
		txt=array_lst[txt_id2]
	except IndexError as e:
		import random
		print (f'IndexError for {(width,txt_id1,txt_id2)}')
		print ( e )
		txt_id2=random.randint(0,len(array_lst)-1)
		print (f'Choosing txt_id2={txt_id2}...')
		txt=array_lst[txt_id2]#-1]

	return txt

def load_buffer(data_dir,**kwargs):
	if data_dir[-4:]=='.npy':
		txt = np.load(data_dir)
		return txt
	elif data_dir[-4:]=='.npz':
		txt = np.load(data_dir,**kwargs)
		txt = txt[txt.files[0]]  #only take the first buffer because there's typically one
		return txt
	else:
		print(f"Warning: file format not supported {data_dir}.")
		raise Exception(f"Warning: file format not supported {data_dir}.")


if __name__=='__main__':
	os.get_cwd()
	for input_fn in sys.argv[1:]:
		download_txt(txt_id,worker_dir)