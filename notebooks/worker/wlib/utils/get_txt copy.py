import os, numpy as np
from .chunk_array import chunk_array
#install dependencies
# os.system('python3 -m pip install --upgrade pip')
# os.system('python3 -m pip install gdown')
#suppose get_txt.sh has run
# os.system('chmod +x get_txt.sh')
# os.system(f'./get_txt.sh {txt_id}')

def download_txt(txt_id,worker_dir):
    '''returns the first gdrive download file found in the directory, worker_dir.'''
    os.chdir(worker_dir)
    if txt_id==0:
        os.system('gdown https://drive.google.com/uc?id=1OYtQNnp5KnGfKMkskk7GeDQSCe3Mo7Gu -O ic/ic1800x1800.npz')#at time, 1210
    if txt_id==1:
        os.system('gdown https://drive.google.com/uc?id=1td_6aQHFWzvunt1kU14ViW5DZ69rUhMD -O ic/ic1800x1800.npz')#at time, 1210
    if txt_id==2:
        os.system('gdown https://drive.google.com/uc?id=12dLQ_YFwSAvuuZc1lhNsKPcv4QXZB86u -O ic/ic1800x1800.npz')#at time, 1210
    if txt_id==3:
        os.system('gdown https://drive.google.com/uc?id=14SipoA-gemvfyuA5v9tAUQRP3Firmu8G -O ic/ic1800x1800.npz')#at time, 1210
    os.chdir(worker_dir)
    txt=load_buffer('ic/ic1800x1800.npz')[0]#,allow_pickle=True)
    return txt


def get_txt(txt_id1,txt_id2,width,height,worker_dir):
    txt_in=download_txt(txt_id1,worker_dir)
    array_lst = chunk_array(txt_in, width, height, typeout='float64')
    N=len(array_lst)
    txt=array_lst[txt_id2]
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
		print(f"\tWarning: Failed to load {data_dir}.")
		raise Exception(f"\tWarning: Failed to load {data_dir}.")


if __name__=='__main__':
	for input_fn in sys.argv[1:]:
		chunk_from_fn(input_fn)
