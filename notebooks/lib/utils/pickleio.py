#pickleio.py
import pickle,os
import scipy.io as sio
def save_to_pkl(input_fn,mat):
    with open(input_fn,'wb') as f: pickle.dump(mat, f)
    return os.path.abspath(input_fn)

def load_from_pkl(input_fn):
    """
    Example Usage:
dict_pkl=load_from_pkl(pkl_fn)
    """
    with open(input_fn,'rb') as f: dict_pkl = pickle.load(f)
    return dict_pkl

#to play well with matlab users
def save_to_mat(input_fn,mdict):
    sio.savemat(
            file_name=input_fn,
            mdict=mdict,appendmat=False,format='5',long_field_names=True,do_compression=True,oned_as='row')
    return os.path.abspath(input_fn)

# Various sizes of files
# - 683925266 Oct 27 19:52 NewPopulationStructure.mat
# - 383019256 Feb 11 19:01 NewPopulationStructure.pkl
# - 4431832018 Sep 16 15:56 PCA_Timeset1_UnitSet1v2Independent.mat
# - 4807197436 Mar 19  2021 MassivePopulationStorageTotal.mat

# Advantages of .pkl Versus .mat pickle
# - .pkl loads into virtual memory in <3 seconds vs. .mat at >7 minutes...
# - .pkl is half the size in memory vs. .mat

# Disadvantages of .pkl it is not json serializable in general
#HINT: see convert_to_dict_recursive

##################################
# aliases
##################################
def load_pkl(input_fn):
    return load_from_pkl(input_fn)
def load_pickle(input_fn):
    return load_from_pkl(input_fn)
def load_from_pickle(input_fn):
    return load_from_pkl(input_fn)
def save_pkl(input_fn,mat):
    return save_to_pkl(input_fn,mat)
def save_pickle(input_fn,mat):
    return save_to_pkl(input_fn,mat)
def save_to_pickle(input_fn,mat):
    return save_to_pkl(input_fn,mat)
