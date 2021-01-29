import os
def make_directories(input_dir):
    '''make f"{worker_dir}/wlib/" or /tmp/ be the absolute directory of the input'''
    worker_dir=os.path.abspath(input_dir+'/worker')
    if not os.path.exists(worker_dir):
        os.mkdir(worker_dir)
    wlib_dir=os.path.abspath(worker_dir+'/wlib')
    if not os.path.exists(wlib_dir):
        os.mkdir(wlib_dir)
    log_dir=os.path.abspath(worker_dir+'/log')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    ic_dir=os.path.abspath(worker_dir+'/ic')
    if not os.path.exists(ic_dir):
        os.mkdir(ic_dir)

if __name__=="__main__":
    input_dir=os.getcwd()
    make_directories(input_dir)