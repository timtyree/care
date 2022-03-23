# viewer_cluster.py
#Programmer: Tim Tyree
#Date: 3.22.2022
#the idea is to generate a lot of textures, batch_size, quickly on gpu, and then to make one task for each batch_size on a cpu processor that does matplotlib
from ..utils.parallel import eval_routine_daskbag

def eval_viewer_cluster(task_lst,routine_to_png,npartitions,printing=True,**kwargs):
    """
    Example Usage:
start=time.time()
retval=eval_viewer_cluster(task_lst=task_lst,routine_to_png=routine_to_png_streaming_tips,npartitions=npartitions,printing=True)
if printing:
    print(f"the apparent run time for plotting was {(time.time()-start)/60:.1f} minutes")
    """
    if printing:
        batch_size=len(task_lst)
        print (f"generating {batch_size} .png files over {npartitions} cores...")
    retval=eval_routine_daskbag(routine=routine_to_png,task_lst=task_lst,npartitions=npartitions,printing=printing,**kwargs)
    return retval
