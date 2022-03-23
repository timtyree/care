#parallel.py
#Programmer: Tim Tyree
#Date: 3.20.2022
import dask.bag as db, time

def eval_routine_daskbag(routine,task_lst,npartitions,printing=True,**kwargs):
    """eval_routine_daskbag returns a list of the values returned by routine, which takes a single argument, task, which is an element of the list, task_lst.
    the integer number of cores requested is npartitions.
    if printing is True, then the overall run time is printed.

    Example Usage:
retval=eval_routine_daskbag(routine,task_lst,npartitions,printing=True)
    """
    if npartitions>1:
        bag = db.from_sequence(task_lst, npartitions=npartitions).map(routine)
        start = time.time()
        retval = list(bag)
    else:
        retval=[]
        for task in task_lst:
            retval.append(routine(task))
    if printing:
        print(f"run time for evaluating routine was {time.time()-start:.2f} seconds, yielding {len(retval)} values returned")
    return retval
