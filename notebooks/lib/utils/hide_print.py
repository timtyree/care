# hide_print.py
#Programmer: Tim Tyree
#Date: 6.29.2022
import os, sys
class HiddenPrints:
    """
    Example Usage:
og_stdout=sys.stdout
with HiddenPrints(): #  (og_stdout=og_stdout):  #kwargs are not for with statements in py39...
    traj = compute_track_tips_pbc(df_log_local, mem=mem, sr=sr, width=width, height=height)

    Example Usage:
 with HiddenPrints():
    retval=tshift_tare_routine(df_R,navg2,max_num_groups=9e9,plotting=False,npartitions=10,R_col='R_nosavgol',printing=True)
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
