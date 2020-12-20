#!/bin/env python3


def print_uniform_circ_motion(n,R,V,omega,T):
    print(f"""for tip trajectory #{n}:
    \tradius of circular spiral  = {R:.2f} pixels,
    \taverage speed  = {V:.2f} pixels/time,
    \tangular velocity  = {omega:.2f} radians/time.
    \timplies the period  = {T:.2f} time units.
    """)
#######################################
###From generating tip trajectory ipynb
#######################################
# #report the bottom line up front
# print(f"time integration complete. time elapsed was {time.time()-start:.2f} seconds")
# beep(3)
# print(f"current time is {tme:.1f}.\n")
# print(f"number of nan pixel voltages is {np.max(sum(np.isnan(txt[...,0])))}.")
# # if len(lst)~=0:
# # print(f"number of tips is = {set([len(q) for q in lst_x[-1]])}.") #most recent number of tips
# print(f"current max voltage is {np.nanmax(txt[...,0]):.4f}.")
# print(f"current max fast variable is {np.nanmax(txt[...,1]):.4f}.")
# print(f"current max slow variable is {np.nanmax(txt[...,2]):.4f}.")
# n_lst, x_lst, y_lst = get_tips(contours_raw, contours_inc)
# tip_states = {'n': n_lst, 'x': x_lst, 'y': y_lst}
# # print(f"tip_states are {tip_states}.")
# # print(f'current tip state is {tip_states}')
#
# if recording:
#     tips = get_tips(contours_raw, contours_inc);
#     print(f"\n number of type 1 contour = {len(contours_raw)},\tnumber of type 2 contour = {len(contours_inc)},")
#     print(f"""the topological tip state is the following:
#     {tips[0]}""")
# beep(3)

# #report the bottom line up front
# print(f"time integration complete. time elapsed was {time.time()-start:.2f} seconds")
# print(f"current time is {tme:.1f}.")
# print(f"number of nan pixel voltages is {np.max(sum(np.isnan(txt[...,0])))}.")
# print(f"current max voltage is {np.nanmax(txt[...,0]):.4f}.")
# print(f"current max fast variable is {np.nanmax(txt[...,1]):.4f}.")
# print(f"current max slow variable is {np.nanmax(txt[...,2]):.4f}.")
# n_lst, x_lst, y_lst = get_tips(contours_raw, contours_inc)
# tip_states = {'n': n_lst, 'x': x_lst, 'y': y_lst}
# # print(f"tip_states are {tip_states}.")
# # print(f'current tip state is {tip_states}')
# # if len(lst)~=0:
# # print(f"number of tips is = {set([len(q) for q in lst_x[-1]])}.") #most recent number of tips
#
# if recording:
#     tips = get_tips(contours_raw, contours_inc);
#     print(f"\n number of type 1 contour = {len(contours_raw)},\tnumber of type 2 contour = {len(contours_inc)},")
#     print(f"""the topological tip state is the following:
#     {tips[0]}""")x slow variable is {np.nanmax(txt[...,2]):.4f}.")
