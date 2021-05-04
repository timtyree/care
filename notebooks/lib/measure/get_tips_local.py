from ..my_initialization import *

# @njit
def compute_all_spiral_tips(img,dimgdt,level1,level2,width=200,height=200):
    #compute all spiral tips present
    retval = find_intersections(img,dimgdt,level1,level2,theta_threshold=theta_threshold)
    # level2=V_threshold
    # retval = find_intersections(img1,img2,level1,level2,theta_threshold=theta_threshold)
    lst_values_x,lst_values_y,lst_values_theta, lst_values_grad_ux, lst_values_grad_uy, lst_values_grad_vx, lst_values_grad_vy = retval
    return format_spiral_tips(lst_values_x,lst_values_y,lst_values_theta, lst_values_grad_ux,
                              lst_values_grad_uy, lst_values_grad_vx, lst_values_grad_vy)
# @njit
def format_spiral_tips(lst_values_x,lst_values_y,lst_values_theta, lst_values_grad_ux, lst_values_grad_uy, lst_values_grad_vx, lst_values_grad_vy):
    x_values = np.array(lst_values_x)
    y_values = np.array(lst_values_y)
    # EP states given by bilinear interpolation with periodic boundary conditions
    v_lst    = interpolate_img(x_values,y_values,width,height,img=img)
    dvdt_lst = interpolate_img(x_values,y_values,width,height,img=dimgdt)
 
    n_tips = x_values.size
    dict_out = {
        't': float(t),
        'n': int(n_tips),
        'x': list(lst_values_x),
        'y': list(lst_values_y),
        'theta': list(lst_values_theta),
        'grad_ux': list(lst_values_grad_ux),
        'grad_uy': list(lst_values_grad_uy),
        'grad_vx': list(lst_values_grad_vx),
        'grad_vy': list(lst_values_grad_vy),
        'v':v_lst,
        'dvdt':dvdt_lst,
    }
    return dict_out

# Example Usage
if __name__=="__main__":
    pass
    # #compute all spiral tips present
    # V_threshold=20.#10.#0.#mV
    # level1=V_threshold
    # # theta_threshold=0.
    # level2=0.
    #
    # #update texture namespace
    # inVc,outVc,inmhjdfx,outmhjdfx,dVcdt=unstack_txt(txt)
    # # txt=stack_txt(inVc,outVc,inmhjdfx,outmhjdfx,dVcdt)
    #
    # img=inVc[...,0]
    # dimgdt=dVcdt[...,0]
    # width=200;height=200
    # dict_out=compute_all_spiral_tips(t,img,dimgdt,level1,level2,width=width,height=height)
    # dict_out_instantaneous=dict_out
