from .. import *
from . import *
def load_buffer_LR(input_file_name, Ca_i_initial = 2*10**-4, Vmax = 35., Vmin = -85.):
    '''
    load_buffer_LR maps the 3 channel/variable initial conditions in txt_ic
    to a not-unreasonable  8 channel/variable buffer initial condition as inVmhjdfx
    and linearly rescales voltage by Vscale and shifts by Vshift.
    
    Ca_i_initial is a float in calcium concentration units of mM.
    input_file_name is a string listing the path to one .npz file with the first entry as a width by height by 3 channel numpy array.
    Example Usage:
    inVmhjdfx = load_buffer_LR(input_file_name)
    
    By default, maps a dimensionless, normalized voltage Vmin from 0 to -84mV and Vmax from 1 to 35mV.
    '''
    txt_ic = load_buffer(input_file_name)
        
    Vscale = Vmax-Vmin
    Vshift = Vmin
    
    #TODO: try setting each field to it's equilibrium value

    Vfield = Vscale*txt_ic[...,0]+Vshift
    ffield = txt_ic[...,1]
    sfield = txt_ic[...,2]



    # - $m$ looks like 1-F
    # - $h$ looks like F
    # - $j$ looks like F
    # - $d$ looks like 1-F
    # - $f$ looks like S
    # - $w$ looks like 1-S

    # mfield = (1.-ffield.copy())**2
    mfield = 1.-ffield.copy()
    hfield = ffield.copy()
    jfield = ffield.copy()
    dfield = (1.-ffield.copy())*sfield.copy()
    Ffield = sfield.copy()
    xfield = 1.-sfield.copy() #1.-sfield.copy() 
    inCa_i = 45/2*Ca_i_initial/(sfield.copy())*np.max(sfield)+Ca_i_initial

    # mfield = ffield.copy()
    # hfield = 1.-ffield.copy()
    # jfield = 1.-ffield.copy()
    # dfield = ffield.copy()
    # Ffield = 1.-ffield.copy()
    # xfield = 0.*sfield.copy()
    # inCa_i = 45/2*Ca_i_initial*(ffield.copy())/np.max(ffield)+Ca_i_initial


    # mfield = 0.*ffield.copy()#+1.
    # hfield = 0.*ffield.copy()+1.
    # jfield = 0.*ffield.copy()+1.
    # dfield = 0.*ffield.copy()
    # Ffield = 0.*sfield.copy()#+1.
    # xfield = 0.*sfield.copy()#+0.2
    # inCa_i = 0.*ffield.copy()+Ca_i_initial

    # Ffield = 0.*ffield.copy()

    #precompute lookup table
    dt=0.1
    arr39=get_arr39(dt,nb_dir)
    v_values=arr39[:,0]
    lookup_params=get_lookup_params(v_values,dv=0.1)
    
    #set each gating field to it's equilibrium value
    width,height=Vfield.shape
    for x in range(width):
        for y in range(height):
            #parse the row linearly interpolated from lookup table with updated voltage
            V = Vfield[x,y]
            arr_interp=lookup_params(V,arr39)
            #parse the linearly interpolated row
            x_infty =arr_interp[1]    # 'xinf1', 
            # tau_x =arr_interp[2]    # 'xtau1', 
            m_infty=arr_interp[3]    # 'xinfm', 
            # tau_m=arr_interp[4]    # 'xtaum', 
            h_infty=arr_interp[5]    # 'xinfh', 
            # tau_h=arr_interp[6]    # 'xtauh', 
            j_infty=arr_interp[7]    # 'xinfj',
            # tau_j=arr_interp[8]    # 'xtauj', 
            d_infty=arr_interp[9]    # 'xinfd', 
            # tau_d=arr_interp[10]    # 'xtaud', 
            f_infty=arr_interp[11]    # 'xinff', 
            # tau_f=arr_interp[12]    # 'xtauf', 

            # mfield[x,y]=m_infty
            hfield[x,y]=h_infty
            # jfield[x,y]=j_infty
            # dfield[x,y]=d_infty
            # Ffield[x,y]=f_infty
            # xfield[x,y]=x_infty

    inVmhjdfxc = np.array([
        Vfield,
        mfield, 
        hfield,
        jfield,
        dfield,
        Ffield,
        xfield,
        inCa_i
    ],dtype=np.double).T
    return inVmhjdfxc

def get_voltage_field_scaled_and_shifted(txt, Vmin_new=0., Vmax_new=1., Vmin_old=-85., Vmax_old=35.):
    '''get_voltage_field_scaled_and_shifted returns the transmembrane voltage field with values scaled/shifted to Vmin_new and Vmax_new.
    txt is a numpy array with transmembrane voltage field in the first channel, and may contain other fields in later channels.
    '''
    Vfield = txt[...,0]
    Vmin_old = np.min(Vfield)
    Vscale = (Vmax_new-Vmin_new)/(Vmax_old-Vmin_old)
    Vshift = Vmax_new/Vscale-Vmax_old
    Vfield_new = Vscale*(Vfield+Vshift)
    return Vfield_new