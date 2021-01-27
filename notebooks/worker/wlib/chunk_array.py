def chunk_array(txt_in,width_out,height_out,typeout='float64'):
    '''returns txt_out_lst
    suppose file_name is a bare string with the extension  ".npz"
    suppose txt_in is a numpy array that is (1800,1800,The_Rest)
    count is the ic number of the most recent ic00X file saved (increments by 1 before saving).
    '''    
    width_in,height_in=txt_in.shape[:2]
    txt_out_lst=[]
    x=0;y=0;
    x_lst=list(range(0,width_in-width_out,width_out))
    y_lst=list(range(0,width_in-width_out,width_out))
    for x in x_lst:
        for y in y_lst:
            arr=txt_in[x:x+width_out,y:y+height_out]
            txt_out_lst.append(arr.astype(typeout))
    return txt_out_lst


if __name__=="__main__":
    width_in=500;height_in=500;chnlno=2
    shape=(width_in,height_in,chnlno)
    txt_in=np.zeros(shape,dtype=np.float64)

    txt_out_lst= chunk_array(txt_in,width_out,height_out,typeout='float64')
    len(txt_out_lst)