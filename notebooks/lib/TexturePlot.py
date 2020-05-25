import numpy as np
import matplotlib.pyplot as plt

def describe_texture(txt):
    print(f"""max value: {np.max(txt)}
    min value: {np.min(txt)}
    mean value: {np.mean(txt)}""")
##Example usage
# describe_texture(txt[...,0])
# describe_texture(txt[...,1])
# describe_texture(txt[...,2])

def display_texture(txt, vmins, vmaxs):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,6))
    ax1.imshow(txt[...,0], cmap='Reds', vmin=vmins[0], vmax=vmaxs[0])
    ax2.imshow(txt[...,1], cmap='Reds', vmin=vmins[1], vmax=vmaxs[1])
    ax3.imshow(txt[...,2], cmap='Reds', vmin=vmins[2], vmax=vmaxs[2])
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    ax1.set_title('voltage/channel 0')
    ax2.set_title('fast_var/channel 1')
    ax3.set_title('slow_var/channel 2')
    plt.show()
##Example usage
# txt = np.load('Data/buffer_test_error.npy')
# dtexture_dt = np.zeros((width, height, channel_no), dtype = np.float64)
# get_time_step(txt , dtexture_dt)
# display_texture(txt, vmins=(0,0,0),vmaxs=(1,1,1))
# display_texture(dtexture_dt, vmins=(0,0,0),vmaxs=(1,1,1))