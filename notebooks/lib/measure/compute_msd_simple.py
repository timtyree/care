import numpy as np
def autocorrFFT(x):
    N=len(x)
    F = np.fft.fft(x, n=2*N)  #2*N because of zero-padding
    PSD = F * F.conjugate()
    res = np.fft.ifft(PSD)
    res= (res[:N]).real   #now we have the autocorrelation in convention B
    n=N*np.ones(N)-np.arange(0,N) #divide res(m) by (N-m)
    return res/n #this is the autocorrelation in convention A

def msd_fft(r):
    N=len(r)
    D=np.square(r).sum(axis=1)
    D=np.append(D,0)
    S2=sum([autocorrFFT(r[:, i]) for i in range(r.shape[1])])
    Q=2*D.sum()
    S1=np.zeros(N)
    for m in range(N):
        Q=Q-D[m-1]-D[N-m]
        S1[m]=Q/(N-m)
    return S1-2*S2

def msd_straight_forward(r):
    shifts = np.arange(len(r))
    msds = np.zeros(shifts.size)

    for i, shift in enumerate(shifts):
        diffs = r[:-shift if shift else None] - r[shift:]
        sqdist = np.square(diffs).sum(axis=1)
        msds[i] = sqdist.mean()

    return msds

if __name__=='__main__':
    #test that msd_fft results in msd_straight_forward to machine precision for an unseeded random walk
    N=1000
    r = np.cumsum(np.random.choice([-1., 0., 1.], size=(N, 3)), axis=0)
    assert ( np.isclose(msd_fft(r)-msd_straight_forward(r),0.).all() )
    numerical_tolerance=np.mean(msd_fft(r)-msd_straight_forward(r))
    print(f"the numerical aggrement between slower and faster methods was numerical_tolerance = {numerical_tolerance}.")
