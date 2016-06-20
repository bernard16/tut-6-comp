import numpy
from matplotlib import pylab as plt
def simulate_lorentzian(t,a=1.0,b=0.25,amp=1,cent=0):
    dat=(a/(b+(t-cent)**2))*amp
    dat+=numpy.random.randn(t.size)
    return dat
    
def get_trial_offset(a):
    return a*numpy.random.randn(a.size)

class Lorentzian:
    def __init__(self,t,a=1.1,b=0.25,amp=1.0,cent=0,offset=0):
        self.t=t
        self.y=simulate_lorentzian(t,a,b,amp,cent)+offset
        self.err=numpy.ones(t.size)
        self.a=a
        self.b=b
        self.amp=amp
        self.cent=cent
        self.offset=offset

    def get_chisq(self,vec):
        a=vec[0]
        b=vec[0]
        amp=vec[1]
        cent=vec[2]
        off=vec[3]

        pred=off+amp*(self.a/(self.b + (self.t-cent)**2))
        chisq=numpy.sum((self.y-pred)**2/self.err**2)
        return chisq

def run_mcmc(data,start_pos,nstep,scale=None):
    nparam=start_pos.size
    params=numpy.zeros([nstep,nparam+1])
    params[0,0:-1]=start_pos
    cur_chisq=data.get_chisq(start_pos)
    cur_pos=start_pos.copy()
    if scale==None:
        scale=numpy.ones(nparam)
    for i in range(1,nstep):
        new_pos=cur_pos+get_trial_offset(scale)
        new_chisq=data.get_chisq(new_pos)
        if new_chisq<cur_chisq:
            accept=True
        else:
            delt=new_chisq-cur_chisq
            prob=numpy.exp(-0.5*delt)
            if numpy.random.rand()<prob:
                accept=True
            else:
                accept=False
        if accept: 
            cur_pos=new_pos
            cur_chisq=new_chisq
        params[i,0:-1]=cur_pos
        params[i,-1]=cur_chisq
    return params


if __name__=='__main__':
    
    #get a realization of a gaussian, with noise added
    t=numpy.arange(-5,5,0.01)
    dat=Lorentzian(t,amp=2.5)

    #pick a random starting position, and guess some errors
    guess=numpy.array([0.3,1.2,0.3,-0.2])
    scale=numpy.array([0.1,0.1,0.1,0.1])
    nstep=100000
    chain=run_mcmc(dat,guess,nstep,scale)
    nn=numpy.round(0.2*nstep)
    chain=chain[nn:,:]
    
    #pull true values out, compare to what we got
    param_true=numpy.array([dat.a,dat.b,dat.amp,dat.cent,dat.offset])
    for i in range(0,param_true.size):
        val=numpy.mean(chain[:,i])
        scat=numpy.std(chain[:,i])
        print [param_true[i],val,scat]


    
