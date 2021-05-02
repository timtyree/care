import numpy as np
from ..utils.dist_func import get_distance_L2_pbc

class ParticlePBC(object):
    r"""Point-like particle that is to be distinguished over a series of observations.
    Enforces periodic boundary conditions"""
    def __init__(self, pid, width, height, **kwargs):
        '''
        - ParticlePBC
            - ParticlePBC(pid) essentially extends a dict_of_lists
                - lesser_arclen_lst
                - greater_arclen_lst
                - lesser_pid_lst
                - greater_pid_lst
                - x_lst
                - y_lst
                - t_lst
                - (optional) state_lst
                - Methods
                    - dist_to(self,point)

        '''
        self.pid = pid
        self.x_lst=[]
        self.y_lst=[]
        self.t_lst=[]
        self.lesser_pid_lst=[]
        self.lesser_arclen_lst=[]
        self.greater_pid_lst=[]
        self.greater_arclen_lst=[]
        self.distance_L2_pbc=get_distance_L2_pbc(width=width,height=height)

    def dist_to(self,point):
        last_point=np.array((self.x_lst[-1],self.y_lst[-1]))
        dist=self.distance_L2_pbc(point,last_point)
        return dist

    def update(self,x,y,t,lesser_pid,lesser_arclen,greater_pid,greater_arclen,**kwargs):
        self.x_lst.append(x)
        self.y_lst.append(y)
        self.t_lst.append(t)
        self.lesser_pid_lst.append(lesser_pid)
        self.lesser_arclen_lst.append(lesser_arclen)
        self.greater_pid_lst.append(greater_pid)
        self.greater_arclen_lst.append(greater_arclen)
        return self

class ParticlePBCDict(dict):
    def __init__(self, dict_tips, width=200.,height=200., **kwargs):
        '''initiate a ParticlePBC for each tip found, incrementing pid_nxt
        supposes dict_tips is a dictionary of lists of values.'''
        x_lst=dict_tips['x']
        y_lst=dict_tips['y']
        t=dict_tips['t']

        lesser_pid_lst=dict_tips['lesser_pid']
        lesser_arclen_lst=dict_tips['lesser_arclen']
        greater_pid_lst=dict_tips['greater_pid']
        greater_arclen_lst=dict_tips['greater_arclen']

        ntips=len(dict_tips['x'])
        for j in range(ntips):
            pid_nxt=j
            particle = ParticlePBC(pid_nxt,width,height)
            particle.update(x=x_lst[j],y=y_lst[j],t=t,
                            lesser_pid=lesser_pid_lst[j],
                            lesser_arclen=lesser_arclen_lst[j],
                            greater_pid=greater_pid_lst[j],
                            greater_arclen=greater_arclen_lst[j])
            self[pid_nxt]=particle
        self.pid_nxt=pid_nxt+1

    def find_closest_particle(self,point):
        #TODO(heretim)
        mindist=9e9
        pid_closest=-9999
        #for each Particle
        pid_lst=list(self.keys())
        for pid in pid_lst:
            #get the distance to point
            dist=self[pid].dist_to(point)
            if dist<mindist:
                mindist=dist
                pid_closest=pid
        #compute tmax
        tmax=max(t_lst)
        return pid_closest

    def get_dead_particles(self):
        #for each Particle
        pid_lst=list(self.keys())
        t_lst=[]
        for pid in pid_lst:
            #get the most recent time
            t_lst.append(self[pid].t_lst[-1])
        #compute tmax
        tmax=max(t_lst)
        #find all particles whose time is not the tmax
        dead_pid_lst=[]
        for pid in pid_lst:
            if t_lst[pid]<tmax:
                dead_pid_lst.append(pid)
        return dead_pid_lst

    def get_alive_particles(self):
        #for each Particle
        pid_lst=list(self.keys())
        t_lst=[]
        for pid in pid_lst:
            #get the most recent time
            t_lst.append(self[pid].t_lst[-1])
        #compute tmax
        tmax=max(t_lst)
        #find all particles whose time is not the tmax
        alive_pid_lst=[]
        for pid in pid_lst:
            if t_lst[pid]==tmax:
                alive_pid_lst.append(pid)
        return alive_pid_lst

    def add_particle(self,x,y,t,lesser_pid,lesser_arclen,greater_pid,greater_arclen,**kwargs):
        pid_nxt=self.pid_nxt
        particle = ParticlePBC(pid_nxt,width,height)
        particle.update(x,y,t,
                        lesser_pid,
                        lesser_arclen,
                        greater_pid,
                        greater_arclen)
        self[pid_nxt]+=1
        #TODO: update lesser,greater_pid with those of self

        return self

    def sort_particles_indices(self, dict_topo):
        '''
        Example Usage:
        in_to_out=sort_particles_indices(self, dict_topo)
        '''
        pid_out_lst=self.get_alive_particles()
        pid_in_lst =dict_topo['pid']
        x_lst=dict_topo['x'];y_lst=dict_topo['y'];
        # make dict that maps dict_topo['pid'] to nearest list(self.keys()) that has tmax as its time
        in_to_out={}
        #for each point[pid_in]
        for pid_in in pid_in_lst:
            point=np.array((x_lst[pid_in],y_lst[pid_in]))
            mindist=9e9
            pid_out_closest=-9999
            #find the closest position[pid_out]
            for pid_out in pid_out_lst:
                dist=self[pid_out].dist_to(point)
                if dist<mindist:
                    pid_out_closest=pid_out
                    mindist=dist

            #and update in_to_out with that pid_out
            in_to_out.update({pid_in:pid_out_closest})
        return in_to_out

    def update_set(self,dict_topo,in_to_out):
        '''append/update self with dict_topo'''
        #extract values from dict_topo
        x_lst =dict_topo['x']
        y_lst =dict_topo['y']
        t     =dict_topo['t']
        greater_pid_lst =dict_topo['greater_pid']
        lesser_pid_lst =dict_topo['lesser_pid']
        greater_arclen_lst =dict_topo['greater_arclen']
        lesser_arclen_lst =dict_topo['lesser_arclen']
        greater_arclen_values_lst =dict_topo['greater_arclen_values']
        lesser_arclen_values_lst =dict_topo['lesser_arclen_values']
        pid_in_lst =dict_topo['pid']

        # append/update self with dict_topo
        for pid_in in pid_in_lst:
            pid_out=in_to_out[pid_in]

            self[pid_out].update(
                x=x_lst[pid_in],
                y=y_lst[pid_in],
                t=t,
                lesser_pid=lesser_pid_lst[pid_in],
                lesser_arclen=lesser_arclen_lst[pid_in],
                greater_pid=greater_pid_lst[pid_in],
                greater_arclen=greater_arclen_lst[pid_in]
            )#,**kwargs)
        return self

    def merge(self,dict_topo):
        ntips=len(dict_topo['x'])
        #merge dict_topo with self, creating new particles for non-matches
        if ntips>0:
            # map apparent index to closest pid in pdict
            in_to_out=self.sort_particles_indices(dict_topo)

            # append/update self with dict_topo
            self.update_set(dict_topo,in_to_out)

            #identify any new spiral tips
            pid_in_set=set(range(ntips))
            pid_matched_set=set(in_to_out.keys())
            pid_born_lst=list(pid_in_set.difference(pid_matched_set))
            for pid_born in pid_born_lst:
                #create a new particle in pdict
                self.add_particle(
                    x=dict_topo['x'][pid_born],
                    y=dict_topo['y'][pid_born],
                    t=t,
                    lesser_pid=dict_topo['lesser_pid'][pid_born],
                    lesser_arclen=dict_topo['lesser_arclen'][pid_born],
                    greater_pid=dict_topo['greater_pid'][pid_born],
                    greater_arclen=dict_topo['greater_arclen'][pid_born]
                )#,**kwargs,)
        return self
