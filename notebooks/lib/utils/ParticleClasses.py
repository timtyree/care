import numpy as np, pandas as pd, json
from ..utils.dist_func import get_distance_L2_pbc

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

class ParticlePBC(dict):
    r"""Point-like particle that is to be distinguished over a series of observations.
    Enforces periodic boundary conditions"""
    def __init__(self, pid, width, height, **kwargs):
        '''
        - ParticlePBC
            - ParticlePBC(pid) essentially extends a dict_of_lists
                - lesser_arclen
                - greater_arclen
                - lesser_pid
                - greater_pid
                - x
                - y
                - t
                - (optional) state
                - Methods
                    - dist_to(self,point)

        '''
        self['x']=[]
        self['y']=[]
        self['t']=[]
        self['lesser_pid']=[]
        self['lesser_arclen']=[]
        self['greater_pid']=[]
        self['greater_arclen']=[]
        self.pid = pid
        self.width=width;self.height=height
        self.distance_L2_pbc=get_distance_L2_pbc(width=self.width,height=self.height)

    def __repr__(self):
        x=self['x'][-1]
        y=self['y'][-1]
        t=self['t'][-1]
        size=len(self['t'])
        return f"(size,t,x,y)=({size:d},{t:.03f},{x:.03f},{y:.03f});"

    def dist_to(self,point):
        last_point=np.array((self['x'][-1],self['y'][-1]))
        dist=self.distance_L2_pbc(point,last_point)
        return dist

    def update(self,x,y,t,lesser_pid,lesser_arclen,greater_pid,greater_arclen,**kwargs):
        self['x'].append(x)
        self['y'].append(y)
        self['t'].append(t)
        self['lesser_pid'].append(lesser_pid)
        self['lesser_arclen'].append(lesser_arclen)
        self['greater_pid'].append(greater_pid)
        self['greater_arclen'].append(greater_arclen)
        return self


    def scale_coordinates(self,scale):
        self['x']=[scale*x for x in self['x']]
        self['y']=[scale*x for x in self['y']]
        self['lesser_arclen']=[scale*x for x in self['lesser_arclen']]
        self['greater_arclen']=[scale*x for x in self['greater_arclen']]
        return self

    def zoom_to_double(self):
        '''
        Example Usage: Zoom to 2**5
        self.zoom_to_double().zoom_to_double().zoom_to_double().zoom_to_double().zoom_to_double()
        '''
        self.width=2*self.width;self.height=2*self.height
        self.distance_L2_pbc=get_distance_L2_pbc(width=self.width,height=self.height)
        # def dist_to(self,point):
        #     last_point=np.array((self.x_lst[-1],self.y_lst[-1]))
        #     dist=self.distance_L2_pbc(point,last_point)
        #     return dist
        return self.scale_coordinates(scale=2.)

    def update_dict_lst(self,kwargs,pid):
        '''
        Example Usage:
        particle.update_dict_lst(dict_tips,2)
        '''
        fields=set(self.keys())
        for key in list(kwargs.keys()):
            if fields.isdisjoint({key}):
                #create new field list
                self[key]=[]
            # append the value
            try:
                self[key].append(kwargs[key][pid])
            except TypeError as e:
                self[key].append(kwargs[key])
        return self

    def to_pandas(self):
        primitive = (int, str, bool, float)
        def is_primitive(thing):
            return isinstance(thing, primitive)

        ds=dict(self)
        dself={}
        #if values are a list of primitives,
        for key in list(ds.keys()):
            values=ds[key]
            if (type(values) is type(list())):#&(is_primitive(values[0])):
                dself[key]=values
        df=pd.DataFrame(dself)
        df['pid']=self.pid
        cols = df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        return df[cols]

    def separate_data_to_dicts(self):
        '''separates contour data from particle data for one particle.
        does not change particle.
        Example Usage:
        dict_particle_out,dict_greater,dict_lesser=separate_data_to_dicts(particle)
        '''
        pid=self.pid
        particle=self
        dict_particle=dict(particle)
        #remove any numpy array objects _values
        keys=set(self.keys())
        keys_lesser_contour_only={
            'lesser_curvature_values',
            'lesser_V_values',
            'lesser_xy_values',
            'lesser_arclen_values'
        }
        keys_greater_contour_only={
            'greater_curvature_values',
            'greater_V_values',
            'greater_xy_values',
            'greater_arclen_values'
        }
        keys_lesser_common={
            't','pid','lesser_pid'
        }
        keys_greater_common={
            't','pid','greater_pid'
        }
        keys_particle=keys.difference(keys_lesser_contour_only).difference(keys_greater_contour_only)
        dict_lesser={}
        try:
            for key in sorted(keys_lesser_contour_only):
                value=dict_particle.pop(key)
                dict_lesser[key]=value
            for key in sorted(keys_lesser_common):
                dict_lesser[key]=dict_particle[key]
            dict_greater={}
            for key in sorted(keys_greater_contour_only):
                value=dict_particle.pop(key)
                dict_greater[key]=value
            for key in sorted(keys_greater_common):
                dict_greater[key]=dict_particle[key]
        except KeyError as e:
            dict_lesser=None
            dict_greater=None

        #TODO(later): update the __init__ method of the ParticlePBCSet class so it inlcudes all keys in dict_topo
        #handle values with a missing first entry
        key_lst_missing_first_entry=['s1','s2','pid','greater_mean_V','lesser_mean_V','greater_mean_curvature','lesser_mean_curvature']
        minlen=9e9
        for key in dict_particle.keys():
            l=len(dict_particle[key])
            if l<minlen:
                minlen=l
        #     print(f"{key}:{l}")
        dict_particle_out={}
        for key in dict_particle.keys():
            v_lst=dict_particle[key]
            l=len(v_lst)
            if l==minlen:
                dict_particle_out[key]=v_lst
            elif l==minlen+1:
                dict_particle_out[key]=v_lst[1:]
        return dict_particle_out,dict_greater,dict_lesser

class ParticlePBCDict(dict):
    def __init__(self, dict_tips, width,height, **kwargs):
        '''a dict with integer keys valued with ParticlePBC instances.
        ParticlePBCDict also has fields:
            - pid_nxt
            - width
            - height
        ParticlePBCDict also has supporting methods.

        To initiate a ParticlePBC for each tip found, ParticlePBCDict increments pid_nxt,
        supposes dict_tips is a dictionary of lists of values.
        #TODO(later): dev classmethod, merge_with_other, which merges pdict1 and pdict2 into pdict_out
        #TODO(later): test-based dev of mapping to/from pandas.DataFrame
        '''
        x_lst=dict_tips['x']
        y_lst=dict_tips['y']
        t=dict_tips['t']

        lesser_pid_lst=dict_tips['lesser_pid']
        lesser_arclen_lst=dict_tips['lesser_arclen']
        greater_pid_lst=dict_tips['greater_pid']
        greater_arclen_lst=dict_tips['greater_arclen']
        pid_nxt=-1
        ntips=len(dict_tips['x'])
        for j in range(ntips):
            pid_nxt=j
            lesser_pid=lesser_pid_lst[j]
            greater_pid=greater_pid_lst[j]
            particle = ParticlePBC(pid_nxt,width,height)
            particle.update(x=x_lst[j],y=y_lst[j],t=t,
                            lesser_pid=lesser_pid,
                            lesser_arclen=lesser_arclen_lst[j],
                            greater_pid=greater_pid,
                            greater_arclen=greater_arclen_lst[j])
            self[pid_nxt]=particle
        self.pid_nxt=pid_nxt+1
        self.width=width
        self.height=height

    def find_closest_particle(self,point):
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
        return pid_closest

    def get_dead_particles(self):
        #for each Particle
        pid_lst=list(self.keys())
        t_lst=[]
        for pid in pid_lst:
            #get the most recent time
            t_lst.append(self[pid]['t'][-1])
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
            t_lst.append(self[pid]['t'][-1])
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
        particle = ParticlePBC(pid_nxt,self.width,self.height)
        particle.update(x,y,t,
                        lesser_pid,
                        lesser_arclen,
                        greater_pid,
                        greater_arclen)
        self[pid_nxt]=particle
        self.pid_nxt+=1

        #TODO: update lesser,greater_pid with those of self
        return self

    def init_particle(self,**kwargs):
        pid_nxt=self.pid_nxt
        particle = ParticlePBC(pid_nxt,self.width,self.height)
        self[pid_nxt]=particle
        self.pid_nxt+=1
        return self

    def sort_particles_indices(self, dict_tips, search_range=40.):
        '''
        Example Usage:
        in_to_out=sort_particles_indices(self, dict_tips)
        '''
        pid_out_lst=self.get_alive_particles()
        pid_in_lst =dict_tips['pid']
        x_lst=dict_tips['x'];y_lst=dict_tips['y'];
        # make dict that maps dict_tips['pid'] to nearest list(self.keys()) that has tmax as its time
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
            if mindist<search_range:
                pid_out_lst.remove(pid_out_closest)
                #and update in_to_out with that pid_out
                in_to_out.update({pid_in:pid_out_closest})
        return in_to_out

    def update_set(self,dict_tips,in_to_out):
        '''append/update self with dict_tips'''
        #extract values from dict_tips
        x_lst =dict_tips['x']
        y_lst =dict_tips['y']
        t     =dict_tips['t']
        greater_pid_lst =dict_tips['greater_pid']
        lesser_pid_lst =dict_tips['lesser_pid']
        greater_arclen_lst =dict_tips['greater_arclen']
        lesser_arclen_lst =dict_tips['lesser_arclen']
        greater_arclen_values_lst =dict_tips['greater_arclen_values']
        lesser_arclen_values_lst =dict_tips['lesser_arclen_values']
        pid_in_lst =list(in_to_out.keys())#dict_tips['pid']

        # append/update self with dict_tips that matched
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

    def merge(self,dict_tips):
        ntips=len(dict_tips['x'])
        #merge dict_tips with self, creating new particles for non-matches
        if ntips>0:
            # map apparent index to closest pid in pdict
            in_to_out=self.sort_particles_indices(dict_tips)

            # append/update self with dict_tips
            self.update_set(dict_tips,in_to_out)

            #identify any new spiral tips
            pid_in_set=set(range(ntips))
            pid_matched_set=set(in_to_out.keys())
            pid_born_lst=list(pid_in_set.difference(pid_matched_set))
            for pid_born in pid_born_lst:
                #create a new particle in pdict
                self.add_particle(
                    x=dict_tips['x'][pid_born],
                    y=dict_tips['y'][pid_born],
                    t=dict_tips['t'],
                    lesser_pid=dict_tips['lesser_pid'][pid_born],
                    lesser_arclen=dict_tips['lesser_arclen'][pid_born],
                    greater_pid=dict_tips['greater_pid'][pid_born],
                    greater_arclen=dict_tips['greater_arclen'][pid_born]
                )#,**kwargs,)
        return self

    def merge_dict(self,dict_tips):
        '''merges all values in the dict of lists, dict_tips'''
        ntips=len(dict_tips['x'])
        #merge dict_tips with self, creating new particles for non-matches
        if ntips>0:
            # map apparent index to closest pid in pdict
            in_to_out=self.sort_particles_indices(dict_tips)

            #identify any new spiral tips
            pid_in_set=set(range(ntips))
            pid_matched_set=set(in_to_out.keys())
            pid_born_lst=list(pid_in_set.difference(pid_matched_set))

            # append/update self with dict_tips with all matches
            for pid_in in list(pid_matched_set):
                self[in_to_out[pid_in]].update_dict_lst(dict_tips,pid_in)

            for pid_born in pid_born_lst:
                #create a new particle in pdict
                pid_nxt=self.pid_nxt
                self.init_particle()[pid_nxt].update_dict_lst(dict_tips,pid_born)


        return self

    def zoom_to_double(self):
        '''
        Example Usage: Zoom to 2**5
        self.zoom_to_double().zoom_to_double().zoom_to_double().zoom_to_double().zoom_to_double()
        '''
        self.width=2*self.width
        self.height=2*self.height
        pid_lst=list(self.keys())
        for pid in pid_lst:
            self[pid].zoom_to_double()
        return self

    def get_current_locations(self):
        '''
        Example Usage:
        x_values,y_values=pdict.get_current_locations()
        '''
        xl=[];yl=[];
        pl=self.get_alive_particles()
        for pid in pl:
            xl.append(self[pid]['x'][-1])
            yl.append(self[pid]['y'][-1])
        x_values=np.array(xl);y_values=np.array(yl);
        pid_values=np.array(pl)
        return x_values,y_values,pid_values

    def to_pandas(self):
        pid_lst=list(self.keys())
        df=pd.concat([self[i].to_pandas()] for i in pid_lst)
        return df

    def to_csv(self,save_fn):
        df=self.to_pandas()
        df.to_csv(save_fn,index=False)

    def read_csv(self,input_fn):
        raise Exception(f"Not yet implemented!")

    def record_tips_return_txt(self,txt,duration,one_step,comp_tips,dt,save_every_n_frames=1,**kwargs):
        '''
        Example Usage:
        txt=pdict.record_tips_return_txt(txt,duration,one_step,comp_tips,dt,save_every_n_frames=1)
        '''
        inVc,outVc,inmhjdfx,outmhjdfx,dVcdt=unstack_txt(txt)
        change_time=0.
        frameno=0
        while change_time<=duration:
            frameno+=1
            change_time+=dt
            one_step(inVc,outVc,inmhjdfx,outmhjdfx,dVcdt)
            if frameno % save_every_n_frames == 0:
                dict_tips=comp_tips(txt=txt)
                self.merge_dict(dict_tips)

        txt=stack_txt(inVc,outVc,inmhjdfx,outmhjdfx,dVcdt)
        return txt

    def find_starting_point(self,pid_pair):
        '''find_starting_point returns xy position of the first particle in pid_pair.
        suppose the pid_pair are birth partners.
        follow pdict back to the birth of this pid_pair.
        '''
        particle=self[pid_pair[0]]
        xy0=np.array((particle['x'][0],particle['y'][0]))
        return xy0

    def separate_data_to_pandas(self):
        '''Example Usage:
        df, dict_greater_dict, dict_lesser_dict=separate_data_to_pandas(pdict)
        '''
        df_lst=[]
        dict_greater_dict={}
        dict_lesser_dict={}
        for pid in sorted(self.keys()):
            particle=self[pid]
            dict_particle_out,dict_greater,dict_lesser=particle.separate_data_to_dicts()
            dict_greater_dict[pid]=dict_greater
            dict_lesser_dict[pid]=dict_lesser
            df_lst.append(pd.DataFrame(dict_particle_out))
        df=pd.concat(df_lst)
        return df, dict_greater_dict, dict_lesser_dict

    def to_csv_and_json(self,modname):
        '''Example Usage:
        modname=f"{nb_dir}/Data/test_data/recursive_death_test"
        to_csv_and_json(pdict,modname)
        '''
        df, dict_greater_dict, dict_lesser_dict=self.separate_data_to_pandas()
        #save all particles in one csv
        save_dir=modname+"_particles_only.csv"
        df.to_csv(save_dir,index=False)

        #save greater/lesser contours as one json, indexed by pid
        save_fn=modname+f"_greater_contours.json"
        # os.system('touch '+save_fn)
        with open(save_fn,"w") as fp:
            json.dump(dict_greater_dict,fp,cls=NpEncoder,indent=0,sort_keys=True)

        save_fn=modname+f"_lesser_contours.json"
        with open(save_fn,"w") as fp:
            json.dump(dict_lesser_dict,fp,cls=NpEncoder,indent=0,sort_keys=True)
        return self

    def get_field_values(self,field,index=0):
        pid_lst=self.get_alive_particles()
        last_particle=self[pid_lst[index]]
        values=np.array(last_particle[field])
        return values

    def get_sigma_max_values(self,ds=5.,field='lesser_arclen',pid=None):
        if pid is None:
            pid_lst=pdict.get_alive_particles()
            last_particle=self[pid_lst[0]]
        else:
            last_particle=self[pid]
        scale=ds/last_particle.width #cm/pixel
        sigma_max_values=scale*np.array(last_particle[field])
        return sigma_max_values

    def get_t_values(self,ds=5.,field='t', index=0,scale=1.):
        pid_lst=self.get_alive_particles()
        last_particle=self[pid_lst[index]]
        # scale=1.#ms per ms
        t_values=scale*np.array(last_particle[field])
        return t_values
