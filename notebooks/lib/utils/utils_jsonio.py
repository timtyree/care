
import json
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

def save_dict_to_json(pdict,save_fn):
    #save all output values as json
    os.system('touch '+save_fn)
    with open(save_fn,"w") as fp:
        json.dump(dict(pdict),fp,cls=NpEncoder,indent=1,sort_keys=True)

def write_parameters_to_json(param_dict, param_file_name):
    '''Example Usage:
    write_parameters_to_json(kwargs, 'lib/param_set_8.json')
    '''
    with open(param_file_name, 'w') as json_file:
        json.dump(param_dict, json_file)
    return True

def read_parameters_from_json(param_file_name):
    '''Example Usage:
    kwargs = read_parameters_from_json('lib/param_set_8.json')
    '''
    with open(param_file_name) as json_file:
        data = json.load(json_file)
        return data

if __name__=='__main__':
  write_json = False
  # define parameter set 8 from a dict named kwargs
  kwargs = {
      'DX'       : 0.025, #cm/pxl
      'diffCoef' : 0.0005, # cm^2 / ms
      'C_m'      : 1.000,  # microFarad/cm^2 (although u channel is adimensionalized voltage.  See Fenton & Karma, (1998))

      #parameter set 8 of FK model from Fenton & Cherry (2002)
      'tau_pv'   : 13.03,
      'tau_v1'   : 19.6,
      'tau_v2'   : 1250,
      'tau_pw'   : 800,
      'tau_mw'   : 40,
      'tau_d'    : 0.45,
      'tau_0'    : 12.5,
      'tau_r'    : 33.25,
      'tau_si'   : 29,
      'K'        : 10,
      'V_sic'    : 0.85,
      'V_c'      : 0.13,
      'V_v'      : 0.04
      }

  #define an dummy example scalar field of initial conditions
  width = 200; height = 200;
  Vin  = np.array([256*x*(y+1) for x in range(width) for y in range(height)]).reshape((width,height))
  field = Vin.copy()
  height,width = field.shape
  kwargs.update({'width':width,'height':height})
  print(kwargs)

  if write_json:
      #input/output of key word arguments for model parameters
      param_file_name = 'lib/param_set_8.json'
      write_parameters_to_json(kwargs, param_file_name)
      # kwargs = read_parameters_from_json('lib/param_set_8.json')
      # print(kwargs)
