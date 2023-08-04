'''
    Last modified: 2023.08.02
    Todo: 
        Add a Pytorch implementation
        Add UV texture mapping
        Save joint data
        Harmonize data structure specification
    pokerlishao@gmail.com
'''
import pickle
import os
import numpy as np
import scipy
import pyassimp
import utils.chumpy as chumpy
import torch
import torch.nn as nn



class SMPL_Loader(pickle.Unpickler, nn.Module):
    '''
    For SMPL v1.1.0
        betas           300 (control body shape)
        pose            72 (control body pose)
        shapedirs       6890 * 3 * 300
        posedirs        6890 * 3 * 207
        v_posed         6890 * 3
        J               24 * 3
        kintree_table   2 * 24 (parent's index and index)
        bs_style        lbs
        weights         6890 * 24
    '''
    def __init__(self,model_path, device=None):
        nn.Module.__init__(self)
        self.model_path = model_path
        self.device = device if device is not None else torch.device('cpu')
        self.load_model()

    def load_model(self):
        with open (self.model_path,'rb') as f:
            super().__init__(f,encoding='latin1')
            self.data = self.load() # pickle.load()

        self.backwards_compatibility_replacements()
        self.trans_shape = [3]
        self.pose_shape = self.data['kintree_table'].shape[1]*3     # 24 * 3
        self.beta_shape = self.data['shapedirs'].shape[-1]          # 300 in v1.1.0 and 10 in v1.0.0
        self.parent = {
            child_id: parent_id
            for child_id, parent_id in zip(self.data['kintree_table'][1],self.data['kintree_table'][0])
        }
        self.set_params()
        self.trans2torch()


    def set_params(self, trans=None, pose=None, betas=None):
        if trans is None:
            trans=torch.zeros(3, dtype=torch.float32)
        if pose is None:
            pose=torch.zeros(self.pose_shape, dtype=torch.float32)
        if betas is None:
            betas=torch.zeros(self.beta_shape, dtype=torch.float32)
        self.data['trans'] = trans
        self.data['pose'] =  pose
        self.data['betas'] = betas


    def cal_shape(self):
        # cal body shapes
        self.data['v_shaped'] = self.data['shapedirs'] @ self.data['betas'] + self.data['v_template'] 
        # cal joint location
        self.J = self.data['J_regressor'] @ self.data['v_shaped']
        # cal rotation matrix for each joint by rodigues
        self.R = self.rodrigues(self.data['pose'].reshape((-1, 1, 3)))
        
        R_cube = self.R[1:]
        I_cube = (torch.eye(3, dtype=torch.float32).unsqueeze(dim=0) + torch.zeros((R_cube.shape[0], 3, 3), dtype=torch.float32))
        lrotmin = (R_cube - I_cube).view(-1)
        # how pose affect body shape in zero pose
        v_posed = self.data['v_shaped'] + self.data['posedirs'] @ lrotmin

        self.data['v'] = []

        # root joint
        self.data['v'].append(torch.cat((torch.cat((self.R[0], torch.reshape(self.J[0, :], (3, 1))), dim=1), 
                                         torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)), dim=0))
        # child joint
        for i in range(1, self.data['kintree_table'].shape[1]):
            self.data['v'].append(
                    self.data['v'][self.parent[i]] @ 
                    torch.cat((torch.cat((self.R[i], torch.reshape(self.J[i, :] - self.J[self.parent[i], :], (3, 1))), dim = 1), 
                                         torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)), dim=0)
                )
                
        stacked = torch.stack(self.data['v'], dim=0)
        tt = stacked @ torch.cat((self.J, torch.zeros((24, 1), dtype=torch.float32)), dim=1).reshape(24, 4, 1)
        self.data['v'] = stacked - torch.cat((torch.zeros((tt.shape[0], 4, 3), dtype=torch.float32), tt), dim=2)

        T = torch.tensordot(self.data['weights'], self.data['v'], dims=[[1], [0]])
        rest_shape_h = torch.cat((v_posed, torch.ones(([v_posed.shape[0], 1]), dtype=torch.float32)), dim = 1)
        v = (T @  rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
        self.data['v'] = v + self.data['trans'].reshape([1, 3])


    # Save to an .obj file       
    def save_obj(self,fname = './test_smpl.obj'):
        with open( fname, 'w') as fp:
            for v in self.data['v']:
                fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )
            # Faces are 1-based, not 0-based in obj files
            for f in self.data['f']+1:
                fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )
        # todo: save UV texture
        print('Save to ', fname)


    # Save to an fbx file
    def seve_fbx(self,fname = './test_smpl.fbx'):
        # transfrom obj to fbx
        self.save_obj('./__temp__.obj')
        with pyassimp.load('./__temp__.obj') as scene:
            pyassimp.export(scene, fname, 'fbx')
        os.remove('./__temp__.obj')
        print('remove ./__temp__.obj')
        print('Save to ', fname)

    # save self.J
    def save_joint(self):
        pass
      
### helper functions
    def rodrigues(self, r):        
        theta = torch.norm(r + torch.randn_like(r) * 1e-8, dim=(1, 2), keepdim=True)
        r_hat = r / theta
        cos = torch.cos(theta)
        z_stick = torch.zeros(theta.shape[0], dtype=torch.float32)
        m = torch.dstack([
            z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
            r_hat[:, 0, 2], z_stick, - r_hat[:, 0, 0],
            -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
        ).reshape([-1, 3, 3])
        i_cube = (torch.eye(3, dtype=torch.float32).unsqueeze(dim=0) \
             + torch.zeros((theta.shape[0], 3, 3), dtype=torch.float32))
        A = r_hat.permute(0, 2, 1)
        dot = A @ r_hat
        R = cos * i_cube + (1 - cos) * dot + torch.sin(theta) * m
        return R


    # help pickle to load pkl file
    def find_class(self, module, name):
        if module == 'chumpy.ch':   # fixed chumpy in local
            return getattr(chumpy.ch, name)
        if module == 'scipy.sparse.csc':    # the `scipy.sparse.csc` namespace is deprecated
            return getattr(scipy.sparse, name) 
        return super().find_class(module, name)
        
    def trans2torch(self):
        for s in ['v_template', 'weights', 'posedirs' ,'shapedirs', 'J']:
            self.data[s] = torch.from_numpy(np.array(self.data[s])).type(torch.float32).to(self.device)
        self.data['J_regressor'] = torch.from_numpy(self.data['J_regressor'].todense()).type(torch.float32).to(self.device)
        self.data['f'] = torch.from_numpy(np.array(self.data['f']).astype(np.int32)).type(torch.float32).to(self.device)
        
    def backwards_compatibility_replacements(self):
        dd = self.data
        # replacements
        if 'default_v' in dd:
            dd['v_template'] = dd['default_v']
            del dd['default_v']
        if 'template_v' in dd:
            dd['v_template'] = dd['template_v']
            del dd['template_v']
        if 'joint_regressor' in dd:
            dd['J_regressor'] = dd['joint_regressor']
            del dd['joint_regressor']
        if 'blendshapes' in dd:
            dd['posedirs'] = dd['blendshapes']
            del dd['blendshapes']
        if 'J' not in dd:
            dd['J'] = dd['joints']
            del dd['joints']

        # defaults
        if 'bs_style' not in dd:
            dd['bs_style'] = 'lbs'
    
    # move to GPU
    def to_device(self, device):
        for name, attr in self.__dict__.items():
            if isinstance(attr, torch.Tensor):
                setattr(self, name, attr.to(device))

def speed_test(smpl,device):
    # about 0.66ms per generation in CPU
    # about 0.89ms per generation in GPU
    import time
    T1 = time.perf_counter()
    for i in range(1000):
        # random pose and shape
        trans = torch.zeros(3, dtype=torch.float32).to(device)
        pose = (torch.rand(smpl.pose_shape, dtype=torch.float32).to(device) - 0.5)
        betas = (torch.rand(smpl.beta_shape, dtype=torch.float32).to(device) - 0.5) * 0.6
        smpl.set_params(trans,pose,betas)
        smpl.cal_shape()
    T2 = time.perf_counter()
    print('Time :%sms' % ((T2 - T1)*1000))  



def main():
    # f_path = 'path/to/smpl_pkl_file'
    f_path = '/home/poker/dataset/SMPL/SMPL_python_v.1.1.0/smpl/models/basicmodel_f_lbs_10_207_0_v1.1.0.pkl'
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.cuda.set_device(0)
    # device = torch.device("cpu")


    # 设置默认的张量类型为CUDA张量

    smpl = SMPL_Loader(f_path, device)
    smpl.to_device(device)
    speed_test(smpl,device)
    
    # random pose and shape
    # trans = torch.zeros(3, dtype=torch.float32)
    # pose = (torch.rand(smpl.pose_shape, dtype=torch.float32) - 0.5)
    # betas = (torch.rand(smpl.beta_shape, dtype=torch.float32) - 0.5) * 0.6
    # smpl.set_params(trans,pose,betas)
    # smpl.cal_shape()

    # smpl.save_obj('0.obj')
    # smpl.seve_fbx('0.fbx')
    
    # smpl.print_data()

if __name__ == "__main__":
    main()