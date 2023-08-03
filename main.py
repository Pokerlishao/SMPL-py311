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



class SMPL_Loader(pickle.Unpickler):
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
    def __init__(self,model_path):
        self.model_path = model_path
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
        self.trans2np()
        
        
    def set_params(self, trans=None, pose=None, betas=None):
        if trans is None:
            trans=np.zeros(3)
        if pose is None:
            pose=np.zeros(self.pose_shape)
        if betas is None:
            betas=np.zeros(self.beta_shape)
        self.data['trans'] = trans
        self.data['pose'] =  pose
        self.data['betas'] = betas
        
    def cal_shape(self):
        # cal body shapes
        self.data['v_shaped'] = self.data['shapedirs'].dot(self.data['betas'])+self.data['v_template']
        # cal joint location
        self.J = self.data['J_regressor'] @ self.data['v_shaped']
        # cal rotation matrix for each joint by rodigues
        pose_cube = self.data['pose'].reshape((-1, 1, 3))
        self.R = self.rodrigues(pose_cube)
        I_cube = np.broadcast_to(
            np.expand_dims(np.eye(3), axis=0),
            (self.R.shape[0]-1, 3, 3)
        )
        lrotmin = (self.R[1:] - I_cube).ravel()
        
        # how pose affect body shape in zero pose
        v_posed = self.data['v_shaped'] + self.data['posedirs'].dot(lrotmin)
        # world transformation of each joint
        G = np.empty((self.data['kintree_table'].shape[1], 4, 4))
        G[0] = np.vstack((np.hstack((self.R[0], self.J[0, :].reshape([3, 1]))), np.array([[0.0, 0.0, 0.0, 1.0]])))
        for i in range(1, self.data['kintree_table'].shape[1]):
            G[i] = G[self.parent[i]].dot(
                np.vstack((
                    np.hstack(
                        [self.R[i],((self.J[i, :]-self.J[self.parent[i],:]).reshape([3,1]))]
                    ), 
                    np.array([[0.0, 0.0, 0.0, 1.0]])
                ))
            )
        
        tG = np.matmul(G,np.hstack([self.J, np.zeros([24, 1])]).reshape([24, 4, 1]))
        G = G - np.dstack((np.zeros((tG.shape[0], 4, 3)), tG))
        # transformation of each vertex
        T = np.tensordot(self.data['weights'], G, axes=[[1], [0]])
        rest_shape_h = np.hstack((v_posed, np.ones([v_posed.shape[0], 1])))
        v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
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
        # pass
        # 本来想试着直接将顶点数据以及mesh数据写入的，但是尝试了一上午发现assimp这个库只适合模型的读取，对于自定义数据的写入极其不友好，所以放弃了
        # transfrom obj to fbx
        self.save_obj('./__temp__.obj')
        with pyassimp.load('./__temp__.obj') as scene:
            # print(len(scene.meshes[0]))
            print(type(scene.meshes[0]))
            pyassimp.export(scene, fname, 'fbx')
        os.remove('./__temp__.obj')
        print('remove ./__temp__.obj')
        print('Save to ', fname)

        # scene = pyassimp.structs.Scene()
        # pyassimp.call_init(scene)
        # root_node = pyassimp.structs.Node()
        # pyassimp.call_init(root_node)
        # mesh = pyassimp.structs.Mesh()
        # pyassimp.call_init(mesh)

        # mesh.vertices = self.data['v']
        # mesh.faces = self.data['f']
        # scene.meshes.append(mesh)
        # pyassimp.recur_pythonize(root_node,scene)
        # pyassimp.export(scene, fname, 'fbx')

        


    # save self.J
    def save_joint(self):
        pass

      
### helper functions
    def rodrigues(self, r):
        theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
        # avoid zero divide
        theta = np.maximum(theta, np.finfo(r.dtype).eps)
        r_hat = r / theta
        cos = np.cos(theta)
        z_stick = np.zeros(theta.shape[0])
        m = np.dstack([
            z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
            r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
            -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
        ).reshape([-1, 3, 3])
        i_cube = np.broadcast_to(
            np.expand_dims(np.eye(3), axis=0),
            [theta.shape[0], 3, 3]
        )
        A = np.transpose(r_hat, axes=[0, 2, 1])
        B = r_hat
        dot = np.matmul(A, B)
        R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
        return R


    # help pickle to load pkl file
    def find_class(self, module, name):
        if module == 'chumpy.ch':   # fixed chumpy in local
            return getattr(chumpy.ch, name)
        if module == 'scipy.sparse.csc':    # the `scipy.sparse.csc` namespace is deprecated
            return getattr(scipy.sparse, name) 
        return super().find_class(module, name)
        
    def trans2np(self):
        for s in ['v_template', 'weights', 'posedirs', 'shapedirs', 'J', 'f']:
                self.data[s] = np.array(self.data[s])
        
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


def main():
    # f_path = 'path/to/smpl_pkl_file'
    f_path = '/home/poker/dataset/SMPL/SMPL_python_v.1.1.0/smpl/models/basicmodel_f_lbs_10_207_0_v1.1.0.pkl'
    smpl = SMPL_Loader(f_path)
    
    # random pose and shape
    trans = np.zeros(3)
    pose =  (np.random.rand(smpl.pose_shape) - 0.5)
    betas = (np.random.rand(smpl.beta_shape) - 0.5) * 0.6
    smpl.set_params(trans,pose,betas)
    
    smpl.cal_shape()
    smpl.save_obj('0.obj')
    smpl.seve_fbx('0.fbx')
    
    # smpl.print_data()

if __name__ == "__main__":
    main()