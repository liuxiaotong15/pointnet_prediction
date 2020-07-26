
import torch

import h5py
import argparse
from ase.visualize import view
from ase import Atoms, Atom
from ase.db import connect
from ase.calculators.morse import MorsePotential
from ase.optimize import QuasiNewton, BFGS
import random
from ase.build import sort
from ase import Atom
from ase.io import read, write
from ase.io.trajectory import Trajectory
from base64 import b64encode, b64decode
import itertools
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
# from pointnet_full import PointNetCls, PointNetReg, feature_transform_reguliarzer


# pointnet++ cost so much memory...
from pointnet_pp import PointNetCls, PointNetReg


seed = 1234
random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.set_default_tensor_type(torch.DoubleTensor)

# properties
A = "rotational_constant_A"
B = "rotational_constant_B"
C = "rotational_constant_C"
mu = "dipole_moment"
alpha = "isotropic_polarizability"
homo = "homo"
lumo = "lumo"
gap = "gap"
r2 = "electronic_spatial_extent"
zpve = "zpve"
U0 = "energy_U0"
U = "energy_U"
H = "enthalpy_H"
G = "free_energy"
Cv = "heat_capacity"

# db = connect('./qm9.db')
# 16.1% memory usage of a 128G machine
# rows = list(db.select(sort='id'))
# rows = list(db.select('id<200'))

atom_names = ['H', 'C', 'O', 'F', 'N']
atom_dict = {'H': 0, 'C':1, 'O':2, 'F':3, 'N':4}
atom_cnt_lst = []
atom_cnt_dict = {}

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training')
    parser.add_argument('--task', type=str, default='reg', help='specify task (\'reg\' or \'cls\')')
    parser.add_argument('--load_data', action='store_true', default=False, help='load data from dataset_morse.hdf5')
    return parser.parse_args()


def get_data_pp(idx, type):
    # extract properties
    prop = 'None'
    # row = db.get(id=idx)
    row = rows[idx-1]
    if(row.id != idx):
        1/0
    # extract from schnetpack source code
    shape = row.data["_shape_" + type]
    dtype = row.data["_dtype_" + type]
    prop = np.frombuffer(b64decode(row.data[type]), dtype=dtype)
    prop = prop.reshape(shape)
    return prop

def gen_potential_data(args, data_count=100, atom_count=2, side_len=20, interval=10):
    input_lst = []
    tgt_lst = []
    atoms_lst = []
    postv = data_count/2
    negtv = data_count/2
    while(len(atoms_lst) < data_count):
        print('current data size is: ', len(atoms_lst))
        last_engy = 0
        atom_coords = []
        atoms = Atoms()
        # atoms.append(Atom('Au', (0.5, 0.5, 0.5)))
        atoms.append(Atom('Au', (0, 0, 0)))
        for _ in range(atom_count-1):
            x = random.random() * side_len
            y = random.random() * side_len
            z = random.random() * side_len
            atoms.append(Atom('Au', (x, y, z)))
        
        morse_calc = MorsePotential()
        
        atoms.set_calculator(morse_calc)
        dyn = BFGS(atoms, trajectory='latest_relax.traj')
        dyn.run(fmax=0.1, steps=100)
        traj = Trajectory('latest_relax.traj')
        for atoms in traj:
            cur_engy = atoms.get_potential_energy()
            if args.task == 'reg':
                if cur_engy < -0.01 and abs(last_engy-cur_engy) > 1:
                    atoms_lst.append(atoms)
                    last_engy = cur_engy
            elif args.task == 'cls':
                if cur_engy < 0 and negtv > 0:
                    negtv -= 1
                    atoms_lst.append(atoms)
                elif cur_engy > 0 and postv > 0:
                    postv -= 1
                    atoms_lst.append(atoms)
                else:
                    pass
            else:
                pass

    random.shuffle(atoms_lst)

    for i in range(len(atoms_lst)):
        atom_coords = list(atoms_lst[i].positions)
        engy = atoms_lst[i].get_potential_energy()
        input_lst.append(atom_coords)
        if args.task == 'reg':
            tgt_lst.append(engy)
        elif args.task == 'cls':
            if engy < 0:
                tgt_lst.append(0)
            else:
                tgt_lst.append(1)
        else:
            pass

        
    if args.task == 'reg':
        # normalize
        tgt_lst = np.array(tgt_lst)
        tgt_lst /= max(abs(tgt_lst))
        tgt_lst = tgt_lst.tolist()

    return input_lst, tgt_lst

def main(args):
    min_vali_loss = 99999
    model = None
    atom_cnt = 128
    if args.load_data:
        f = h5py.File('dataset_morse.hdf5', 'r')
        data_lst = f['dset1'][:]
        data_tgt = f['dset2'][:]
        f.close()
        print('load data finished.')
    else:
        data_lst, data_tgt = gen_potential_data(args=args, data_count=10000, atom_count=atom_cnt)

        f = h5py.File('dataset_morse.hdf5', 'w')
        f.create_group('/grp1') # or f.create_group('grp1')
        f.create_dataset('dset1', compression='gzip', data=np.array(data_lst))
        f.create_dataset('dset2', compression='gzip', data=np.array(data_tgt))
        f.close()
        print('save data finished.')

    train_lst, train_tgt = data_lst[0:int(len(data_lst)*0.8)], data_tgt[0:int(len(data_lst)*0.8)]
    vali_lst, vali_tgt = data_lst[int(len(data_lst)*0.8):int(len(data_lst)*0.9)], data_tgt[int(len(data_lst)*0.8):int(len(data_lst)*0.9)]
    test_lst, test_tgt = data_lst[int(len(data_lst)*0.9):len(data_lst)], data_tgt[int(len(data_lst)*0.9):len(data_lst)]

    
    if args.task == 'reg':
        model = PointNetReg()
        criterion = torch.nn.MSELoss() # Defined loss function
    elif args.task == 'cls':
        model = PointNetCls()
        criterion = torch.nn.CrossEntropyLoss() # Defined loss function
    else:
        pass
    x_data = torch.from_numpy(np.array(train_lst).transpose(0, 2, 1)).to('cpu')
    y_data = torch.from_numpy(np.array(train_tgt)).to('cpu')
    x_data_vali = torch.from_numpy(np.array(vali_lst).transpose(0, 2, 1)).to('cpu')
    y_data_vali = torch.from_numpy(np.array(vali_tgt)).to('cpu')
    x_data_test = torch.from_numpy(np.array(test_lst).transpose(0, 2, 1)).to('cpu')
    y_data_test = torch.from_numpy(np.array(test_tgt)).to('cpu')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) # Defined optimizer
    for epoch in range(1000):
        for i in range(0, len(train_lst), args.batch_size):
            # Forward pass
            y_pred, trans_feat = model(x_data[i:i+args.batch_size])
            # print(y_pred.shape)
        
            # Compute loss
            loss = criterion(torch.squeeze(y_pred), y_data[i:i+args.batch_size]) # + 0.001 * feature_transform_reguliarzer(trans_feat) 
 
            # Zero gradients
            optimizer.zero_grad()
            # perform backward pass
            loss.backward()
            # update weights
            optimizer.step()
           
        with torch.no_grad():
            loss_vali_val = 0
            for i in range(0, len(vali_lst), args.batch_size):
                # Forward pass vali
                y_pred_vali, trans_feat = model(x_data_vali[i:i+args.batch_size])
    
                # Compute loss vali
                loss_vali = criterion(torch.squeeze(y_pred_vali), y_data_vali[i:i+args.batch_size]) # + 0.001 * feature_transform_reguliarzer(trans_feat) 
                loss_vali_val += loss_vali.item()
            loss_vali_val /= (len(vali_lst)/args.batch_size)
            print(epoch, loss.item(), loss_vali_val)

            if loss_vali_val < min_vali_loss:
                min_vali_loss = loss_vali_val
                # save model
                print('model saved...')
                torch.save(model.state_dict(), 'best_vali.pth')

    # load model
    model.load_state_dict(torch.load('best_vali.pth'))

    with torch.no_grad():
        success = 0
        err = 0
        for i in range(0, len(test_lst), args.batch_size):
            # inference on test dataset
            y_pred_test, _ = model(x_data_test[i:i+args.batch_size])
            if args.task == 'cls':
                for j in range(args.batch_size):
                    if y_pred_test[j][0] > y_pred_test[j][1] and 0 == test_tgt[i+j]:
                        success += 1
                    elif y_pred_test[j][0] < y_pred_test[j][1] and 1 == test_tgt[i+j]:
                        success += 1
                    else:
                        pass
            elif args.task == 'reg':
                test_pred = y_pred_test.detach().numpy()
                for j in range(args.batch_size):
                    err += abs((test_pred[j] - test_tgt[i+j])/test_tgt[i+j])
            else:
                pass
        
        if args.task == 'cls':
            print('classify success: ', success, 'of', len(test_tgt))
        elif args.task == 'reg':
            print('percentage of regression error: ', err/len(test_tgt)*100, '%')

if __name__ == '__main__':
    args = parse_args()
    main(args)
