
import torch

import argparse
from ase.visualize import view
from ase import Atoms, Atom
from ase.db import connect
from ase.calculators.morse import MorsePotential

import random
from ase.build import sort
from ase import Atom
from ase.io import read, write
from base64 import b64encode, b64decode
import itertools
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from pointnet import PointNetCls, PointNetReg


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

db = connect('./qm9.db')
# 16.1% memory usage of a 128G machine
# rows = list(db.select(sort='id'))
rows = list(db.select('id<200'))

atom_names = ['H', 'C', 'O', 'F', 'N']
atom_dict = {'H': 0, 'C':1, 'O':2, 'F':3, 'N':4}
atom_cnt_lst = []
atom_cnt_dict = {}

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--task', type=str, default='reg', help='specify task (\'reg\' or \'cls\')')
    # parser.add_argument('--normal', action='store_true', default=True, help='Whether to use normal information [default: False]')
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

def gen_potential_data(args, data_count=100, atom_count=2, side_len=5):
    input_lst = []
    tgt_lst = []
    postv = data_count/2
    negtv = data_count/2
    while(len(tgt_lst) < data_count):
        atom_coords = []
        atoms = Atoms()
        atoms.append(Atom('Au', (0.5, 0.5, 0.5)))
        for _ in range(atom_count-1):
            x = random.random() * side_len
            y = random.random() * side_len
            z = random.random() * side_len
            atoms.append(Atom('Au', (x, y, z)))
            atom_coords.append([x, y, z])
        
        morse_calc = MorsePotential()
        
        atoms.set_calculator(morse_calc)
        engy = atoms.get_potential_energy()

        if args.task == 'reg':
            if engy < 0:
                input_lst.append(atom_coords)
                tgt_lst.append(min(engy, 0))
        elif args.task == 'cls':
            if engy < 0 and negtv > 0:
                negtv-=1
                input_lst.append(atom_coords)
                tgt_lst.append(0)
            elif engy > 0 and postv > 0:
                postv -= 1
                input_lst.append(atom_coords)
                tgt_lst.append(1)
            else:
                pass
        else:
            pass
    return input_lst, tgt_lst

def main(args):
    min_vali_loss = 99999
    model = None
    atom_cnt = 3
    train_lst, train_tgt = gen_potential_data(args=args, data_count=1000, atom_count=atom_cnt)
    vali_lst, vali_tgt = gen_potential_data(args=args, data_count=1000, atom_count=atom_cnt)
    test_lst, test_tgt = gen_potential_data(args=args, data_count=1000, atom_count=atom_cnt)
    
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
        # Forward pass
        y_pred = model(x_data)
        # print(y_pred.shape)
    
        # Compute loss
        loss = criterion(y_pred, y_data)
        
        # Forward pass vali
        y_pred_vali = model(x_data_vali)
    
        # Compute loss vali
        loss_vali = criterion(y_pred_vali, y_data_vali)
        print(epoch, loss.item(), loss_vali.item())

        if loss_vali.item() < min_vali_loss:
            min_vali_loss = loss_vali.item()
            # save model
            print('model saved...')
            torch.save(model.state_dict(), 'best_vali.pth')

        # Zero gradients
        optimizer.zero_grad()
        # perform backward pass
        loss.backward()
        # update weights
        optimizer.step()
    # load model
    model.load_state_dict(torch.load('best_vali.pth'))
    # inference on test dataset
    y_pred_test = model(x_data_test)
    success = 0
    for i in range(len(test_tgt)):
        if y_pred_test[i][0] > y_pred_test[i][1] and 0 == test_tgt[i]:
            success += 1
        elif y_pred_test[i][0] < y_pred_test[i][1] and 1 == test_tgt[i]:
            success += 1
        else:
            pass
    print(success)

if __name__ == '__main__':
    args = parse_args()
    main(args)
