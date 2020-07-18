
import torch
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
from pointnet import PointNetEncoder, PointNetReg


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

model = PointNetReg()
# train_lst = []
# train_tgt = []

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

def gen_potential_data(data_count=100, atom_count=2, side_len=5):
    input_lst = []
    tgt_lst = []
    for _ in range(data_count):
        atom_coords = []
        atoms = Atoms()
        for _ in range(atom_count):
            x = random.random() * side_len
            y = random.random() * side_len
            z = random.random() * side_len
            atoms.append(Atom('Au', (x, y, z)))
            atom_coords.append([x, y, z])
        morse_calc = MorsePotential()
        atoms.set_calculator(morse_calc)
        engy = atoms.get_potential_energy()
        
        input_lst.append(atom_coords)
        tgt_lst.append(engy)
    return input_lst, tgt_lst

if __name__ == '__main__':
    # for row in rows:
    #     xyz = row.toatoms().get_positions()
    #     if train_lst.shape[0] == 0:
    #         train_lst = np.array([xyz])
    #         print(train_lst.shape)
    #     else:
    #         print(xyz.shape)
    #         train_lst = np.dstack((train_lst, xyz))
    #     train_tgt = np.append(train_tgt, get_data_pp(row.id, G))
    #     # print(row.id, get_data_pp(row.id, G))
    train_lst, train_tgt = gen_potential_data(data_count=1000, atom_count=2)
    
    criterion = torch.nn.MSELoss() # Defined loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Defined optimizer
    x_data = torch.from_numpy(np.array(train_lst).transpose(0, 2, 1)).to('cpu')
    y_data = torch.from_numpy(np.array(train_tgt)).to('cpu')

    print(x_data.shape, y_data.shape)
    for epoch in range(1000):
        # Forward pass
        y_pred = model(x_data)
        print(y_pred.shape)
    
        # Compute loss
        loss = criterion(y_pred, y_data)
        '''
        # Forward pass vali
        y_pred_vali = model(x_data_vali.double())
    
        # Compute loss vali
        loss_vali = criterion(y_pred_vali, y_data_vali)
        '''
        print(epoch, loss.item())

        # Zero gradients
        optimizer.zero_grad()
        # perform backward pass
        loss.backward()
        # update weights
        optimizer.step()
