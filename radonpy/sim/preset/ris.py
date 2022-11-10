#  Copyright (c) 2022. RadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

# ******************************************************************************
# sim.preset.ris module
# ******************************************************************************

import os
import glob
import re
import datetime
import numpy as np
import itertools
from rdkit import Chem
from ...core import poly, utils
from ...sim.psi4_wrapper import Psi4w
import multiprocessing as mp

__version__ = '0.2.3'

class Dihedralscan():
    def __init__(self, mol, prefix='', work_dir=None, save_dir=None, **kwargs):
        self.mol = utils.deepcopy_mol(mol)
        self.prefix = prefix
        self.work_dir        = work_dir if work_dir is not None else './'
        self.save_dir        = save_dir if save_dir is not None else os.path.join(self.work_dir, 'ris')
        os.makedirs(self.save_dir, exist_ok=True)

        self.oligomer_dir    = os.path.join(self.work_dir, 'oligomer')

        self.scan_dihedral_dir = os.path.join(self.work_dir, 'dihedralscan')
        self.eachenergy_dir    = os.path.join(self.work_dir, 'eachenergy')

        self.oligomer_name      = kwargs.get('oligomer_name', '%soligomer.pickle' % self.prefix)
        self.scan_result        = kwargs.get('dihedralscan_name', '%sdihedralscan.npy' % self.prefix)
        self.dihedralset_result = kwargs.get('dihedralset_name', '%sdihedralset.npy' % self.prefix)

        self.ter = kwargs.get('ter', '*C')
        self.dihedral_sets = None
        self.polymer = None
        self.dihedral_lib = None

    
    def load(self):
        # polymer
        polymerpath = os.path.join(self.save_dir, self.oligomer_name)
        if os.path.isfile(polymerpath):
            self.polymer = utils.pickle_load(polymerpath)
        
        # dihedral_sets
        dihedral_setspath = os.path.join(self.save_dir, self.dihedralset_result)
        if os.path.isfile(dihedral_setspath):
            self.dihedral_sets = np.load(dihedral_setspath, allow_pickle=True)
        
        # dihedral_lib
        dihedral_libpath = os.path.join(self.save_dir, self.scan_result)
        if os.path.isfile(dihedral_libpath):
            self.dihedral_lib = np.atleast_1d(np.load(dihedral_libpath, allow_pickle=True))[0]

        return


    def _triplewise(self, l):
        "Return overlapping triplets from an list"
        # triplewise('ABCDEFG') -> ABC BCD CDE DEF EFG FGA GAB
        iter = list(itertools.islice(itertools.cycle(l), len(l)+2))
        triple = []
        for i in range(len(l)):
            triple.append((iter[i], iter[i+1], iter[i+2]))
        return triple


    def get_rotable_bonds(self):
        # get shortest path between head-tail
        mol = poly.polymerize_rw(self.mol, 1)
        head_ne_idx = mol.GetIntProp('head_ne_idx') 
        tail_ne_idx = mol.GetIntProp('tail_ne_idx') 
        shortest_path = Chem.rdmolops.GetShortestPath(mol, head_ne_idx, tail_ne_idx)

        # for each bonds
        rotable_bonds = []
        spans         = []
        inspan_flag   = False
        for i in range(len(shortest_path)-1):
            a, b = shortest_path[i], shortest_path[i+1]
            bond = mol.GetBondBetweenAtoms(a, b)
            # span flag
            if inspan_flag is False:
                span_head = a
                inspan_flag = True
            
            # not ring and single bond = rotable
            if (bond.IsInRing() is False) and (str(bond.GetBondType())=='SINGLE'):
                rotable_bonds.append((a, b))
                spans.append((span_head, b))
                inspan_flag = False

        # the bond between monomer and monomer must be single bond (rotable)
        a, b = shortest_path[-1], shortest_path[0]
        if inspan_flag is False:
                span_head = a
        rotable_bonds.append((a, b))
        spans.append((span_head, b))
        
        # numbering main chain atoms as Rdkit IntProp
        atom_dic = {}
        for i, idx in enumerate(shortest_path):
            atom = mol.GetAtomWithIdx(idx)
            atom.SetIntProp('RIS_atom_id', i)
            atom_dic[idx] = i

        # renumbering
        rotable_bonds = [(atom_dic[a], atom_dic[b]) for a,b in rotable_bonds]
        spans         = [(atom_dic[a], atom_dic[b]) for a,b in spans]

        # rotable_bonds = 1 -> 4mer, rotable_bonds = 2 -> trimer, rotable_bonds > 1 -> dimer 
        if len(rotable_bonds) == 1:
            polymer = poly.polymerize_rw(mol, 4)
            utils.radon_print('oligomer for RIS calc: 4mer', level=1)
        elif len(rotable_bonds) == 2:
            polymer = poly.polymerize_rw(mol, 3)
            utils.radon_print('oligomer for RIS calc: 3mer', level=1)
        else:
            polymer = poly.polymerize_rw(mol, 2)
            utils.radon_print('oligomer for RIS calc: 2mer', level=1)

        # termination with CH3
        ter     = utils.mol_from_smiles(self.ter)
        polymer = poly.terminate_rw(polymer, ter) 

        # get shortest path between head-tail of oligomer
        head_idx = polymer.GetIntProp('terminal_idx1')
        tail_idx = polymer.GetIntProp('terminal_idx2')
        shortest_path = Chem.rdmolops.GetShortestPath(polymer, head_idx, tail_idx)

        # make new atom number dict
        atom_dics = []
        last_id = -1
        ad = {}
        for idx in shortest_path:
            atom = polymer.GetAtomWithIdx(idx)
            if 'RIS_atom_id' in atom.GetPropNames():
                atom_id = atom.GetIntProp('RIS_atom_id')
                if atom_id < last_id:
                    atom_dics.append(ad)
                    ad = {}
                ad[atom_id] = idx
                last_id = atom_id
        atom_dics.append(ad)

        #function toget new atom idx for all dihedrals
        def atom_renumbering(triples):
            renumbered_bond_set = []    
            for bond_set in triples:
                cycle = 0
                prev_atom = 0
                renumbered_bonds = []
                for bond in bond_set:
                    new_bond = []
                    for atom in bond:
                        if atom < prev_atom:
                            cycle += 1
                        new_bond.append(atom_dics[cycle][atom])
                        prev_atom = atom
                    renumbered_bonds.append(tuple(new_bond))
                renumbered_bond_set.append(tuple(renumbered_bonds))
            return renumbered_bond_set
        
        renumbered_spans         = atom_renumbering(self._triplewise(spans))
        renumbered_rotable_bonds = atom_renumbering(self._triplewise(rotable_bonds))

        triple_dihedral = []
        for triple in renumbered_rotable_bonds:
            dihedrals = []
            for bond in triple:
                ind = shortest_path.index(bond[0])
                dihedral = (shortest_path[ind-1], shortest_path[ind], shortest_path[ind+1], shortest_path[ind+2])
                dihedrals.append(dihedral)
            triple_dihedral.append(tuple(dihedrals))

        dihedral_sets = np.array([])
        for span, rotable_bond, span_atomid, dihedral in zip(self._triplewise(spans), renumbered_rotable_bonds, renumbered_spans, triple_dihedral):
            d = {'spans': span,
                 'rotable_bonds': rotable_bond,
                 'span_vectors': span_atomid,
                 'dihedral': dihedral}
            dihedral_sets = np.append(dihedral_sets, d)

        self.dihedral_sets = dihedral_sets
        self.polymer       = polymer
        np.save(os.path.join(self.save_dir, self.dihedralset_result), self.dihedral_sets)
        return dihedral_sets


    def opt_oligomer(self, **kwargs):
        if self.polymer is None:
            utils.radon_print('oligomer is not found.', level=3)
            return None
        else:
            os.makedirs(self.oligomer_dir, exist_ok=True)
            psi4mol = Psi4w(self.polymer, work_dir=self.oligomer_dir, **kwargs)
            # run psi4 opt to get optimized coord
            energy, coord = psi4mol.optimize(**kwargs)
            if psi4mol.error_flag == True:
                utils.radon_print('psi4 error.', level=3)
                return None
            else:
                self.polymer = utils.deepcopy_mol(psi4mol.mol)
                utils.pickle_dump(self.polymer, os.path.join(self.save_dir, self.oligomer_name))
                return self.polymer


    def scan_dihedrals(self, step=15, geom_iter=200, **kwargs):
        dihedral_lib = {}
        for dihedral_set in self.dihedral_sets:
            dihedral_data = {}
            span     = dihedral_set['spans'][1]
            dihedral = dihedral_set['dihedral'][1]
            
            dihedral_data['scaned_dihedral'] = dihedral

            temp_work_dir = os.path.join(self.scan_dihedral_dir, 'span_%d_%d'%span)
            os.makedirs(temp_work_dir, exist_ok=True)

            psi4mol = Psi4w(self.polymer, work_dir=temp_work_dir, **kwargs)
            scaned_energy, scaned_coord, localminima_energy, localminima_coord, localminima_dihedrals = psi4mol.localminima_dihedrals_scan(dihedral=dihedral, step=step, geom_iter=geom_iter)

            dihedral_data['scaned_energy']         = scaned_energy - np.min(localminima_energy)
            dihedral_data['scaned_coord']          = scaned_coord
            dihedral_data['num_localminima']       = len(localminima_energy)
            dihedral_data['localminima_energy']    = localminima_energy - np.min(localminima_energy)
            dihedral_data['localminima_coord']     = localminima_coord
            dihedral_data['localminima_dihedrals'] = localminima_dihedrals

            dihedral_lib[span] = dihedral_data

        self.dihedral_lib = dihedral_lib
        np.save(os.path.join(self.save_dir, self.scan_result), self.dihedral_lib)
        return self.dihedral_lib

    
    def calc_alltriplets(self, geom_iter=200, **kwargs):
        def sub_psi4(polymer, dihedrals, alpha, beta, gamma, energy, coord, work_dir, geom_iter=200, **kwargs):
            # set coord of 3 dihedrals
            temp_polymer = utils.deepcopy_mol(polymer)
            conf = temp_polymer.GetConformer(0)
            
            Chem.rdMolTransforms.SetDihedralDeg(conf, dihedrals[0][0], dihedrals[0][1], dihedrals[0][2], dihedrals[0][3], float(alpha))
            Chem.rdMolTransforms.SetDihedralDeg(conf, dihedrals[1][0], dihedrals[1][1], dihedrals[1][2], dihedrals[1][3], float(beta))
            Chem.rdMolTransforms.SetDihedralDeg(conf, dihedrals[2][0], dihedrals[2][1], dihedrals[2][2], dihedrals[2][3], float(gamma))
            
            # MMFF
            prop = Chem.AllChem.MMFFGetMoleculeProperties(temp_polymer)
            ff = Chem.AllChem.MMFFGetMoleculeForceField(temp_polymer, prop, confId=0)
            error = 0.5
            ff.MMFFAddTorsionConstraint(dihedrals[0][0], dihedrals[0][1], dihedrals[0][2], dihedrals[0][3], False, float(alpha)-error, float(alpha)+error, 1e5)
            ff.MMFFAddTorsionConstraint(dihedrals[1][0], dihedrals[1][1], dihedrals[1][2], dihedrals[1][3], False, float(beta)-error, float(beta)+error, 1e5)
            ff.MMFFAddTorsionConstraint(dihedrals[2][0], dihedrals[2][1], dihedrals[2][2], dihedrals[2][3], False, float(gamma)-error, float(gamma)+error, 1e5)
            ff.Minimize()
            
            Chem.rdMolTransforms.SetDihedralDeg(conf, dihedrals[0][0], dihedrals[0][1], dihedrals[0][2], dihedrals[0][3], float(alpha))
            Chem.rdMolTransforms.SetDihedralDeg(conf, dihedrals[1][0], dihedrals[1][1], dihedrals[1][2], dihedrals[1][3], float(beta))
            Chem.rdMolTransforms.SetDihedralDeg(conf, dihedrals[2][0], dihedrals[2][1], dihedrals[2][2], dihedrals[2][3], float(gamma))                      
            
            os.makedirs(work_dir, exist_ok=True)

            psi4mol = Psi4w(temp_polymer, work_dir=work_dir, **kwargs)
            e, c = psi4mol.optimize(freeze=dihedrals, geom_iter=geom_iter, ignore_conv_error=True, **kwargs)
            energy.append(e)
            coord.append(c)

        manager = mp.Manager()
        for i, dihedral_set in enumerate(self.dihedral_sets):
            state_num = []
            angles    = []
            for span in dihedral_set['spans']:
                dihedral_data = self.dihedral_lib[span]
                state_num.append(dihedral_data['num_localminima'])
                angles.append(dihedral_data['localminima_dihedrals'])

            energies = []
            coords   = []
            for a, alpha in enumerate(angles[0]):
                energy_b = []
                coord_b  = []
                for b, beta in enumerate(angles[1]):
                    energy_g = []
                    coord_g  = []
                    for g, gamma in enumerate(angles[2]):
                        e = manager.Value
                        c = manager.Value
                        dihedrals = dihedral_set['dihedral']
                        temp_work_dir = os.path.join(self.eachenergy_dir, 'set_%d_triplet_%d_%d_%d'%(i,a,b,g))

                        j = mp.Process(target=sub_psi4, 
                                       args=(self.polymer, dihedrals, alpha, beta, gamma, e, c, temp_work_dir, geom_iter),
                                       kwargs=kwargs)
                        j.start()                    
                        j.join()

                        if j.exitcode == 1:
                            energy_g.append(e)
                            coord_g.append(c)
                        else:
                            energy_g.append(np.nan)
                            coord_g.append(np.nan)
                        
                    energy_b.append(energy_g)
                    coord_b.append(coord_g)
                energies.append(energy_b)
                coords.append(coord_b)

            energies = energies - np.min(energies)
            dihedral_set['energy_matrix'] = energies
            dihedral_set['coord_matrix']  = coords
            self.dihedral_sets[i] = dihedral_set

        np.save(os.path.join(self.save_dir, self.dihedralset_result), self.dihedral_sets)
        return self.dihedral_sets