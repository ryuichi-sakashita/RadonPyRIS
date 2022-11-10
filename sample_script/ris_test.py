import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

#import py3Dmol
import sys
sys.path.append( "./RadonPy" )

from radonpy.core import utils
from radonpy.sim.preset import ris


#smiles = "*CC(*)c1ccccc1"
#smiles = "*Oc1ccc(cc1)S(=O)(=O)c1ccc(cc1)Oc1ccc(cc1)C(c1ccc(cc1)*)(C)C"
smiles = "*/C=C/CC*"
#smiles = "*c1ccc(*)cc1"

mol = utils.mol_from_smiles(smiles)

# RIS二面角スキャン用のインスタンスを作成
scan = ris.Dihedralscan(mol, work_dir='work_dir')

# 途中までのデータがある場合はロード
scan.load()

# 回転可能な二面角を求める
if scan.dihedral_sets is None:
    dihedral_sets = scan.get_rotable_bonds()
else:
    dihedral_sets = scan.dihedral_sets
print(dihedral_sets)

# オリゴマーの再安定構造を計算
if scan.polymer is None:
    scan.opt_oligomer(omp=30, memory=10000)
else:
    print('polymer loaded')

# 各二面角を回転させて安定二面角を求める 
if scan.dihedral_lib is None:
    dihedral_lib = scan.scan_dihedrals(step=15, omp=30, memory=10000)
else:
    dihedral_lib = scan.dihedral_lib
print(dihedral_lib)

# 安定二面角3つの組み合わせについてエネルギーを算出
# dihedral_sets = scan.calc_alltriplets(omp=30, memory=10000)
print(dihedral_sets)