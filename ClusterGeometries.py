from ClusterSim import input_classes
from sympy.utilities.iterables import multiset_permutations
import numpy as np

iko=input_classes.Ikosaeder()

def basis_vector_distances(layers,lattice_const):
    
    for i in range(1,len(layers)):
        if i==1:
            all_layers=np.concatenate((np.array(layers[i-1]),np.array(layers[i])))
        else:
            all_layers=np.concatenate((all_layers,np.array(layers[i])))
    
    shape=all_layers.shape

    out=np.zeros((len(all_layers),shape[0]))
    for i in range(len(all_layers)):
        out[i,:]=np.linalg.norm(iko.get_xyz(all_layers[i]-all_layers,lattice_const),axis=1)
        
    return out

def get_possible_distances(n_atoms,all_dists):
    
    dists=[]
    #get all possible distances
    temp_vec=3*[1]+(n_atoms-3)*[0]
    temp_permutations=np.asarray(list(multiset_permutations(temp_vec)))
    
    for permutation in temp_permutations:
        bool_ones=permutation==1
        ones=list(np.where(bool_ones)[0])
        for i in range(0,len(ones)):
            for j in range(0,len(ones)):
                temp_dist=round(all_dists[ones[i],ones[j]],5)
                if not(temp_dist in dists):
                    dists.append(temp_dist)
    return dists

def get_unique_configurations(n_atoms_per_type,all_dists,possible_dists):
    
    configurations=[]
    dist_configs=[]
    iter_vec=n_atoms_per_type[0]*[1]+n_atoms_per_type[1]*[0]
    all_permutations=np.asarray(list(multiset_permutations(iter_vec)))

    for permutation in all_permutations:
        bool_ones=permutation==1
        ones=list(np.where(bool_ones)[0]) 
        if len(ones)<2:
            ones=[0]+ones
            
        dist_count=[0]*len(possible_dists)
        for i in range(0,len(ones)):
            for j in range(0,len(ones)):
                dist=round(all_dists[ones[i],ones[j]],5)
                dist_idx=possible_dists.index(dist)
                dist_count[dist_idx]+=1
        if not(dist_count in dist_configs):
            dist_configs.append(dist_count)
            configurations.append(permutation)

    return configurations

def create_QE_configurations(configs,xyzs,atom_types):
    
    out_configs=[]
    #for first atom species
    for config in configs:
        geom=[]
        for a_type,atom in zip(config,xyzs):
            this_atom=(atom_types[a_type],atom)
            geom.append(this_atom)
        out_configs.append(geom)
    #for second atom species
    for config in configs:
        geom=[]
        for a_type,atom in zip(config,xyzs):
            this_atom=(atom_types[1-a_type],atom)
            geom.append(this_atom)
        out_configs.append(geom)
        
    return out_configs
            


def create_unique_geometries(n_layers,atom_types=["Ni","Au"],lattice_const=1):

    iko.build_layers(n_layers)
    xyzs=iko.get_xyz(iko.layers,lattice_const)
    n_atoms=iko.get_magic_nr(n_layers)
    all_dists=basis_vector_distances(iko.layers,lattice_const)
    possible_dists=get_possible_distances(n_atoms,all_dists)
    
    unique_configs=[]
    for i in range(0,int(n_atoms/2)-1): #from 0 to half of the atoms (other half is equivalent to changing atom species) 
        n_atoms_per_type=[i,n_atoms-i]
        unique_configs+=get_unique_configurations(n_atoms_per_type,all_dists,possible_dists)
    
    return create_QE_configurations(unique_configs,xyzs,atom_types)


def make_files(geometries,randomization_factor=0,path=''):
    
    header="&CONTROL\n  calculation  = 'scf'\n  prefix       = 'geom_opt',\n  pseudo_dir   = '/usr/share/espresso/pseudo',\n outdir       = '/home/afuchs/.tempdir'\n/\n&SYSTEM\n ibrav     = 1,\n celldm(1) = 40\n  nat       = $natom$,\n  ntyp      = 2,\n  ecutwfc   = 25.D0,\n  ecutrho   = 100.D0,\n occupations='smearing', smearing='methfessel-paxton',degauss=0.01\n/\n&ELECTRONS\n  conv_thr    = 1.D-5,\n  mixing_beta = 0.15D0,\n  electron_maxstep = 250\n/\nATOMIC_SPECIES\nNi 58.69 Ni.pbe-n-kjpaw_psl.0.1.UPF\nAu 197.00 Au.pbe-dn-kjpaw_psl.0.1.UPF\nATOMIC_POSITIONS {angstrom}\n"
    n_atoms=len(geometries[0])
    header=header.replace("$natom$",str(n_atoms))
    for i,geom in enumerate(geometries):
        with open(path+'NiAu_'+str(i)+".in",'w') as file:
            file.write(header)
            for j in range(0,len(geom)):
                rnd=np.random.rand(3)*randomization_factor
                line=str(geom[j][0])+" "+str(geom[j][1][0]+rnd[0])+" "+str(geom[j][1][1]+rnd[1])+" "+str(geom[j][1][2]+rnd[2])
                file.writelines(line+'\n')
            file.write("K_POINTS Gamma")
            file.close()