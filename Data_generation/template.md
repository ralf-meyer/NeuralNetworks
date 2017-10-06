
units          metal
dimension      3
atom_style     atomic
boundary       f  f  f

# atoms
region mybox sphere 0 0 0 5
create_box	2 mybox
create_atoms	1 random 6 999 NULL
create_atoms	2 random 7 888 NULL
mass	1 58.6934
mass	2 196.96

#Potential
pair_style     eam/alloy
pair_coeff * * NiAu_Zhou.eam.alloy Ni Au

# Output

velocity       all zero angular
fix            1 all momentum 100 linear 1 1 1 angular
# Variables

variable	eng equal pe
variable	fx atom fx
variable	fy atom fy
variable	fz atom fz
variable	x atom x
variable	y atom y
variable	z atom z
variable	type atom type


# Run the simulation
run            1

