# Stochastic Simulation Algorithm input file

# Reactions
R1:
    mRNA > 2 mRNA
    Ksyn*mRNA*kd

R2:
    mRNA > $pool
    Kdeg*mRNA
 
# Fixed species
 
# Variable species

 
# Parameters
kd = 100
Ksyn = 2.9
Kdeg = 3
