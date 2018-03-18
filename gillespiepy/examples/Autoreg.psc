# Generated by PySCeS 0.7.0 (2010-10-07 19:43)
 
# Keywords
Description: Auto-regulatory network
Modelname: AutoRegulatoryNetwork
Output_In_Conc: True
Species_In_Conc: False
 
# GlobalUnitDefinitions
UnitVolume: litre, 1.0, 0, 1
UnitLength: metre, 1.0, 0, 1
UnitSubstance: item, 1.0, 0, 1
UnitArea: metre, 1.0, 0, 2
UnitTime: second, 1.0, 0, 1
 
# Compartments
Compartment: Cell, 1.0, 3 
 
# Reactions
RepressionBindingCell:
    P2 + Gene > P2Gene
    RepressionBinding_k1*Gene*P2

ProteinDegradationCell:
    P > $pool
    ProteinDegradation_k6*P

DimerisationCell:
    2.0P > P2
    Dimerisation_k4*0.5*P*(P-1)

DissociationCell:
    P2 > 2.0P
    Dissociation_k4r*P2

TranscriptionCell:
    $pool > Rna
    Transcription_k2*Gene

RnaDegradationCell:
    Rna > $pool
    RnaDegradation_k5*Rna

TranslationCell:
    $pool > P
    Translation_k3*Rna

ReverseRepressionBindingCell:
    P2Gene > P2 + Gene
    ReverseRepressionBinding_k1r*P2Gene
 
# Fixed species
 
# Variable species
P2Cell = 0.0
PCell = 0.0
RnaCell = 0.0
GeneCell = 10.0
P2GeneCell = 0.0
 
# Parameters
RepressionBinding_k1 = 1.0
ProteinDegradation_k6 = 0.01
Dimerisation_k4 = 1.0
Dissociation_k4r = 1.0
Transcription_k2 = 0.01
RnaDegradation_k5 = 0.1
Translation_k3 = 10.0
ReverseRepressionBinding_k1r = 10.0 