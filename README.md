ExpMetSim is a very intuitive code to use. It is thoroughly described in “Design, assembly, and test of G-CLEF's 
exposure meter I: design trade-off and first conclusions” in preparation to be published at SPIE Optical Systems 
Design 2024 Proceedings.

There is a separate section where you can choose inputs according to your need. The following parameters are 
customizable: 

##############################################################################
##############################################################################
#################                   INPUTS                   #################
##############################################################################
FirstTime1=' ' ## 'YES'/'NO' Is this the first time using any of the following parameters?
HIP_ID = ' ' ## Object ID from Hiparcus Catalog; use 'HIP****'
HIP_number =  ## Object ID from Hiparcus Catalog; use only the number
mag= ## Stellar apparent magnitude
SpectralType = ' ' ## Stellar spectral type: 'F', 'G', 'K', 'M' 
BlockBandsFilter=' ' ## Block selected bands as if there were filters in the system? 
Filter1=' ' ## 'YES'/'NO' Filter blocking from 6250 to 6340
Filter2=' ' ## 'YES'/'NO' Filter blocking from 6850 to 6970
Filter3=' ' ## 'YES'/'NO' Filter blocking from 7141 to 7380
Filter4=' ' ## 'YES'/'NO' Filter blocking from 7575 to 7735
Filter5=' ' ## 'YES'/'NO' Filter blocking from 8095 to 8385
Filter6=' ' ## 'YES'/'NO' Filter blocking from 8905 to 9890
ObservingMode=' ' ## Observing mode: 'PRV', 'NPRV'
BlockBandsAfter=' ' ## 'YES'/'NO' Block selected bands after detection but before fitting?
##############################      Prism      ###############################
BeamSpeed=3 ## Beam speed entering the exposure meter (ex: for F/3 insert only 3)
Diam_fiber=400e-6 ## Entrance fiber diameter in m
Diam_coll=0.025 ## Collimator diameter in m
t_prism=0.025 ## Prism base width in m
r=1 ## Anamorphic magnification factor
PrismIncAngle= 0 ## Prism inclination angle in degree; For 0 deg: prism base is parallel to incoming beam 
(positive inclination clockwise considering collimator on the left)
Apex=40 ## Prism apex angle in degree; it is considered an isosceles triangle
###############################     CCD     ##################################
DkC=0.1 ## dark current [e/pixel]
pix_size=6.5e-6 ## pixel size in m
RoN=0.9 ## read out noise [e]
##############################################################################
FirstTime2=' ' ## 'YES'/'NO' Is this the first time using any of the following parameters?
JDTime=2459453.726658 ## Beginning of the observation in Julian Date
GCLEFtime=1800 ## Time of exposure on G-CLEF in seconds
EXPMETtime=10.0 ## Exposure time of each exposure on the exposure meter in seconds
Directory='Magellan_BC_HIP90850' ## Name of the directory containing the output files
##############################################################################
##############################################################################


Once you set up the desired parameters, the code is good to run.
