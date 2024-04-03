# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 14:53:13 2020

@author: henrique
"""
#import scipy as *
import numpy as np
#import pylab as plt
from astropy.io import fits
import matplotlib.pyplot as plt
import os

#from barycorrpy import sample_script 
from barycorrpy import get_BC_vel 
#from barycorrpy import utc_tdb 

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.coordinates import Angle, Latitude, Longitude

import math

##   Wavelength range   ##
rangelambA = np.arange(3500,10000, 0.025)  ## A //  e-10 m

##   Constants   ##
h=6.62907015e-34 ## J s // Plank's constant
c=299792458 ## m/s  // Speed of light
kb=1.380649e-23 ## J/K  // Boltzmann constant
Agmt=368 ##m2  // Effective collecting area of GMT

##############################################################################
##############################################################################
#################                   INPUTS                   #################
##############################################################################
FirstTime1='NO' ## YES/NO Is this the first time using all the following parameters?
HIP_ID = 'HIP34596' ## Object ID from Hiparcus Catalog; use 'HIP****'
HIP_number = 34596  ## Object ID from Hiparcus Catalog; use only the number
mag= 7.17 ## Stellar apparent magnitude
SpectralType = 'G' ## Stellar spectral type: 'F', 'G', 'K', 'M' ## change to temperature?
BlockBandsFilter='NO' ## Block selected bands as if there were filters in the system? 
ObservingMode='PRV' ## Observing mode: 'PRV', 'NPRV'
#Glass='PBM2Y' ## Prism glass: PBM2Y, PBM8Y, PBM18Y
BlockBandsAfter='NO' ## Block selected bands after detection but before fitting?
##############################      Prism      ###############################
BeamSpeed=3 ## Beam speed entering the exposure meter (ex: for F/3 insert only 3)
Diam_fiber=300e-6 ## Entrance fiber diameter in m
Diam_coll=0.025 ## Collimator diameter in m
t_prism=0.025 ## Prism base width in m
r=1 ## Anamorphic magnification factor
PrismIncAngle= -30 ## Prism inclination angle in degree; For 0 deg: prism base is parallel to incoming beam (positive inclination clockwise considering collimator on the left)
Apex=60 ## Prism apex angle in degree; it is considered an isoceles triangle
##############################################################################
FirstTime2='NO' ## YES/NO Is this the first time using all the following parameters?
JDTime=2458988.45747800 ## Begining of the observation in Julian Date
GCLEFtime=1800 ## Time of exposure on G-CLEF in seconds
EXPMETtime=0.25 ## Exposure time of each exposure on the exposure meter in seconds
Directory='BC_HIP34596' ## Name of the directory containing the output files
##############################################################################
##############################################################################

##############  Atmosphere transmission depending on airmass  ################
def Transmittance(X):
    Tparanal=np.loadtxt('DATA/sky_trans.ascii',usecols=(1),unpack=True)
    lambdaKappaA=10*np.loadtxt('DATA/sky_trans.ascii', usecols=(0), unpack=True) ##Wavelength in A (10*nm in the file)
    Kappa=(-1)*np.log(Tparanal)  ## Extinction coefficient for Cerro Paranal
    #X = (1/cos(z))[1-0.0012*((1/cos(z))**2-1)]
    Tatm= np.exp(-X*Kappa) 
    Tatm= np.interp(rangelambA, lambdaKappaA, Tatm)  ##interpolation to fit the grid
    return Tatm

###########################     Star Through GMT     #########################
def StellarSpec(StellarType,mag): #TIREI SPECTYPE
    print('Calculating stellar signal')
    if StellarType == 'F':
        T=7000 ##K
        StarSpec=fits.getdata('DATA/PHOENIX/lte07000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')

    if StellarType == 'G':
        T=5300 ##K
        StarSpec=fits.getdata('DATA/PHOENIX/lte05300-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')
        
    if StellarType == 'K':
        T=4500 ##K
        StarSpec=fits.getdata('DATA/PHOENIX/lte04500-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')

    if StellarType == 'M':
        T=3000 ##K
        StarSpec=fits.getdata('DATA/PHOENIX/lte03000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')
        
    WLPHOENIX=fits.getdata('DATA/PHOENIX/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits') ## Wavelength array for PHOENIX files, in A
    StarSpec=np.interp(rangelambA,WLPHOENIX,StarSpec) ## Interpolate to fit the grid  in equally spaced resolution elements

    ###    Johnson V filter    ###
    lamb_filter=np.loadtxt("DATA/JohnsonV.txt", usecols=(0), unpack=True)
    T_filter=np.loadtxt("DATA/JohnsonV.txt", usecols=(1), unpack=True)

    lamb_filter=lamb_filter[::-1]  ### was 1000nm to 300nm now 300nm to 1000nm 
    T_filter=T_filter[::-1]

    VFilter=np.empty(len(lamb_filter))
    VFilter = 0.01*np.interp(rangelambA/10, lamb_filter, T_filter)  ## interpolation so that the filter description can fit the grid

    AreaFilter=np.trapz(VFilter,rangelambA) ## Integration over the transmittance curve of the filter
    ###############################
    
    ### BASE FOR MAG FLUX CONVERTION ###    
    F_o=2.091e-08 ###W/m2 Zero mag flux for Johnson V filter
    Flux=F_o*10**(-mag/2.5) ## W/m2 ## over the filter
    ##Mag->Flux converter
    Star_V=StarSpec*VFilter  ## Blackbody of Star through Johnson V filter 
    Integral=np.trapz(Star_V,rangelambA) 
    C=Flux*AreaFilter/Integral 
    Star=StarSpec*C/rangelambA ## Spectra in W/m2/A of a "StellarType" Star of magnitude mag and Blackbody Temperature T
    #Star=StarSpec*C  ## Spectra in W/m2 of a "StellarType" Star of magnitude mag and Blackbody Temperature T
      
    return Star

######################   Efficiency of Optical System   ######################
def Eff_Inst(ObservingMode,Signal,BlockBands):
    
    Signal=Signal*Agmt # Star going into GMT [W]
    
    ###     Transmitance of Optical System    ###

    lambErrorBudget,T_GMT,T_FrontEnd,PRV_Fiber,NPRV_Fiber,InsideSpec=np.loadtxt("DATA/T_OpticalSystem.txt",usecols=(0,1,2,3,4,5),unpack=True)
    ## info in T_OpticalSystem comes from G-CLEF's Throughput budget

    T_PRV=T_GMT*T_FrontEnd*PRV_Fiber*InsideSpec
    T_NPRV=T_GMT*T_FrontEnd*NPRV_Fiber*InsideSpec

    T_PRV= np.interp(rangelambA, lambErrorBudget, T_PRV)
    T_NPRV= np.interp(rangelambA, lambErrorBudget, T_NPRV)

    if ObservingMode =='PRV':
        Signal=Signal*T_PRV
        
    if ObservingMode =='NPRV':
        Signal=Signal*T_NPRV
        
        
    Signal=Signal*0.001  ## 0.1% of light getting into G-CLEF's echelle grating will be redirected to the exposure meter
    ##Signal [W/A]
    
    QE=np.loadtxt('DATA/Zyla44QE.txt',usecols=(1),unpack=True) ##CMOS quantum efficiency in certain conditions. We will consider 95% of this number: 
    Signal=Signal*QE*0.01*0.95 ##Signal [W/A]
    
    if BlockBands == 'YES':
         #####   BLOCKING SELECTED BANDS WITH FILTERS  ####        
        i=0
        while i<len(rangelambA):
            
            if 6250<= rangelambA[i]<=6340 or 6850<=rangelambA[i]<=6970 or 7141<=rangelambA[i]<=7380 or 7575<=rangelambA[i]<=7735 or 8095<=rangelambA[i]<=8385 or 8905<=rangelambA[i]<=9890:
                Signal[i]=0
                
            i=i+1
    rangelambm = rangelambA*1e-10 ## m
    E=h*c/rangelambm ##Energy of 1 photon of wavelength lambda
    Counts=Signal/E ## Counts: [photons s-1 A-1]
        
    return Counts   

##  Calculation of flux on each channel and airmass during the observation  ##   
def ExpMet(Glass,Signal,JDo,targetname,GCLEFtime,EXPMETtime):
    ## JDo is the time the observation starts
    ## We are considering a G-CLEF observation of GCLEFtime in seconds 
    ## Each exposure of the exposure meter of EXPMETtime in seconds
    ## Target HIP identification number
    print('Calculating flux per exposure for %s'%Glass)
    ChannelLamb,channels=DevideInChannels(Glass,rangelambA)
    NumbExp = GCLEFtime/EXPMETtime ##Number of exposures on the exposure meter
    NumbExp = int(NumbExp)
    
    if Glass=='PBM2Y':
        lambA,T_PBM2Y=np.loadtxt('DATA/OHARAGlasses/TransmittancePBM2Y.txt',usecols=(0,2),skiprows=1,unpack=True)
        T_PBM2Y= np.interp(rangelambA, lambA, T_PBM2Y)
        Signal=Signal*T_PBM2Y
    if Glass=='PBM8Y':        
        lambA,T_PBM8Y=np.loadtxt('DATA/OHARAGlasses/TransmittancePBM8Y.txt',usecols=(0,2),skiprows=1,unpack=True)
        T_PBM8Y= np.interp(rangelambA, lambA, T_PBM8Y)
        Signal=Signal*T_PBM8Y
    if Glass=='PBM18Y':        
        lambA,T_PBM18Y=np.loadtxt('DATA/OHARAGlasses/TransmittancePBM18Y.txt',usecols=(0,2),skiprows=1,unpack=True)
        T_PBM18Y= np.interp(rangelambA, lambA, T_PBM18Y)
        Signal=Signal*T_PBM18Y
    
    print(channels,'channels')
    
    TargetCoord=SkyCoord.from_name(targetname)  ### Retrieving target's coordinates RA DEC
    LCO_GMT = EarthLocation(lat=-29.028333*u.deg, lon=-70.685833*u.deg, height=2516*u.m) ## GMT location
        
    t=0 ## index over time array// time stamp

    JD=JDo
    SumFluxChannel=np.zeros(shape=(NumbExp,channels)) 
    AirmassArray=np.zeros(NumbExp)
    
    while t<NumbExp: 
        
        ch=0 ## index over number of channels
        JDMean=(2*JD+((1/86400)*EXPMETtime))/2
        time = Time(format='jd',val=JDMean,scale='tt')
        TargetCoordAltAz = TargetCoord.transform_to(AltAz(obstime=time,location=LCO_GMT))
        z=90 - TargetCoordAltAz.alt.degree ## Zenith angle in degree
        z=z*np.pi/180 ## Zenith angle in rad
        X = (1/np.cos(z))*(1-0.0012*((1/np.cos(z))**2-1)) ### Airmass        
        
        T_atm=Transmittance(X)
        F=Signal*EXPMETtime*T_atm  ## Signal=[counts/sA] now flux is in counts/A
        
        FChannels,channels=DevideInChannels(Glass,F)
      
        AirmassArray[t]=X
        
        while ch<channels:
            SumFluxChannel[t][ch]=np.trapz(FChannels[ch],ChannelLamb[ch]) #integrating in wavelength resulting in [counts]
            ch=ch+1
        
        if t/100-int(t/100)==0:print('Exposure',t, 'of',NumbExp)
        
        t=t+1
        JD=JD+((1/86400)*EXPMETtime)
    
    return SumFluxChannel,AirmassArray
    
##############   Definition number of lines per channel for Prisms    #########
def LinesPerChannel(Glass):
    WL,D_PBM2Y,D_PBM8Y,D_PBM18Y=np.loadtxt("DATA/OHARAGlasses/DispersionOHARAGlasses.txt",usecols=(0,1,2,3),unpack=True,skiprows=1) ## Dispersion in (mu m)^-1

    D_PBM2Y=np.interp(rangelambA,WL,D_PBM2Y)
    D_PBM8Y=np.interp(rangelambA,WL,D_PBM8Y)
    D_PBM18Y=np.interp(rangelambA,WL,D_PBM18Y)

    AppertureAng=2*math.atan(1/(2*BeamSpeed)) ## Apperture angle given a beam speed in rad
    beta=Apex/2+PrismIncAngle ## degree
    beta=beta*math.pi/180 ## rad
    BeamDiamPrism=(1/math.cos(beta))*Diam_coll  ## Size of beam projected on prism face
    
    Res_PBM2Y=-1*rangelambA*D_PBM2Y*Diam_coll*t_prism/(r*AppertureAng*BeamDiamPrism*Diam_fiber*10000)
    Res_PBM8Y=-1*rangelambA*D_PBM8Y*Diam_coll*t_prism/(r*AppertureAng*BeamDiamPrism*Diam_fiber*10000)
    Res_PBM18Y=-1*rangelambA*D_PBM18Y*Diam_coll*t_prism/(r*AppertureAng*BeamDiamPrism*Diam_fiber*10000)
    
    if Glass=='PBM2Y':
        Resolution=Res_PBM2Y
        Dispersion=D_PBM2Y
        DeltaLamb=rangelambA/Resolution
        filename='%s/LinesPerChannelPBM2Y.txt'%Directory
        f= open(filename,"w+")
    if Glass=='PBM8Y':
        Resolution=Res_PBM8Y
        Dispersion=D_PBM8Y
        DeltaLamb=rangelambA/Resolution
        filename='%s/LinesPerChannelPBM8Y.txt'%Directory
        f= open(filename,"w+")
    if Glass=='PBM18Y':
        Resolution=Res_PBM18Y
        Dispersion=D_PBM18Y
        DeltaLamb=rangelambA/Resolution
        filename='%s/LinesPerChannelPBM18Y.txt'%Directory
        f= open(filename,"w+")
    i=0 
    j=0      
    ch=0
    N=0
    while i in range(len(rangelambA)):
        while i in range(len(rangelambA)) and rangelambA[i]<rangelambA[j]+DeltaLamb[j]:
            N=N+1
            i=i+1
            
        f.write("%f    %f\r\n" % (ch,N))
        j=i
        N=0
        ch=ch+1
    return(ch)

####################   Definition of Channels for Prisms    ##################
def DevideInChannels(Glass,Array2BDevided):
    if Glass=='PBM2Y':
        flag=LinesPerChannel(Glass)
        filename='%s/LinesPerChannelPBM2Y.txt'%Directory
        ch,lines=np.loadtxt(filename,usecols=(0,1),unpack=True)
        
    if Glass=='PBM8Y':
        flag=LinesPerChannel(Glass)
        filename='%s/LinesPerChannelPBM8Y.txt'%Directory
        ch,lines=np.loadtxt(filename,usecols=(0,1),unpack=True)
        
    if Glass=='PBM18Y':
        flag=LinesPerChannel(Glass)
        filename='%s/LinesPerChannelPBM18Y.txt'%Directory
        ch,lines=np.loadtxt(filename,usecols=(0,1),unpack=True)
        
    if Glass=='GCLEF':
        ch,lines=np.loadtxt('DATA/LinesPerChannelGCLEF.txt',usecols=(0,1),unpack=True)
        
    chi=0 ## index i for channels
    linesj=0 ## index j for lines on the whole band rangelambA
    lineslim=0 ##
    j=0 ## line index inside the channels
     
    while chi<=ch[-1]: 
        lineslim=lineslim+lines[chi]
        globals()["Channel" + str(chi)] = np.zeros(int(lines[chi]))
        while j<lines[chi]: 
            globals()["Channel" + str(chi)][j]=Array2BDevided[linesj]
            j=j+1
            linesj=linesj+1
        j=0
        chi=chi+1
            
        
    ChannelLamb=np.array_split(Array2BDevided,chi)
    i=0
    while i<chi:
        ChannelLamb[i]=globals()["Channel" + str(i)]
        i=i+1
        
    return ChannelLamb,chi

#################### Calculation of Barycentric Correction ###################   
def BaryCorr(GCLEFtime,EXPMETtime,JDTime,HIP_ID):
    print('Calculating barycentric correction per exposure')
    NumbExp = GCLEFtime/EXPMETtime ##Number of exposures on the exposure meter
    NumbExp = int(NumbExp)
    velocities=np.zeros(NumbExp)
    x=np.zeros(NumbExp)
    i=0
    filename='BC_HIP34596/BCPerExposure%s.txt'%FLAGEXPOSURE
#    if FLAGEXPOSURE=='250ms':
#        filename='%s/BCPerExposure250ms.txt'%Directory
#    if FLAGEXPOSURE=='500ms':
#        filename='%s/BCPerExposure500ms.txt'%Directory
#    if FLAGEXPOSURE=='1s':
#        filename='%s/BCPerExposure1s.txt'%Directory
#    if FLAGEXPOSURE=='10s':
#        filename='%s/BCPerExposure10s.txt'%Directory
#    if FLAGEXPOSURE=='50s':
#        filename='%s/BCPerExposure50s.txt'%Directory
#    if FLAGEXPOSURE=='100s':
#        filename='%s/BCPerExposure100s.txt'%Directory
    f= open(filename,"w+")
    while i<NumbExp:
        JDTimeMean=(2*JDTime+((1/86400)*EXPMETtime))/2 
        vel,warning,flag = get_BC_vel(JDUTC=JDTimeMean,hip_id=HIP_ID,obsname='LCO',ephemeris='de430',zmeas=0)
        velocities[i]=vel
        #print(JDTime,"    ",vel)
        f.write("%f    %f\r\n" % (JDTimeMean,vel))
        x[i]=i
        if i/100-int(i/100)==0:print('BC for exposure',i,'of',NumbExp)
        i=i+1
        JDTime=JDTime+((1/86400)*EXPMETtime)   
    f.close()
    return(i)

######### Definition of one value of BC correction per each channel ##########
def BaryCorrPerChannel(Glass,SumFluxChannel,BLOCK):
    print('Calculating BC per channel for %s' %Glass)
    ChannelLamb,channels=DevideInChannels(Glass,rangelambA)
    
    barycorrperchannel=np.zeros(channels)
    effwlchannel=np.zeros(channels)
    t=0
    ch=0
    SUM=0
    WeightedBC=0
    #filename='%s/BCPerExposure.txt'%Directory
    filename='BC_HIP34596/BCPerExposure%s.txt'%FLAGEXPOSURE
#    if FLAGEXPOSURE=='250ms':
#        filename='%s/BCPerExposure250ms.txt'%Directory
#    if FLAGEXPOSURE=='500ms':
#        filename='%s/BCPerExposure500ms.txt'%Directory
#    if FLAGEXPOSURE=='1s':
#        filename='%s/BCPerExposure1s.txt'%Directory
#    if FLAGEXPOSURE=='10s':
#        filename='%s/BCPerExposure10s.txt'%Directory
#    if FLAGEXPOSURE=='50s':
#        filename='%s/BCPerExposure50s.txt'%Directory
#    if FLAGEXPOSURE=='100s':
#        filename='%s/BCPerExposure100s.txt'%Directory
    
    BaryCorrVel=np.loadtxt(filename,usecols=(1),unpack=True)

    while ch<channels:
        while t<len(BaryCorrVel):
            WeightedBC=WeightedBC+BaryCorrVel[t]*SumFluxChannel[t][ch]
            SUM=SUM+SumFluxChannel[t][ch]
            t=t+1
        barycorrperchannel[ch]=WeightedBC/SUM
        if Glass=='GCLEF':
            if ch/10000-int(ch/10000)==0:print('BC for channel',ch, 'of',channels)
        else: 
            print('BC for channel',ch, 'of',channels)
        ch=ch+1
        t=0
        WeightedBC=0
        SUM=0
    
    if BLOCK=='YES':
####   BLOCKING SELECTED BANDS AFTER DETECTION BEFORE FITTING  ####        
        ch=0
        while ch<channels:
            effwlchannel[ch]=(ChannelLamb[ch][-1]+ChannelLamb[ch][0])/2
            if 6250<= effwlchannel[ch]<=6340 or 6850<=effwlchannel[ch]<=6970 or 7141<=effwlchannel[ch]<=7380 or 7575<=effwlchannel[ch]<=7735 or 8095<=effwlchannel[ch]<=8385 or 8905<=effwlchannel[ch]<=9890:
                barycorrperchannel[ch]=float('NaN')
            ch=ch+1
####   REPLACE NAN VALUES FOR INTERPOLATIONS   ####
        ch=0
        while ch<channels:
            if math.isnan(barycorrperchannel[ch]):
                chnext=ch+1
                foundnumber=0
                while foundnumber==0 and chnext<channels:
                    if math.isnan(barycorrperchannel[chnext]):
                        chnext=chnext+1
                    else:
                        barycorrperchannel[ch]=(barycorrperchannel[ch-1]+barycorrperchannel[chnext])/2
                        foundnumber=1
                    if foundnumber==0 and chnext==channels: barycorrperchannel[ch]=barycorrperchannel[ch-1]
            ch=ch+1
    else: 
        ch=0
        while ch<channels:
            effwlchannel[ch]=(ChannelLamb[ch][-1]+ChannelLamb[ch][0])/2
            ch=ch+1
    return barycorrperchannel, effwlchannel

#########   Assigning effective wavelength for each G-CLEF channel   #########
def EffWLChannel(channels):
    effwlchannel=np.zeros(channels)
    ChannelLamb=np.array_split(rangelambA,channels)
    ch=0
    while ch<channels:
        effwlchannel[ch]=(ChannelLamb[ch][-1]+ChannelLamb[ch][0])/2
        ch=ch+1
        
    return effwlchannel

##############################################################################
##############################################################################

Star=StellarSpec(SpectralType,mag) ## Chosen star spectrum
Signal=Eff_Inst(ObservingMode ,Star, BlockBandsFilter) ## After passing through optical train [W]

if FirstTime1=='YES' and FirstTime2=='YES':
    os.mkdir(Directory)
    filename='%s/ReadMe_Header.txt'%Directory
    f= open(filename,"w+")
    f.write('##############################################################################\r\n')
    f.write('#################                   INPUTS                   #################\r\n')
    f.write('##############################################################################\r\n')
    f.write('FirstTime1=%s ## Is this the first time using all the following parameters?\r\n'%FirstTime1)
    f.write('HIP_ID = %s ## Object ID from Hiparcus Catalog\r\n'%HIP_ID)
    f.write('mag= %f ## Stellar apparent magnitude\r\n'%mag)
    f.write('SpectralType = %s ## Stellar spectral type: F, G, K, M ## change to temperature?\r\n'%SpectralType)
    f.write('BlockBandsFilter= %s ## Block selected bands as if there were filters in the system?\r\n'%BlockBandsFilter)
    f.write('ObservingMode= %s ## Observing mode: PRV, NPRV\r\n'%ObservingMode)
    f.write('BlockBandsAfter=%s ## Block selected bands after detection but before fitting?\r\n'%BlockBandsAfter)
    f.write('##############################      Prism      ###############################\r\n')
    f.write('BeamSpeed=%f ## Beam speed entering the exposure meter (ex: for F/3 insert only 3)\r\n'%BeamSpeed)
    f.write('Diam_fiber=%f ## Entrance fiber diameter in m\r\n'%Diam_fiber)
    f.write('Diam_coll=%f ## Collimator diameter in m\r\n'%Diam_coll)
    f.write('t_prism=%f ## Prism base width in m\r\n'%t_prism)
    f.write('r=%f ## Anamorphic magnification factor\r\n'%r)
    f.write('PrismIncAngle= %f ## Prism inclination angle in degree; For 0 deg: prism base is parallel to incoming beam (inclination clockwise considering collimator on the left)\r\n'%PrismIncAngle)
    f.write('##############################################################################\r\n')
    f.write('Apex=%f ## Prism apex angle in degree; it is considered an isoceles triangle\r\n'%Apex)
    f.write('FirstTime2=%s ## Is this the first time using all the following parameters?\r\n'%FirstTime2)
    f.write('JDTime=%f ## Begining of the observation in Julian Date\r\n'%JDTime)
    f.write('GCLEFtime=%f## Time of exposure on G-CLEF in seconds\r\n'%GCLEFtime)
    f.write('EXPMETtime=%f ## Exposure time of each exposure on the exposure meter in seconds\r\n'%EXPMETtime)
    f.write('Directory=%s ## Name of the directory containing the output files\r\n'%Directory)
    f.write('##############################################################################\r\n')
    f.close()

if FirstTime2=='YES':
    filename = "%s/FluxGCLEF" % Directory
    SumFluxGCLEFChannel,AirmassArray=ExpMet('GCLEF',Signal,JDTime,HIP_ID,GCLEFtime,EXPMETtime)
    np.save(filename,SumFluxGCLEFChannel)
    FLAG=BaryCorr(GCLEFtime, EXPMETtime, JDTime, HIP_number)
    BCGCLEF,WLGCLEFCHANNEL=BaryCorrPerChannel('GCLEF',SumFluxGCLEFChannel,'NO')
    np.save('%s/BCGCLEF' % Directory ,BCGCLEF)
    np.save('%s/WLGCLEF' % Directory, WLGCLEFCHANNEL)
else:    
    SumFluxGCLEFChannel=np.load('%s/FluxGCLEF.npy' % Directory)
    BCGCLEF=np.load('%s/BCGCLEF.npy' % Directory)
    WLGCLEFCHANNEL=np.load('%s/WLGCLEF.npy' % Directory)

if FirstTime1 == 'YES':
    #SumFluxPBM2Y,AirmassArray=ExpMet('PBM2Y',Signal,JDTime,HIP_ID,GCLEFtime,EXPMETtime)
    #np.save("%s/FluxPBM2Y" % Directory,SumFluxPBM2Y)
    #BCPBM2Y,WLPBM2Y=BaryCorrPerChannel('PBM2Y',SumFluxPBM2Y,BlockBandsAfter)
    #np.save('%s/BCPBM2Y' % Directory ,BCPBM2Y)
    #np.save('%s/WLPBM2Y' % Directory, WLPBM2Y)
    
    SumFluxPBM2Y250ms,AirmassArray=ExpMet('PBM2Y',Signal,JDTime,HIP_ID,GCLEFtime,0.25)
    np.save("%s/FluxPBM2Y250ms" % Directory,SumFluxPBM2Y250ms)
    FLAGEXPOSURE='250ms'
    FLAG=BaryCorr(GCLEFtime, 0.25, JDTime, HIP_number)
    BCPBM2Y250ms,WLPBM2Y250ms=BaryCorrPerChannel('PBM2Y',SumFluxPBM2Y250ms,BlockBandsAfter)
    np.save('%s/BCPBM2Y250ms' % Directory ,BCPBM2Y250ms)
    np.save('%s/WLPBM2Y250ms' % Directory, WLPBM2Y250ms)
    
    SumFluxPBM2Y500ms,AirmassArray=ExpMet('PBM2Y',Signal,JDTime,HIP_ID,GCLEFtime,0.5)
    np.save("%s/FluxPBM2Y500ms" % Directory,SumFluxPBM2Y500ms)
    FLAGEXPOSURE='500ms'
    FLAG=BaryCorr(GCLEFtime, 0.5, JDTime, HIP_number)
    BCPBM2Y500ms,WLPBM2Y500ms=BaryCorrPerChannel('PBM2Y',SumFluxPBM2Y500ms,BlockBandsAfter)
    np.save('%s/BCPBM2Y500ms' % Directory ,BCPBM2Y500ms)
    np.save('%s/WLPBM2Y500ms' % Directory, WLPBM2Y500ms)
    
    SumFluxPBM2Y750ms,AirmassArray=ExpMet('PBM2Y',Signal,JDTime,HIP_ID,GCLEFtime,0.75)
    np.save("%s/FluxPBM2Y750ms" % Directory,SumFluxPBM2Y750ms)
    FLAGEXPOSURE='750ms'
    FLAG=BaryCorr(GCLEFtime, 0.75, JDTime, HIP_number)
    BCPBM2Y750ms,WLPBM2Y750ms=BaryCorrPerChannel('PBM2Y',SumFluxPBM2Y750ms,BlockBandsAfter)
    np.save('%s/BCPBM2Y750ms' % Directory ,BCPBM2Y750ms)
    np.save('%s/WLPBM2Y750ms' % Directory, WLPBM2Y750ms)
    
    SumFluxPBM2Y900ms,AirmassArray=ExpMet('PBM2Y',Signal,JDTime,HIP_ID,GCLEFtime,0.9)
    np.save("%s/FluxPBM2Y900ms" % Directory,SumFluxPBM2Y900ms)
    FLAGEXPOSURE='900ms'
    FLAG=BaryCorr(GCLEFtime, 0.90, JDTime, HIP_number)
    BCPBM2Y900ms,WLPBM2Y900ms=BaryCorrPerChannel('PBM2Y',SumFluxPBM2Y900ms,BlockBandsAfter)
    np.save('%s/BCPBM2Y900ms' % Directory ,BCPBM2Y900ms)
    np.save('%s/WLPBM2Y900ms' % Directory, WLPBM2Y900ms)
    
    SumFluxPBM2Y1s,AirmassArray=ExpMet('PBM2Y',Signal,JDTime,HIP_ID,GCLEFtime,1)
    np.save("%s/FluxPBM2Y1s" % Directory,SumFluxPBM2Y1s)    
    FLAGEXPOSURE='1s'
    FLAG=BaryCorr(GCLEFtime, 1, JDTime, HIP_number)
    BCPBM2Y1s,WLPBM2Y1s=BaryCorrPerChannel('PBM2Y',SumFluxPBM2Y1s,BlockBandsAfter)
    np.save('%s/BCPBM2Y1s' % Directory ,BCPBM2Y1s)
    np.save('%s/WLPBM2Y1s' % Directory, WLPBM2Y1s)
    
    SumFluxPBM2Y10s,AirmassArray=ExpMet('PBM2Y',Signal,JDTime,HIP_ID,GCLEFtime,10)
    np.save("%s/FluxPBM2Y10s" % Directory,SumFluxPBM2Y10s)
    FLAGEXPOSURE='10s'
    FLAG=BaryCorr(GCLEFtime, 10, JDTime, HIP_number)
    BCPBM2Y10s,WLPBM2Y10s=BaryCorrPerChannel('PBM2Y',SumFluxPBM2Y10s,BlockBandsAfter)
    np.save('%s/BCPBM2Y10s' % Directory ,BCPBM2Y10s)
    np.save('%s/WLPBM2Y10s' % Directory, WLPBM2Y10s)
    
    SumFluxPBM2Y50s,AirmassArray=ExpMet('PBM2Y',Signal,JDTime,HIP_ID,GCLEFtime,50)
    np.save("%s/FluxPBM2Y50s" % Directory,SumFluxPBM2Y50s)
    FLAGEXPOSURE='50s'
    FLAG=BaryCorr(GCLEFtime, 50, JDTime, HIP_number)
    BCPBM2Y50s,WLPBM2Y50s=BaryCorrPerChannel('PBM2Y',SumFluxPBM2Y50s,BlockBandsAfter)
    np.save('%s/BCPBM2Y50s' % Directory ,BCPBM2Y50s)
    np.save('%s/WLPBM2Y50s' % Directory, WLPBM2Y50s)
    
    SumFluxPBM2Y100s,AirmassArray=ExpMet('PBM2Y',Signal,JDTime,HIP_ID,GCLEFtime,100)
    np.save("%s/FluxPBM2Y100s" % Directory,SumFluxPBM2Y100s)
    FLAGEXPOSURE='100s'
    FLAG=BaryCorr(GCLEFtime, 100, JDTime, HIP_number)
    BCPBM2Y100s,WLPBM2Y100s=BaryCorrPerChannel('PBM2Y',SumFluxPBM2Y100s,BlockBandsAfter)
    np.save('%s/BCPBM2Y100s' % Directory ,BCPBM2Y100s)
    np.save('%s/WLPBM2Y100s' % Directory, WLPBM2Y100s)
    
else:  
    
    SumFluxPBM2Y250ms=np.load("%s/FluxPBM2Y250ms.npy" % Directory)
    BCPBM2Y250ms=np.load('%s/BCPBM2Y250ms.npy' % Directory)
    WLPBM2Y250ms=np.load('%s/WLPBM2Y250ms.npy' % Directory)
    
    SumFluxPBM2Y500ms=np.load("%s/FluxPBM2Y500ms.npy" % Directory)
    BCPBM2Y500ms=np.load('%s/BCPBM2Y500ms.npy' % Directory)
    WLPBM2Y500ms=np.load('%s/WLPBM2Y500ms.npy' % Directory)
    
    SumFluxPBM2Y750ms=np.load("%s/FluxPBM2Y750ms.npy" % Directory)
    BCPBM2Y750ms=np.load('%s/BCPBM2Y750ms.npy' % Directory)
    WLPBM2Y750ms=np.load('%s/WLPBM2Y750ms.npy' % Directory)
    
    SumFluxPBM2Y900ms=np.load("%s/FluxPBM2Y900ms.npy" % Directory)
    BCPBM2Y900ms=np.load('%s/BCPBM2Y900ms.npy' % Directory)
    WLPBM2Y900ms=np.load('%s/WLPBM2Y900ms.npy' % Directory)
    
    SumFluxPBM2Y600ms,AirmassArray=ExpMet('PBM2Y',Signal,JDTime,HIP_ID,GCLEFtime,0.6)
    np.save("%s/FluxPBM2Y600ms" % Directory,SumFluxPBM2Y600ms)
    FLAGEXPOSURE='600ms'
    FLAG=BaryCorr(GCLEFtime, 0.60, JDTime, HIP_number)
    BCPBM2Y600ms,WLPBM2Y600ms=BaryCorrPerChannel('PBM2Y',SumFluxPBM2Y600ms,BlockBandsAfter)
    np.save('%s/BCPBM2Y600ms' % Directory ,BCPBM2Y600ms)
    np.save('%s/WLPBM2Y600ms' % Directory, WLPBM2Y600ms)

    SumFluxPBM2Y700ms,AirmassArray=ExpMet('PBM2Y',Signal,JDTime,HIP_ID,GCLEFtime,0.7)
    np.save("%s/FluxPBM2Y700ms" % Directory,SumFluxPBM2Y700ms)
    FLAGEXPOSURE='700ms'
    FLAG=BaryCorr(GCLEFtime, 0.70, JDTime, HIP_number)
    BCPBM2Y700ms,WLPBM2Y700ms=BaryCorrPerChannel('PBM2Y',SumFluxPBM2Y700ms,BlockBandsAfter)
    np.save('%s/BCPBM2Y700ms' % Directory ,BCPBM2Y700ms)
    np.save('%s/WLPBM2Y700ms' % Directory, WLPBM2Y700ms)
    
    SumFluxPBM2Y800ms,AirmassArray=ExpMet('PBM2Y',Signal,JDTime,HIP_ID,GCLEFtime,0.8)
    np.save("%s/FluxPBM2Y800ms" % Directory,SumFluxPBM2Y800ms)
    FLAGEXPOSURE='800ms'
    FLAG=BaryCorr(GCLEFtime, 0.80, JDTime, HIP_number)
    BCPBM2Y800ms,WLPBM2Y800ms=BaryCorrPerChannel('PBM2Y',SumFluxPBM2Y800ms,BlockBandsAfter)
    np.save('%s/BCPBM2Y800ms' % Directory ,BCPBM2Y800ms)
    np.save('%s/WLPBM2Y800ms' % Directory, WLPBM2Y800ms)
    

    SumFluxPBM2Y1s=np.load("%s/FluxPBM2Y1s.npy" % Directory)
    BCPBM2Y1s=np.load('%s/BCPBM2Y1s.npy' % Directory)
    WLPBM2Y1s=np.load('%s/WLPBM2Y1s.npy' % Directory)
    
    SumFluxPBM2Y1200ms,AirmassArray=ExpMet('PBM2Y',Signal,JDTime,HIP_ID,GCLEFtime,1.2)
    np.save("%s/FluxPBM2Y1200ms" % Directory,SumFluxPBM2Y1200ms)    
    FLAGEXPOSURE='1200ms'
    FLAG=BaryCorr(GCLEFtime, 1.2, JDTime, HIP_number)
    BCPBM2Y1200ms,WLPBM2Y1200ms=BaryCorrPerChannel('PBM2Y',SumFluxPBM2Y1200ms,BlockBandsAfter)
    np.save('%s/BCPBM2Y1200ms' % Directory ,BCPBM2Y1200ms)
    np.save('%s/WLPBM2Y1200ms' % Directory, WLPBM2Y1200ms)
    
    SumFluxPBM2Y10s=np.load("%s/FluxPBM2Y10s.npy" % Directory)
    BCPBM2Y10s=np.load('%s/BCPBM2Y10s.npy' % Directory)
    WLPBM2Y10s=np.load('%s/WLPBM2Y10s.npy' % Directory)
    
    SumFluxPBM2Y50s=np.load("%s/FluxPBM2Y50s.npy" % Directory)
    BCPBM2Y50s=np.load('%s/BCPBM2Y50s.npy' % Directory)
    WLPBM2Y50s=np.load('%s/WLPBM2Y50s.npy' % Directory)
    
    SumFluxPBM2Y100s=np.load("%s/FluxPBM2Y100s.npy" % Directory)
    BCPBM2Y100s=np.load('%s/BCPBM2Y100s.npy' % Directory)
    WLPBM2Y100s=np.load('%s/WLPBM2Y100s.npy' % Directory)
    

#######   Search for minimum signal-to-noise ratio on our detections   #######
MinSN_PBM2Y=np.sqrt(np.amin(SumFluxPBM2Y250ms))
#MinSN_PBM8Y=np.sqrt(any(min(SumFluxPBM8Y)))
#MinSN_PBM18Y=np.sqrt(any(min(SumFluxPBM18Y)))
#print('Minimum S/N - PBM2Y:%f; PBM8Y:%f; PBM18Y:%f' %MinSN_PBM2Y,MinSN_PBM8Y,MinSN_PBM18Y)
print('Minimum S/N - PBM2Y:%f' %MinSN_PBM2Y)

###################   Relative Barycentric correction   ######################
DeltaBC=min(BCPBM2Y250ms)
RBCPBM2Y250ms=BCPBM2Y250ms-DeltaBC #relative barycentric correction
RBCPBM2Y500ms=BCPBM2Y500ms-DeltaBC
RBCPBM2Y600ms=BCPBM2Y600ms-DeltaBC
RBCPBM2Y700ms=BCPBM2Y700ms-DeltaBC
RBCPBM2Y750ms=BCPBM2Y750ms-DeltaBC
RBCPBM2Y800ms=BCPBM2Y800ms-DeltaBC
RBCPBM2Y900ms=BCPBM2Y900ms-DeltaBC
RBCPBM2Y1s=BCPBM2Y1s-DeltaBC
RBCPBM2Y1200ms=BCPBM2Y1200ms-DeltaBC
RBCPBM2Y10s=BCPBM2Y10s-DeltaBC
RBCPBM2Y50s=BCPBM2Y50s-DeltaBC
RBCPBM2Y100s=BCPBM2Y100s-DeltaBC
#RBCGCLEF=BCGCLEF-DeltaBC

##########  Linear fitting of barycentric correction vs wavelength ###########
FITPBM2Y250ms=np.interp(rangelambA,WLPBM2Y250ms,RBCPBM2Y250ms)
FITPBM2Y500ms=np.interp(rangelambA,WLPBM2Y500ms,RBCPBM2Y500ms)
FITPBM2Y600ms=np.interp(rangelambA,WLPBM2Y600ms,RBCPBM2Y600ms)
FITPBM2Y700ms=np.interp(rangelambA,WLPBM2Y700ms,RBCPBM2Y700ms)
FITPBM2Y750ms=np.interp(rangelambA,WLPBM2Y750ms,RBCPBM2Y750ms)
FITPBM2Y800ms=np.interp(rangelambA,WLPBM2Y800ms,RBCPBM2Y800ms)
FITPBM2Y900ms=np.interp(rangelambA,WLPBM2Y900ms,RBCPBM2Y900ms)
FITPBM2Y1s=np.interp(rangelambA,WLPBM2Y1s,RBCPBM2Y1s)
FITPBM2Y1200ms=np.interp(rangelambA,WLPBM2Y1200ms,RBCPBM2Y1200ms)
FITPBM2Y10s=np.interp(rangelambA,WLPBM2Y10s,RBCPBM2Y10s)
FITPBM2Y50s=np.interp(rangelambA,WLPBM2Y50s,RBCPBM2Y50s)
FITPBM2Y100s=np.interp(rangelambA,WLPBM2Y100s,RBCPBM2Y100s)
#FITGCLEF=np.interp(rangelambA,WLGCLEFCHANNEL,RBCGCLEF)

#plt.figure()
#plt.plot(rangelambA, FITGCLEF)
#lt.plot(rangelambA, FITPBM2Y)
#plt.plot(rangelambA, FITPBM18Y)
#plt.plot(rangelambA, FITPBM18Y)


########################## Residuals calculation #############################
RESIDUALSPBM2Y500ms=FITPBM2Y250ms-FITPBM2Y500ms
RESIDUALSPBM2Y600ms=FITPBM2Y250ms-FITPBM2Y600ms
RESIDUALSPBM2Y700ms=FITPBM2Y250ms-FITPBM2Y700ms
RESIDUALSPBM2Y750ms=FITPBM2Y250ms-FITPBM2Y750ms
RESIDUALSPBM2Y800ms=FITPBM2Y250ms-FITPBM2Y800ms
RESIDUALSPBM2Y900ms=FITPBM2Y250ms-FITPBM2Y900ms
RESIDUALSPBM2Y1s=FITPBM2Y250ms-FITPBM2Y1s
RESIDUALSPBM2Y1200ms=FITPBM2Y250ms-FITPBM2Y1200ms
RESIDUALSPBM2Y10s=FITPBM2Y250ms-FITPBM2Y10s
RESIDUALSPBM2Y50s=FITPBM2Y250ms-FITPBM2Y50s
RESIDUALSPBM2Y100s=FITPBM2Y250ms-FITPBM2Y100s

#plt.figure()
#plt.imshow(SumFluxPBM2Y250ms, interpolation='none' )
#plt.show()


plt.figure()
plt.title('Residuals relative to 0.25s exposure')

plt.plot(0,0,color='C0')
plt.plot(0,0,color='C1')
plt.plot(0,0,color='C2')
plt.plot(0,0,color='C3')
plt.plot(0,0,color='C4')
plt.plot(0,0,color='C5')
plt.plot(0,0,color='C6')
plt.plot(0,0,color='C7')
plt.plot(0,0,color='C8')
plt.plot(0,0,color='C9')
plt.plot(0,0,color='C10')


plt.legend(['500ms','600ms','700ms','750ms','800ms','900ms','1s','1.2s','10s','50s','100s'],loc=0)

plt.plot(rangelambA,RESIDUALSPBM2Y500ms*100 ,color='C0',linewidth=1)
plt.plot(rangelambA,RESIDUALSPBM2Y600ms*100 ,color='C1',linewidth=1)
plt.plot(rangelambA,RESIDUALSPBM2Y700ms*100 ,color='C2',linewidth=1)
plt.plot(rangelambA,RESIDUALSPBM2Y750ms*100 ,color='C3',linewidth=1)
plt.plot(rangelambA,RESIDUALSPBM2Y800ms*100 ,color='C4',linewidth=1)
plt.plot(rangelambA,RESIDUALSPBM2Y900ms*100 ,color='C5',linewidth=1)
plt.plot(rangelambA,RESIDUALSPBM2Y1s*100 ,color='C6',linewidth=1) #*100 to go from m/s to cm/s
plt.plot(rangelambA,RESIDUALSPBM2Y1200ms*100 ,color='C7',linewidth=1)
plt.plot(rangelambA,RESIDUALSPBM2Y10s*100 ,color='C8',linewidth=1)
plt.plot(rangelambA,RESIDUALSPBM2Y50s*100 ,color='C9',linewidth=1)
plt.plot(rangelambA,RESIDUALSPBM2Y100s*100 ,color='C10',linewidth=1)

plt.grid(True)
#plt.yscale('symlog', linthreshy=0.01)
plt.xlabel('Wavelength $\AA$')
plt.ylabel('Residuals (cm/s)')
plt.xlim(3500,10000)
#plt.ylim(-2,100)

#plt.figure()
#time,BaryCorrVel1=np.loadtxt('%s/BCPerExposure250ms.txt'%Directory,usecols=(0,1),unpack=True)
#plt.plot(time,BaryCorrVel1)
#time,BaryCorrVel1=np.loadtxt('%s/BCPerExposure1s.txt'%Directory,usecols=(0,1),unpack=True)
#plt.plot(time,BaryCorrVel1)
#time,BaryCorrVel1=np.loadtxt('%s/BCPerExposure10s.txt'%Directory,usecols=(0,1),unpack=True)
#plt.plot(time,BaryCorrVel1)
#time,BaryCorrVel1=np.loadtxt('%s/BCPerExposure50s.txt'%Directory,usecols=(0,1),unpack=True)
#plt.plot(time,BaryCorrVel1)
#time,BaryCorrVel1=np.loadtxt('%s/BCPerExposure100s.txt'%Directory,usecols=(0,1),unpack=True)
#plt.plot(time,BaryCorrVel1)


#plt.figure()
#plt.plot(rangelambA, FITPBM2Y250ms)
#plt.plot(rangelambA, FITPBM2Y1s)
#plt.plot(rangelambA, FITPBM2Y10s)
#plt.plot(rangelambA, FITPBM2Y50s)
#plt.plot(rangelambA, FITPBM2Y100s)












###############################     Plots     ################################
#plt.figure()
#plt.subplot(211)
#plt.title('Flux Weighted Barycentric Corrections - Residuals (%s)' %HIP_ID)

#plt.plot(0,0,color='C0')
#plt.plot(0,0,color='C4')
#plt.plot(0,0,color='C2')
#plt.plot(0,0,color='C1')

#plt.legend(['G-CLEF (130k channels)','PBM2Y','PBM8Y','PBM18Y'],loc=0)

#plt.plot(WLGCLEFCHANNEL,RBCGCLEF,'.',color='C0',markersize=2)
#plt.plot(WLPBM2Y,RBCPBM2Y,'.',color='C4',markersize=6)
#plt.plot(WLPBM8Y,RBCPBM8Y,'.',color='C2',markersize=6)
#plt.plot(WLPBM18Y,RBCPBM18Y,'.',color='C1',markersize=6)

#plt.grid(True)
##plt.xlabel('Wavelength $\AA$')
#plt.ylabel('Relative BC (m/s)')
#plt.xlim(3500,10000)
##plt.ylim(-0.05,0.55)

#if BlockBandsAfter=='YES' or BlockBandsFilter=='YES':
#    plt.fill_betweenx([-0.1,0.6],6250,6340,color='r',alpha=0.1) ### m/s scale
#    plt.fill_betweenx([-0.1,0.6],6850,6970,color='r',alpha=0.1)
#    plt.fill_betweenx([-0.1,0.6],7141,7380,color='r',alpha=0.1)
#    plt.fill_betweenx([-0.1,0.6],7575,7735,color='r',alpha=0.1)
#    plt.fill_betweenx([-0.1,0.6],8095,8385,color='r',alpha=0.1)
#    plt.fill_betweenx([-0.1,0.6],8905,9890,color='r',alpha=0.1)


#plt.subplot(212)
#plt.plot(rangelambA,RESIDUALSPBM8Y*100 ,color='C2',linewidth=1) #*100 to go from m/s to cm/s
#plt.plot(rangelambA,RESIDUALSPBM18Y*100 ,color='C1',linewidth=1)
#plt.plot(rangelambA,RESIDUALSPBM2Y*100 ,color='C4',linewidth=1)

#plt.grid(True)
#plt.yscale('symlog', linthreshy=0.01)
#plt.xlabel('Wavelength $\AA$')
#plt.ylabel('Residuals (cm/s)')
#plt.xlim(3500,10000)
#plt.ylim(-2,100)

#if BlockBandsAfter=='YES' or BlockBandsFilter=='YES':
#    plt.fill_betweenx([-100,100],6250,6340,color='r',alpha=0.1) ### cm/s scale
#    plt.fill_betweenx([-100,100],6850,6970,color='r',alpha=0.1)
#    plt.fill_betweenx([-100,100],7141,7380,color='r',alpha=0.1)
#    plt.fill_betweenx([-100,100],7575,7735,color='r',alpha=0.1)
#    plt.fill_betweenx([-100,100],8095,8385,color='r',alpha=0.1)
#    plt.fill_betweenx([-100,100],8905,9890,color='r',alpha=0.1)

##lines to indicate 1cm/s##
#plt.plot([3500,10000],[1,1],color='r')
#plt.plot([3500,10000],[-1,-1],color='r')

##lines to indicate 0.1cm/s##
#plt.plot([3500,10000],[0.1,0.1],color='r',linestyle='--')
#plt.plot([3500,10000],[-0.1,-0.1],color='r',linestyle='--')


