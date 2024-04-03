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
Agmt=31.5239187834 ##m2 // Effective collecting are of Clay Magellan (pi x (6.5/2)^2)*0.95; 6.5m diam with 5% obscuration

##############################################################################
##############################################################################
#################                   INPUTS                   #################
##############################################################################
FirstTime1='NO' ## 'YES'/'NO' Is this the first time using any of the following parameters?
HIP_ID = 'HIP90850' ## Object ID from Hiparcus Catalog; use 'HIP****'
HIP_number = 90850  ## Object ID from Hiparcus Catalog; use only the number
mag= 10.0 ## Stellar apparent magnitude
SpectralType = 'M' ## Stellar spectral type: 'F', 'G', 'K', 'M' 
BlockBandsFilter='YES' ## Block selected bands as if there were filters in the system? 
Filter1='NO' ## Filter blocking from 6250 to 6340
Filter2='NO' ## Filter blocking from 6850 to 6970
Filter3='NO' ## Filter blocking from 7141 to 7380
Filter4='NO' ## Filter blocking from 7575 to 7735
Filter5='NO' ## Filter blocking from 8095 to 8385
Filter6='NO' ## Filter blocking from 8905 to 9890
ObservingMode='PRV' ## Observing mode: 'PRV', 'NPRV'
BlockBandsAfter='NO' ## Block selected bands after detection but before fitting?
##############################      Prism      ###############################
BeamSpeed=3 ## Beam speed entering the exposure meter (ex: for F/3 insert only 3)
Diam_fiber=400e-6 ## Entrance fiber diameter in m
Diam_coll=0.025 ## Collimator diameter in m
t_prism=0.025 ## Prism base width in m
r=1 ## Anamorphic magnification factor
PrismIncAngle= 0 ## Prism inclination angle in degree; For 0 deg: prism base is parallel to incoming beam (positive inclination clockwise considering collimator on the left)
Apex=40 ## Prism apex angle in degree; it is considered an isosceles triangle
###############################     CCD     ##################################
DkC=0.1 ## dark current [e/pixel]
pix_size=6.5e-6 ## pixel size in m
RoN=0.9 ## read out noise [e]
##############################################################################
FirstTime2='NO' ## 'YES'/'NO' Is this the first time using any of the following parameters?
JDTime=2459453.726658 ## Begining of the observation in Julian Date
GCLEFtime=1800 ## Time of exposure on G-CLEF in seconds
EXPMETtime=10.0 ## Exposure time of each exposure on the exposure meter in seconds
Directory='Magellan_BC_HIP90850' ## Name of the directory containing the output files
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
    lamb_filter,T_filter=np.loadtxt("DATA/JohnsonV.txt", usecols=(0,1), unpack=True)

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
    Star_V=StarSpec*VFilter  ## Stellar spectrum through Johnson V filter 
    Integral=np.trapz(Star_V,rangelambA) 
    C=Flux*AreaFilter/Integral 
    Star=StarSpec*C/rangelambA ## Spectra in W/m2/A of a "StellarType" Star of magnitude mag and Blackbody Temperature T
    #Star=StarSpec*C  ## Spectra in W/m2 of a "StellarType" Star of magnitude mag and Blackbody Temperature T
      
    return Star

######################   Efficiency of Optical System   ######################
def Eff_Inst(ObservingMode,Star,BlockBands):
    
    Signal=Star*Agmt # Star going into GMT [W]
    
    ###     Transmitance of Optical System    ###

    lambErrorBudget,T_GMT,T_FrontEnd,PRV_Fiber,NPRV_Fiber,InsideSpec=np.loadtxt("DATA/T_OpticalSystem.txt",usecols=(0,1,2,3,4,5),unpack=True)
    ## info in T_OpticalSystem comes from G-CLEF's Throughput budget
    
    T_PRV=T_GMT*T_FrontEnd*PRV_Fiber*InsideSpec ## Efficiency of the system in PRV mode
    T_NPRV=T_GMT*T_FrontEnd*NPRV_Fiber*InsideSpec ## Efficiency of the system in N-PRV mode
    
    T_PRV= np.interp(rangelambA, lambErrorBudget, T_PRV)  ## To fit the grid
    T_NPRV= np.interp(rangelambA, lambErrorBudget, T_NPRV)
    
    if ObservingMode =='PRV':
        Signal=Signal*T_PRV
        
    if ObservingMode =='NPRV':
        Signal=Signal*T_NPRV
        
        
    Signal=Signal*0.001  ## 0.1% of light getting into G-CLEF's echelle grating will be redirected to the exposure meter
    ##Signal [W/A]
    
    WLQE,QE=np.loadtxt('DATA/Zyla44QE.txt',usecols=(0,1),unpack=True) ##CMOS quantum efficiency in specific conditions. We will consider 95% of this number: 
    QE=np.interp(rangelambA,WLQE,QE)
    Signal=Signal*QE*0.01*0.95 ##Signal [W/A]
    
    #####   BLOCKING SELECTED BANDS WITH FILTERS  ####        
    if BlockBands == 'YES':
        i=0
        while i<len(rangelambA):
            if 6250<= rangelambA[i]<=6340 and Filter1=='YES':
                Signal[i]=0
            if 6850<=rangelambA[i]<=6970 and Filter2=='YES':
                Signal[i]=0
            if 7141<=rangelambA[i]<=7380 and Filter3=='YES':
                Signal[i]=0
            if 7575<=rangelambA[i]<=7735 and Filter4=='YES':
                Signal[i]=0
            if 8095<=rangelambA[i]<=8385 and Filter5=='YES':
                Signal[i]=0
            if 8905<=rangelambA[i]<=9890 and Filter6=='YES':
                Signal[i]=0
            i=i+1
            
    rangelambm = rangelambA*1e-10 ##  wavelength in [m]
    E=h*c/rangelambm ##Energy of 1 photon of wavelength lambda
    Counts=Signal/E ## Counts: [photons s-1 A-1]
        
    return Counts   

##  Calculation of flux on each channel and airmass during the observation  ##   
def ExpMet(Glass,Signal,JDo,targetname,GCLEFtime,EXPMETtime):
    ## JDo is the time the observation starts
    ## We are considering a G-CLEF observation of 'GCLEFtime' in seconds 
    ## Each exposure of the exposure meter of 'EXPMETtime' in seconds
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
    if Glass!= 'GCLEF':
        Noise=np.zeros(shape=(NumbExp,channels)) #creating an array with the same size of SumFluxChannel
        Sigma=np.zeros(shape=(NumbExp,channels)) ## creating an array with the same size of SumfluxChannel
        TotalFlux=np.zeros(shape=(NumbExp,channels)) 
        SNR=np.zeros(shape=(NumbExp,channels)) 
    
    if (int(Diam_fiber*r/(pix_size))-(Diam_fiber*r/(pix_size)))==0:
        PIX=int(Diam_fiber*r/(pix_size)) ## pixels/channel 
    else:
        PIX=int(Diam_fiber*r/(pix_size))+1 ## pixels/channel 
    
    AirmassArray=np.zeros(NumbExp)
    
    while t<NumbExp: 
        
        ch=0 ## index over number of channels
        JDMean=JD+((1/86400)*EXPMETtime)/2 ## JDMean - mean time of exposure
        time = Time(format='jd',val=JDMean,scale='tt')
        TargetCoordAltAz = TargetCoord.transform_to(AltAz(obstime=time,location=LCO_GMT))
        z=90 - TargetCoordAltAz.alt.degree ## Zenith angle in degree
        z=z*np.pi/180 ## Zenith angle in rad
        X = (1/np.cos(z))*(1-0.0012*((1/np.cos(z))**2-1)) ### Airmass       
        
        T_atm=Transmittance(X)
        F=Signal*EXPMETtime*T_atm  ## Signal=[counts/sA] now F= [counts/A]
        
        FChannels,channels=DevideInChannels(Glass,F)
      
        AirmassArray[t]=X
        
        while ch<channels:
            SumFluxChannel[t][ch]=np.trapz(FChannels[ch],ChannelLamb[ch]) #integrating over wavelength resulting in [counts]
            if Glass != 'GCLEF':
                Sigma[t][ch]=np.sqrt(SumFluxChannel[t][ch] + DkC*EXPMETtime*PIX + (RoN)**2)
                s=np.random.normal(SumFluxChannel[t][ch],Sigma[t][ch],100)
                Noise[t][ch]=np.std(s)
                TotalFlux[t][ch]=SumFluxChannel[t][ch]+Noise[t][ch]
                SNR[t][ch]=SumFluxChannel[t][ch]/Noise[t][ch]
            ch=ch+1
        
        if t/100-int(t/100)==0:print('Exposure',t, 'of',NumbExp)
        
        t=t+1
        JD=JD+((1/86400)*EXPMETtime)
    if Glass != 'GCLEF':
        return SumFluxChannel,Noise,Sigma,TotalFlux,SNR,AirmassArray
    else:    
        return SumFluxChannel,AirmassArray
##############   Definition number of lines per channel for Prisms    #########
def LinesPerChannel(Glass):
    WL,D_PBM2Y,D_PBM8Y,D_PBM18Y=np.loadtxt("DATA/OHARAGlasses/DispersionOHARAGlasses.txt",usecols=(0,1,2,3),unpack=True,skiprows=1) ## Dispersion in (mu m)^-1

    AppertureAng=2*math.atan(1/(2*BeamSpeed)) ## Apperture angle given a beam speed in [rad]
    beta=Apex/2+PrismIncAngle ## degree
    beta=beta*math.pi/180 ## rad
    BeamDiamPrism=(1/math.cos(beta))*Diam_coll  ## Size of beam projected on prism face
    
    if Glass=='PBM2Y':
        D_PBM2Y=np.interp(rangelambA,WL,D_PBM2Y) ## Glass dispersion
        Resolution=-1*rangelambA*D_PBM2Y*Diam_coll*t_prism/(r*AppertureAng*BeamDiamPrism*Diam_fiber*10000)   ## Prism resolution
        DeltaLamb=rangelambA/Resolution
        filename='%s/LinesPerChannelPBM2Y.txt'%Directory
        f= open(filename,"w+")
    if Glass=='PBM8Y':
        D_PBM8Y=np.interp(rangelambA,WL,D_PBM8Y)
        Resolution=-1*rangelambA*D_PBM8Y*Diam_coll*t_prism/(r*AppertureAng*BeamDiamPrism*Diam_fiber*10000)
        DeltaLamb=rangelambA/Resolution
        filename='%s/LinesPerChannelPBM8Y.txt'%Directory
        f= open(filename,"w+")
    if Glass=='PBM18Y':
        D_PBM18Y=np.interp(rangelambA,WL,D_PBM18Y)
        Resolution=-1*rangelambA*D_PBM18Y*Diam_coll*t_prism/(r*AppertureAng*BeamDiamPrism*Diam_fiber*10000)
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
    return

####################   Definition of Channels for Prisms    ##################
def DevideInChannels(Glass,Array2BDevided):
    if Glass=='PBM2Y':
        LinesPerChannel(Glass)
        filename='%s/LinesPerChannelPBM2Y.txt'%Directory
        ch,lines=np.loadtxt(filename,usecols=(0,1),unpack=True)
        
    if Glass=='PBM8Y':
        LinesPerChannel(Glass)
        filename='%s/LinesPerChannelPBM8Y.txt'%Directory
        ch,lines=np.loadtxt(filename,usecols=(0,1),unpack=True)
        
    if Glass=='PBM18Y':
        LinesPerChannel(Glass)
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
    filename='%s/BCPerExposure.txt'%Directory
    f= open(filename,"w+")
    while i<NumbExp:
        JDTimeMean=JDTime+((1/86400)*EXPMETtime)/2 ## JDTimeMean - mean time of exposure
        vel,warning,flag = get_BC_vel(JDUTC=JDTimeMean,hip_id=HIP_ID,obsname='LCO',ephemeris='de430',zmeas=0)
        velocities[i]=vel
        f.write("%f    %f\r\n" % (JDTimeMean,vel))
        x[i]=i
        if i/100-int(i/100)==0:print('BC for exposure',i,'of',NumbExp)
        i=i+1
        JDTime=JDTime+((1/86400)*EXPMETtime)    
    f.close()
    return

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
    filename='%s/BCPerExposure.txt'%Directory
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
    
########## REVISE THE FOLLOWING SECTION - DON'T THINK IT IS EFFECTIVE  #######    
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

##############################################################################
##############################################################################

if FirstTime1=='YES' or FirstTime2=='YES':
    Star=StellarSpec(SpectralType,mag) ## Chosen star spectrum
    
if FirstTime1=='YES' and FirstTime2=='YES':
    os.mkdir(Directory)
    filename='%s/Header.txt'%Directory
    f= open(filename,"w+")
    f.write('##############################################################################\r\n')
    f.write('#################                   INPUTS                   #################\r\n')
    f.write('##############################################################################\r\n')
    f.write('FirstTime1=%s ## Is this the first time using all the following parameters?\r\n'%FirstTime1)
    f.write('HIP_ID = %s ## Object ID from Hiparcus Catalog\r\n'%HIP_ID)
    f.write('mag= %f ## Stellar apparent magnitude\r\n'%mag)
    f.write('SpectralType = %s ## Stellar spectral type: F, G, K, M \r\n'%SpectralType)
    f.write('BlockBandsFilter= %s ## Block selected bands as if there were filters in the system?\r\n'%BlockBandsFilter)
    f.write('Filter1=%s ## Filter blocking from 6250 to 6340'%Filter1)
    f.write('Filter2=%s ## Filter blocking from 6850 to 6970'%Filter2)
    f.write('Filter3=%s ## Filter blocking from 7141 to 7380'%Filter3)
    f.write('Filter4=%s ## Filter blocking from 7575 to 7735'%Filter4)
    f.write('Filter5=%s ## Filter blocking from 8095 to 8385'%Filter5)
    f.write('Filter6=%s ## Filter blocking from 8905 to 9890'%Filter6)
    f.write('ObservingMode= %s ## Observing mode: PRV, NPRV\r\n'%ObservingMode)
    f.write('BlockBandsAfter=%s ## Block selected bands after detection but before fitting?\r\n'%BlockBandsAfter)
    f.write('##############################      Prism      ###############################\r\n')
    f.write('BeamSpeed=%f ## Beam speed entering the exposure meter (ex: for F/3 insert only 3)\r\n'%BeamSpeed)
    f.write('Diam_fiber=%f ## Entrance fiber diameter in m\r\n'%Diam_fiber)
    f.write('Diam_coll=%f ## Collimator diameter in m\r\n'%Diam_coll)
    f.write('t_prism=%f ## Prism base width in m\r\n'%t_prism)
    f.write('r=%f ## Anamorphic magnification factor\r\n'%r)
    f.write('PrismIncAngle= %f ## Prism inclination angle in degree; For 0 deg: prism base is parallel to incoming beam (inclination clockwise considering collimator on the left)\r\n'%PrismIncAngle)
    f.write('Apex=%f ## Prism apex angle in degree; it is considered an isosceles triangle\r\n'%Apex)
    f.write('##################################    CCD    #################################\r\n')
    f.write('DkC=%f ## dark current [e/pixel]\r\n'%DkC)
    f.write('pix_size=%f ## pixel size in m\r\n'%pix_size)    
    f.write('RoN=%f ## read out noise [e]\r\n'%RoN)
    f.write('##############################################################################\r\n')
    f.write('FirstTime2=%s ## Is this the first time using all the following parameters?\r\n'%FirstTime2)
    f.write('JDTime=%f ## Begining of the observation in Julian Date\r\n'%JDTime)
    f.write('GCLEFtime=%f## Time of exposure on G-CLEF in seconds\r\n'%GCLEFtime)
    f.write('EXPMETtime=%f ## Exposure time of each exposure on the exposure meter in seconds\r\n'%EXPMETtime)
    f.write('Directory=%s ## Name of the directory containing the output files\r\n'%Directory)
    f.write('##############################################################################\r\n')
    f.close()

if FirstTime2=='YES':
    filename = "%s/FluxGCLEF" % Directory
    Signal=Eff_Inst(ObservingMode ,Star, 'NO') ## After passing through optical train [photons s-1 A-1]
    SumFluxGCLEFChannel,AirmassArray=ExpMet('GCLEF',Signal,JDTime,HIP_ID,GCLEFtime,EXPMETtime)
    np.save(filename,SumFluxGCLEFChannel)
    BaryCorr(GCLEFtime, EXPMETtime, JDTime, HIP_number)
    BCGCLEF,WLGCLEFCHANNEL=BaryCorrPerChannel('GCLEF',SumFluxGCLEFChannel,'NO')
    np.save('%s/BCGCLEF' % Directory ,BCGCLEF)
    np.save('%s/WLGCLEF' % Directory, WLGCLEFCHANNEL)
else:    
    SumFluxGCLEFChannel=np.load('%s/FluxGCLEF.npy' % Directory)
    BCGCLEF=np.load('%s/BCGCLEF.npy' % Directory)
    WLGCLEFCHANNEL=np.load('%s/WLGCLEF.npy' % Directory)

if FirstTime1 == 'YES':
    Signal=Eff_Inst(ObservingMode ,Star, BlockBandsFilter) ## After passing through optical train [photons s-1 A-1]
    SumFluxPBM2Y,NoisePBM2Y,SigmaPBM2Y,TotalFluxPBM2Y,SNRPBM2Y,AirmassArray=ExpMet('PBM2Y',Signal,JDTime,HIP_ID,GCLEFtime,EXPMETtime)
    np.save("%s/FluxPBM2Y" % Directory,SumFluxPBM2Y)
    np.save('%s/NoisePBM2Y'%Directory, NoisePBM2Y)
    np.save('%s/SigmaPBM2Y'%Directory, SigmaPBM2Y)
    np.save('%s/TotalFluxPBM2Y'%Directory, TotalFluxPBM2Y)
    np.save('%s/SNRPBM2Y'%Directory, SNRPBM2Y)
    BCPBM2Y,WLPBM2Y=BaryCorrPerChannel('PBM2Y',TotalFluxPBM2Y ,BlockBandsAfter)
    np.save('%s/BCPBM2Y' % Directory ,BCPBM2Y)
    np.save('%s/WLPBM2Y' % Directory, WLPBM2Y)
    
#    SumFluxPBM8Y,NoisePBM8Y,SigmaPBM8Y,TotalFluxPBM8Y,SNRPBM8Y,AirmassArray=ExpMet('PBM8Y',Signal,JDTime,HIP_ID,GCLEFtime,EXPMETtime)
#    np.save("%s/FluxPBM8Y" % Directory,SumFluxPBM8Y)
#    np.save('%s/NoisePBM8Y'%Directory, NoisePBM8Y)
#    np.save('%s/SigmaPBM8Y'%Directory, SigmaPBM8Y)
#    np.save('%s/TotalFluxPBM8Y'%Directory, TotalFluxPBM8Y)
#    np.save('%s/SNRPBM8Y'%Directory, SNRPBM8Y)
#    BCPBM8Y,WLPBM8Y=BaryCorrPerChannel('PBM8Y',TotalFluxPBM8Y,BlockBandsAfter)
#    np.save('%s/BCPBM8Y' % Directory ,BCPBM8Y)
#    np.save('%s/WLPBM8Y' % Directory, WLPBM8Y)
    
#    SumFluxPBM18Y,NoisePBM18Y,SigmaPBM18Y,TotalFluxPBM18Y,SNRPBM18Y,AirmassArray=ExpMet('PBM18Y',Signal,JDTime,HIP_ID,GCLEFtime,EXPMETtime)
#    np.save("%s/FluxPBM18Y" % Directory,SumFluxPBM18Y)
#    np.save('%s/NoisePBM18Y'%Directory, NoisePBM18Y)
#    np.save('%s/SigmaPBM18Y'%Directory, SigmaPBM18Y)
#    np.save('%s/TotalFluxPBM18Y'%Directory, TotalFluxPBM18Y)
#    np.save('%s/SNRPBM18Y'%Directory, SNRPBM18Y)
#    BCPBM18Y,WLPBM18Y=BaryCorrPerChannel('PBM18Y',TotalFluxPBM18Y,BlockBandsAfter)
#    np.save('%s/BCPBM18Y' % Directory ,BCPBM18Y)
#    np.save('%s/WLPBM18Y' % Directory, WLPBM18Y)
else:  
    SumFluxPBM2Y=np.load("%s/FluxPBM2Y.npy" % Directory) 
    NoisePBM2Y=np.load('%s/NoisePBM2Y.npy'%Directory)
    SigmaPBM2Y=np.load('%s/SigmaPBM2Y.npy'%Directory)
    TotalFluxPBM2Y=np.load('%s/TotalFluxPBM2Y.npy'%Directory)
    SNRPBM2Y=np.load('%s/SNRPBM2Y.npy'%Directory)
    BCPBM2Y=np.load('%s/BCPBM2Y.npy' % Directory)
    WLPBM2Y=np.load('%s/WLPBM2Y.npy' % Directory)
    
#    SumFluxPBM8Y=np.load("%s/FluxPBM8Y.npy" % Directory) 
#    NoisePBM8Y=np.load('%s/NoisePBM8Y.npy'%Directory)
#    SigmaPBM8Y=np.load('%s/SigmaPBM8Y.npy'%Directory)
#    TotalFluxPBM8Y=np.load('%s/TotalFluxPBM8Y.npy'%Directory)
#    SNRPBM8Y=np.load('%s/SNRPBM8Y.npy'%Directory)
#    BCPBM8Y=np.load('%s/BCPBM8Y.npy' % Directory)
#    WLPBM8Y=np.load('%s/WLPBM8Y.npy' % Directory)
    
#    SumFluxPBM18Y=np.load("%s/FluxPBM18Y.npy" % Directory)
#    NoisePBM18Y=np.load('%s/NoisePBM18Y.npy'%Directory)
#    SigmaPBM18Y=np.load('%s/SigmaPBM18Y.npy'%Directory)
#    TotalFluxPBM18Y=np.load('%s/TotalFluxPBM18Y.npy'%Directory)
#    SNRPBM18Y=np.load('%s/SNRPBM18Y.npy'%Directory)
#    BCPBM18Y=np.load('%s/BCPBM18Y.npy' % Directory)
#    WLPBM18Y=np.load('%s/WLPBM18Y.npy' % Directory)

#######   Search for minimum signal-to-noise ratio on our detections   #######
MinSN_PBM2Y=np.amin(SNRPBM2Y)
#MinSN_PBM8Y=np.amin(SNRPBM8Y)
#MinSN_PBM18Y=np.amin(SNRPBM18Y)
print('Minimum S/N - PBM2Y:%f' %MinSN_PBM2Y)
#print('Minimum S/N - PBM8Y:%f' %MinSN_PBM8Y)
#print('Minimum S/N - PBM18Y:%f' %MinSN_PBM18Y)
ii=np.where(SNRPBM2Y == MinSN_PBM2Y)
print('Signal:', SumFluxPBM2Y[ii])
print('Random Noise:',NoisePBM2Y[ii])
print('Calculated Sigma:', SigmaPBM2Y[ii])

filename='%s/Output.txt'%Directory
f= open(filename,"w+")
f.write('##############################      OUTPUT     ###############################\r\n')
f.write('Minimum S/N - PBM2Y:%f \r\n' %MinSN_PBM2Y)
f.write('Signal:%f \r\n' %SumFluxPBM2Y[ii])
f.write('Calculated sigma:%f \r\n'%SigmaPBM2Y[ii])
f.write('Random noise:%f \r\n'%NoisePBM2Y[ii])
f.close()

###################   Relative Barycentric correction   ######################
DeltaBC=min(BCPBM2Y)
RBCPBM2Y=BCPBM2Y-DeltaBC #relative barycentric correction
#RBCPBM8Y=BCPBM8Y-DeltaBC
#RBCPBM18Y=BCPBM18Y-DeltaBC
RBCGCLEF=BCGCLEF-DeltaBC

##########  Linear fitting of barycentric correction vs wavelength ###########
FITPBM2Y=np.interp(rangelambA,WLPBM2Y,RBCPBM2Y)
#FITPBM8Y=np.interp(rangelambA,WLPBM8Y,RBCPBM8Y)
#FITPBM18Y=np.interp(rangelambA,WLPBM18Y,RBCPBM18Y)  
FITGCLEF=np.interp(rangelambA,WLGCLEFCHANNEL,RBCGCLEF)


########################## Residuals calculation #############################
RESIDUALSPBM2Y=FITGCLEF-FITPBM2Y
#RESIDUALSPBM8Y=FITGCLEF-FITPBM8Y
#RESIDUALSPBM18Y=FITGCLEF-FITPBM18Y

###############################     Plots     ################################
plt.figure()
plt.subplot(211)
plt.title('Flux Weighted Barycentric Corrections - Residuals (%s)' %HIP_ID)

plt.plot(0,0,color='gray')
plt.plot(0,0,color='k')
#plt.plot(0,0,color='C2')
#plt.plot(0,0,color='C1')

#plt.legend(['G-CLEF','PBM2Y','PBM8Y','PBM18Y'],loc=1)
plt.legend(['G-CLEF','PBM2Y'],loc=1)


plt.plot(WLGCLEFCHANNEL,RBCGCLEF,'.',color='gray',markersize=2)
plt.errorbar(WLPBM2Y,RBCPBM2Y,yerr=SigmaPBM2Y)
#plt.plot(WLPBM2Y,RBCPBM2Y,'.',color='k',markersize=6)
#plt.plot(WLPBM8Y,RBCPBM8Y,'.',color='C2',markersize=6)
#plt.plot(WLPBM18Y,RBCPBM18Y,'.',color='C1',markersize=6)
plt.plot(rangelambA,FITPBM2Y)

plt.grid(True)
#plt.xlabel('Wavelength $\AA$')
plt.ylabel('Relative BC (m/s)')
plt.xlim(3500,10000)
plt.ylim(-0.05,0.55)

BlockBandsAfter='YES'

if BlockBandsAfter=='YES' or BlockBandsFilter=='YES':
    plt.fill_betweenx([-0.1,0.6],6250,6340,color='r',alpha=0.1) ### m/s scale
    plt.fill_betweenx([-0.1,0.6],6850,6970,color='r',alpha=0.1)
    plt.fill_betweenx([-0.1,0.6],7141,7380,color='r',alpha=0.1)
    plt.fill_betweenx([-0.1,0.6],7575,7735,color='r',alpha=0.1)
    plt.fill_betweenx([-0.1,0.6],8095,8385,color='r',alpha=0.1)
    plt.fill_betweenx([-0.1,0.6],8905,9890,color='r',alpha=0.1)
    if Filter1=='YES':
         plt.fill_betweenx([-0.1,0.6],6250,6340,color='r',alpha=0.2) ### m/s scale
    if Filter2=='YES':
        plt.fill_betweenx([-0.1,0.6],6850,6970,color='r',alpha=0.2)
    if Filter3=='YES':
        plt.fill_betweenx([-0.1,0.6],7141,7380,color='r',alpha=0.2)
    if Filter4=='YES':
        plt.fill_betweenx([-0.1,0.6],7575,7735,color='r',alpha=0.2)
    if Filter5=='YES':
        plt.fill_betweenx([-0.1,0.6],8095,8385,color='r',alpha=0.2)
    if Filter6=='YES':
        plt.fill_betweenx([-0.1,0.6],8905,9890,color='r',alpha=0.2)
    

plt.subplot(212)
#plt.plot(rangelambA,RESIDUALSPBM8Y*100 ,color='C2',linewidth=1) #*100 to go from m/s to cm/s
#plt.plot(rangelambA,RESIDUALSPBM18Y*100 ,color='C1',linewidth=1)
plt.plot(rangelambA,RESIDUALSPBM2Y*100 ,color='k',linewidth=1)

plt.grid(True)
plt.yscale('symlog', linthreshy=0.01)
plt.xlabel('Wavelength $\AA$')
plt.ylabel('Residuals (cm/s)')
plt.xlim(3500,10000)
plt.ylim(-20,100)

if BlockBandsAfter=='YES' or BlockBandsFilter=='YES':
    plt.fill_betweenx([-100,100],6250,6340,color='r',alpha=0.1) ### cm/s scale
    plt.fill_betweenx([-100,100],6850,6970,color='r',alpha=0.1)
    plt.fill_betweenx([-100,100],7141,7380,color='r',alpha=0.1)
    plt.fill_betweenx([-100,100],7575,7735,color='r',alpha=0.1)
    plt.fill_betweenx([-100,100],8095,8385,color='r',alpha=0.1)
    plt.fill_betweenx([-100,100],8905,9890,color='r',alpha=0.1)
    if Filter1=='YES':
        plt.fill_betweenx([-100,100],6250,6340,color='r',alpha=0.2) ### cm/s scale
    if Filter2=='YES':
        plt.fill_betweenx([-100,100],6850,6970,color='r',alpha=0.2)
    if Filter3=='YES':
        plt.fill_betweenx([-100,100],7141,7380,color='r',alpha=0.2)
    if Filter4=='YES':
        plt.fill_betweenx([-100,100],7575,7735,color='r',alpha=0.2)
    if Filter5=='YES':
        plt.fill_betweenx([-100,100],8095,8385,color='r',alpha=0.2)
    if Filter6=='YES':
        plt.fill_betweenx([-100,100],8905,9890,color='r',alpha=0.2)


##lines to indicate 1cm/s##
plt.plot([3500,10000],[1,1],color='r')
plt.plot([3500,10000],[-1,-1],color='r')

##lines to indicate 0.1cm/s##
plt.plot([3500,10000],[0.1,0.1],color='r',linestyle='--')
plt.plot([3500,10000],[-0.1,-0.1],color='r',linestyle='--')


