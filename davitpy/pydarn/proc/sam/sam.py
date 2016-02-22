"""
*********************
**Module**: pydarn.proc.sam.sam
   :synopsis: Using SuperDARN Assimilative Mapping procedure to calculate fitted 
   convection electric fields, velocities, or potential
*********************

**Class**:
    * :class:`pydarn.proc.sam.sam.SamConv`: 

**Functions**:
    * :func:`pydarn.proc.sam.sam.calcSam`

"""

import numpy as np
import datetime as dt
from pydarn.proc.sam.cs10 import *
from pydarn.proc.sam.basis import *

from pydarn.sdio.sdDataTypes import sdDataPtr,gridData
from pydarn.sdio.sdDataRead  import *    
import gme.ind, utils.timeUtils, models.aacgm

import matplotlib.pyplot as plt

class SamConv():
    """ Basic container for holding proccessed SAM output 

    **Attrs**:
    * **sTime** (`datetime <http://tinyurl.com/bl352yx>`_): start time of the record
    * **eTime** (`datetime <http://tinyurl.com/bl352yx>`_): end time of the record
    * **IMFave* (int): # minutes over which IMF was averaged  
    * **IMFdelay** (int):  # minutes by which IMF data was delayed
    * **IMFBy** (float): the By component of the IMF (nT)
    * **IMFBz** (float): the Bz component of the IMF (nT)
    * **Vsw** (float): the Eartward component of the solar wind velocity (km/s)
    * **tilt** (float): dipole tilt, positive sunward (deg)
    * **hemi** (string): the hemisphere, 'north' or 'south'
    * **fit_coeffs** (1D float array): fitted coefficients of AMIE basis functions
    * **fit_cov** (2D float array): fitted error covariance in terms of AMIE basis functions

    **Methods**:
    * :func:`calcFitVel`
    * :func:`calcFitEfield`
    * :func:`calcFitPot`
    
    **Example**:
        ::
            import datetime as dt
            import numpy as np
            
            samList = pydarn.proc.sam.sam.calcSam(dt.datetime(2011,1,1,12,0),dt.datetime(2011,1,1,12,56),2)

            (lat_grd,lon_grd,vel_th,vel_ph,verr_th,verr_ph) = samList[0].calcFitVel(dlat=2,dlon=10,lat0=50)
            (lat_grd,lon_grd,el_th,el_ph,eerr_th,eerr_ph)   = samList[0].calcFitEfield(dlat=2,dlon=10,lat0=50)
            (lat_grd,lon_grd,pot,perr)                      = samList[0].calcFitPot(dlat=2,dlon=10,lat0=50)
                    
            lat = np.ones(24)*70.
            lon = np.linspace(0,345,24)
            (lat,lon,el_th,el_ph,eerr_th,eerr_ph)   = samList[0].calcFitEfield(lat=lat,lon=lon)

    written by Ellen D. P. Cousins, 2014-08
    """

    def __init__(self,sTime=None,eTime=None,hemi=None,IMFave=None,IMFdelay=None,IMFBy=None,IMFBz=None,\
                     Vsw=None,tilt=None,fit_coeffs=None,fit_cov=None,radEarth=6371.,alt=110.):

        self.sTime = sTime
        self.eTime = eTime
        self.hemi = hemi
        self.IMFave = IMFave
        self.IMFdelay = IMFdelay
        self.IMFBy = IMFBy
        self.IMFBz = IMFBz
        self.Vsw = Vsw
        self.tilt = tilt
        self.fit_coeffs = fit_coeffs
        self.fit_cov = fit_cov
        self.radEarth = radEarth
        self.alt = alt

    def calcFitVel(self,lat=None,lon=None,dlat=None,dlon=None,lat0=None,CS10path=None,bmod=None):
        """Calculate fitted convection velocity and errors from SAM output
        
        **Belongs to**: :class:`SAMConv`

        **Args**:
        * **[lat]** (1D float array): Latitudes at which to calc sol'n           (specify lat & lon or dlat,
        * **[lon]** (1D float array): Longitudes at which to calc sol'n          (dlon, & lat0)
        * **[dlat]** (float): Latitude step of regular grid on which to calc sol'n
        * **[dlon]** (float): Longitude step of regular grid on which to calc sol'n
        * **[lat0]** (float): Equatorward latitude of regular grid on which to calc sol'n
        * **[CS10path]** (str): Directory where CS10 model files can be found
        * **[bmod]** (str): Geomagnetic field model to use when calc'ing vel (can be 'dip' (dipole), 'igrf', 'const')


        **Returns**:
            1D float arrays of Lat,Lon,Fitted velocity (theta (positive equatorward) & phi (positive eastward) components),and errors
        """

        # Wrapper
        (lat,lon,y_fit,y_err) = self.calcFitOutput(lat=lat,lon=lon,dlat=dlat,dlon=dlon,lat0=lat0,
                                              typ='vel',CS10path=CS10path,bmod=bmod)
        
        (vel_th,vel_ph) = y_fit
        (err_th,err_ph) = y_err

        return lat,lon,vel_th,vel_ph,err_th,err_ph

    def calcFitEfield(self,lat=None,lon=None,dlat=None,dlon=None,lat0=None,CS10path=None):
        """Calculate fitted convection electric field and errors from SAM output
        
        **Belongs to**: :class:`SAMConv`

        **Args**: (described under calcFitVel)

        **Returns**:
            1D float arrays of Lat,Lon,Fitted Electric Field (theta (positive equatorward) & phi (positive eastward) components),and errors
        """

        # Wrapper
        (lat,lon,y_fit,y_err) = self.calcFitOutput(lat=lat,lon=lon,dlat=dlat,dlon=dlon,lat0=lat0,
                                              typ='efield',CS10path=CS10path)
        
        (el_th,el_ph) = y_fit
        (err_th,err_ph) = y_err

        return lat,lon,el_th,el_ph,err_th,err_ph

    def calcFitPot(self,lat=None,lon=None,dlat=None,dlon=None,lat0=None,CS10path=None):
        """Calculate fitted convection electric potential and errors from SAM output
        
        **Belongs to**: :class:`SAMConv`

        **Args**: (described under calcFitVel)
        
        **Returns**:
            1D float arrays of Lat,Lon,Fitted potential,and errors
        """

        # Wrapper
        (lat,lon,pot,err) = self.calcFitOutput(lat=lat,lon=lon,dlat=dlat,dlon=dlon,lat0=lat0,
                                              typ='pot',CS10path=CS10path)

        return lat,lon,pot,err


    def calcFitOutput(self,lat=None,lon=None,dlat=None,dlon=None,lat0=None,typ='pot',CS10path=None,bmod=None):
        """Internal function to calculate fitted convection parameters and errors from SAM output

        **Belongs to**: :class:`SAMConv`

        **Args**: (described under calcFitVel)

        **Returns**: (described under other calcFitPot() func's)
        """

        if self.hemi == 'north' :
            hemisphere = 1
        else :
            hemisphere = -1
            
        if (lat != None) or (lon != None):
            assert( (lon != None) and (lat != None) and (len(lat) == len(lon))),\
                'Both lat & lon must be specified & must be same length'
        else:
            assert( (dlon != None) and (dlat != None) and (lat0 != None)),\
                'If lat & lon are not specified, then dlon, dlat, and lat0 must all be specified'
        
        if (lat == None): #Construct regular grid & flatten into lat,lon arrays
            lat0 = hemisphere*abs(lat0)

            nlat = np.floor((hemisphere*90-lat0)/dlat)+1
            nlon = np.floor(360/dlon)
            lats = np.linspace(lat0,hemisphere*90,nlat)
            lons = np.linspace(0,360,nlon)

            (lat2d,lon2d) = np.meshgrid(lats,lons)

            lat = np.ravel(lat2d)
            lon = np.ravel(lon2d)
        
        # Get CS10 model coeffs for given conds
        (sdcoeffs,sdlmin) = calc_cs10_coeffs(self.hemi, self.IMFBy, self.IMFBz, self.Vsw, self.tilt, path=CS10path, silent=1)
       
        # Evaluate SH & AMIE basis functions at desired locations
        if typ == 'efield': 
            x_sh   = eval_efield_sh(lat,lon,sdlmin)
            x_amie = eval_efield_amie(lat,lon)
        elif typ == 'vel':
            x_sh   = eval_vel_sh(lat,lon,sdlmin,bmod=bmod)
            x_amie = eval_vel_amie(lat,lon,bmod=bmod)
        else:
            x_sh   = eval_pot_sh(lat,lon,sdlmin)
            x_amie = eval_pot_amie(lat,lon)

        # Multiply by coeffs
        y_fit = np.dot(x_sh, sdcoeffs) + np.dot(x_amie, self.fit_coeffs)
        
        if typ != 'pot': #Need to separate theta & phi components
            y_fit = (y_fit[0,:], y_fit[1,:])

        # Errors from Matrix multiplication: x_amie x fit_cov x x_amie' 
        if typ == 'pot':
            err_matrix = np.dot(np.dot(x_amie, self.fit_cov), np.transpose(x_amie))
            y_err = np.sqrt(np.diag(err_matrix))
        else: #Need to separate theta & phi components
            err_mat1 = np.dot(np.dot(np.squeeze(x_amie[0,:,:]), self.fit_cov), np.transpose(np.squeeze(x_amie[0,:,:])))
            err_mat2 = np.dot(np.dot(np.squeeze(x_amie[1,:,:]), self.fit_cov), np.transpose(np.squeeze(x_amie[1,:,:])))

            y_err_th = np.sqrt(np.diag(err_mat1))
            y_err_ph = np.sqrt(np.diag(err_mat2))
            y_err    = (y_err_th, y_err_ph)

        return lat,lon,y_fit,y_err



def calcSam(sTime,eTime,deltaT,hemi='north',fileType='grdex',src=None,fileName=None, \
               noCache=False,tablesDir=None):    #remove this keyword custType='grdex' by Xueling 2/19/2016

    """ Calculate best-fit basis function coeffs using SAM procedure

    **Args**:
    * **sTime** (`datetime <http://tinyurl.com/bl352yx>`_): start time of desired interval
    * **eTime** (`datetime <http://tinyurl.com/bl352yx>`_): end time of desired interval
    * **detaT** (int): Time step (resolution) for analysis in mins (should be multiple of 2)
    * **hemi** (string): the hemisphere, 'north' or 'south'
    * **[fileType]** (str):  The type of data you want to read.  valid inputs are: 'grd','grdex'
    * **[src]** (str): the source of the data.  valid inputs are 'local' 'sftp'.  if this is set to None, it will try all possibilites sequentially.  default = None
    * **[fileName]** (str): the name of a specific file which you want to open.  If this is set, we will not look for cached files.  default=None
    * **[custType]** (str): if fileName is specified, the filetype of the file.  default = 'grdex'
    * **[noCache]** (boolean): flag to indicate that you do not want to check first for cached files.  default = False.
    * **[tablesDir]** (str): Directory where SAM EOF files, CS10 model files, & AMIE files can be found
    
    **Returns**:
    Array of SamConv objects, 1 per time step within time interval
    """

    #deltaT must be multiple of 2
    deltaT = max((int(2*np.floor(deltaT/2.)),2))

    #First read in SuperDARN grd data
    sdList = sdGridLoad(sTime,eTime,deltaT,hemi,fileType,src,fileName,noCache,estd=False)  #remove this keyword custType  by Xueling 2/19/2016
    nrecs = len(sdList)

    #Get solar wind etc. conditions, average over 45 min, ending 10 min prior to given time
    IMFdelay = 10
    IMFave   = 45
    (by, bz, vsw, tilt) = get_conds(sTime,eTime,deltaT,IMFave,IMFdelay)

    #Load complete set of CS10 model coefficients in advance to speed up loop
    cs10mod = ModCoeff()

    samList = []
    for nn in range(nrecs):
        # Get CS10 model coeffs
        (sdcoeffs,sdlmin) = calc_cs10_coeffs(hemi,by[nn],bz[nn],vsw[nn],tilt[nn],cs10mod=cs10mod,path=tablesDir,silent=1)

        lat  = np.asarray(sdList[nn].vector.mlat)
        lon  = np.asarray(sdList[nn].vector.mlon)
        vlos = np.asarray(sdList[nn].vector.velmedian)
        verr = np.asarray(sdList[nn].vector.velsd)
        azm  = np.asarray(sdList[nn].vector.kvect)

        if nn == 0: print verr

        # Get into MLT coords
        epoch = utils.timeUtils.datetimeToEpoch(sTime+dt.timedelta(minutes=deltaT/2.))
        mltDef = models.aacgm.mltFromEpoch(epoch,0.0) * 15. 
        # mltDef is the rotation that needs to be applied, and lon is the AACGM longitude.
        # use modulo so new longitude is between 0 & 360
        mlt_lon = np.mod((lon + mltDef), 360.)

        # Do assimilative fit
        (fit_coeffs, fit_cov) = samDoAssim(lat,mlt_lon,vlos,verr,azm,sdcoeffs,sdlmin,sdorder=8)

        sam_i = SamConv(sdList[nn].sTime,sdList[nn].eTime,hemi,IMFave,IMFdelay,by[nn],bz[nn],\
                            vsw[nn],tilt[nn],fit_coeffs,fit_cov)

        samList.append(sam_i)
        
    return samList

            
def samDoAssim(lat,lon,vlos,verr,azm,sdcoeffs,sdlmin,sdorder=8):
    """Internal function to SAM assimilative fit
    """

    # Get coefficients that define EOFs
    eof_coeffs = load_eofs()
    (nbasis,neof) = eof_coeffs.shape
    
    #Construct EOF covariance matrix
    # (diagonal matrix with variance defined by power law:
    #  C[n,n] = a*n^-b , with
    #  power-law parameters a = 80 kV^2, b = 1)
    apl = 80.*1e6 #kV -> V
    bpl = 1.
    n = np.arange(neof)+1
    eof_var = apl*n**(-bpl)
    eof_cov = np.diag(eof_var)
    eof_cov_inv = np.diag(1./eof_var)

    # If no SuperDARN data:
    if len(lat) == 0:
        #Return original error covariance in terms of AMIE basis functions
        fit_cov = np.dot(np.dot(eof_coeffs,eof_cov),np.transpose(eof_coeffs))
        return (np.zeros(nbasis),fit_cov)

    #--------------------------------------------------------------------
    # Convert SuperDARN Vlos to E-field
    # (Should be using IGRF...just using constant instead)
    Bconst = 0.5e-4

    eobs   = vlos*Bconst
    cdev   = np.sin(np.radians(azm))
    sdev   = np.cos(np.radians(azm)) 
    eerr   = verr*Bconst
    
    # Evaluate AMIE basis functions & data locs
    X_org_tp = eval_efield_amie(lat,lon)
    cdev_arr = np.tile(np.reshape(cdev,(len(cdev),1)),(1,nbasis))
    sdev_arr = np.tile(np.reshape(sdev,(len(cdev),1)),(1,nbasis))
    # Dot product to project onto LOS direction
    X_org = cdev_arr*np.squeeze(X_org_tp[0,...]) + sdev_arr*np.squeeze(X_org_tp[1,...])

    Y = eobs
    #--------------------------------------------------------------------
    # Remove Model
    efield_sh = eval_efield_sh(lat,lon,sdlmin,order=sdorder)
    mod_efld = np.dot(efield_sh,sdcoeffs)

    #subtract off model in obs direction (dot product)
    Ymod = (mod_efld[0,:]*cdev+mod_efld[1,:]*sdev)
    Y = Y - Ymod
    
    #--------------------------------------------------------------------
    # Fit EOFs
    EOF = np.dot(X_org,eof_coeffs)

    # Construct obs. error covariance matrix
    npts = len(vlos)
    nfac = npts**(.25)
    err_cov = np.diag(nfac*eerr**2) 
    err_cov_inv = np.diag(1./(nfac*eerr**2))

    # Matrix operations:
    # ((EOF' x err_cov_inv x EOF) + eof_cov_inv)^-1 x (EOF' x err_cov_inv)
    # Eqn 3 in SAM paper
    gain = np.dot(np.linalg.inv(np.dot(np.dot(np.transpose(EOF),err_cov_inv),EOF) + eof_cov_inv), \
                      np.dot(np.transpose(EOF),err_cov_inv))

    #Best-fit EOF coeffs: alpha
    alpha = np.squeeze(np.transpose(np.dot(gain,Y)))

    #Best-fit coeffs in terms of full set of AMIE basis functions
    fit_coeffs = np.dot(eof_coeffs,alpha)

    #Error covariance of fitted EOF coefficents (eqn 5 in SAM paper)
    fit_cov_alp = np.dot((np.eye(neof) - np.dot(gain,EOF)),eof_cov)
    #Error covariance in terms of AMIE basis functions
    fit_cov     = np.dot(np.dot(eof_coeffs,fit_cov_alp),np.transpose(eof_coeffs))
    
    #Yfit = np.dot(EOF,alpha)+Ymod
    #plt.figure
    #plt.plot(lat,eobs,'r+',lat,Ymod,'b+',lat,Yfit,'g+')
    #plt.show()

    return (fit_coeffs,fit_cov)

def get_conds(sTime,eTime,deltaT,IMFave=0,IMFdelay=0,path=None):
    """Internal function to get IMF, solar wind, & dipole tilt conditions for desired times
    """

    from models import tsyganenko as ts

    # Offset time to allow for backward averaging & delay 
    sTimeIMF = sTime-dt.timedelta(minutes=IMFdelay)-dt.timedelta(minutes=(IMFave-1))
    eTimeIMF = eTime-dt.timedelta(minutes=IMFdelay)

    # Get 1-min resolution OMNI data
    omniList = gme.ind.readOmni(sTimeIMF,eTimeIMF,res=1)

    if omniList != None:
        by_in  =  np.asarray([omniI.bym for omniI in omniList],dtype=np.float64)
        bz_in  =  np.asarray([omniI.bzm for omniI in omniList],dtype=np.float64)
        vsw_in = -np.asarray([omniI.vxe for omniI in omniList],dtype=np.float64)

        # Find and replace missing values
        # Replacing with 0 for now, should do something fancier like interpolation
        by_in  = np.nan_to_num(by_in)
        bz_in  = np.nan_to_num(bz_in)
        vsw_in = np.nan_to_num(vsw_in)

        # Do boxcar smoothing - this throws out first floor(IMFave/2) entries
        if IMFave != 0:
            win = np.ones(IMFave)/IMFave
            by_in  = np.convolve(win,by_in,mode='valid')
            bz_in  = np.convolve(win,bz_in,mode='valid')
            vsw_in = np.convolve(win,vsw_in,mode='valid')

        # Adjust for delay
        imf_time  = [(omniI.time + dt.timedelta(minutes=IMFdelay)) for omniI in omniList]
        # And we want backward windowed smoothing, so throw out first IMFave-1 entries
        imf_time  = imf_time[int(IMFave-1):]

    tilt_arr  = []
    by_arr    = []
    bz_arr    = []
    vsw_arr   = []

    iT = sTime
    while iT <= eTime:

        # Get dipole tilt
        ts.tsygFort.recalc_08(iT.year, iT.timetuple().tm_yday, iT.hour, iT.minute, iT.second, -400,0,0)
        itilt = np.degrees(np.arcsin(ts.tsygFort.geopack1.sps))
        tilt_arr.append(itilt)

        if omniList != None:
            # Find OMNI data for time
            try: iIMF = imf_time.index(iT)
            except:
                findT = [(iTimf - iT).seconds for iTimf in imf_time]
                iIMF = np.argmin(np.fabs(np.asarray(findT)/60.))

            by_arr.append(by_in[iIMF])
            bz_arr.append(bz_in[iIMF])
            vsw_arr.append(vsw_in[iIMF])
        else:
            # Couldn't get OMNI data, just use zeros
            by_arr.append(0.)
            bz_arr.append(0.)
            vsw_arr.append(0.)

        iT = iT + dt.timedelta(minutes=deltaT)
    

    return (by_arr, bz_arr, vsw_arr, tilt_arr)


def sdGridLoad(sTime,eTime,deltaT,hemi,fileType,src,fileName,noCache,estd=False): #custType removed by Xueling 2/19/2016
    """Internal function to get SuperDARN grid data for desired times, and adjust time resolution if necessary
    """
    
    # Read in SuperDARN grid data
    myPtr  = sdDataOpen(sTime=sTime,hemi=hemi,eTime=eTime,fileType=fileType,src=src,
                        fileName=fileName,noCache=noCache)  #custType=custType removed by Xueling 2/19/2016
    sdList = sdDataReadAll(myPtr)
    #remove by Xueling on 2/19/2016
    #if not myPtr.ptr.closed:
    #    myPtr.ptr.close()

    dTnative = (sdList[0].eTime - sdList[0].sTime).seconds/60

    if (deltaT > dTnative): #will need to concatentate multiple records
        #Create new list using longer time steps
        sdListNew = []

        nn = 0
        iT = sTime
        while iT <= eTime:
            sdI = gridData()
            sdI.sTime = iT
            sdI.eTime = (iT + dt.timedelta(minutes=deltaT))
            sdI.vector.mlat      = []
            sdI.vector.mlon      = []
            sdI.vector.velmedian = []
            sdI.vector.velsd     = []
            sdI.vector.kvect     = []
            sdI.vector.stid      = []

            # Concatenate entries within larger time window
            while (nn < len(sdList)) and (sdList[nn].eTime <= (iT + dt.timedelta(minutes=deltaT))):
                sdI.vector.mlat.extend(sdList[nn].vector.mlat)
                sdI.vector.mlon.extend(sdList[nn].vector.mlon)
                sdI.vector.velmedian.extend(sdList[nn].vector.velmedian)
                sdI.vector.velsd.extend(sdList[nn].vector.velsd)
                sdI.vector.kvect.extend(sdList[nn].vector.kvect)
                sdI.vector.stid.extend(sdList[nn].vector.stid)
                nn += 1

            sdListNew.append(sdI)
            iT += dt.timedelta(minutes=deltaT)

        sdList = sdListNew

    if estd: # Need to recalulated errors based on regional std dev values
        err_min = 50.
        for sdI in sdList:

            lat = np.asarray(sdI.vector.mlat)
            lon = np.asarray(sdI.vector.mlon)
            vel = np.asarray(sdI.vector.velmedian)
            kaz = np.asarray(sdI.vector.kvect)
            
            rid_arr = np.asarray(sdI.vector.stid)
            rid_set = np.unique(rid_arr)
            for rad in rid_set:
                q = np.where(rid_arr == rad)
                q = q[0]

                for i in range(len(q)):
                    # Find neighbooring data
                    qq = np.where(np.logical_and(abs(lat[q] - lat[q[i]]) <= 1,
                                                 abs(lon[q] - lon[q[i]]) <= 7.5))
                    qq = qq[0]

                    velqq = vel[q[qq]]
                    kazqq = kaz[q[qq]]

                    # Not enough to calc variance
                    if (len(qq) < 5): 
                        sdI.vector.velsd[q[i]] *= 2

                    else:
                        var0 = np.std(velqq) #Simple std dev
                        
                        # Try magnitude variance taking into account variance in directions
                        (mvel,mkaz) = merge_one(velqq,kazqq)
                        if mvel != None:
                            var1 = np.std(velqq/np.cos(np.radians(kazqq-mkaz)))
                            sdI.vector.velsd[q[i]] = min((var0,var1))
                        else: 
                            sdI.vector.velsd[q[i]] = var0

                    #err_min < err < 1000
                    sdI.vector.velsd[q[i]] = max((min((sdI.vector.velsd[q[i]],1000)),err_min))

    return sdList

def merge_one(vlos,kaz):    
    """Internal function to regress for vector velocity magnitude & direction given set of LOS obs
    """
    
    #L-shell fitting

    azm_min =  25.           #azimuth range required
    ang_l   =  azm_min
    ang_u   =  180. - azm_min
    
    vlos_in =  vlos
    kaz_in  =  kaz

    q = np.where(kaz_in < -180.)
    if (len(q) > 0): kaz_in[q] += 360

    q = np.where(kaz_in >  180.)
    if (len(q) > 0): kaz_in[q] -= 360

    # Check for sufficient azm var'n
    i = np.argmin(abs(kaz_in))
    test_arr = 1-np.cos(np.radians(kaz_in-kaz_in[i]))
    if max(test_arr) < (1-np.cos(np.radians(azm_min))):
        return (None,None)

    # Linear regression
    ang_lsh = np.radians(90 - kaz_in)
    sx2  = np.sum(np.sin(ang_lsh)**2)
    cx2  = np.sum(np.cos(ang_lsh)**2)
    cxsx = np.sum(np.cos(ang_lsh)*np.sin(ang_lsh))
    
    ysx  = np.sum(vlos_in*np.sin(ang_lsh))
    ycx  = np.sum(vlos_in*np.cos(ang_lsh))

    den  =  sx2*cx2 - cxsx**2
    vpar = (sx2*ycx - cxsx*ysx)/den
    vper = (cx2*ysx - cxsx*ycx)/den
    
    mvel =  np.sqrt(vpar**2 + vper**2)
    mkaz =  np.degrees(np.arctan2(vpar,vper))

    return (mvel,mkaz)
