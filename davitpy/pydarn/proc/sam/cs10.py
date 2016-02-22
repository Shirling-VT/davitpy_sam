"""
*********************
**Module**: pydarn.proc.sam.cs10
   :synopsis: Functions for loading & calculating CS10 statistical model coefficients
*********************

**Functions**:
    * :func:`pydarn.proc.sam.cs10.calc_coeff`

**Classes**:
    * :class:`pydarn.proc.sam.cs10.ModCoeff` : Container for set of stat. model coefficients

    written by Ellen D. P. Cousins, 2014-08
"""

import numpy as np
import scipy,os,glob

class ModCoeff():
    """This class is the basic container for holding & loading the set of statistical model coefficients.
    """

    def __init__(self,model='CS10',path=None):
        self.model = model
        self.order = None
        self.coeff_arr = None
        self.latmin_arr = None

        if model is 'CS10':
            self.load_cs10(path)
        else:
            print 'Only model currently supported is CS10'

    def load_cs10(self,path=None):
        if path == None:
            path = os.environ['DAVITPY'] + '/tables/model/'

        self.model = 'CS10'
        self.order = 8

        # Parameters for CS10 model
        hem_strs = ['north','south']
        tlt_strs = ['DP-','DP0','DP+']
        mag_strs = ['0.00t1.20','1.20t1.70','1.70t2.20','2.20t2.90','2.90t4.10','4.10t20.00']
        ang_strs = ['Bz+','Bz+_By+','By+','Bz-_By+','Bz-','Bz-_By-','By-','Bz+_By-']

        prefix = 'omni'
        ext    = '.bsph'
        order  = self.order

        coeff_arr = np.ndarray(shape=(len(hem_strs),len(tlt_strs),len(mag_strs),len(ang_strs),(order+1)**2))
        latmin_arr = np.ndarray(shape=(len(hem_strs),len(tlt_strs),len(mag_strs),len(ang_strs)))

        # Loop thru all categories & read files
        for hh in range(len(hem_strs)):
            hemi = hem_strs[hh]
            for tt in range(len(tlt_strs)):
                tlt = tlt_strs[tt]
                for mm in range(len(mag_strs)):
                    mag = mag_strs[mm]
                    for aa in range(len(ang_strs)):
                        ang = ang_strs[aa]
                        
                        fname =  path+prefix+'.'+hemi+'_'+mag+'_'+ang+'_'+tlt+ext 
                        if(not os.path.isfile(fname)):
                            print 'problem reading',fname,':file does not exist'
                            return None

                        f = open(fname,'r')
                        for i in range(4): #Skip top 4 lines
                            f.readline()
                        res = f.readline().split()
                        lat0 = float(res[0])
                        f.close()
                        resarr = np.loadtxt(fname,skiprows=6)
                        
                        coeff_arr[hh,tt,mm,aa,:] = resarr[:,2]
                        latmin_arr[hh,tt,mm,aa]  = lat0
        
        self.coeff_arr = coeff_arr
        self.latmin_arr = latmin_arr


def calc_cs10_coeffs(hemi,by,bz,vsw,tilt,cs10mod=None,path=None,silent=0):
    """Calculate sph. har. coeffs for given conditions by interpolating between categories
    
    **Args**:
        * **hemi** (str) Hemisphere, can be 'north' or 'south'
        * **by**   (float) IMF By component in GSM coordinates in nT
        * **bz**   (float) IMF Bz component in GSM coordinates in nT
        * **vsw**  (float) Earthward component of solar wind in km/s
        * **tilt** (float) Earth's dipole tilt in deg

    **Returns**:
        * Array of spherical harmonic coeffs & equatorward latitude limit
     

    **Example**:
    (coeffs,latmin) = pydarn.proc.sam.cs10.calc_cs10_coeffs('north',2,2,400,5)
    """

    if cs10mod == None:
        cs10mod = ModCoeff() #populate with complete set of CS10 coeffs

    if hemi == 'north':
        hh = 0
    else:
        hh = 1

    coeff_arr  = np.squeeze(cs10mod.coeff_arr[hh])
    latmin_arr = np.squeeze(cs10mod.latmin_arr[hh])

    #Set up definitions of model categories
    mag_l_arr =   np.array([0, 1.2, 1.7, 2.2, 2.9, 4.1])
    mag_u_arr =   np.array([1.2, 1.7, 2.2, 2.9, 4.1,20])
    n_mags    =   len(mag_l_arr)

    mlow  = .5*(mag_l_arr[0:(n_mags-1)]+mag_u_arr[0:(n_mags-1)])
    mhgh  = .5*(mag_l_arr[1:n_mags]+mag_u_arr[1:n_mags])
    mhgh[n_mags-2] = 7.5

    ang_low_arr =  np.array([-25, 25, 70, 110, 155, 205, 250, 290])
    ang_hgh_arr =  np.array([25, 70, 110, 155, 205, 250, 290, 335])
    ang_ref     =  (ang_low_arr + ang_hgh_arr)/2.
    n_angs      =  len(ang_ref)

    alow  = ang_ref
    ahgh  = np.append(ang_ref[1:n_angs],ang_ref[0]+360)
   
    tilt_l_arr = np.array([-35,-10, 10])
    tilt_u_arr = np.array([-10, 10, 35])
    n_tilts = len(tilt_l_arr)
    tlow  = np.array([-20, 0])
    thgh  = np.array([ 0, 20])

    # Prep input condition data
    bt   = np.sqrt(by**2 + bz**2)
    cang = np.degrees(np.arctan2(by,bz))
    mag   = bt*vsw*1.e-3

    # angle ranges from bottom of first bin to top of last bin
    if (cang > (ahgh[n_angs-1])):
        cang = cang-360.
    if (cang < (alow[0])):
        cang = cang+360.

    if hh == 1:
        tilt = -tilt   #swap sign for shemi

    # Find which categories bracket input condition
    q = np.where(np.logical_and((tlow <= tilt),(thgh > tilt)))[0]
    if len(q) != 0:
        it = q[0]
    else: 
        if tilt > thgh[n_tilts-2]:
            it = n_tilts - 2
            tilt = thgh[it]
        else:
            it = 0
            tilt = tlow[it]

    
    q = np.where(np.logical_and(cang >= alow, cang < ahgh))[0]
    if len(q) == 0 :
        print 'unable to classify clk angle, setting to 0'
        cang = 0
        ia = 0
    else:
        ia = q[0]

    q = np.where(np.logical_and(mag >= mlow, mag < mhgh))[0]
    if len(q) != 0 : 
        im = q[0]
    else:
        if mag >= mhgh[n_mags-2] :
            im = n_mags - 2
            mag = mhgh[im]
        elif mag <= mlow[0] :
            im = 0
            mag = mlow[im]
        if silent != 1 :
            print 'model saturated at Esw= ',mag,' mV/m'


    if ((im == (n_mags-2)) and (ia >= 2 and ia <= 5 and not (ia == 2 and cang == alow[ia]))) :
         #top mag bin, Bz<0 no defined, use next lower mag bin
        im = n_mags - 3
        mag = mhgh[im]
    
        if silent != 1 :
            print 'model saturated at Esw= ',mag,' mV/m'
    
    # Nonlinear relationship with ang -> linear in sin(ang/2)
    if (ia == n_angs-1):
        ia2 = 0
    else:
        ia2 = ia+1

    afac_h = abs(np.sin(np.radians(ahgh[ia])/2.))
    afac_l = abs(np.sin(np.radians(alow[ia])/2.))
    afac   = abs(np.sin(np.radians(cang)/2.))

    # Do tri-linear interpolation
    denom = (afac_h-afac_l)*(mhgh[im]-mlow[im])*(thgh[it]-tlow[it])
    A = np.squeeze(coeff_arr[it,im,  ia, :]/denom)
    B = np.squeeze(coeff_arr[it,im,  ia2,:]/denom)
    C = np.squeeze(coeff_arr[it,im+1,ia, :]/denom)
    D = np.squeeze(coeff_arr[it,im+1,ia2,:]/denom)
    E = np.squeeze(coeff_arr[it+1,im,  ia, :]/denom)
    F = np.squeeze(coeff_arr[it+1,im,  ia2,:]/denom)
    G = np.squeeze(coeff_arr[it+1,im+1,ia, :]/denom)
    H = np.squeeze(coeff_arr[it+1,im+1,ia2,:]/denom)

    coeffs = A*(afac_h-afac)*(mhgh[im]-mag)*(thgh[it]-tilt) \
           + B*(afac-afac_l)*(mhgh[im]-mag)*(thgh[it]-tilt) \
           + C*(afac_h-afac)*(mag-mlow[im])*(thgh[it]-tilt) \
           + D*(afac-afac_l)*(mag-mlow[im])*(thgh[it]-tilt) \
           + E*(afac_h-afac)*(mhgh[im]-mag)*(tilt-tlow[it]) \
           + F*(afac-afac_l)*(mhgh[im]-mag)*(tilt-tlow[it]) \
           + G*(afac_h-afac)*(mag-mlow[im])*(tilt-tlow[it]) \
           + H*(afac-afac_l)*(mag-mlow[im])*(tilt-tlow[it])

    #Interpolate HMB lat also
    A = float(latmin_arr[it,im,  ia])/denom
    B = float(latmin_arr[it,im,  ia2])/denom
    C = float(latmin_arr[it,im+1,ia])/denom
    D = float(latmin_arr[it,im+1,ia2])/denom
    E = float(latmin_arr[it+1,im,  ia])/denom
    F = float(latmin_arr[it+1,im,  ia2])/denom
    G = float(latmin_arr[it+1,im+1,ia])/denom
    H = float(latmin_arr[it+1,im+1,ia2])/denom

    latmin = A*(afac_h-afac)*(mhgh[im]-mag)*(thgh[it]-tilt) \
           + B*(afac-afac_l)*(mhgh[im]-mag)*(thgh[it]-tilt) \
           + C*(afac_h-afac)*(mag-mlow[im])*(thgh[it]-tilt) \
           + D*(afac-afac_l)*(mag-mlow[im])*(thgh[it]-tilt) \
           + E*(afac_h-afac)*(mhgh[im]-mag)*(tilt-tlow[it]) \
           + F*(afac-afac_l)*(mhgh[im]-mag)*(tilt-tlow[it]) \
           + G*(afac_h-afac)*(mag-mlow[im])*(tilt-tlow[it]) \
           + H*(afac-afac_l)*(mag-mlow[im])*(tilt-tlow[it])

    return (coeffs,latmin)
