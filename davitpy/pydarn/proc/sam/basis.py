"""
*********************
**Module**: pydarn.proc.sam.basis
   :synopsis: Functions for evaluting spherical harmonic & AMIE basis functions
*********************

**Functions**:
    * :func:`pydarn.proc.sam.basis.load_eofs`
    * :func:`pydarn.proc.sam.basis.eval_pot_amie`
    * :func:`pydarn.proc.sam.basis.eval_efield_amie`
    * :func:`pydarn.proc.sam.basis.eval_pot_sh`
    * :func:`pydarn.proc.sam.basis.eval_efield_sh`

**Classes**:
    * :class:`pydarn.proc.sam.basis.AMIEqset` : Container for set of AMIE basis functions

    written by Ellen D. P. Cousins, 2014-08
"""

import numpy as np
import scipy,os

class AMIEqset():
    """This class is the basic container for holding & loading the set of AMIE basis functions.
    **Example**:
         qset = AMIEqset()
    """

    def __init__(self,path=None):

        # Set parameters
        # qset244.24
        # the following parameters are for 36 basis functions, down to 44deg
        # From A. Richmond, NCAR-HAO

        if path == None:
            path = os.environ['DAVITPY'] + '/tables/amie/'

        fname1 = path + 'qset16ascii24424bf_1.dat'
        fname2 = path + 'qset16ascii24424bf_2.dat'
        fname3 = path + 'qset16ascii24424bf_3.dat'
        fname4 = path + 'qset16ascii24424bf_4.dat'

        if(not os.path.isfile(fname1)):
            print 'problem reading',fname1,':file does not exist'
            return None


        f = open(fname4,'r')
        res = f.readline().split()
        f.close()

        self.ithmx   = int(res[0])
        self.mrow    = int(res[1])
        self.kmx     = int(res[2])
        self.mmx     = int(res[3])
        self.nq      = int(res[4])
        self.nqs     = int(res[5])
        self.mxnmx   = int(res[6])
        self.ithtrns = int(res[7])
        self.dth     = float(res[8])


        f = open(fname1,'r')
        data = np.loadtxt(f)
        f.close()

        nq = self.nq
        ithmx = self.ithmx
        self.q  = data[0:nq,1:ithmx+1]     # using 1:ithmx+1 rather than 0:ithmx+1
        self.dq = data[nq:2*nq,1:ithmx+1]  # to match bug in original code

        f = open(fname2,'r')
        data = np.loadtxt(f,dtype='int')
        f.close()

        self.ibm = data[:,0]-1 #from 1-based to 0-based indexing
        self.iem = data[:,1]-1

        f = open(fname3,'r')
        data = np.loadtxt(f,dtype='int')
        f.close()

        self.ns  = data[:,0]-1 #from 1-based to 0-based indexing
        self.nmx = data[:,1]
        self.nss = data[:,2]
             

def load_eofs(filename=None):
    """Load AMIE coefficients defining a set of EOFs and construct EOF covariance matrix
    
    **Args**:
        * **filename** Name of file containing coefficients defining the EOFs

    **Returns**:
        * Array of set of basis function values at locations 

    **Example**:
        eof_coeffs = load_eofs()
    """
    
    if filename == None:
        path = os.environ['DAVITPY'] + '/tables/model/'
        filename = path + 'sam_eof_coeffs.dat'

    if(not os.path.isfile(filename)):
        print 'problem reading',filename,':file does not exist'
        return (None, None)

    f = open(filename,'r')
    eof_coeffs = np.loadtxt(f)
    f.close()

    return np.transpose(eof_coeffs)


def fcmp(mmx,cp,sp):
    # Internal function
    # Converted from AMIE FORTRAN originally by A. Richmond, NCAR-HAO

    nlont = len(cp)
    assert(nlont == len(sp)),'Length of two vectors, sp and cp have to be the same'

    fp = np.ndarray(shape=(mmx+1,nlont),dtype=np.float64)
    fm = np.ndarray(shape=(mmx+1,nlont),dtype=np.float64)

    fp[0,:] = 1.
    fm[0,:] = 1.
    fp[1,:] = np.sqrt(2)*sp 
    fm[1,:] = np.sqrt(2)*cp

    for m in range(2,mmx+1):
        fp[m,:] = cp*fp[m-1,:] + sp*fm[m-1,:]
        fm[m,:] = cp*fm[m-1,:] - sp*fp[m-1,:]

    f = np.vstack((fm[mmx:0:-1,:],fp[0:mmx+1,:]))

    return f


def eval_pot_amie(lat,lon,qset=None):
    """Evalute AMIE basis functions in terms of potential at given locations

    **Args**:
        * **lat** array of latitudes in degrees
        * **lon** array of longitudes in degrees

    **Returns**:
        * (len(lat),244) Array of set of basis function values at locations 
     
    **Example**:
         import numpy as np
         lat = np.ones(24)*70.
         lon = np.linspace(0,345,24)
         pot_arr = pydarn.proc.sam.basis.eval_pot_amie(lat,lon)
    """
    # Converted from AMIE FORTRAN originally by A. Richmond, NCAR-HAO

    if qset == None:
        qset = AMIEqset()
    
    mmx = qset.mmx
    iem = qset.iem
    ibm = qset.ibm
    ns  = qset.ns
    
    theta = np.radians(90.-abs(lat))
    phi   = np.radians(lon)

    sp = np.sin(phi)
    cp = np.cos(phi)

    ff = fcmp(mmx,cp,sp)

    sth = np.sin(theta)
    
    x   = theta/qset.dth
    ith = x
    ith = np.around(np.clip(ith,1,qset.ithmx-2)).astype(int)

    x   = x - ith

    xm1 = x*(-2. + x*(3. - x))/6.
    x0  = 1.+ x*(-.5 + x*(-1. + .5*x))
    xp1 = x*( 1. + x*(.5 - .5*x))
    xp2 = x*(-1. + x*x)/6
    
    potarr = np.ndarray(shape=(len(phi),iem[2*mmx]+1),dtype=np.float64)

    for m in range(-mmx,mmx+1):
        mmm = m + mmx

        fint = ff[mmm,:]

        mm = abs(m)
        for i in range(ibm[mmm],iem[mmm]+1):
            ix = i - ibm[mmm]
            potarr[:,i] = (xm1*qset.q[2*ix+ns[mm],ith-1] +
                           x0*qset.q[2*ix+ns[mm],ith] +
                           xp1*qset.q[2*ix+ns[mm],ith+1] +
                           xp2*qset.q[2*ix+ns[mm],ith+2])*fint

    return potarr


def eval_efield_amie(lat,lon,radEarth=6371.,alt=110.,qset=None):
    """Evalute AMIE basis functions in terms of electric field at given locations

    **Args**:
        * **lat** array of latitudes in degrees
        * **lon** array of longitudes in degrees

    **Returns**:
        * (2,len(lat),244) Array of set of Etheta,Ephi values at locations 

    **Example**:
         import numpy as np
         lat = np.ones(24)*70.
         lon = np.linspace(0,345,24)
         el_arr = pydarn.proc.sam.basis.eval_efield_amie(lat,lon)
    """
    # Converted from AMIE FORTRAN originally by A. Richmond, NCAR-HAO

    if qset == None:
        qset = AMIEqset()
    
    mmx = qset.mmx
    iem = qset.iem
    ibm = qset.ibm
    ns  = qset.ns

    RI = (radEarth + alt)*1000.

    theta = np.radians(90.-abs(lat))
    phi   = np.radians(lon)

    sp = np.sin(phi)
    cp = np.cos(phi)

    ff = fcmp(mmx,cp,sp)

    sth = np.sin(theta)
    
    x   = theta/qset.dth
    ith = x
    ith = np.around(np.clip(ith,1,qset.ithmx-2)).astype(int)

    x   = x - ith
    xm1 = x*(-2. + x*(3. - x))/6.
    x0  = 1.+ x*(-.5 + x*(-1. + .5*x))
    xp1 = x*( 1. + x*(.5 - .5*x))
    xp2 = x*(-1. + x*x)/6
    
    earr = np.ndarray(shape=(2,len(phi),iem[2*mmx]+1),dtype=np.float64)

    for m in range(-mmx,mmx+1):
        mmm = m + mmx

        fint = ff[mmm,:]/RI
        dfint= m*ff[24-mmm,:]/sth/RI

        mm = abs(m)
        for i in range(ibm[mmm],iem[mmm]+1):
            ix = i - ibm[mmm]

            # Dirivative in Theta direction
            earr[0,:,i] = (xm1*qset.dq[2*ix+ns[mm],ith-1] +
                           x0*qset.dq[2*ix+ns[mm],ith] +
                           xp1*qset.dq[2*ix+ns[mm],ith+1] +
                           xp2*qset.dq[2*ix+ns[mm],ith+2])*fint

            # Dirivative in Phi direction
            earr[1,:,i] = (xm1*qset.q[2*ix+ns[mm],ith-1] +
                           x0*qset.q[2*ix+ns[mm],ith] +
                           xp1*qset.q[2*ix+ns[mm],ith+1] +
                           xp2*qset.q[2*ix+ns[mm],ith+2])*dfint
    
    # Zero at 90deg
    q = np.array(np.where(theta == 0))
    q = q[0]
    if (len(q) != 0):
        earr[:,q,:] = 0

    earr = -earr

    return earr

def eval_vel_amie(lat,lon,radEarth=6371.,alt=110.,bmod='const',qset=None):
    """Evalute AMIE basis functions in terms of velocity at given locations

    **Args**:
        * **lat** 1D array of latitudes in degrees
        * **lon** 1D array of longitudes in degrees
        * **bmod** Geomagnetic field model to use, can be 'const', 'dip', 'igrf'

    **Returns**:
        * (2,len(lat),(244) Array of set of Vtheta,Vphi values at locations

    **Example**:
         import numpy as np
         lat = np.ones(24)*70.
         lon = np.linspace(0,345,24)
         v_arr = pydarn.proc.sam.basis.eval_vel_amie(lat,lon)
    """

    lat = np.squeeze(lat)
    lon = np.squeeze(lon)

    #First get efields
    earr = eval_efield_amie(lat,lon,radEarth=radEarth,alt=alt,qset=qset)
    nbasis = earr.shape[2]

    if bmod == 'igrf':
        print 'IGRF not implemnted yet, using constant Bfield'
        bmod = 'const'

    if bmod == 'dip':
        theta = np.radians(90.-abs(lat))
        bFldPolar = 0.62e-4 
        bvals = bFldPolar*(1. - 3.*alt/radEarth)*np.sqrt(3.*np.square(np.cos(theta)) + 1.)/2
        bvals = tile(bvals,(nbasis,1))
    else: #bmod == 'const':
        bvals = 0.5e-4

    varr = np.zeros_like(earr)
    varr[0,:,:] = -earr[1,:,:]/bvals
    varr[1,:,:] =  earr[0,:,:]/bvals

    return varr


def eval_pot_sh(lat,lon,latmin,order=8):
    """Evalute spherical harmonic basis functions in terms of potential at given locations

    **Args**:
        * **lat** array of latitudes in degrees
        * **lon** array of longitudes in degrees
        * **latmin** equatorward limit of defined region (positive for North, negative for South)
        * **order**  maximum order of SH expansion

    **Returns**:
        * (len(lat),(order+1)^2) Array of set of basis function values at locations 

    **Example**:
         import numpy as np
         lat = np.ones(24)*70.
         lon = np.linspace(0,345,24)
         pot_arr = pydarn.proc.sam.basis.eval_pot_sh(lat,lon,60)

    .Note. 
    This assumes the unnormalized, real formulation of the spherical harmonic expansion (sines & cosines)
    The SuperDARN RST C code used the normalized complex expansion, while IDL code used real formluation
     
    """

    theta = np.radians(90.-abs(lat))
    phi   = np.radians(lon)

    # Map 0->thetamax to 0->2pi
    thetaMax = np.radians((90.0-abs(latmin)))
    tPrime = (np.pi/thetaMax)*theta
    x = np.cos(tPrime)


    # Here we evaluate the associated legendre polynomials..from order 0 to order
    # we use scipy.special.lpmn() function to get the assciated legendre polynomials...but it doesnt
    # accept an array...so do loop calculate the leg.pol for each value of x and append these arrays to a new array
    for j in range(len(x)):
        (plmTemp,dplmTemp) = scipy.special.lpmn(order, order, x[j])

        if j == 0 :
            plmFit = plmTemp
        else :
            plmFit = np.dstack((plmFit,plmTemp))

    # Evaluate potential for each basis function
    potarr = np.ndarray(shape=(len(phi),(order+1)**2))

    # use a lambda function to convert from 2D l,m to 1D index
    indexLgndr = lambda l,m :( m == 0 and l**2 ) or \
        ( (l != 0 ) and (m != 0) and l**2 + 2*m - 1 ) or 0

    for m in range(order+1) :
        for L in range( m,order+1) :
            k = indexLgndr( L, m )
            if m == 0 :
                potarr[:,k] = plmFit[0,L,:]
            else :
                potarr[:,k]   = np.cos( m*phi )*plmFit[m,L,:]
                potarr[:,k+1] = np.sin( m*phi )*plmFit[m,L,:]

    
    # Zero below latmin / above thetaMax
    q = np.array(np.where(theta > thetaMax))
    q = q[0]
    if (len(q) != 0):
        potarr[q,:] = 0

    return potarr


def eval_efield_sh(lat,lon,latmin,order=8,radEarth=6371.,alt=110.):
    """Evalute spherical harmonic basis functions in terms of electric field at given locations

    **Args**:
        * **lat** array of latitudes in degrees
        * **lon** array of longitudes in degrees
        * **latmin** equatorward limit of defined region (positive for North, negative for South)
        * **order**  maximum order of SH expansion

    **Returns**:
        * (2,len(lat),(order+1)^2) Array of set of Etheta,Ephi values at locations

    **Example**:
         import numpy as np
         lat = np.ones(24)*75.
         lon = np.linspace(0,345,24)
         el_arr = pydarn.proc.sam.basis.eval_efield_sh(lat,lon,60)

    .Note. 
    This assumes the unnormalized, real formulation of the spherical harmonic expansion (sines & cosines)
    The SuperDARN RST C code used the normalized complex expansion, while IDL code used real formluation
    """

    RI = (radEarth + alt)*1000.

    theta = np.radians(90.-abs(lat))
    phi   = np.radians(lon)
    sth   = np.sin(theta)

    # Map 0->thetamax to 0->pi
    thetaMax = np.radians((90.0-abs(latmin)))
    alpha  = (np.pi/thetaMax)
    tPrime = alpha*theta
    x = np.cos(tPrime)

    # Here we evaluate the associated legendre polynomials..from order 0 to order
    # we use scipy.special.lpmn() function to get the assciated legendre polynomials...but it doesnt
    # accept an array...so do loop calculate the leg.pol for each value of x and append these arrays to a new array
    for j in range(len(x)):
        (plmTemp,dplmTemp) = scipy.special.lpmn(order, order, x[j])
        
        if j == 0 :
            plmFit  = plmTemp
        else :
            plmFit  = np.dstack((plmFit,  plmTemp))

    # Evaluate el field for each basis function
    earr = np.ndarray(shape=(2,len(phi),(order+1)**2))

    # use a lambda function to convert from 2D l,m to 1D index
    indexLgndr = lambda l,m :( m == 0 and l**2 ) or \
        ( (l != 0 ) and (m != 0) and l**2 + 2*m - 1 ) or 0

    for m in range(order+1) :
        for L in range(m,order+1) :
            k = indexLgndr( L, m )
            if m == 0 :
                # Dirivatives in both direction = 0
                earr[0,:,k]   = np.zeros(len(x))
                earr[1,:,k]   = np.zeros(len(x))
            else :
                # Derivative in Theta direction
                # Using recursion relationship for deriv. of Asoc. Legn. Poly.
                #Note L can't == 0 since m > 0
                dplm = (L*x*plmFit[m,L,:] - (L+m)*plmFit[m,L-1,:])/np.sin(tPrime)
                earr[0,:,k]   = alpha*np.cos(m*phi)*dplm/RI
                earr[0,:,k+1] = alpha*np.sin(m*phi)*dplm/RI

                # Dirivative in Phi direction
                earr[1,:,k]   = -m*np.sin(m*phi)*plmFit[m,L,:]/sth/RI
                earr[1,:,k+1] =  m*np.cos(m*phi)*plmFit[m,L,:]/sth/RI

    # Zero at 90deg
    q = np.array(np.where(theta == 0))
    q = q[0]
    if (len(q) != 0):
        earr[:,q,:] = 0
    
    # I'm not sure if we should take negative, I can't find it in original IDL code...         
    earr = -earr

    # Zero below latmin / above thetaMax
    q = np.array(np.where(theta > thetaMax))
    q = q[0]
    if (len(q) != 0):
        earr[:,q,:] = 0

    return earr


def eval_vel_sh(lat,lon,latmin,order=8,radEarth=6371.,alt=110.,bmod='const'):
    """Evalute spherical harmonic basis functions in terms of velocity at given locations

    **Args**:
        * **lat** 1D array of latitudes in degrees
        * **lon** 1D array of longitudes in degrees
        * **latmin** equatorward limit of defined region (positive for North, negative for South)
        * **order**  maximum order of SH expansion
        * **bmod** Geomagnetic field model to use, can be 'const', 'dip', 'igrf'

    **Returns**:
        * (2,len(lat),(order+1)^2) Array of set of Vtheta,Vphi values at locations

    **Example**:
         import numpy as np
         lat = np.ones(24)*70.
         lon = np.linspace(0,345,24)
         varr = pydarn.proc.sam.basis.eval_vel_sh(lat,lon,60)

    .Note. 
    This assumes the unnormalized, real formulation of the spherical harmonic expansion (sines & cosines)
    The SuperDARN RST C code used the normalized complex expansion, while IDL code used real formluation
    """

    lat = np.squeeze(lat)
    lon = np.squeeze(lon)

    #First get efields
    earr = eval_efield_sh(lat,lon,latmin,order=order,radEarth=radEarth,alt=alt)
    nbasis = earr.shape[2]

    if bmod == 'igrf':
        print 'IGRF not implemnted yet, using constant Bfield'
        bmod = 'const'

    if bmod == 'dip':
        theta = np.radians(90.-abs(lat))
        bFldPolar = 0.62e-4 
        bvals = bFldPolar*(1. - 3.*alt/radEarth)*np.sqrt(3.*np.square(np.cos(theta)) + 1.)/2
        bvals = tile(bvals,(nbasis,1))
    else: #bmod == 'const':
        bvals = 0.5e-4

    #IDL code put the negative sign here
    #bvals = -bvals

    varr = np.zeros_like(earr)
    varr[0,:,:] = -earr[1,:,:]/bvals
    varr[1,:,:] =  earr[0,:,:]/bvals

    return varr
    
