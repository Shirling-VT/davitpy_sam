from pydarn.sdio.radDataRead import *
from utils import *
import datetime as dt
import pydarn
import models.aacgm as aacgm
import scipy.io as sio
import os
from math import *
import matplotlib.pyplot as plt
import numpy as np


def calc_azm(lat, lon, txlat, txlon):

    """lat & lon are position of observation in degrees (use AACGM)
    TXlat & TXlon are position of radar in degress (use AACGM)

    calculate the azimuth by taking the bearing from
    the current observation to the radar

    returns azm angle in degrees with range (0, 360)
    """

    dlon = (lon - txlon)
    dtor = pi/180
    #zonal distance
    ty = sin(dlon*dtor)*cos(txlat*dtor)  
    #meridional dist      
    tx = cos(lat*dtor)*sin(txlat*dtor)-sin(lat*dtor)*cos(txlat*dtor)*cos(dlon*dtor)

    azm = atan2(-ty, tx)/dtor
    if (azm<0):
        azm = azm+360

    return azm


def vel_plot(latO, lonO_MLT, vels, azm, sTime, eTime):

    '''plot velocities'''
    
    ds = u'\N{DEGREE SIGN}'

    f = plt.figure()
    ax = f.add_subplot(111, polar=True)
    dtor = pi/180
    theta = (np.array(lonO_MLT)-90)*dtor
    r = (90-np.array(latO))*dtor
    dr = np.cos(np.array(azm)*dtor)*np.array(vels)
    dt = np.sin(np.array(azm)*dtor)*np.array(vels)
    C = abs(np.array(vels))
    cax = ax.quiver(theta, r, -dr*np.cos(theta)-dt*np.sin(theta), -dr*np.sin(theta)+dt*np.cos(theta), C, \
        scale = 1e4)
    #cax = ax.quiver(theta[[1,30]], r[[1,30]], np.array([0.1,1]), np.array([0.1,0]), scale=10)
    ax.yaxis.set_ticks([10*dtor, 20*dtor, 30*dtor, 40*dtor])
    ax.set_xticklabels(['6 MLT', '', '12 MLT', '', '18 MLT', '', '0 MLT', ''])
    ax.set_yticklabels(['80'+ds, '70'+ds, '60'+ds, '50'+ds])
    ax.set_title(str(sTime)+'-'+str(eTime))
    cbar = f.colorbar(cax)
    cbar.set_label('Velocity [m/s]')
    plt.show(f)

    return 0


def read_data(sTime, eTime, rad):

    # remove the file
    #try:
    #    os.remove('data.mat')
    #except OSError:
    #    pass

    #cp_list = [150, 151, 153, 157, 9050, 3501]

    #####################################
    # start time
    # sTime = dt.datetime(2011,10,8,4,0)

    # rad list
    # rad = ['pgr']

    # end time
    # eTime = dt.datetime(2011,10,8,4,1)
    ####################################

    # specify channel as 'a'
    channel = 'a'

    # aacgm coord
    coords = 'mag'

    bmnum = None

    cp = None

    # complete
    fileType = 'fitacf'

    filtered = False

    # to be specified as local in the future
    src = None

    # starts main function
    myFiles = []
    # go through all the radars
    for i in range(len(rad)):
        f = radDataOpen(sTime, rad[i], eTime=eTime, channel=channel, bmnum=bmnum, cp=cp, fileType=fileType, filtered=filtered, src=src)
        if (f!=None):
            myFiles.append(f)
    allBeams = ['']*len(myFiles)
    # explanation of parameters
    # vels records velocity
    # ts records time
    # latR, lonR record the location of radars (fixed for 'mag' coord)
    # latO, lonO record the location of observations in 'mag' coord
    # azm records azm angles (0, 360)
    # lonO_MLT is converted from lonO in MLT coord
    vels, ts, latR, lonR, latO, lonO, azm, lonO_MLT, vel_err = [], [], [], [], [], [], [], [], []
    data_dict = {}
    for i in range(len(myFiles)):
        allBeams[i] = radDataReadRec(myFiles[i])
        # only need to read it once since it is fixed w.r.t. time
        t = allBeams[i].time
        site = pydarn.radar.site(radId=allBeams[i].stid, dt=t)
        myFov = pydarn.radar.radFov.fov(site=site, rsep=allBeams[i].prm.rsep,\
                ngates=allBeams[i].prm.nrang+1, nbeams=site.maxbeam, coords=coords)
        # x records the location of radars (i-th)
        if (coords=='mag'):
            x = aacgm.aacgmConv(site.geolat, site.geolon, 0., t.year, 0)
        # read in data until the end of file
        while (allBeams[i]!=None):
            if (abs(allBeams[i].cp) in cp_list):
                t = allBeams[i].time
                # go through all the observations which are located along one particular beam
                # throw out the first 10 "range gates" of the data grid
                for k in range(10, len(allBeams[i].fit.slist)):
                    # gate number r
                    r = allBeams[i].fit.slist[k]
                    # throw away outliers
                    if (allBeams[i].fit.qflg[k]!=1):
                        continue
                    if (allBeams[i].fit.gflg[k]==1):
                        continue
                    if (allBeams[i].fit.v_e[k]>150):
                        continue
                    if (allBeams[i].fit.p_l[k]<3):
                        continue
                    vels.append(allBeams[i].fit.v[k])
                    # calculate the measurement error
                    vel_err.append(allBeams[i].fit.w_l[k]/sqrt(allBeams[i].fit.nlag[k]))
                    ts.append(t)
                    latR.append(x[0])
                    lonR.append(x[1])
                    # allBeams[i].bmnum represents the beam number
                    latO_ele = myFov.latCenter[allBeams[i].bmnum, r]
                    lonO_ele = myFov.lonCenter[allBeams[i].bmnum, r]
                    latO.append(latO_ele)
                    lonO.append(lonO_ele)
                    # obtain the azm angle
                    azm.append(calc_azm(latO_ele, lonO_ele, x[0], x[1]))
                    # longitude coordinate conversion from aacgm to MLT
                    epoch = timeUtils.datetimeToEpoch(sTime)
                    mltDef = aacgm.mltFromEpoch(epoch, 0.0)*15. 
                    # mltDef is the rotation that needs to be applied, and lonO_ele is the AACGM longitude.
                    # use modulo so new longitude is between 0 & 360
                    lonO_MLT.append(np.mod((lonO_ele+mltDef), 360. ))
            allBeams[i] = radDataReadRec(myFiles[i])
    # sio.savemat('data.mat', {'latO': latO, 'lonO_MLT': lonO_MLT, 'vels': vels, 'azm': azm, 'vel_err': vel_err})        
    data_dict = {'latO': latO, 'lonO_MLT': lonO_MLT, 'vels': vels, 'azm': azm, 'vel_err': vel_err}

    return data_dict

sTime = dt.datetime(2014,9,25,6,0)
eTime = dt.datetime(2014,9,25,6,30)
rad = ['cve']

read_data(sTime, eTime, rad)
vel_plot(latO, lonO_MLT, vels, azm, sTime, eTime)
