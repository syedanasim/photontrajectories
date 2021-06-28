import numpy as np
from numpy import pi,cos,sin,arctan,arcsin,arctan2,arccos
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''Constants'''
degrees=180/pi #div for deg, mul for rad

msun= 1477 #[m]    #mass and time are in length units becasue geometrized units; it is measuring mass in units of Rs
M_star= 1.4*msun #[m]   #Mass neutron star
Rs= 2*M_star #[m]
R_star= 12000 #[m]

# t = time
# r = radius
# th = theta
# ph = phi8
# M = mass
# a = spin
# b = as in eqn 9 & 10
# v_r = dr/dlambda
# v_th = dth/dlambda
# alpha_0 = initial condition
# beta_0 = initial condition
# zeta_0 = initial condition

'''Sphere Plotting'''
theta_sphere, phi_sphere = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
x_sphere = R_star*sin(theta_sphere)*cos(phi_sphere)
y_sphere = R_star*sin(theta_sphere)*sin(phi_sphere)
z_sphere = R_star*cos(theta_sphere)

'''Metric'''
def delta(r,M,a):
    delta=(r**2)-(2*M*r)+(a**2)
    return delta
def sigma(r,th,a):
    sigma=(r**2)+((a**2)*(cos(th)**2))
    return sigma
def gtt(r,th,M,a):
    gtt=((2*M*r)/sigma(r,th,a))-1
    return gtt 
def grr(r,th,M,a):
    grr=sigma(r,th,a)/delta(r,M,a)
    return grr
def gthth(r,th,a):
    gthth=sigma(r,th,a)
    return gthth
def gphph(r,th,M,a):
    gphph=((r**2)+(a**2)+((2*M*(a**2)*r*(sin(th)**2))/sigma(r,th,a)))*(sin(th)**2)
    return gphph
def gtph(r,th,M,a):
    gtph=-(2*M*a*r*(sin(th)**2))/sigma(r,th,a)
    return gtph
def gpht(r,th,M,a):
    gpht=-(2*M*a*r*(sin(th)**2))/sigma(r,th,a)
    return gpht
def gtr():
    g=0
    return g
def gtth():
    g=0
    return g
def grt():
    g=0
    return g
def grth():
    g=0
    return g
def grph():
    g=0
    return g
def gtht():
    g=0
    return g
def gthr():
    g=0
    return g
def gthph():
    g=0
    return g
def gphr():
    g=0
    return g
def gphth():
    g=0
    return g
def Metric(r,th,M,a):
    matrix=[[[gtt(r,th,M,a)],[gtr()],[gtth()],[gtph(r,th,M,a)]],
            [[grt()],[grr(r,th,M,a)],[grth()],[grph()]],
            [[gtht()],[gthr()],[gthth(r,th,a)],[gthph()]],
            [[gpht(r,th,M,a)],[gphr()],[gphth()],[gphph(r,th,M,a)]]]
    return matrix

'''Christoffel'''
def Gamma_r_tt(r,th,M,a):
    gamma=-(M*((a**2)+r*(-2*M+r)))*(((a**2)*(cos(th)**2))-(r**2))/(((r**2)+(a**2)*(cos(th)**2))**3)
    return gamma
def Gamma_r_rr(r,th,M,a):
    gamma=(r*(a**2-M*r)+a**2*(M-r)*cos(th)**2)/((a**2+r*(-2*M+r))*(r**2+a**2*cos(th)**2))
    return gamma
def Gamma_r_thth(r,th,M,a):
    gamma=-((r*(a**2+r*(-2*M+r)))/(r**2+a**2*cos(th)**2))
    return gamma
def Gamma_r_phph(r,th,M,a):
    gamma=-(((a**2+r*(-2*M+r))*sin(th)**2*(r**5+a**4*r*cos(th)**4-a**2*M*r**2*sin(th)**2+cos(th)**2*(2*a**2*r**3+a**4*M*sin(th)**2)))/(r**2+a**2*cos(th)**2)**3)
    return gamma
def Gamma_r_pht(r,th,M,a):
    gamma=(a*M*(a**2+r*(-2*M+r))*(-r**2+a**2*cos(th)**2)*sin(th)**2)/(r**2+a**2*cos(th)**2)**3
    return gamma
def Gamma_r_thr(r,th,a):
    gamma=-((a**2*cos(th)*sin(th))/(r**2+a**2*cos(th)**2))
    return gamma
def Gamma_th_tt(r,th,M,a):
    gamma=-(2*a**2*M*r*cos(th)*sin(th))/(r**2+a**2*cos(th)**2)**3
    return gamma
def Gamma_th_rr(r,th,M,a):
    gamma=(a**2*cos(th)*sin(th))/((a**2+r*(-2*M+r))*(r**2+a**2*cos(th)**2))
    return gamma
def Gamma_th_thth(r,th,a):
    gamma=-((a**2*cos(th)*sin(th))/(r**2+a**2*cos(th)**2))
    return gamma
def Gamma_th_phph(r,th,M,a):
    gamma=-((cos(th)*sin(th)*(2*a**2*r**2*(a**2+r**2)*cos(th)**2+a**4*(a**2+r**2)*cos(th)**4+r*(a**2*r**3+r**5+4*a**2*M*r**2*sin(th)**2+2*a**4*M*sin(th)**4+a**4*M*sin(2*th)**2)))/(r**2+a**2*cos(th)**2)**3)
    return gamma
def Gamma_th_pht(r,th,M,a):
    gamma=(a*M*r*(a**2+r**2)*sin(2*th))/(r**2+a**2*cos(th)**2)**3
    return gamma
def Gamma_th_thr(r,th,a):
    gamma=r/(r**2+a**2*cos(th)**2)
    return gamma

'''Eqn 9-12''' #ODEs
def dt_dlambda(r,th,M,a,b):
    eqn=(-gphph(r,th,M,a)-(b*gtph(r,th,M,a)))/((gphph(r,th,M,a)*gtt(r,th,M,a))-(gtph(r,th,M,a)**2))
    return eqn
def dph_dlambda(r,th,M,a,b):
    eqn=((b*gtt(r,th,M,a))+gtph(r,th,M,a))/((gphph(r,th,M,a)*gtt(r,th,M,a))-(gtph(r,th,M,a)**2))
    return eqn

def dr_dlambda(r,th,M,a,b,v_r):
    eqn=v_r
    return eqn
def dth_dlambda(r,th,M,a,b,v_th):
    eqn=v_th
    return eqn

def dvr_dlambda(r,th,M,a,b,v_r,v_th):
    eqn=-(Gamma_r_tt(r,th,M,a)*(dt_dlambda(r,th,M,a,b)**2))
    -(Gamma_r_rr(r,th,M,a)*(dr_dlambda(r,th,M,a,b,v_r)**2))
    -(Gamma_r_thth(r,th,M,a)*(dth_dlambda(r,th,M,a,b,v_th)**2))
    -(Gamma_r_phph(r,th,M,a)*(dph_dlambda(r,th,M,a,b)**2))
    -(Gamma_r_pht(r,th,M,a)*(dph_dlambda(r,th,M,a,b)*dt_dlambda(r,th,M,a,b)))
    -(Gamma_r_thr(r,th,a)*(dth_dlambda(r,th,M,a,b,v_th)*dr_dlambda(r,th,M,a,b,v_r)))
    return eqn
def dvth_dlambda(r,th,M,a,b,v_r,v_th):
    eqn=-(Gamma_th_tt(r,th,M,a)*(dt_dlambda(r,th,M,a,b)**2))
    -(Gamma_th_rr(r,th,M,a)*(dr_dlambda(r,th,M,a,b,v_r)**2))
    -(Gamma_th_thth(r,th,a)*(dth_dlambda(r,th,M,a,b,v_th)**2))
    -(Gamma_th_phph(r,th,M,a)*(dph_dlambda(r,th,M,a,b)**2))
    -(Gamma_th_pht(r,th,M,a)*(dph_dlambda(r,th,M,a,b)*dt_dlambda(r,th,M,a,b)))
    -(Gamma_th_thr(r,th,a)*(dth_dlambda(r,th,M,a,b,v_th)*dr_dlambda(r,th,M,a,b,v_r)))
    return eqn

'''b (impact parameter)'''
def imp_par(d,alpha_0,beta_0,zeta_0,M,a):
    dph_dl=k_ph(d,alpha_0,beta_0,zeta_0)
    y_initial=initial(d,alpha_0,beta_0,zeta_0)
    r=y_initial[1]
    th=y_initial[2]
    b=((dph_dl*((gphph(r,th,M,a)*gtt(r,th,M,a))-(gtph(r,th,M,a)**2)))-gtph(r,th,M,a))/gtt(r,th,M,a)
    return b

'''Eqn 15-20''' #initial conditions

def r_i(d,alpha_0,beta_0):
    eqn=((d**2)+(alpha_0**2)+(beta_0**2))**0.5
    return eqn
def costh_i(d,alpha_0,beta_0,zeta_0):
    eqn=((d*cos(zeta_0))+(beta_0*sin(zeta_0)))/r_i(d,alpha_0,beta_0)
    return eqn
def tanph_i(d,alpha_0,beta_0,zeta_0):
    eqn=alpha_0*((d*cos(zeta_0))-(beta_0*sin(zeta_0)))**-1
    return eqn
def k_r(d,alpha_0,beta_0):
    eqn=d/r_i(d,alpha_0,beta_0)
    return eqn
def k_th(d,alpha_0,beta_0,zeta_0):
    eqn=(-cos(zeta_0)+(((d*cos(zeta_0))+(beta_0*sin(zeta_0)))*d/(r_i(d,alpha_0,beta_0)**2)))*((r_i(d,alpha_0,beta_0)**2)-(((d*cos(zeta_0))+(beta_0*sin(zeta_0)))**2))**-0.5
    return eqn
def k_ph(d,alpha_0,beta_0,zeta_0):
    eqn=-(alpha_0*sin(zeta_0))/((((d*sin(zeta_0))-(beta_0*cos(zeta_0)))**2)+alpha_0**2)
    return eqn

def initial(d,alpha_0,beta_0,zeta_0):
    t=0
    r=r_i(d,alpha_0,beta_0)
    costh=costh_i(d,alpha_0,beta_0,zeta_0)
    th=arccos(costh)
    tanph=tanph_i(d,alpha_0,beta_0,zeta_0)
    ph=arctan(tanph)
    kr=-k_r(d,alpha_0,beta_0)
    kth=-k_th(d,alpha_0,beta_0,zeta_0)
    #kph=k_ph(d,alpha_0,beta_0,zeta_0)
    i_vec=(t,r,th,ph,kr,kth)
    return i_vec

'''Eqn 14''' #will use as a check for after integration

def xi(yvec):
    M=M_star
    t=yvec[0]
    r=yvec[1]
    th=yvec[2]
    ph=yvec[3]
    v_r=yvec[4]
    v_th=yvec[5]
    xi=(grr(r,th,M,a)*(dr_dlambda(r,th,M,a,b,v_r)**2)+gphph(r,th,M,a)*(dph_dlambda(r,th,M,a,b)**2)+gthth(r,th,a)*(dth_dlambda(r,th,M,a,b,v_th)**2)+2*gtph(r,th,M,a)*(dt_dlambda(r,th,M,a,b)*dph_dlambda(r,th,M,a,b)))/(gtt(r,th,M,a)*(dt_dlambda(r,th,M,a,b)**2))
    if xi==-1:
        print('xi =',xi,'true')
    elif xi<=-0.99 and xi>-1:
        print('xi =',xi,'almost true')
    elif xi<-1 and xi>=-1.01:
        print('xi =',xi,'almost true')
    elif xi<-1.01:
        print('xi =',xi,'false')
    elif xi>-0.99:
        print('xi =',xi,'false')
def xi_check(yvec):
    check=[]
    for i in range(len(yvec[0])):
        check.append(xi(yvec[:,i]))#,M,a,b))

def trajectory(time,y,M,a,b):
    r=y[1]
    th=y[2]
    v_r=y[4]
    v_th=y[5]
    dt=dt_dlambda(r,th,M,a,b)
    dr=dr_dlambda(r,th,M,a,b,v_r)
    dth=dth_dlambda(r,th,M,a,b,v_th)
    dph=dph_dlambda(r,th,M,a,b)
    dvr=dvr_dlambda(r,th,M,a,b,v_r,v_th)
    dvth=dvth_dlambda(r,th,M,a,b,v_r,v_th)
    return (dt,dr,dth,dph,dvr,dvth)

def star(time,yvec,M,a,b):
    return yvec[1]-R_star #when r-R_star=0
star.terminal = True
star.direction = -1