import numpy as np
from numpy import pi,cos,sin,arctan,arcsin,arctan2,arccos
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import njit, jit, prange

'''Constants'''
degrees=180/pi #div for deg, mul for rad
#mass and time are in length units becasue geometrized units; it is measuring mass in units of Rs
msun= 1477.0145901865496 #[m]    
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
@njit(fastmath=True)
def delta(r,M,a):
    delta=(r**2)-(2*M*r)+(a**2)
    return delta
@njit(fastmath=True)
def sigma(r,th,a):
    sigma=(r**2)+((a**2)*(cos(th)**2))
    return sigma
@njit(fastmath=True)
def gtt(r,th,M,a):
    gtt=((2*M*r)/sigma(r,th,a))-1
    return gtt 
@njit(fastmath=True)
def grr(r,th,M,a):
    grr=sigma(r,th,a)/delta(r,M,a)
    return grr
@njit(fastmath=True)
def gthth(r,th,a):
    gthth=sigma(r,th,a)
    return gthth
@njit(fastmath=True)
def gphph(r,th,M,a):
    gphph=((r**2)+(a**2)+((2*M*(a**2)*r*(sin(th)**2))/sigma(r,th,a)))*(sin(th)**2)
    return gphph
@njit(fastmath=True)
def gtph(r,th,M,a):
    gtph=-(2*M*a*r*(sin(th)**2))/sigma(r,th,a)
    return gtph
@njit(fastmath=True)
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
@njit(fastmath=True)
def Metric(r,th,M,a):
    matrix=[[[gtt(r,th,M,a)],[gtr()],[gtth()],[gtph(r,th,M,a)]],
            [[grt()],[grr(r,th,M,a)],[grth()],[grph()]],
            [[gtht()],[gthr()],[gthth(r,th,a)],[gthph()]],
            [[gpht(r,th,M,a)],[gphr()],[gphth()],[gphph(r,th,M,a)]]]
    return matrix

'''Christoffel'''
@njit(fastmath=True)
def Gamma_r_tt(r,th,M,a):
    gamma=-(M*((a**2)+r*(-2*M+r)))*(((a**2)*(cos(th)**2))-(r**2))/(((r**2)+(a**2)*(cos(th)**2))**3)
    return gamma
@njit(fastmath=True)
def Gamma_r_rr(r,th,M,a):
    gamma=(r*(a**2-M*r)+a**2*(M-r)*cos(th)**2)/((a**2+r*(-2*M+r))*(r**2+a**2*cos(th)**2))
    return gamma
@njit(fastmath=True)
def Gamma_r_thth(r,th,M,a):
    gamma=-((r*(a**2+r*(-2*M+r)))/(r**2+a**2*cos(th)**2))
    return gamma
@njit(fastmath=True)
def Gamma_r_phph(r,th,M,a):
    gamma=-(((a**2+r*(-2*M+r))*sin(th)**2*(r**5+a**4*r*cos(th)**4-a**2*M*r**2*sin(th)**2+cos(th)**2*(2*a**2*r**3+a**4*M*sin(th)**2)))/(r**2+a**2*cos(th)**2)**3)
    return gamma
@njit(fastmath=True)
def Gamma_r_pht(r,th,M,a):
    gamma=(a*M*(a**2+r*(-2*M+r))*(-r**2+a**2*cos(th)**2)*sin(th)**2)/(r**2+a**2*cos(th)**2)**3
    return gamma
@njit(fastmath=True)
def Gamma_r_thr(r,th,a):
    gamma=-((a**2*cos(th)*sin(th))/(r**2+a**2*cos(th)**2))
    return gamma
@njit(fastmath=True)
def Gamma_th_tt(r,th,M,a):
    gamma=-(2*a**2*M*r*cos(th)*sin(th))/(r**2+a**2*cos(th)**2)**3
    return gamma
@njit(fastmath=True)
def Gamma_th_rr(r,th,M,a):
    gamma=(a**2*cos(th)*sin(th))/((a**2+r*(-2*M+r))*(r**2+a**2*cos(th)**2))
    return gamma
@njit(fastmath=True)
def Gamma_th_thth(r,th,a):
    gamma=-((a**2*cos(th)*sin(th))/(r**2+a**2*cos(th)**2))
    return gamma
@njit(fastmath=True)
def Gamma_th_phph(r,th,M,a):
    gamma=-((cos(th)*sin(th)*(2*a**2*r**2*(a**2+r**2)*cos(th)**2+a**4*(a**2+r**2)*cos(th)**4+r*(a**2*r**3+r**5+4*a**2*M*r**2*sin(th)**2+2*a**4*M*sin(th)**4+a**4*M*sin(2*th)**2)))/(r**2+a**2*cos(th)**2)**3)
    return gamma
@njit(fastmath=True)
def Gamma_th_pht(r,th,M,a):
    gamma=(a*M*r*(a**2+r**2)*sin(2*th))/(r**2+a**2*cos(th)**2)**3
    return gamma
@njit(fastmath=True)
def Gamma_th_thr(r,th,a):
    gamma=r/(r**2+a**2*cos(th)**2)
    return gamma

'''Eqn 9-12''' #ODEs
@njit(fastmath=True)
def dt_dlambda(r,th,M,a,b):
    eqn=(-gphph(r,th,M,a)-(b*gtph(r,th,M,a)))/((gphph(r,th,M,a)*gtt(r,th,M,a))-(gtph(r,th,M,a)**2))
    return eqn
@njit(fastmath=True)
def dph_dlambda(r,th,M,a,b):
    eqn=((b*gtt(r,th,M,a))+gtph(r,th,M,a))/((gphph(r,th,M,a)*gtt(r,th,M,a))-(gtph(r,th,M,a)**2))
    return eqn

@njit(fastmath=True)
def dr_dlambda(r,th,M,a,b,v_r):
    eqn=v_r
    return eqn
@njit(fastmath=True)
def dth_dlambda(r,th,M,a,b,v_th):
    eqn=v_th
    return eqn

@njit(fastmath=True)
def dvr_dlambda(r,th,M,a,b,v_r,v_th):
    eqn=-(Gamma_r_tt(r,th,M,a)*(dt_dlambda(r,th,M,a,b)**2)) -(Gamma_r_rr(r,th,M,a)*(dr_dlambda(r,th,M,a,b,v_r)**2)) -(Gamma_r_thth(r,th,M,a)*(dth_dlambda(r,th,M,a,b,v_th)**2)) -(Gamma_r_phph(r,th,M,a)*(dph_dlambda(r,th,M,a,b)**2)) -(2*Gamma_r_pht(r,th,M,a)*(dph_dlambda(r,th,M,a,b)*dt_dlambda(r,th,M,a,b))) -(2*Gamma_r_thr(r,th,a)*(dth_dlambda(r,th,M,a,b,v_th)*dr_dlambda(r,th,M,a,b,v_r)))
    return eqn
@njit(fastmath=True)
def dvth_dlambda(r,th,M,a,b,v_r,v_th):
    eqn=-(Gamma_th_tt(r,th,M,a)*(dt_dlambda(r,th,M,a,b)**2)) -(Gamma_th_rr(r,th,M,a)*(dr_dlambda(r,th,M,a,b,v_r)**2)) -(Gamma_th_thth(r,th,a)*(dth_dlambda(r,th,M,a,b,v_th)**2)) -(Gamma_th_phph(r,th,M,a)*(dph_dlambda(r,th,M,a,b)**2)) -(2*Gamma_th_pht(r,th,M,a)*(dph_dlambda(r,th,M,a,b)*dt_dlambda(r,th,M,a,b))) -(2*Gamma_th_thr(r,th,a)*(dth_dlambda(r,th,M,a,b,v_th)*dr_dlambda(r,th,M,a,b,v_r)))
    return eqn

'''b (impact parameter)'''
@njit(fastmath=True)
def imp_par(d,alpha_0,beta_0,zeta_0,M,a):
    dph_dl=k_ph(d,alpha_0,beta_0,zeta_0)
    y_initial=initial(d,alpha_0,beta_0,zeta_0)
    r=y_initial[1]
    th=y_initial[2]
    b=-((dph_dl*((gphph(r,th,M,a)*gtt(r,th,M,a))-(gtph(r,th,M,a)**2)))-gtph(r,th,M,a))/gtt(r,th,M,a)
    return b

'''Eqn 15-20''' #initial conditions

@njit(fastmath=True)
def r_i(d,alpha_0,beta_0):
    eqn=((d**2)+(alpha_0**2)+(beta_0**2))**0.5
    return eqn
@njit(fastmath=True)
def costh_i(d,alpha_0,beta_0,zeta_0):
    eqn=((d*cos(zeta_0))+(beta_0*sin(zeta_0)))/r_i(d,alpha_0,beta_0)
    return eqn
@njit(fastmath=True)
def tanph_i(d,alpha_0,beta_0,zeta_0):
    eqn=alpha_0*((d*sin(zeta_0))-(beta_0*cos(zeta_0)))**-1
    return eqn
@njit(fastmath=True)
def k_r(d,alpha_0,beta_0):
    eqn=d/r_i(d,alpha_0,beta_0)
    return eqn
@njit(fastmath=True)
def k_th(d,alpha_0,beta_0,zeta_0):
    eqn=(-cos(zeta_0)+(((d*cos(zeta_0))+(beta_0*sin(zeta_0)))*d/(r_i(d,alpha_0,beta_0)**2)))*((r_i(d,alpha_0,beta_0)**2)-(((d*cos(zeta_0))+(beta_0*sin(zeta_0)))**2))**-0.5
    return eqn
@njit(fastmath=True)
def k_ph(d,alpha_0,beta_0,zeta_0):
    eqn=-(alpha_0*sin(zeta_0))/((((d*sin(zeta_0))-(beta_0*cos(zeta_0)))**2)+alpha_0**2)
    return eqn

@njit(fastmath=True)
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

@njit(fastmath=True)
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
    elif xi<=-0.9999 and xi>-1:
        print('xi =',xi,'almost true')
    elif xi<-1 and xi>=-1.0001:
        print('xi =',xi,'almost true')
    elif xi<-1.0001:
        print('xi =',xi,'false')
    elif xi>-0.9999:
        print('xi =',xi,'false')
def xi_check(yvec):
    check=[]
    for i in range(len(yvec[0])):
        check.append(xi(yvec[:,i]))#,M,a,b))      
@njit(fastmath=True)
def xi_vals(yvec):
    M=M_star
    t=yvec[0]
    r=yvec[1]
    th=yvec[2]
    ph=yvec[3]
    v_r=yvec[4]
    v_th=yvec[5]
    xi=(grr(r,th,M,a)*(dr_dlambda(r,th,M,a,b,v_r)**2)+gphph(r,th,M,a)*(dph_dlambda(r,th,M,a,b)**2)+gthth(r,th,a)*(dth_dlambda(r,th,M,a,b,v_th)**2)+2*gtph(r,th,M,a)*(dt_dlambda(r,th,M,a,b)*dph_dlambda(r,th,M,a,b)))/(gtt(r,th,M,a)*(dt_dlambda(r,th,M,a,b)**2))
    return xi
@njit(fastmath=True)
def trajectory(time,y,M,a,b,R_star):
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
@njit(fastmath=True)
def star(time,yvec,M,a,b,R_star):
    return yvec[1]-R_star #when r-R_star=0
star.terminal = True
star.direction = -1

@jit(fastmath=True, nopython=False)
def final_state(M,a,d,alpha_0,beta_0,zeta_0,R_star):
    y_initial = initial(d,alpha_0,beta_0,zeta_0)
    b = imp_par(d,alpha_0,beta_0,zeta_0,M,a)
    t_final=2*d
    yarr=solve_ivp(trajectory,[0,t_final],y_initial,args=(M,a,b,R_star),method='RK45',atol=1e-6,rtol=1e-6,dense_output=True,events=star)
    y_fin=yarr.y[:,-1] #t,r,th,ph,kr,kth
    output_vec=[alpha_0,beta_0,y_fin[0],y_fin[1],y_fin[2],y_fin[3],y_fin[4],y_fin[5]] #add zenith angle & xi deviation
    return output_vec #alpha_0,beta_0,t_fin,r_fin,th,ph,kr,kth
@jit(parallel=True, fastmath=True, nopython=False)
def set_final_states(pixa,pixb,zeta_0,M,R_star):
    alpha_vec=np.linspace(-1.6*R_star, 1.6*R_star,pixa)
    beta_vec=np.linspace(-1.6*R_star, 1.6*R_star,pixb)
    output_matrix=[]
    for j in prange(len(beta_vec)):
        for i in range(len(alpha_vec)):
            output_matrix.append(final_state(M,a,d,alpha_vec[i],beta_vec[j],zeta_0,R_star))
    return np.array(output_matrix)

def myround(x, base=2):
    return base * np.round(x/base)

def plot_trajectory(M,a,d,alpha_0,beta_0,zeta_0):
    R_star= 12000
    y_initial = initial(d,alpha_0,beta_0,zeta_0)
    b = imp_par(d,alpha_0,beta_0,zeta_0,M,a)
    t_final=2*d
    yarr=solve_ivp(trajectory,[0,t_final],y_initial,args=(M,a,b),method='RK45',atol=1e-10,rtol=1e-10,dense_output=True,events=star)
    R_final=yarr.y[1,-1]
    if R_final<=R_star: #hits star
        t_end=yarr.t_events
        time_vec=np.linspace(0, t_end[0][0],10000)
        yarr_new=yarr.sol(time_vec)
        Rad=yarr_new[1,:]
        Theta=yarr_new[2,:]
        Phi=yarr_new[3,:]
        X=Rad*sin(Theta)*cos(Phi)
        Y=Rad*sin(Theta)*sin(Phi)
        Z=Rad*cos(Theta)
        cart=(X,Y,Z)
    else: #misses star
        time_vec=np.linspace(0, 2*d,10000)
        yarr_new=yarr.sol(time_vec)
        Rad=yarr_new[1,:]
        Theta=yarr_new[2,:]
        Phi=yarr_new[3,:]
        X=Rad*sin(Theta)*cos(Phi)
        Y=Rad*sin(Theta)*sin(Phi)
        Z=Rad*cos(Theta)
        cart=(X,Y,Z) 
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_sphere, y_sphere, z_sphere,  rstride=1, cstride=1, color='c', alpha=0.3, linewidth=0)
    ax.scatter(X,Y,Z, c='b', marker='.')
    ax.set_xlim([-R_star,R_star])
    ax.set_ylim([-R_star,R_star])
    ax.set_zlim([-R_star,R_star])
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    ax.view_init(80, -45)
    plt.tight_layout()
def plot_hits():
    fig = plt.figure(figsize=(6,6))
    ax=plt.subplot()
    alpha_plot_hit=test_hit[:,0]
    beta_plot_hit=test_hit[:,1]
    alpha_plot_miss=test_miss[:,0]
    beta_plot_miss=test_miss[:,1]
    plt.scatter(alpha_plot_hit,beta_plot_hit,c='royalblue',s=1)
    plt.scatter(alpha_plot_miss,beta_plot_miss,c='darkgrey',s=1)
    ax.set_xlabel('alpha$_0$',fontsize=12)
    ax.set_ylabel('beta$_0$',fontsize=12)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
def plot_gridline():
    fig = plt.figure(figsize=(6,6))
    ax=plt.subplot()
    alpha_plot_hit_theta=test_theta[:,0]
    beta_plot_hit_theta=test_theta[:,1]
    alpha_plot_hit_phi=test_phi[:,0]
    beta_plot_hit_phi=test_phi[:,1]
    plt.scatter(alpha_plot_hit_theta,beta_plot_hit_theta,c='royalblue',s=1)
    plt.scatter(alpha_plot_hit_phi,beta_plot_hit_phi,c='royalblue',s=1)
    ax.set_xlabel('alpha$_0$',fontsize=12)
    ax.set_ylabel('beta$_0$',fontsize=12)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
def plot_star():
    radius_fin=test_hit[:,3]
    theta_fin=test_hit[:,4]
    phi_fin=test_hit[:,5]

    X=radius_fin*sin(theta_fin)*cos(phi_fin)
    Y=radius_fin*sin(theta_fin)*sin(phi_fin)
    Z=radius_fin*cos(theta_fin)
    
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_sphere, y_sphere, z_sphere,  rstride=1, cstride=1, color='c', alpha=0.3, linewidth=0)
    ax.scatter(X,Y,Z, c='b', marker='.',s=1)
    ax.set_xlim([-2*R_star,2*R_star])
    ax.set_ylim([-2*R_star,2*R_star])
    ax.set_zlim([-2*R_star,2*R_star])
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    ax.view_init(0, 90)
    plt.tight_layout()