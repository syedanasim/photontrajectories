import numpy as np
from numpy import pi,cos,sin,arctan,arcsin,arctan2

msun=1.989*(10**30) #[kg]


'''Metric'''

def Metric(t,r,th,ph,M,a): 
    #(time,radius,theta,phi,mass,spin)
    
    delta=(r**2)-(2*M*r)+(a**2)
    sigma=(r**2)+((a**2)*(cos(th)**2))
    
    gtt=((2*M*r)/sigma)-1
    grr=sigma/delta
    gthth=sigma
    gphph=((r**2)+(a**2)+((2*M*(a**2)*r*(sin(th)**2))/sigma))*(sin(th)**2)
    
    gtph=-(2*M*a*r*(sin(th)**2))/sigma
    gpht=-(2*M*a*r*(sin(th)**2))/sigma
    
    gtr=0
    gtth=0
    grt=0
    grth=0
    grph=0
    gtht=0
    gthr=0
    gthph=0
    gphr=0
    gphth=0
    
    matrix=[[[gtt],[gtr],[gtth],[gtph]],
            [[grt],[grr],[grth],[grph]],
            [[gtht],[gthr],[gthth],[gthph]],
            [[gpht],[gphr],[gphth],[gphph]]]
    
    return matrix

'''Christoffel'''

def Gamma_r_tt(t,r,th,ph,M,a): 
    #(time,radius,theta,phi,mass,spin)
    gamma=-((M*(a**2 + r*(-2*M + r))*(-r**2 + a**2*cos(th)**2))/(r**2 + a**2*cos(th)**2)**3)
    return gamma

def Gamma_r_rr(t,r,th,ph,M,a): 
    #(time,radius,theta,phi,mass,spin)
    gamma=(r*(a**2 - M*r) + a**2*(M - r)*cos(th)**2)/((a**2 + r*(-2*M + r))*(r**2 + a**2*cos(th)**2))
    return gamma

def Gamma_r_thth(t,r,th,ph,M,a): 
    #(time,radius,theta,phi,mass,spin)
    gamma=-((r*(a**2 + r*(-2*M + r)))/(r**2 + a**2*cos(th)**2))
    return gamma

def Gamma_r_phph(t,r,th,ph,M,a): 
    #(time,radius,theta,phi,mass,spin)
    gamma=-(((a**2 + r*(-2*M + r))*sin(th)**2*(r**5 + a**4*r*cos(th)**4 - a**2*M*r**2*sin(th)**2 + cos(th)**2*(2*a**2*r**3 + a**4*M*sin(th)**2)))/(r**2 + a**2*cos(th)**2)**3)
    return gamma

def Gamma_r_pht(t,r,th,ph,M,a): 
    #(time,radius,theta,phi,mass,spin)
    gamma=(a*M*(a**2 + r*(-2*M + r))*(-r**2 + a**2*cos(th)**2)*sin(th)**2)/(r**2 + a**2*cos(th)**2)**3
    return gamma

def Gamma_r_thr(t,r,th,ph,M,a): 
    #(time,radius,theta,phi,mass,spin)
    gamma=-((a**2*cos(th)*sin(th))/(r**2 + a**2*cos(th)**2))
    return gamma

def Gamma_th_tt(t,r,th,ph,M,a): 
    #(time,radius,theta,phi,mass,spin)
    gamma=-(2*a**2*M*r*cos(th)*sin(th))/(r**2 + a**2*cos(th)**2)**3
    return gamma

def Gamma_th_rr(t,r,th,ph,M,a): 
    #(time,radius,theta,phi,mass,spin)
    gamma=(a**2*cos(th)*sin(th))/((a**2 + r*(-2*M + r))*(r**2 + a**2*cos(th)**2))
    return gamma

def Gamma_th_thth(t,r,th,ph,M,a): 
    #(time,radius,theta,phi,mass,spin)
    gamma=-((a**2*cos(th)*sin(th))/(r**2 + a**2*cos(th)**2))
    return gamma

def Gamma_th_phph(t,r,th,ph,M,a): 
    #(time,radius,theta,phi,mass,spin)
    gamma=-((cos(th)*sin(th)*(2*a**2*r**2*(a**2 + r**2)*cos(th)**2 + a**4*(a**2 + r**2)*cos(th)**4 + r*(a**2*r**3 + r**5 + 4*a**2*M*r**2*sin(th)**2 + 2*a**4*M*sin(th)**4 + a**4*M*sin(2*th)**2)))/(r**2 + a**2*cos(th)**2)**3)
    return gamma

def Gamma_th_pht(t,r,th,ph,M,a): 
    #(time,radius,theta,phi,mass,spin)
    gamma=(a*M*r*(a**2 + r**2)*sin(2*th))/(r**2 + a**2*cos(th)**2)**3
    return gamma

def Gamma_th_thr(t,r,th,ph,M,a): 
    #(time,radius,theta,phi,mass,spin)
    gamma=r/(r**2 + a**2*cos(th)**2)
    return gamma