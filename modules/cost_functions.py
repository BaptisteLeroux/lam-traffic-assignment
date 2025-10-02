from modules import *

def bpr_function(V, t0, C, alpha, beta) :
    return t0 *(1+alpha*(V/C)**beta)

def linearised_bpr_function(V, t0, C, alpha, beta, eps) :
    return t0*(1+alpha*eps**beta*(1-beta)) + (t0*alpha*beta*eps**(beta-1)/C)*V