from qit import *
import numpy as np
import scipy as sp
import pyquil.api as api
from pyquil.paulis import PauliTerm, PauliSum
from scipy.optimize import minimize
#from grove.pyqaoa.qaoa import QAOA
from qaoa_mio_noise import * 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib
import matplotlib.pyplot as plt
from pyquil.noise import add_decoherence_noise

pauli_channel = [0.01, 0.01, 0.01] #depolarizing noise: probabilities of applying X,Y,Z gates after each gate
#qvm = api.QVMConnection(gate_noise=pauli_channel)
qvm = api.QVMConnection()


matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size'   : 12})
#matplotlib.rc('font', **{'family' : "sans-serif"})
matplotlib.rc('text', usetex=True)
params = {'text.latex.preamble' : [r'\usepackage{amsmath}']}
plt.rcParams.update(params)

def print_fun(x):
    print(x)


def Pplus(i):
    return PauliTerm('I', i, 0.5) + PauliTerm('Z', i, 0.5) 

def Pminus(i):
    return PauliTerm('I', i, 0.5) - PauliTerm('Z', i, 0.5)


def Cost_operator(n):
    Op = PauliTerm('I', 0, -1.0)
    for i in range(n):
        Op = Op*Pplus(i)
    return Op 
            

def plot_prob3D(nbits, probs_per_p, p, fig, ax):

    x = np.full(len(p), nbits)
    y = probs_per_p
    z = p
    ax.set_xlabel('Number of Qbits')#, fontsize=20, rotation=150)
    ax.set_ylabel('Probability of solution')
    ax.set_zlabel('Number of QAOA steps p')#, fontsize=30, rotation=60)
    ax.plot3D(x, y, z, 'blue')
    ax.scatter3D(x, y, z, c=z, cmap='Greens');
    
def Plus_braket(i):
    ## :return: state |+><+| on bit i
    return PauliTerm('I', i, 0.5) + PauliTerm('X', i, 0.5) 

def Minus_braket():    
    ## :return: state |-><-| on bit i
    return PauliTerm('I', i, 0.5) - PauliTerm('X', i, 0.5)
    
def grover_driver(n):
    Op = PauliTerm('I', 0, -1.0)
    for i in range(n):
        Op = Op*Plus_braket(i)
    return Op
    

def toy_nbit_Grover(n_qbits = 1, steps=1, rand_seed=None, connection=None, samples=None,
                initial_beta=None, initial_gamma=None, minimizer_kwargs=None,
                vqe_option=None):
    cost_operators = []
    driver_operators = []
    
    Op = Cost_operator(n_qbits)
    cost_operators.append(Op)
    #for i in range(n_qbits):
    #    driver_operators.append(PauliSum([PauliTerm("X", i, -1.0)]))
    Op = grover_driver(n_qbits)  
    driver_operators.append(Op)  
        
    if minimizer_kwargs is None:
        minimizer_kwargs = {'method': 'Nelder-Mead',
                            'options': {'ftol': 1.0e-2, 'xtol': 1.0e-2,
                                        'disp': False}}
    if vqe_option is None:
        vqe_option = {'disp': print_fun, 'return_all': True,
                      'samples': samples}
            
    qaoa_inst = QAOA(qvm=qvm, n_qubits=n_qbits, steps=steps, cost_ham=cost_operators,
                     ref_hamiltonian=driver_operators, store_basis=True,
                     rand_seed=rand_seed,
                     init_betas=initial_beta,
                     init_gammas=initial_gamma,
                     minimizer=minimize,
                     minimizer_kwargs=minimizer_kwargs,
                     vqe_options=vqe_option)

    return qaoa_inst    


def toy_nbit_qaoa(n_qbits = 1, steps=1, rand_seed=None, connection=None, samples=None,
                initial_beta=None, initial_gamma=None, minimizer_kwargs=None,
                vqe_option=None):
    cost_operators = []
    driver_operators = []
    
    Op = Cost_operator(n_qbits)
    cost_operators.append(Op)
    #for i in range(n_qbits):
    #    driver_operators.append(PauliSum([PauliTerm("X", i, -1.0)]))
    Op = grover_driver(n_qbits)  
    driver_operators.append(Op)        
        
    if minimizer_kwargs is None:
        minimizer_kwargs = {'method': 'Nelder-Mead',
                            'options': {'ftol': 1.0e-2, 'xtol': 1.0e-2,
                                        'disp': False}}
    if vqe_option is None:
        vqe_option = {'disp': print_fun, 'return_all': True,
                      'samples': samples}
            
    qaoa_inst = QAOA(qvm=qvm, n_qubits=n_qbits, steps=steps, cost_ham=cost_operators,
                     ref_hamiltonian=driver_operators, store_basis=True,
                     rand_seed=rand_seed,
                     init_betas=initial_beta,
                     init_gammas=initial_gamma,
                     minimizer=minimize,
                     minimizer_kwargs=minimizer_kwargs,
                     vqe_options=vqe_option)

    return qaoa_inst    


def maxprob_grover(n_qbits, n_steps):
    inst_grover = toy_nbit_Grover(n_qbits=n_qbits, steps=n_steps)
    betas_grover = np.array([np.pi]*n_steps)
    gammas_grover = np.array([np.pi]*n_steps)
    probs_grover = inst_grover.probabilities(np.hstack((betas_grover, gammas_grover)))
    probs_grover = probs_grover.flatten().real
    print(probs_grover)
    max_value = np.amax(probs_grover)
    max_index = np.argmax(probs_grover)
    print("Grover p =" + str(n_steps))
    print(max_value, max_index, inst_grover.states[max_index])
    return max_value
    

def maxprob_qaoa(n_qbits, n_steps):
    inst = toy_nbit_qaoa(n_qbits=n_qbits, steps=n_steps)
    betas, gammas = inst.get_angles()
    probs = inst.probabilities(np.hstack((betas, gammas)))
    most_freq_string, sampling_results = inst.get_string(betas, gammas)
    print(most_freq_string)
    probs = probs.flatten().real
    print(probs)
    max_value = np.amax(probs)
    max_index = np.argmax(probs)
    print("QAOA p =" + str(n_steps))
    print(max_value, max_index, inst.states[max_index])
    return max_value

def stats_grover(n_qbits, n_steps, t1, t2):
    inst_grover = toy_nbit_Grover(n_qbits=n_qbits, steps=n_steps)
    betas_grover = np.array([np.pi]*n_steps)
    gammas_grover = np.array([np.pi]*n_steps)
    #probs_grover = inst_grover.probabilities2grover(np.hstack((betas_grover, gammas_grover)), t1, t2)
    probs_grover = inst_grover.probabilities(np.hstack((betas_grover, gammas_grover)))   
    return probs_grover
    
    
def stats_qaoa(n_qbits, n_steps, t1, t2):
    inst = toy_nbit_qaoa(n_qbits=n_qbits, steps=n_steps)
    betas, gammas = 0,0# inst.get_angles()  
    probs = inst.probabilities2(np.hstack((betas, gammas)), t1, t2)
    #probs = inst.probabilities(np.hstack((betas, gammas)))
    #most_freq_string, sampling_results = inst.get_string(betas, gammas)
    return probs
    
    
def stats_new(n_qbits, n_steps, t1, t2):
    inst = toy_nbit_qaoa(n_qbits=n_qbits, steps=n_steps)
    probs = inst.get_probs(t1, t2)
    return probs
    
    
def stats_new_grover(n_qbits, n_steps, t1, t2):
    inst = toy_nbit_Grover(n_qbits=n_qbits, steps=n_steps)
    probs = inst.get_probs_grover(t1, t2)
    return probs
    
    
def plot_prob(n_qbits, p):
    probs_grover = []
    probs_qaoa = []
    for i in range(1, p+1): 
        probs_grover.append(maxprob_grover(n_qbits, i))
        probs_qaoa.append(maxprob_qaoa(n_qbits, i))
    fig = plt.figure()
    plt.plot(list(range(1,p+1)),probs_grover)
    plt.scatter(list(range(1,p+1)),probs_grover, label='Grover')
    plt.plot(list(range(1,p+1)),probs_qaoa)
    plt.scatter(list(range(1,p+1)),probs_qaoa, label='QAOA')
    plt.legend() 
    xint = []
    locs, labels = plt.xticks()
    for each in locs:
        xint.append(int(each))
    plt.xticks(xint)
    plt.title('Number of qubits = ' + str(n_qbits))
    plt.xlabel('p step')
    plt.ylabel('Probability of solution')
    plt.savefig('plot_Grover-vs-Qaoa.pdf')  
        
        
#def plot_prob_diffp(n_qbits, max_p):

#    for p in range(1,max_p+1):
#        probs_grover = []
#        probs_qaoa = []                



        
def plot_bitstat(n_qbits, p):
    stat_grover = []
    stat_qaoa = []
    
    t1s = np.logspace(-6, -2, num=512)
    np.save('t1s.npy', t1s)
    #t1s = np.linspace(10**-6,10**-3, num=8)
    t2s = np.flip(np.logspace(-6, -2, num=512), 0)
    np.save('t2s.npy', t2s)
    #t2s = np.flip(np.linspace(10**-6, 10**-3, num=8), 0)
    for t2 in t2s:
        a = []
        b = []
        for t1 in t1s:
            prob_qaoa = stats_new(n_qbits, p, t1=t1, t2=t2)
            prob_grover = stats_new_grover(n_qbits, p, t1=t1, t2=t2)           
            a.append(prob_qaoa)
            b.append(prob_grover)
        stat_qaoa.append(a)
        stat_grover.append(b) 
    print(type(np.array(stat_grover)[0][0]))
    #fig, ax = plt.subplots()
    #im = ax.imshow(stats_qaoa)
    x, y = np.meshgrid(t1s, t2s) 
    plt.pcolormesh(x, y, np.array(stat_qaoa))
    np.save('stat_qaoa.npy', np.array(stat_qaoa))
    plt.yscale('log')
    plt.xscale('log')
    plt.colorbar()
    plt.title('Variational Algorithm')
    plt.xlabel('$T_1$ ($\mu s$)')
    plt.ylabel('$T_2$ ($\mu s$)')
    #plt.show()
    plt.savefig('qaoa_noise.pdf')      
    #prob_qaoa = stats_new(n_qbits, p, t1=0.000001, t2=0.00001)
    #prob_grover = stats_new_grover(n_qbits, p, t1=0.000001, t2=0.00001)
    #print(prob_qaoa)
    
def main():
    #plot_prob(n_qbits=3, p=3)
    plot_bitstat(n_qbits=3, p=2)
    
        
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
