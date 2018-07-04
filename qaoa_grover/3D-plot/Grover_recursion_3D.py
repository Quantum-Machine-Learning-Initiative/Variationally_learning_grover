import numpy as np
import qit
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import math
from matplotlib.ticker import MaxNLocator
import matplotlib


matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size'   : 12})
#matplotlib.rc('font', **{'family':'serif', 'serif':['Computer Modern Roman'], 
#                                'monospace': ['Computer Modern Typewriter'], 'size'  : 12})
matplotlib.rc('text', usetex=True)


#matplotlib.rcParams['mathtext.fontset'] = 'custom'
#matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
#matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
#matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
#matplotlib.rcParams['mathtext.fontset'] = 'stix'
#matplotlib.rcParams['font.family'] = 'STIXGeneral'


def Grover_iter(n_bits, n_iterations):
    p = n_iterations
    N = 2**n_bits
    A_array = []
    B_array = []
    Ai = np.sqrt(N-1)/np.sqrt(N)
    Bi = (1/np.sqrt(N))
    A_next = Ai
    B_next = Bi
    A_array.append(Ai)
    B_array.append(Bi)
    for i in range(p):
        Ai = (1-(2/N))*A_next -2*(np.sqrt(N-1)/N)*B_next
        Bi = 2*(np.sqrt(N-1)/N)*A_next + (1-(2/N))*B_next
        A_next = Ai
        B_next = Bi
        A_array.append(Ai)
        B_array.append(Bi)
    Ampl = np.amax(B_array)
    prob = Ampl*np.conj(Ampl)
    #print('prob = ', prob)
    return prob




def qaoa_iter(n_bits, n_iterations, alpha):
    a = 1+np.exp(1j*alpha)
    p = n_iterations
    N = 2**n_bits    
    A_array = []
    B_array = []
    Ai = np.sqrt(N-1)/np.sqrt(N)
    Bi = (1/np.sqrt(N))
    A_next = Ai
    B_next = Bi
    A_array.append(Ai)
    B_array.append(Bi)
    for i in range(1,p+1):
        #Ai = (np.exp(1j*alpha) - (1+np.exp(1j*alpha))/N )*A_next - (1+np.exp(1j*alpha))*np.exp(1j*alpha)*(np.sqrt(N-1)/N)*B_next
        #Bi = (1+np.exp(1j*alpha))*(np.sqrt(N-1)/N)*A_next + (np.exp(1j*alpha) - (1/N)*np.exp(1j*alpha)*(1+np.exp(1j*alpha)) )*B_next    
        #Ai = (np.exp(1j*alpha) - a/N)*A_next - a*np.exp(1j*alpha)*(np.sqrt(N-1)/N)*B_next
        #Bi = a*(np.sqrt(N-1)/N)*A_next + (np.exp(1j*alpha) - np.exp(1j*alpha)*a/N)*B_next       
        A_next = Ai
        B_next = Bi            
        A_array.append(Ai)
        B_array.append(Bi)    
    return B_array
           

def qaoa_iter_matrix(n_bits, n_iterations, alpha):
    a = (np.exp(1j*alpha) - 1)
    b = a
    p = n_iterations
    N = 2**n_bits    
    A_array = []
    B_array = []
    Ai = np.sqrt(N-1)/np.sqrt(N)
    Bi = (1/np.sqrt(N))
    vec_amp = np.array([Ai, Bi])
    U = np.array([ [(1+a*(N-1)/N) , a*(b+1)*np.sqrt(N-1)/N], [a*np.sqrt(N-1)/N , (b+1)*(1+a/N)] ])
    A_next = Ai
    B_next = Bi
    A_array.append(Ai)
    B_array.append(Bi)
    for i in range(1,p+1):
        new_vec = np.dot(U,vec_amp)
        B_next = new_vec[1]            
        B_array.append(B_next)
        vec_amp = new_vec    
    return B_array
           




def Grover_upto(n):
    max_probs_array = []
    qbits_array = []
    for nqbits in range(1,(n+1)):
        print('n qubits = ', nqbits)   
        steps = int(np.sqrt(2**nqbits)) + 3       
        prob = Grover_iter(nqbits, steps)
        max_probs_array.append(prob)
        qbits_array.append(nqbits)
    #print(max_probs_array)
    #print(qbits_array)
    newProbs = np.ones(len(max_probs_array)) - max_probs_array
    exp_probs = np.exp(newProbs) 
    print(exp_probs[4:])
    

    plt.plot(qbits_array[10:], exp_probs[10:],label='grover')
    plt.scatter(qbits_array[10:], exp_probs[10:],label='grover')
    plt.title('Grover optimal probability $e^{(1-P_{GROVER})}$')
    plt.xlabel('number of qubits')
    plt.ylabel(' Optimal Probability ')
    plt.savefig('grover_recursion.pdf')
    plt.show()
        
        
        
        
###########################################################################33
def plot_alpha():
    prob_array = []
    p = 2
    n_bits = 3
    alpha_array = np.linspace(0, 2*np.pi, num=100)
    for alpha in alpha_array:
        B_array = qaoa_iter_matrix(n_bits, p, alpha)
        Ampl = B_array[-1]
        prob = Ampl*np.conj(Ampl)         
        prob_array.append(prob)
    #plt.plot(alpha_array, prob_array)
    #plt.show()

######################  3D Plots      #############################################################
def Get_prob_array(n_bits, p):
    alpha_array = np.linspace(0, 2*np.pi, num=200)
    prob_array = []
    for alpha in alpha_array:
        B_array = qaoa_iter_matrix(n_bits, p, alpha)
        Ampl = B_array[-1]
        prob = Ampl*np.conj(Ampl)         
        prob_array.append(prob)
    return prob_array    


def Get_prob_matrix(number_bits, p_steps):
    P_matrix = []
    for n, p in zip(number_bits, p_steps):
        P_matrix.append(Get_prob_array(n, p))
    return P_matrix
    
    
'''
def alpha3D():
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    number_bits = [3, 4, 5, 6]
    p_steps = [2, 3, 4, 6]
    alpha_array = np.linspace(0, 2*np.pi, num=100)
    Prob_array = Get_prob_matrix(number_bits, p_steps)
    for i, n in enumerate(number_bits):
        L = len(alpha_array)
        y = [n]*L
        x = alpha_array
        z = Prob_array[i]
        p =  p_steps[i] 
        print(p) 
        string = 'p = ' + str(p)
        ax.plot3D(x, y, z, label=string)
    
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=False))
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

    plt.legend()
    plt.show()
    
'''
    
def alpha3D():
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    number_bits = [3, 4, 5, 6]
    p_steps = [2, 3, 4, 6]
    number_bits.reverse()
    p_steps.reverse()
    alpha_array = np.linspace(0, 2*np.pi, num=200)
    Prob_array = Get_prob_matrix(number_bits, p_steps)
    for i, n in enumerate(number_bits):
        L = len(alpha_array)
        y = [n]*L
        x = alpha_array
        z = Prob_array[i]   
        p =  p_steps[i] 
        print(p) 
        string = 'step p = ' + str(p)
        ax.plot3D(x, y, z, label=string)
    #plt.xlabel('Variational angle (radians)')
    #plt.ylabel('number of qubits')
    ax.set_xlabel(r'variational angle $\alpha$ (radians)')
    ax.set_ylabel(r'number of qubits')
    ax.set_zlabel(r'solution probability')
    
    
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=False))
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

    np.save('Prob_array', Prob_array)
    np.save('alpha_array', alpha_array)
    np.save('p_steps', p_steps)
    np.save('number_bits', number_bits)

    plt.legend()
    plt.savefig('alpha3d.pdf')
    plt.show()    

def main():
    Grover_upto(30)
    
 
    

          
if __name__ == "__main__":    
    #main()
    #plot_alpha()
    alpha3D()
        
        
        
        
        
