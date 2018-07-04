
import numpy as np
import qit
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import math
from matplotlib.ticker import MaxNLocator
import matplotlib

matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size'   : 12})
#matplotlib.rc('font', **{'family' : "sans-serif"})
matplotlib.rc('text', usetex=True)
params = {'text.latex.preamble' : [r'\usepackage{amsmath}']}
plt.rcParams.update(params)

Prob_array = np.load('Prob_array.npy')
alpha_array = np.load('alpha_array.npy')
p_steps = np.load('p_steps.npy')
number_bits = np.load('number_bits.npy')



def slice_plot():
    fig = plt.figure()
    plt.axhline(y=1.0, color='gray', linestyle='dashed')
    plt.axhline(y=0.945, color='black', linestyle='dashed')
    extraticksx = [[2.1268, np.pi, 4.155], [r'\alpha_1', r'$\pi $', r'$\alpha_2$']]
    plt.xticks(*extraticksx)
    L = len(alpha_array)
    #print(number_bits[3])
    y = Prob_array[3][40:160]
    x = alpha_array[40:160]
    print(len(x))
    plt.plot(x, y, 'r')
    plt.show()



def alpha3D():
    fig = plt.figure()
    ax = plt.axes(projection='3d')

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
    
    plt.legend()
    plt.savefig('alpha3d.pdf')
    plt.show()    
    
    
    
if __name__ == "__main__":    
    #alpha3D()
    slice_plot()
