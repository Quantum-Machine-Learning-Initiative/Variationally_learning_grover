import numpy as np
import qit
import matplotlib.pyplot as plt
import pandas as pd
import scipy




def initial_state(n_qubits):
    plus_state=(qit.state('0')+qit.state('1'))/np.sqrt(2)
    state=(qit.state('0')+qit.state('1'))/np.sqrt(2)
    for i in range(1,n_qubits):
        state=qit.state.tensor(state,plus_state)
    return state
    
    

########################### Grover functions ####################################
#################################################################################
def grover_u_w(state, n_qubits):
    I=qit.tensor(qit.lmap(np.eye(2)))
    for i in range(1,n_qubits):
        I=qit.tensor(I,qit.lmap(np.eye(2)))
    return -2*qit.state.projector(state)+I
    
    
def grover_u_s(n_qubits):
    plus_state=(qit.state('0')+qit.state('1'))/np.sqrt(2)
    state=(qit.state('0')+qit.state('1'))/np.sqrt(2)
    for i in range(1,n_qubits):
        state=qit.state.tensor(state,plus_state)
    I=qit.tensor(qit.lmap(np.eye(2)))
    for i in range(1,n_qubits):
        I=qit.tensor(I,qit.lmap(np.eye(2)))
    return -1*I+2*qit.state.projector(state)
    
def grover_step(state, target_state, n_qubits):
    U=grover_u_s(n_qubits)*grover_u_w(target_state, n_qubits)
    return state.kraus_propagate([U.data])   
    

def GROVER(iterations, state, n_qubits):
    final_state=initial_state(n_qubits)
    for i in range(iterations):
        final_state=grover_step(final_state,state,n_qubits)
    return qit.state.fidelity(state,final_state)**2  
    
###########################  QAOA Functions   ###################################
#################################################################################
def C_operator(target_state, n_qubits):
       return -1*qit.state.projector(target_state)

def B_operator(n_qubits):
    plus_state=(qit.state('0')+qit.state('1'))/np.sqrt(2)
    state=(qit.state('0')+qit.state('1'))/np.sqrt(2)
    for i in range(1,n_qubits):
        state=qit.state.tensor(state,plus_state)
    return -1*qit.state.projector(state)


def qaoa_step(state, target_state, n_qubits, params):
    state=state.propagate(C_operator(target_state,n_qubits).data,params[0])
    return state.propagate(B_operator(n_qubits).data,params[1])  

def F_function(target_state, n_qubits, p, params):
    ini_state=initial_state(n_qubits)
    for i in range(p):
        ini_state=qaoa_step(ini_state,target_state,n_qubits,params=[params[2*i],params[2*i+1]])
    return ini_state.ev(C_operator(target_state,n_qubits).data), ini_state
    
    
    
def get_params(target_state, n_qubits, p, params_0):
#     function to optimimize
    def fun(x):
#         we minimize f to find max for F 
        return F_function(target_state, n_qubits, p, params=x)[0]
# starting point
#     params_0=[0.25*np.pi for i in range(2*p)]
    params_min=[0 for i in range(2*p)]
    params_max=[2*np.pi if i%2==0 else np.pi for i in range(2*p)]
    # the bounds required by L-BFGS-B
    bounds = [(low, high) for low, high in zip(params_min, params_max)]

# use method L-BFGS-B because the problem is smooth and bounded
    minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds)
    result = scipy.optimize.basinhopping(fun, params_0, minimizer_kwargs=minimizer_kwargs)
    return [result.x[i] for i in range(2*p)]    

###################################################################################
###################################################################################


def calc_n_p(n_qbits,p_steps):
    n = n_qbits
    number_iterations = p_steps
    qaoa_approximation = []
    grover_approximation = []
    p_array = []
    state=qit.state('000')
    params=[0.5*np.pi,0.5*np.pi]
    for i in range(1,number_iterations):
        params=get_params(target_state=state,p=i,n_qubits=n,params_0=params)
        state_2=F_function(target_state=state,params=params,p=i,n_qubits=n)[1]
        qaoa_approximation.append(qit.state.fidelity(state,state_2)**2)
   
        grover_approximation.append(GROVER(iterations=i,state=state,n_qubits=n))
        p_array.append(i)
        print('Iteration ', i)
        params.extend([0,0])


    np.save('qaoa_data', qaoa_approximation)
    np.save('grover_data', grover_approximation)
    np.save('p_data', p_array)

def main():
    calc_n_p(3,5)
    
          
if __name__ == "__main__":
    main()    


    
    
