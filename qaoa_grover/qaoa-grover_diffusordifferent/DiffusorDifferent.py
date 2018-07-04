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
    

###########################  QAOA (All angles equal) Functions   ###################################
#################################################################################
def C_operator_all(target_state, n_qubits):
       return -1*qit.state.projector(target_state)

def B_operator_all(n_qubits):
    plus_state=(qit.state('0')+qit.state('1'))/np.sqrt(2)
    state=(qit.state('0')+qit.state('1'))/np.sqrt(2)
    for i in range(1,n_qubits):
        state=qit.state.tensor(state,plus_state)
    return -1*qit.state.projector(state)


def qaoa_step_all(state, target_state, n_qubits, params):
    #state=state.propagate(C_operator_all(target_state,n_qubits).data,params)
    state=state.propagate(C_operator_all(target_state,n_qubits).data,[np.pi])
    return state.propagate(B_operator_all(n_qubits).data, params)  

def F_function_all(target_state, n_qubits, p, params):
    ini_state=initial_state(n_qubits)
    for i in range(p):
        ini_state=qaoa_step_all(ini_state,target_state,n_qubits,params=params)
    return ini_state.ev(C_operator_all(target_state,n_qubits).data), ini_state
    
    
    
def get_params_all(target_state, n_qubits, p, params_0):
#     function to optimimize
    def fun(x):
#         we minimize f to find max for F 
        return F_function_all(target_state, n_qubits, p, params=x)[0]
# starting point
#     params_0=[0.25*np.pi for i in range(2*p)]
    params_min=[0 for i in range(2*p)]
    params_max=[2*np.pi if i%2==0 else np.pi for i in range(2*p)]
    # the bounds required by L-BFGS-B
    bounds = [(0, 2*np.pi)]

# use method L-BFGS-B because the problem is smooth and bounded
    minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds)
    result = scipy.optimize.basinhopping(fun, params_0, minimizer_kwargs=minimizer_kwargs)
    print(result.x)
    return result.x[0]    
        
        
    
    
    


    
    

###################################################################################
###################################################################################


def calc_n_p_allequal(n_qbits,p_steps):
    n = n_qbits
    number_iterations = p_steps
    qaoa_approximation = []
    grover_approximation = []
    param_array = []
    p_array = []
    state=qit.state('0000')
    params=0.5*np.pi
    for i in range(1,number_iterations):
        params=get_params_all(target_state=state,p=i,n_qubits=n,params_0=params)
        state_2=F_function_all(target_state=state,params=params,p=i,n_qubits=n)[1]
        qaoa_approximation.append(qit.state.fidelity(state,state_2)**2)
        param_array.append(params)
        grover_approximation.append(GROVER(iterations=i,state=state,n_qubits=n))
        p_array.append(i)
        print('Iteration ', i)
        print('angle', params)
        #params.extend([0,0])


    np.save('qaoa_data', qaoa_approximation)
    np.save('grover_data', grover_approximation)
    np.save('p_data', p_array)
    np.save('angle_data_allequal', param_array)


def Just_Grover(n_qbits, p_steps):
    n = n_qbits
    number_iterations = p_steps
    grover_approximation = [] 
    p_array = []
    string = '0'*n_qbits
    state=qit.state(string)    
    for i in range(1,number_iterations):   
        grover_approximation.append(GROVER(iterations=i,state=state,n_qubits=n))
        print(grover_approximation[-1])
        p_array.append(i)
        print('Iteration ', i) 
   
    print('Grover probs')
    print(grover_approximation)
    print('n steps')
    print(p_array)
    
    plt.plot(p_array, grover_approximation,label='grover')
    plt.show()
        

	

def main():
    calc_n_p_allequal(4,10)
    

         
if __name__ == "__main__":
    #Just_Grover(12,70)    
    main()

    
    
