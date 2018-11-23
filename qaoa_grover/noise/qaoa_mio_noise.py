##############################################################################
# Copyright 2016-2017 Rigetti Computing
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
##############################################################################

from collections import Counter
from scipy import optimize
import numpy as np
from grove.pyvqe.vqe import VQE
from pyquil.quil import Program
import pyquil.quil as pq
from pyquil.gates import H, RX, RZ, CNOT, CZ
from pyquil.paulis import exponential_map, PauliSum
from functools import reduce
from pyquil.noise import add_decoherence_noise
from pyquil.api import WavefunctionSimulator


pi = np.pi

def add_gate_program(p, gate):
    erase = ['(',')', '0','1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '-'] 
    gate_data = gate.split(' ')
    name_gate = gate_data[0]
    for char in erase:
        name_gate = name_gate.replace(char, '')
        
    if name_gate == 'H':
        p.inst(RZ(pi/2, int(gate_data[1])), RX(pi/2, int(gate_data[1])), RZ(pi/2, int(gate_data[1])) )
    if name_gate == 'X':
        p.inst(RZ(pi/2, int(gate_data[1])), RX(pi/2, int(gate_data[1])), RZ(pi, int(gate_data[1])), RX(-pi/2, int(gate_data[1])), RZ(-pi/2, int(gate_data[1])) )
    if name_gate == 'CNOT':
        p.inst(RZ(pi/2, int(gate_data[2])), RX(pi/2, int(gate_data[2])), CZ(int(gate_data[1]), int(gate_data[2])), RX(-pi/2, int(gate_data[2])), RZ(-pi/2, int(gate_data[2])) )
    if name_gate == 'RZ':
        angle = float(gate_data[0].replace('RZ(', '').replace(')', '') )
        p.inst( RZ(angle, int(gate_data[1])) )
    if name_gate == 'PHASE':
        angle = float(gate_data[0].replace('PHASE(', '').replace(')', '') )
        p.inst( RZ(angle, int(gate_data[1])) )    
    return p
    
    
def add_gate_program_grover3(p, gate):
    erase = ['(',')', '0','1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '-'] 
    gate_data = gate.split(' ')
    name_gate = gate_data[0]
    for char in erase:
        name_gate = name_gate.replace(char, '')
        
    if name_gate == 'H':
        p.inst(RZ(pi/2, int(gate_data[1])), RX(pi/2, int(gate_data[1])), RZ(pi/2, int(gate_data[1])) )
    if name_gate == 'X':
        p.inst(RZ(pi/2, int(gate_data[1])), RX(pi/2, int(gate_data[1])), RZ(pi, int(gate_data[1])), RX(-pi/2, int(gate_data[1])), RZ(-pi/2, int(gate_data[1])) )
    if name_gate == 'CNOT':
        p.inst(RZ(pi/2, int(gate_data[2])), RX(pi/2, int(gate_data[2])), CZ(int(gate_data[1]), int(gate_data[2])), RX(-pi/2, int(gate_data[2])), RZ(-pi/2, int(gate_data[2])) )
    if name_gate == 'RZ':
        angle = float(gate_data[0].replace('RZ(', '').replace(')', '') )
        p.inst( RZ(-pi/4, int(gate_data[1])) )
    if name_gate == 'PHASE':
        angle = float(gate_data[0].replace('PHASE(', '').replace(')', '') )
        p.inst( RZ(pi/8, int(gate_data[1])) )    
    return p
        

    
def add_gate_program_grover4(p, gate):
    erase = ['(',')', '0','1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '-'] 
    gate_data = gate.split(' ')
    name_gate = gate_data[0]
    for char in erase:
        name_gate = name_gate.replace(char, '')
        
    if name_gate == 'H':
        p.inst(RZ(pi/2, int(gate_data[1])), RX(pi/2, int(gate_data[1])), RZ(pi/2, int(gate_data[1])) )
    if name_gate == 'X':
        p.inst(RZ(pi/2, int(gate_data[1])), RX(pi/2, int(gate_data[1])), RZ(pi, int(gate_data[1])), RX(-pi/2, int(gate_data[1])), RZ(-pi/2, int(gate_data[1])) )
    if name_gate == 'CNOT':
        p.inst(RZ(pi/2, int(gate_data[2])), RX(pi/2, int(gate_data[2])), CZ(int(gate_data[1]), int(gate_data[2])), RX(-pi/2, int(gate_data[2])), RZ(-pi/2, int(gate_data[2])) )
    if name_gate == 'RZ':
        angle = float(gate_data[0].replace('RZ(', '').replace(')', '') )
        p.inst( RZ(-pi/8, int(gate_data[1])) )
    if name_gate == 'PHASE':
        angle = float(gate_data[0].replace('PHASE(', '').replace(')', '') )
        p.inst( RZ(0.19634954084936207, int(gate_data[1])) )    
    return p
        
    
    
def new_circuit():
    p = Program()
    lines = [line.rstrip('\n') for line in open('gates_used_3bits.txt')]      
    for gate in lines:
        p = add_gate_program(p, gate)
    return p 


def new_circuit_grover3():
    p = Program()
    lines = [line.rstrip('\n') for line in open('gates_used_3bits.txt')]      
    for gate in lines:
        p = add_gate_program_grover3(p, gate)
    return p 
    
    
def new_circuit_grover4():
    p = Program()
    lines = [line.rstrip('\n') for line in open('gates_used_4bits.txt')]      
    for gate in lines:
        p = add_gate_program_grover4(p, gate)
    return p 


class QAOA(object):
    def __init__(self, qvm, n_qubits, steps=1, init_betas=None,
                 init_gammas=None, cost_ham=[],
                 ref_hamiltonian=[], driver_ref=None,
                 minimizer=None, minimizer_args=[],
                 minimizer_kwargs={}, rand_seed=None,
                 vqe_options={}, store_basis=False):
        """
        QAOA object.

        Contains all information for running the QAOA algorthm to find the
        ground state of the list of cost clauses.

        :param qvm: (Connection) The qvm connection to use for the algorithm.
        :param n_qubits: (int) The number of qubits to use for the algorithm.
        :param steps: (int) The number of mixing and cost function steps to use.
                      Default=1.
        :param init_betas: (list) Initial values for the beta parameters on the
                           mixing terms. Default=None.
        :param init_gammas: (list) Initial values for the gamma parameters on the
                            cost function. Default=None.
        :param cost_ham: list of clauses in the cost function. Must be
                    PauliSum objects
        :param ref_hamiltonian: list of clauses in the cost function. Must be
                    PauliSum objects
        :param driver_ref: (pyQuil.quil.Program()) object to define state prep
                           for the starting state of the QAOA algorithm.
                           Defaults to tensor product of \|+> states.
        :param rand_seed: integer random seed for initial betas and gammas
                          guess.
        :param minimizer: (Optional) Minimization function to pass to the
                          Variational-Quantum-Eigensolver method
        :param minimizer_kwargs: (Optional) (dict) of optional arguments to pass to
                                 the minimizer.  Default={}.
        :param minimizer_args: (Optional) (list) of additional arguments to pass to the
                               minimizer. Default=[].
        :param minimizer_args: (Optional) (list) of additional arguments to pass to the
                               minimizer. Default=[].
        :param vqe_options: (optinal) arguents for VQE run.
        :param store_basis: (optional) boolean flag for storing basis states.
                            Default=False.
        """
        self.qvm = qvm
        self.steps = steps
        self.n_qubits = n_qubits
        self.nstates = 2 ** n_qubits
        if store_basis:
            self.states = [np.binary_repr(i, width=self.n_qubits) for i in range(
                           self.nstates)]
        self.betas = init_betas
        self.gammas = init_gammas
        self.vqe_options = vqe_options

        if driver_ref is not None:
            if not isinstance(driver_ref, pq.Program):
                raise TypeError("""Please provide a pyQuil Program object as a
                                   to generate initial state""")
            else:
                self.ref_state_prep = driver_ref
        else:
            ref_prog = pq.Program()
            for i in range(self.n_qubits):
                ref_prog.inst(H(i))
            self.ref_state_prep = ref_prog

        if not isinstance(cost_ham, (list, tuple)):
            raise TypeError("""cost_hamiltonian must be a list of PauliSum
                               objects""")
        if not all([isinstance(x, PauliSum) for x in cost_ham]):
            raise TypeError("""cost_hamiltonian must be a list of PauliSum
                                   objects""")
        else:
            self.cost_ham = cost_ham

        if not isinstance(ref_hamiltonian, (list, tuple)):
            raise TypeError("""cost_hamiltonian must be a list of PauliSum
                               objects""")
        if not all([isinstance(x, PauliSum) for x in ref_hamiltonian]):
            raise TypeError("""cost_hamiltonian must be a list of PauliSum
                                   objects""")
        else:
            self.ref_ham = ref_hamiltonian

        if minimizer is None:
            self.minimizer = optimize.minimize
        else:
            self.minimizer = minimizer
        # minimizer_kwargs initialized to empty dictionary
        if len(minimizer_kwargs) == 0:
            self.minimizer_kwargs = {'method': 'Nelder-Mead',
                                     'options': {'disp': True,
                                                 'ftol': 1.0e-2,
                                                 'xtol': 1.0e-2}}
        else:
            self.minimizer_kwargs = minimizer_kwargs

        self.minimizer_args = minimizer_args

        if rand_seed is not None:
            np.random.seed(rand_seed)
        if self.betas is None:
            self.betas = np.random.uniform(0, np.pi, self.steps)[::-1]
        if self.gammas is None:
            self.gammas = np.random.uniform(0, 2*np.pi, self.steps)

    def get_parameterized_program(self):
        """
        Return a function that accepts parameters and returns a new Quil
        program

        :returns: a function
        """
        cost_para_programs = []
        driver_para_programs = []

        for idx in range(self.steps):
            cost_list = []
            driver_list = []
            for cost_pauli_sum in self.cost_ham:
                for term in cost_pauli_sum.terms:
                    cost_list.append(exponential_map(term))

            for driver_pauli_sum in self.ref_ham:
                for term in driver_pauli_sum.terms:
                    driver_list.append(exponential_map(term))

            cost_para_programs.append(cost_list)
            driver_para_programs.append(driver_list)

        def psi_ref(params):
            """Construct a Quil program for the vector (beta, gamma).

            :param params: array of 2 . p angles, betas first, then gammas
            :return: a pyquil program object
            """
            if len(params) != 2*self.steps:
                raise ValueError("""params doesn't match the number of parameters set
                                    by `steps`""")
            betas = params[:self.steps]
            gammas = params[self.steps:]

            prog = pq.Program()
            prog += self.ref_state_prep
            for idx in range(self.steps):
                for fprog in cost_para_programs[idx]:
                    prog += fprog(gammas[idx])

                for fprog in driver_para_programs[idx]:
                    prog += fprog(betas[idx])

            return prog

        return psi_ref

    def get_angles(self):
        """
        Finds optimal angles with the quantum variational eigensolver method.

        Stored VQE result

        :returns: ([list], [list]) A tuple of the beta angles and the gamma
                  angles for the optimal solution.
        """
        stacked_params = np.hstack((self.betas, self.gammas))
        vqe = VQE(self.minimizer, minimizer_args=self.minimizer_args,
                  minimizer_kwargs=self.minimizer_kwargs)
        cost_ham = reduce(lambda x, y: x + y, self.cost_ham)
        # maximizing the cost function!
        param_prog = self.get_parameterized_program()
        result = vqe.vqe_run(param_prog, cost_ham, stacked_params, qvm=self.qvm,
                             **self.vqe_options)
        self.result = result
        betas = result.x[:self.steps]
        gammas = result.x[self.steps:]
        return betas, gammas

    def probabilities(self, angles):
        """
        Computes the probability of each state given a particular set of angles.

        :param angles: [list] A concatenated list of angles [betas]+[gammas]
        :return: [list] The probabilities of each outcome given those angles.
        """
        if isinstance(angles, list):
            angles = np.array(angles)

        assert angles.shape[0] == 2 * self.steps, "angles must be 2 * steps"
        param_prog = self.get_parameterized_program()
        prog = param_prog(angles)
        #prog = new_gs_circuit()
        #f = open("myfile.txt","wb")
        #for g in prog.instructions:
        #    print(str(g), file=f)
        #print(type(prog.instructions[0]))
        #q = Program().inst(prog.instructions)
        print(prog)
        #noisy = add_decoherence_noise(prog, T1=t1, T2=t2) #.inst([MEASURE(0, 0),MEASURE(1, 1),])
        #wf = self.qvm.wavefunction(noisy)
        wf = self.qvm.wavefunction(prog)
        wf = wf.amplitudes.reshape((-1, 1))
        probs = np.zeros_like(wf)
        for xx in range(2 ** self.n_qubits):
            probs[xx] = np.conj(wf[xx]) * wf[xx]
        return probs
 
    def get_probs(self, t1, t2):
        prog = new_circuit()
        prog = add_decoherence_noise(prog, T1=t1, T2=t2)
        #result = self.qvm.run_and_measure(prog, [0,1,2,3,4,5,6,7], trials=100000)      
        #print(len(result))
        #print(result.count([0]*(2**3)) )
        print('Hola')
        #wf = self.qvm.wavefunction(prog)
        #wf = wf.amplitudes.reshape((-1, 1))
        #probs = np.zeros_like(wf)
        #for xx in range(2 ** self.n_qubits):
        #0    probs[xx] = np.conj(wf[xx]) * wf[xx]        
        #print(probs[0][0].real)
        #return probs[0][0].real        
        
    def get_probs_grover(self, t1, t2):
        prog = new_circuit_grover3()
        prog = add_decoherence_noise(prog, T1=t1, T2=t2)
        #result = self.qvm.run_and_measure(prog, [0,1,2,3,4,5,6,7], trials=100000)      
        #print(len(result))
        #print(result.count([0]*(2**3)) )
        wf = self.qvm.wavefunction(prog)
        wf = wf.amplitudes.reshape((-1, 1))
        probs = np.zeros_like(wf)
        for xx in range(2 ** self.n_qubits):
            probs[xx] = np.conj(wf[xx]) * wf[xx]        
        print(probs[0][0].real)
        return probs[0][0].real 

    def probabilities2(self, angles, t1, t2):
        prog = new_circuit()
        prog = add_decoherence_noise(prog, T1=t1, T2=t2)
        wf = self.qvm.wavefunction(prog)
        wf = wf.amplitudes.reshape((-1, 1))
        probs = np.zeros_like(wf)
        for xx in range(2 ** self.n_qubits):
            probs[xx] = np.conj(wf[xx]) * wf[xx]
        return probs
               
        #print(prog)
        #return prog


    def probabilities2grover(self, angles, t1, t2):
        prog = new_circuit_grover3()
        print(prog)
        wf = self.qvm.wavefunction(prog)
        wf = wf.amplitudes.reshape((-1, 1))
        probs = np.zeros_like(wf)
        for xx in range(2 ** self.n_qubits):
            probs[xx] = np.conj(wf[xx]) * wf[xx]
        return probs      


    def get_string(self, betas, gammas, samples=100):
        """
        Compute the most probable string.

        The method assumes you have passed init_betas and init_gammas with your
        pre-computed angles or you have run the VQE loop to determine the
        angles.  If you have not done this you will be returning the output for
        a random set of angles.

        :param betas: List of beta angles
        :param gammas: List of gamma angles
        :param samples: (int, Optional) number of samples to get back from the
                        QVM.
        :returns: tuple representing the bitstring, Counter object from
                  collections holding all output bitstrings and their frequency.
        """
        if samples <= 0 and not isinstance(samples, int):
            raise ValueError("samples variable must be positive integer")
        param_prog = self.get_parameterized_program()
        stacked_params = np.hstack((betas, gammas))
        sampling_prog = param_prog(stacked_params)
        for i in range(self.n_qubits):
            sampling_prog.measure(i, [i])

        bitstring_samples = self.qvm.run_and_measure(sampling_prog,
                                                     list(range(self.n_qubits)),
                                                     trials=samples)
        bitstring_tuples = list(map(tuple, bitstring_samples))
        freq = Counter(bitstring_tuples)
        most_frequent_bit_string = max(freq, key=lambda x: freq[x])
        return most_frequent_bit_string, freq
