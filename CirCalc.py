import numpy as np
from itertools import product

sqr2 = 1 / np.sqrt(2) 

class cirbuild():
    identity = np.array([[1,0],[0,1]])
    x_gate = np.array([[0,1],[1,0]])
    y_gate = np.array([[0,-1j],[1j,0]])
    z_gate = np.array([[1,0],[0,-1]])
    h_gate = np.array([[sqr2,sqr2],[sqr2,-sqr2]])
    


    outer_zero = np.array([[1,0],[0,0]])
    outer_one = np.array([[0,0],[0,1]])


    
    def __init__(self,n):
        self.num_qubits = n
        self.state = '|'
        self.times_state_called = 0
        self.circuit_matrix = np.zeros((2 ** self.num_qubits,1),dtype=int)
        self.circuit_matrix[0][0] = 1
        self.state_list = cirbuild.states(self)

    def states(self):
        state_list = []
        state_fini_list = []
        state_init_list = list(product([0,1], repeat= self.num_qubits))
        for stut in state_init_list:
            state_fini_list.append((''.join(map(str, stut))))
        for state in state_fini_list:
            state_list.append(f'|{state}>')
        return state_list
        

    def circuit_init(self):
        for i in range(self.num_qubits):
            self.state += '0'
        self.state += '>'

    def x(self,qubit):
        qubit_set = self.circuit_matrix
        before = self.num_qubits - qubit -1
        qubit_pos = cirbuild.x_gate
        after = self.num_qubits - before - 1
        gate_set = 1
        for num in range(before):
            gate_set = np.kron(cirbuild.identity,gate_set)

        gate_set = np.kron(cirbuild.x_gate,gate_set)

        for num in range(after):
            gate_set = np.kron(cirbuild.identity,gate_set)
        
        self.circuit_matrix = np.dot(gate_set,qubit_set)
        
    
    def h(self,qubit):
        qubit_set = self.circuit_matrix
        before = self.num_qubits - qubit -1
        qubit_pos = cirbuild.h_gate
        after = self.num_qubits - before - 1
        gate_set = 1
        for num in range(before):
            gate_set = np.kron(cirbuild.identity,gate_set)

        gate_set = np.kron(cirbuild.h_gate,gate_set)

        for num in range(after):
            gate_set = np.kron(cirbuild.identity,gate_set)
        
        self.circuit_matrix = np.dot(gate_set,qubit_set)


    def y(self,qubit):
        qubit_set = self.circuit_matrix
        before = self.num_qubits - qubit -1
        qubit_pos = cirbuild.y_gate
        after = self.num_qubits - before - 1
        gate_set = 1
        for num in range(before):
            gate_set = np.kron(cirbuild.identity,gate_set)

        gate_set = np.kron(cirbuild.y_gate,gate_set)

        for num in range(after):
            gate_set = np.kron(cirbuild.identity,gate_set)
        
        self.circuit_matrix = np.dot(gate_set,qubit_set)

    def z(self,qubit):
        qubit_set = self.circuit_matrix
        before = self.num_qubits - qubit -1
        qubit_pos = cirbuild.z_gate
        after = self.num_qubits - before - 1
        gate_set = 1
        for num in range(before):
            gate_set = np.kron(cirbuild.identity,gate_set)

        gate_set = np.kron(cirbuild.z_gate,gate_set)

        for num in range(after):
            gate_set = np.kron(cirbuild.identity,gate_set)

        
        self.circuit_matrix = np.dot(gate_set,qubit_set)

    def rx(self,qubit,theta):
        theta_by_two = np.deg2rad(float(theta)/2)
        rx_gate = np.array([[np.cos(theta_by_two), -1j * np.sin(theta_by_two)],[-1j * np.sin(theta_by_two), np.cos(theta_by_two)]])
        qubit_set = self.circuit_matrix
        before = self.num_qubits - qubit -1
        qubit_pos = cirbuild.x_gate
        after = self.num_qubits - before - 1
        gate_set = 1
        for num in range(before):
            gate_set = np.kron(cirbuild.identity,gate_set)

        gate_set = np.kron(rx_gate,gate_set)

        for num in range(after):
            gate_set = np.kron(cirbuild.identity,gate_set)
        
        self.circuit_matrix = np.dot(gate_set,qubit_set)

    def ry(self,qubit,theta):
        theta_by_two = np.deg2rad(float(theta)/2)
        rx_gate = np.array([[np.cos(theta_by_two), -1 * np.sin(theta_by_two)],[ np.sin(theta_by_two), np.cos(theta_by_two)]])
        qubit_set = self.circuit_matrix
        before = self.num_qubits - qubit -1
        qubit_pos = cirbuild.x_gate
        after = self.num_qubits - before - 1
        gate_set = 1
        for num in range(before):
            gate_set = np.kron(cirbuild.identity,gate_set)

        gate_set = np.kron(rx_gate,gate_set)

        for num in range(after):
            gate_set = np.kron(cirbuild.identity,gate_set)
        
        self.circuit_matrix = np.dot(gate_set,qubit_set)

    def rz(self,qubit,theta):
        thetabytwo = np.deg2rad(float(theta)/2)
        coes= np.cos(thetabytwo)
        sint = 1j * np.sin(thetabytwo)
        if abs(sint) < 1.0e-2:
            sint = 0
        euler = coes + sint
        antieuler = coes - sint
        phase_gate = np.array([[antieuler,0],[0,euler]])
        qubit_set = self.circuit_matrix
        before = self.num_qubits - qubit -1
        qubit_pos = cirbuild.x_gate
        after = self.num_qubits - before - 1
        gate_set = 1
        for num in range(before):
            gate_set = np.kron(cirbuild.identity,gate_set)

        gate_set = np.kron(phase_gate,gate_set)

        for num in range(after):
            gate_set = np.kron(cirbuild.identity,gate_set)
        
        current_cmatrix = np.dot(gate_set,qubit_set)

        self.circuit_matrix = np.where(
        (np.abs(current_cmatrix.real) > 0) & (np.abs(current_cmatrix.real) < 1.0e-3), 
            1j * current_cmatrix.imag, current_cmatrix)
    
    def cx(self,cqubit,tqubit):
        qubit_set = self.circuit_matrix
        big_one = max(cqubit,tqubit)
        small_one = min(cqubit,tqubit) 
        before = small_one
        inbetween = abs(tqubit - cqubit) - 1
        after = self.num_qubits - big_one - 1

        if cqubit > tqubit:
            first_1= cirbuild.x_gate
            first_0 = cirbuild.identity
            second_0 = cirbuild.outer_zero
            second_1 = cirbuild.outer_one
        else:
            second_1 = cirbuild.x_gate
            second_0 = cirbuild.identity
            first_0 = cirbuild.outer_zero
            first_1 = cirbuild.outer_one

        gate_set = 1
        for num in range(before):
            gate_set = np.kron(gate_set,cirbuild.identity)

        zero_position = np.kron(gate_set,first_0)
        one_position = np.kron(gate_set,first_1)
        
        for num in range(inbetween):
            zero_position = np.kron(zero_position,cirbuild.identity)
            one_position = np.kron(one_position,cirbuild.identity)


        zero_position = np.kron(zero_position,second_0)
        one_position = np.kron(one_position,second_1)


        for num in range(after):
            zero_position = np.kron(zero_position,cirbuild.identity)
            one_position = np.kron(one_position,cirbuild.identity)

        final_gate_matrix = zero_position + one_position
        self.circuit_matrix = np.dot(final_gate_matrix,qubit_set)

    def p(self,qubit,theta):
        theta = np.deg2rad(float(theta))
        coes= np.cos(theta)
        sint = 1j * np.sin(theta)
        if abs(sint) < 1.0e-2:
            sint = 0
        euler = coes + sint
        phase_gate = np.array([[1,0],[0,euler]])
        qubit_set = self.circuit_matrix
        before = self.num_qubits - qubit -1
        qubit_pos = cirbuild.x_gate
        after = self.num_qubits - before - 1
        gate_set = 1
        for num in range(before):
            gate_set = np.kron(cirbuild.identity,gate_set)

        gate_set = np.kron(phase_gate,gate_set)

        for num in range(after):
            gate_set = np.kron(cirbuild.identity,gate_set)
        
        current_cmatrix = np.dot(gate_set,qubit_set)

        self.circuit_matrix = np.where(
        (np.abs(current_cmatrix.real) > 0) & (np.abs(current_cmatrix.real) < 1.0e-3), 
            1j * current_cmatrix.imag, current_cmatrix)


    def cz(self,cqubit,tqubit):
        qubit_set = self.circuit_matrix
        big_one = max(cqubit,tqubit)
        small_one = min(cqubit,tqubit) 
        before = small_one
        inbetween = abs(tqubit - cqubit) - 1
        after = self.num_qubits - big_one - 1

        if cqubit > tqubit:
            first_1= cirbuild.z_gate
            first_0 = cirbuild.identity
            second_0 = cirbuild.outer_zero
            second_1 = cirbuild.outer_one
        else:
            second_1 = cirbuild.z_gate
            second_0 = cirbuild.identity
            first_0 = cirbuild.outer_zero
            first_1 = cirbuild.outer_one

        gate_set = 1
        for num in range(before):
            gate_set = np.kron(gate_set,cirbuild.identity)

        zero_position = np.kron(gate_set,first_0)
        one_position = np.kron(gate_set,first_1)
        
        for num in range(inbetween):
            zero_position = np.kron(zero_position,cirbuild.identity)
            one_position = np.kron(one_position,cirbuild.identity)


        zero_position = np.kron(zero_position,second_0)
        one_position = np.kron(one_position,second_1)


        for num in range(after):
            zero_position = np.kron(zero_position,cirbuild.identity)
            one_position = np.kron(one_position,cirbuild.identity)

        final_gate_matrix = zero_position + one_position
        self.circuit_matrix = np.dot(final_gate_matrix,qubit_set)

    def ccx(self,cqubit1,cqubit2,tqubit):
        qubit_set = self.circuit_matrix
        bigc = max(cqubit1,cqubit2)
        smallc = min(cqubit1,cqubit2)

        if tqubit > bigc:
            before = smallc
            inbetween_1 = bigc - smallc - 1
            inbetween_2 = tqubit - bigc - 1
            after = self.num_qubits - tqubit - 1

            first_0 = cirbuild.outer_zero
            first_1 = cirbuild.outer_one
            second_0 = cirbuild.outer_zero
            second_1 = cirbuild.outer_one
            third_0 = cirbuild.identity
            thrid_1 = cirbuild.x_gate


        elif tqubit < smallc:
            before = tqubit
            inbetween_1 = smallc - tqubit - 1
            inbetween_2 = bigc - smallc - 1
            after = self.num_qubits - bigc - 1

            first_0 = cirbuild.identity
            first_1 = cirbuild.x_gate
            second_0 = cirbuild.outer_zero
            second_1 = cirbuild.outer_one
            third_0 = cirbuild.outer_zero
            thrid_1 = cirbuild.outer_one

        else: 
            before = smallc
            inbetween_1 = tqubit - smallc - 1
            inbetween_2 = bigc - tqubit - 1
            after = self.num_qubits - bigc - 1

            first_0 = cirbuild.outer_zero
            first_1 = cirbuild.outer_one
            second_0 = cirbuild.identity
            second_1 = cirbuild.x_gate
            third_0 = cirbuild.outer_zero
            thrid_1 = cirbuild.outer_one
        

        gate_set = 1
        for num in range(before):
            gate_set = np.kron(gate_set,cirbuild.identity)
        zero_position = np.kron(gate_set,first_0)
        one_position = np.kron(gate_set,first_1)

        for num in range(inbetween_1):
            zero_position = np.kron(zero_position,cirbuild.identity)
            one_position = np.kron(one_position,cirbuild.identity)

        zero_position = np.kron(zero_position,second_0)
        one_position = np.kron(one_position,second_1)

        for num in range(inbetween_2):
            zero_position = np.kron(zero_position,cirbuild.identity)
            one_position = np.kron(one_position,cirbuild.identity)

        zero_position = np.kron(zero_position,third_0)
        one_position = np.kron(one_position,thrid_1)


        for num in range(after):
            zero_position = np.kron(zero_position,cirbuild.identity)
            one_position = np.kron(one_position,cirbuild.identity)

        final_gate_matrix = zero_position + one_position
        self.circuit_matrix = np.dot(final_gate_matrix,qubit_set)


    def circ_state(self):
        new_list = []
        total_states = 2 ** self.num_qubits
        for state in range(total_states):
            coeff = (self.circuit_matrix[state][0])
            if abs(coeff) < 1.0e-3:
                continue
            if abs(coeff.imag) > 0 and abs(coeff.real) > 0:
                z_list = str(coeff).split('+')
                new_list.append(f"({z_list[0][:6].replace('(','')} + {z_list[1][:5].replace(')','')}j){self.state_list[state]}")


            elif abs(coeff.imag) > 0:

                new_list.append(f"{str(coeff.imag)[:6].replace('(','').replace(')','')}j{self.state_list[state]}")

                
            
            elif abs(coeff.real) > 0:
                new_list.append(f"{str(coeff.real)[:6].replace('(','').replace(')','')}{self.state_list[state]}")

            
            else:
                continue

        
        final_string = ''
        for item in new_list:
            if item == new_list[-1]:
                final_string += item
                continue
            final_string += f'{item} + '
        print(final_string)

    def probs(self):
        total_states = 2 ** self.num_qubits
        print('###########')
        for state in range(total_states):
            prob = np.abs(self.circuit_matrix[state][0]) ** 2
            if abs(prob) < 1.0e-3:

                continue
            print((f'{str(prob)[:6]} probability of state {self.state_list[state]}'))
        print('###########')

    def cmatrix(self):
        print(self.circuit_matrix)





