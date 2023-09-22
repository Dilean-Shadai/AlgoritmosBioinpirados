'''Algoritmos Bioinspirados | Sep 2023 | Dilean Shadai García
Algoritmo de Recocido Simulado para minimizar una función: min F(x) = x2

Notas: 
El éxito del Recocido Simulado se basa en la escogencia de una buena temperatura inicial (TO)
y una adecuada velocidad de enfriamiento (r).

'''

#Librerias necesarias 
import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt  
import matplotlib as mpl

def annealing(random_start, #Función para llevar acabo el recocido
              cost_function,
              random_neighbour,
              acceptance,
              temperature,
              maxsteps=1000, #Realizara mil iteraciones 
              debug=True):
    """ Optimiza la función de caja negra con el algoritmo de recocido simulado"""
    state = random_start()
    cost = cost_function(state)
    states, costs = [state], [cost]
    for step in range(maxsteps):
        fraction = step / float(maxsteps)
        T = temperature(fraction)
        new_state = random_neighbour(state, fraction)
        new_cost = cost_function(new_state)
       
        if acceptance_probability(cost, new_cost, T) > rn.random():
            state, cost = new_state, new_cost
            states.append(state)
            costs.append(cost)
           
    return state, cost_function(state), states, costs

#Funcion objetivo
interval = (-10, 10)

def f(x):
    """ Function to minimize."""
    return x ** 2

def clip(x):
    """ Force x to be in the interval."""
    a, b = interval
    return max(min(x, b), a)
def cost_function(x):
    """ Cost of x = f(x)."""
    return f(x)

#vacindario usa un vecino aleatorio
def random_start():
    #Punto aleatorio intermedio en el intervalo
    a, b = interval
    return a + (b - a) * rn.random_sample()
def random_neighbour(x, fraction=1):
    """Move a little bit x, from the left or the right."""
    amplitude = (max(interval) - min(interval)) * fraction / 10
    delta = (-amplitude/2.) + amplitude * rn.random_sample()
    return clip(x + delta)

#Probabilidad de aceptación
def acceptance_probability(cost, new_cost, temperature):
    if new_cost < cost:
        return 1
    else:
        p = np.exp(- (new_cost - cost) / temperature)
        return p
    
#Temperatura
def temperature(fraction):
    """ Example of temperature dicreasing as the process goes on."""
    return max(0.01, min(1, 1 - fraction))

#ver el resultado con 100 iteraciones
state, c, states, costs = annealing(random_start, cost_function, random_neighbour, acceptance_probability, temperature, maxsteps=100, debug=False)

print(state,"-", c)

annealing(random_start, cost_function, random_neighbour, acceptance_probability, temperature, maxsteps=100, debug=True)


