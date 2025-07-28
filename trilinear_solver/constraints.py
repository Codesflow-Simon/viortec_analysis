string_to_index_trilinear = {
    'k_1': 0,
    'k_2': 1,
    'k_3': 2,
    'x_1': 3,
    'x_2': 4
}

string_to_index_blankevoort = {
    'e_t': 0,
    'k_1': 1,
}

def to_param(a, params):
    if isinstance(a, str):
        return params[string_to_index_trilinear[a]]
    else:
        return a

    
def greater_than(a, b):
    func =  lambda params: to_param(a, params) - to_param(b, params)
    const = {'type': 'ineq', 'fun': func}
    return const

def n_a_greater_m_b(n, a, m, b):
    func = lambda params: n * to_param(a, params) - m * to_param(b, params)
    const = {'type': 'ineq', 'fun': func}
    return const


# Set up constraints as inequality constraints (g(x) >= 0)
trilinear_param_names = ['k_1', 'k_2', 'k_3', 'x_1', 'x_2']
trilinear_constraints = [
    # # Ordering constraints
    greater_than('x_2', 'x_1'),

    greater_than('k_3', 'k_2'),
    greater_than('k_2', 'k_1'),

    # # From models, k_2 is generally twice k_1
    n_a_greater_m_b(2, 'k_1', 1, 'k_2'),
    n_a_greater_m_b(1, 'k_2', 2, 'k_1'),

    # Some bounds to stop the optimisation from exploding
    greater_than('x_1', 0.05),
    greater_than(0.25, 'x_2'),
    n_a_greater_m_b(1, 'x_2', 2, 'x_1'),

]

blankevoort_param_names = ['e_t', 'k_1']
blankevoort_constraints = [
    greater_than('e_t', 0.01),
]




