# string_to_index = {
#     'k_1': 0,
#     'k_2': 1,
#     'k_3': 2,
#     'x_0': 3,
#     'x_1': 4,
#     'x_2': 5
# }

string_to_index = {
    'transition_length': 0,
    'k_1': 1,
    'x_0': 2
}

def to_param(a, params):
    if isinstance(a, str):
        return params[string_to_index[a]]
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
# constraints = [
#     # # Ordering constraints
#     greater_than('x_2', 'x_1'),
#     greater_than('x_1', 'x_0'),
#     greater_than('k_3', 'k_2'),
#     greater_than('k_2', 'k_1'),

#     # # Slack length is zero
#     greater_than('x_0', 0),
#     greater_than(0, 'x_0'),

#     # # Stop it from being close to zero
#     greater_than('x_1', 0.5),

#     # # From models, k_2 is generally twice k_1
#     n_a_greater_m_b(2, 'k_1', 1, 'k_2'),
#     n_a_greater_m_b(1, 'k_2', 2, 'k_1'),

#     # # From models, x_2 is generally twice x_1
#     n_a_greater_m_b(1, 'x_2', 2, 'x_1'),
# ]

constraints = [
    greater_than('x_0', 0),
    greater_than(0, 'x_0'),
]



