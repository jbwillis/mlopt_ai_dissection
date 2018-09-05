include("../src/MyModule.jl")

# Generate data
# -------------
problem = MyModule.Assignment()
problem.A = 50
theta_dim = 50  # A

# Generate training data points
theta_bar = zeros(theta_dim)

N_train = 1000
N_test = 100
radius = 1.
theta_train = MyModule.sample(problem, theta_bar, radius, N=N_train)
theta_test = MyModule.sample(problem, theta_bar, radius, N=N_test)

# Learn
# -----
srand(1)

# Get active_constr for each point
y_train, enc2active_constr = MyModule.encode(MyModule.active_constraints(theta_train, problem))

# Learn tree
lnr = MyModule.tree(theta_train, y_train, export_tree=true, problem=problem)


# Test
# ------
# Evaluate performance
df, df_detail = MyModule.eval_performance(theta_test, lnr, problem, enc2active_constr)

# Store results
MyModule.write_output(df, df_detail, problem)
