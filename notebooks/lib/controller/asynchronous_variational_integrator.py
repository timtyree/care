#asynchronous_variational_integrator.py
####################################################################
# Tim Tyree
# 8.28.2020
# asynchronous_variational_integrator algorithm
# Molecular Dynamics optimized (small number of time step sizes)
####################################################################
# Nota bene: for structural dynamics, 
# - consider only structural forces (a = [spring_forces])
# - consider more carfully your collection of time steps. 
# See 'Stability of AVIs.pdf', Fig. 1/??

# Input
#initial priority also time?/state?
theta0
#initial configuration
x0
#initial velocity
v0
#set of all potential times
t[j][k] = #np.linspace(TMIN,TMAX,DT), with K substeps at uniform spacing

# Output
# for all i
theta[i]
x[i]
v[i]

##############################################
# Initialization
##############################################
i = 0
v = v0
x = x0
theta_old = theta0

# with K substeps at uniform spacing
K = 7
k_values = np.linspace(0,DT,DT/K)
for k in k_values:
	Push (t[0][k]) into the priority_queue#, Q
	M[k] = #size of array t[j][k]

##############################################
# Integrate the system over the time interval [0,T]
##############################################
while not priority_queue.is_empty() do:
	Pop the top element (t[j][k]) from priority_queue
	theta_new = t[j][k]

	# if the new priority is greater than the old priority
	if theta_new>theta_old:
		# Half-kick
		for all a do:
			v[a] = v[a] + 0.5*F_m[a]/m[a]
		x[i] = x; v[i] = v; theta[i] = theta_old
		i = i + 1

		# Half-kick
		for all a do:
			v[a] = v[a] + 0.5*F_p[a]/m[a]

		# Drift
		x = x ( theta_new - theta_old) * v
		# F[+1/2] = F[-1/2] = 0
		F_p = F_m = 0
	theta_old = theta_new
	if j>0:
		F_m = F_m - (t[j][k] - t[j-1][k])*grad(V[k](x))
	if j<M[k]:
		F_p = F_p - (t[j+1][k] - t[j][k])*grad(V[k](x))
		Push (t[j+1][k]) into priority_queue

# Half-kick
for all a do:
	v[a] = v[a] + 0.5*F_m[a]/m[a]
x[i] = x; v[i] = v; theta[i] = theta_old


