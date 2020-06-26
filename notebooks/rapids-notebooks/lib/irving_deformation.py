#pseudocode from irving 2002 on invertable deformation : re : triangles
import numpy as np

#define edge vectors for each triangle
dm1 = X1 - X0  #material coordinates
dm2 = X2 - X0
ds1 = x1 - x0  #world coordinates
ds2 = x2 - x0

Dm = np.concatenate(dm1, dm2)
Ds = np.concatenate(ds1, ds2)

#deformation map is 2x2 in 2 dimensions, 3x3 in 3 dimensions
F = Ds/Dm #Dm is matrix inverse, which should be precomputed

#then, the spatial coordinates are Ds = F*Dm
#inverted triangle has det F < 0
#isotropic material means we can do a QR decomposition on Dm

# I am calculating the force due to pressure correctly, 
# but I should precompute bi, the part that can be done in 
# the unchanging material coordinates.
gi = -P(A1 + A2 + A3 + ...)/3 = P * bi

Bm = (b1,b2,b3)#precompute
#then the nodal forces are linearly related to P
G = P * Bm

#P is the first Piola-Kirchoff stress, here

#the pressure force I was so excited about appears to be 
# the first Piola- Kirchhoff stress.  But wait... that's a stress not a pressure... !!

#isotropic material iff P is rotationally invariant.

# use the modified neo-hookean constitutive relation
# "we modify the constitutive model near the origin to remove the 
# singularity by either lin- earizing at a given compression 
# imit or simply clamping the stress at some maximum value.
# The resulting model is identical to the phys- ical model
 # most of the time, and allows the simulation to continue 
 # if a few tetrahedrons invert. Furthermore, our ex- tensions 
 # provide C0 or C1 continuity around the flat case, which 
 # avoids sudden jumps or oscillations which might effect 
 # neighboring elements.
# " It is exceedingly dif- ficult to measure material 
# response in situations of extreme compression, so 
# constitutive models are often measured for moderate 
# deformation and continued heuristically down to the flat cases."

#HOW to model constitutive relations for biological tissues
# For example, most biological material is soft under small 
# defor- mation, but becomes stiffer as the deformation 
# increases. A simple model capturing this behavior is given 
# y choosing threshold values for compression and elongation, specifying the slope of the stress curve outside these threshold values and at the undeformed state, and using a cubic spline to in- terpolate between them.

###--> Hence, it is perhaps unpublished the angular momentum imparted by cardiac models exhibiting anisotropic diffusion tensors.




