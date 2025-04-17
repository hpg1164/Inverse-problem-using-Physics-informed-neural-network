The Poisson equation on a unit square 
Ω
=
[
0
,
1
]
×
[
0
,
1
]
Ω=[0,1]×[0,1] with zero Dirichlet boundary condition can be written as:

−
∇
⋅
(
𝑐
∇
𝑢
)
=
1
in 
Ω
,
𝑢
=
0
on 
∂
Ω
,
−∇⋅(c∇u)=1in Ω,u=0on ∂Ω,
where 
𝑐
c is the thermal diffusivity. This equation is relevant in thermodynamics, where it represents the steady-state heat equation. The boundary of the square is held at a constant temperature of 0, the right-hand side represents a uniform volumetric heat source, and 
𝑐
c governs how heat diffuses through the material.

When 
𝑐
=
1
c=1, one possible exact solution to this equation on the unit square is:

𝑢
(
𝑥
,
𝑦
)
=
1
2
𝜋
2
sin
⁡
(
𝜋
𝑥
)
sin
⁡
(
𝜋
𝑦
)
u(x,y)= 
2π 
2
 
1
​
 sin(πx)sin(πy)
(Note: This is a classical example solution for illustrative purposes, as exact solutions on the square generally involve series solutions.)

In the forward problem, we solve for 
𝑢
(
𝑥
,
𝑦
)
u(x,y) given the thermal diffusivity 
𝑐
c, source term, and boundary conditions. However, in the inverse problem, we do the opposite: given a known temperature distribution 
𝑢
(
𝑥
,
𝑦
)
u(x,y) at a set of points, known source term, and boundary conditions, the goal is to determine the unknown coefficient 
𝑐
c.

We can approach this using a Physics-Informed Neural Network (PINN), where we optimize both the parameters of the neural network representing 
𝑢
(
𝑥
,
𝑦
)
u(x,y) and the scalar parameter 
𝑐
c. In this example, we assume the data is given by the exact solution above. In practical scenarios, the exact solution is typically unknown, but measurement data for 
𝑢
u at scattered points in the domain is available instead.
