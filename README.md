## 🧠 Inverse Problem Using Physics-Informed Neural Networks (PINNs)

This project addresses the **Poisson equation** on a **unit square domain** \(\Omega = [0,1] \times [0,1]\) with **zero Dirichlet boundary conditions**:
![image](https://github.com/user-attachments/assets/d02992c6-25bd-4bf6-8978-1c2fbd8aa539)


Here, `c` is the **thermal diffusivity**. This equation is commonly encountered in **thermodynamics**, representing the **steady-state heat conduction** problem. The boundary of the square is held at a constant temperature of 0, and the source term on the right-hand side represents a **uniform volumetric heat source**. The coefficient `c` determines how heat diffuses through the material.

When `c = 1`, a classical example of an exact solution to this equation is:
![image](https://github.com/user-attachments/assets/5ef853bd-9577-45cb-bd15-52c7719c2125)


In the typical **forward problem**, the goal is to compute the temperature distribution `u(x, y)` given the coefficient `c`, source term, and boundary conditions.

In contrast, we focus on the **inverse problem**:

> Given the solution `u(x, y)` at certain points in the domain, the known source term, and the boundary conditions, **can we recover the unknown thermal diffusivity `c`?**

To solve this, we use a **Physics-Informed Neural Network (PINN)**. The network learns the solution `u(x, y)` and simultaneously optimizes the unknown parameter `c` by minimizing a loss function that incorporates the governing PDE, boundary conditions, and available data.

In this example, we use synthetic data generated from the known exact solution. However, the same approach can be extended to real-world applications where the solution is measured experimentally rather than known analytically.
