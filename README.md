# Taiho Solver 

> A lightweight, parallel-friendly CFD solver inspired by the legendary **IJN Taihō (大鳳)** and *Kantai Collection*.  
> Designed with love for large-scale incompressible flow simulations. Built for speed, stability, and science! 

---

## ✨ Overview

**Taiho** is a C++ based CFD solver specialized for **incompressible flows**, supporting both **steady-state** and **transient** simulations.  
Named after the armored aircraft carrier *Taihō*, she's armored, reliable, and surprisingly nimble — just like the solver.

Whether you're commanding a single-core destroyer or a full-parallel battlefleet of CPUs, Taiho adapts gracefully. 

---

## 💡 Features

- ✅ **Simple Algorithm (SIMPLE)** for steady-state problems  
- ✅ **PISO & SIMPLE** schemes for transient simulations  
- ✅ Supports **structured mesh** generation via built-in C++ tool  
- ✅ Includes **parallel CG** and **parallel BiCGStab** solvers for large linear systems  
- ✅ **OpenMP** and **MPI**-enabled for parallel computation  
- ✅ Friendly Python postprocessing via Jupyter Notebook (`.ipynb`)

---
