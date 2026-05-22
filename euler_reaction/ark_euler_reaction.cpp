/* -----------------------------------------------------------------------------
 * Programmer(s): Daniel R. Reynolds @ SMU
 * -----------------------------------------------------------------------------
 * SUNDIALS Copyright Start
 * Copyright (c) 2002-2024, Lawrence Livermore National Security
 * and Southern Methodist University.
 * All rights reserved.
 *
 * See the top-level LICENSE and NOTICE files for details.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * SUNDIALS Copyright End
 * -----------------------------------------------------------------------------
 * This example solves the 1D compressible Euler equations in conserved
 * variables, along with reactive transport of 3 chemical species,
 * over the domain (t,x) in [0, 0.2] x [0, 1].
 *
 * This problem should be run with homogeneous Neumann boundary conditions.
 *
 * The system is advanced in time using an additive Runge--Kutta method from
 * ARKStep, where the compressible Euler equations are evolved explicitly, and the
 * reaction terms are evolved implicitly.  For the implicit systems, we use a
 * custom "matrix embedded linear solver", that leverages the block diagonal
 * Jacobian structure of the reaction operator.
 *
 * Several additional command line options are available to change the
 * initial conditions and integrator settings. Use the flag --help for more
 * information.
 * ---------------------------------------------------------------------------*/

#include "ark_euler_reaction.hpp"
using namespace std;

int main(int argc, char* argv[])
{
  // SUNDIALS context object for this simulation
  sundials::Context ctx;

  // -----------------
  // Setup the problem
  // -----------------

  EulerData udata(ctx);
  ARKODEParameters uopts;

  vector<string> args(argv + 1, argv + argc);

  int flag = ReadInputs(args, udata, uopts, ctx);
  if (check_flag(flag, "ReadInputs")) { return 1; }
  if (flag > 0) { return 0; }

  flag = PrintSetup(udata, uopts);
  if (check_flag(flag, "PrintSetup")) { return 1; }

  // Create state vector and set initial condition
  N_Vector vecs[NSPECIES];
  for (int i = 0; i < NSPECIES; i++)
  {
    vecs[i] = N_VNew_Serial((sunindextype)udata.nx, ctx); // rho (density)
    if (check_ptr(vecs[i], "N_VNew_Serial")) { return 1; }
  }
  N_Vector y = N_VNew_ManyVector(NSPECIES, vecs, ctx);
  if (check_ptr(y, "N_VNew_ManyVector")) { return 1; }

  flag = SetIC(y, udata);
  if (check_flag(flag, "SetIC")) { return 1; }

  // --------------------
  // Setup the integrator
  // --------------------
  void* arkode_mem = nullptr;

  // ARKODE memory structure
  arkode_mem = ARKStepCreate(fErhs, fIrhs, udata.t0, y, ctx);
  if (check_ptr(arkode_mem, "ARKStepCreate")) { return 1; }

  // Select ARK method
  flag = ARKStepSetTableName(arkode_mem, uopts.dirk_table.c_str(),
                             uopts.erk_table.c_str());
  if (check_flag(flag, "ARKStepSetTableName")) { return 1; }

  // Specify tolerances
  flag = ARKodeSStolerances(arkode_mem, uopts.rtol, uopts.atol);
  if (check_flag(flag, "ARKodeSStolerances")) { return 1; }

  // Attach user data
  flag = ARKodeSetUserData(arkode_mem, &udata);
  if (check_flag(flag, "ARKodeSetUserData")) { return 1; }

  // Set fixed step size, or initial step size if adaptive
  if (uopts.fixed_h > ZERO)
  {
    flag = ARKodeSetFixedStep(arkode_mem, uopts.fixed_h);
    if (check_flag(flag, "ARKodeSetFixedStep")) { return 1; }
  }
  else if (uopts.h0 > ZERO)
  {
    flag = ARKodeSetInitStep(arkode_mem, uopts.h0);
    if (check_flag(flag, "ARKodeSetInitStep")) { return 1; }
  }

  // Set max steps between outputs
  flag = ARKodeSetMaxNumSteps(arkode_mem, uopts.maxsteps);
  if (check_flag(flag, "ARKodeSetMaxNumSteps")) { return 1; }

  // Set stopping time
  flag = ARKodeSetStopTime(arkode_mem, udata.tf);
  if (check_flag(flag, "ARKodeSetStopTime")) { return 1; }

  // Set up custom linear solver
  SUNLinearSolver LS = MatrixEmbeddedLS(arkode_mem, ctx);
  if (check_ptr(LS, "MatrixEmbeddedLS")) { return 1; }

  // Attach linear solver
  flag = ARKodeSetLinearSolver(arkode_mem, LS, nullptr);
  if (check_flag(flag, "ARKodeSetLinearSolver")) { return 1; }

  // Specify that IVP is autonomous
  flag = ARKodeSetAutonomous(arkode_mem, SUNTRUE);
  if (check_flag(flag, "ARKodeSetAutonomous")) { return 1; }

  // ----------------------
  // Evolve problem in time
  // ----------------------

  // Initial time, time between outputs, output time
  sunrealtype t     = ZERO;
  sunrealtype dTout = udata.tf / uopts.nout;
  sunrealtype tout  = dTout;

  // initial output
  flag = OpenOutput(udata, uopts);
  if (check_flag(flag, "OpenOutput")) { return 1; }

  long int nst, nst_cur;
  nst_cur = 0;
  flag = ARKodeGetNumSteps(arkode_mem, &nst);
  if (check_flag(flag, "ARKodeGetNumSteps")) { return 1; }

  flag = WriteOutput(t, y, nst_cur, udata, uopts);
  if (check_flag(flag, "WriteOutput")) { return 1; }

  // Loop over output times
  for (int iout = 0; iout < uopts.nout; iout++)
  {
    // Evolve
    if (uopts.output == 3)
    {
      // Stop at output time (do not interpolate output)
      flag = ARKodeSetStopTime(arkode_mem, tout);
      if (check_flag(flag, "ARKodeSetStopTime")) { return 1; }
    }

    //   Advance in time
    flag = ARKodeEvolve(arkode_mem, tout, y, &t, ARK_NORMAL);
    if (check_flag(flag, "ARKodeEvolve")) { break; }

    //  Get number of steps taken and elapsed
    nst_cur = nst;
    flag = ARKodeGetNumSteps(arkode_mem, &nst);
    if (check_flag(flag, "ARKodeGetNumSteps")) { return 1; }
    nst_cur = nst - nst_cur;

    // Output solution
    flag = WriteOutput(t, y, nst_cur, udata, uopts);
    if (check_flag(flag, "WriteOutput")) { return 1; }

    // Update output time
    tout += dTout;
    tout = (tout > udata.tf) ? udata.tf : tout;
  }

  // Close output
  flag = CloseOutput(uopts);
  if (check_flag(flag, "CloseOutput")) { return 1; }

  // ------------
  // Output stats
  // ------------

  if (uopts.output)
  {
    cout << "Final integrator statistics:" << endl;
    flag = ARKodePrintAllStats(arkode_mem, stdout, SUN_OUTPUTFORMAT_TABLE);
  }

  // --------
  // Clean up
  // --------

  ARKodeFree(&arkode_mem);
  for (int i = 0; i < NSPECIES; i++) { N_VDestroy(vecs[i]); }
  N_VDestroy(y);

  return 0;
}

// -----------------------------------------------------------------------------
// Functions called by the integrator
// -----------------------------------------------------------------------------

// Explicit ODE RHS function
int fErhs(sunrealtype t, N_Vector y, N_Vector f, void* user_data)
{
  // Access problem data
  EulerData* udata = (EulerData*)user_data;

  // initialize output to zeros
  N_VConst(ZERO, f);

  // Access data arrays
  sunrealtype* rho = N_VGetSubvectorArrayPointer_ManyVector(y, 0);
  if (check_ptr(rho, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* mx = N_VGetSubvectorArrayPointer_ManyVector(y, 1);
  if (check_ptr(mx, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* my = N_VGetSubvectorArrayPointer_ManyVector(y, 2);
  if (check_ptr(my, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* mz = N_VGetSubvectorArrayPointer_ManyVector(y, 3);
  if (check_ptr(mz, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* et = N_VGetSubvectorArrayPointer_ManyVector(y, 4);
  if (check_ptr(et, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* u = N_VGetSubvectorArrayPointer_ManyVector(y, 5);
  if (check_ptr(u, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* v = N_VGetSubvectorArrayPointer_ManyVector(y, 6);
  if (check_ptr(v, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* w = N_VGetSubvectorArrayPointer_ManyVector(y, 7);
  if (check_ptr(w, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }

  sunrealtype* rhodot = N_VGetSubvectorArrayPointer_ManyVector(f, 0);
  if (check_ptr(rhodot, "N_VGetSubvectorArrayPointer_ManyVector"))
  {
    return -1;
  }
  sunrealtype* mxdot = N_VGetSubvectorArrayPointer_ManyVector(f, 1);
  if (check_ptr(mxdot, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* mydot = N_VGetSubvectorArrayPointer_ManyVector(f, 2);
  if (check_ptr(mydot, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* mzdot = N_VGetSubvectorArrayPointer_ManyVector(f, 3);
  if (check_ptr(mzdot, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* etdot = N_VGetSubvectorArrayPointer_ManyVector(f, 4);
  if (check_ptr(etdot, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* udot = N_VGetSubvectorArrayPointer_ManyVector(f, 5);
  if (check_ptr(udot, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* vdot = N_VGetSubvectorArrayPointer_ManyVector(f, 6);
  if (check_ptr(vdot, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* wdot = N_VGetSubvectorArrayPointer_ManyVector(f, 7);
  if (check_ptr(wdot, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }

  // Set shortcut variables
  const long int nx    = udata->nx;
  const sunrealtype dx = udata->dx;
  sunrealtype* flux    = udata->flux;

  // compute face-centered fluxes over domain interior: pack 1D x-directional array
  // of variable shortcuts, and compute flux at lower x-directional face
  for (long int i = 3; i < nx - 2; i++)
  {
    udata->pack1D(rho, mx, my, mz, et, u, v, w, i);
    face_flux(udata->w1d, &(flux[i * NSPECIES]), *udata);
  }

  // compute face-centered fluxes at left boundary
  for (long int i = 0; i < 3; i++)
  {
    udata->pack1D_bdry(rho, mx, my, mz, et, u, v, w, i);
    face_flux(udata->w1d, &(flux[i * NSPECIES]), *udata);
  }

  // compute face-centered fluxes at right boundary
  for (long int i = nx - 2; i <= nx; i++)
  {
    udata->pack1D_bdry(rho, mx, my, mz, et, u, v, w, i);
    face_flux(udata->w1d, &(flux[i * NSPECIES]), *udata);
  }

  // iterate over subdomain, updating RHS
  for (long int i = 0; i < nx; i++)
  {
    rhodot[i] -= (flux[(i + 1) * NSPECIES + 0] - flux[i * NSPECIES + 0]) / dx;
    mxdot[i] -= (flux[(i + 1) * NSPECIES + 1] - flux[i * NSPECIES + 1]) / dx;
    mydot[i] -= (flux[(i + 1) * NSPECIES + 2] - flux[i * NSPECIES + 2]) / dx;
    mzdot[i] -= (flux[(i + 1) * NSPECIES + 3] - flux[i * NSPECIES + 3]) / dx;
    etdot[i] -= (flux[(i + 1) * NSPECIES + 4] - flux[i * NSPECIES + 4]) / dx;
    udot[i] -= (flux[(i + 1) * NSPECIES + 5] - flux[i * NSPECIES + 5]) / dx;
    vdot[i] -= (flux[(i + 1) * NSPECIES + 6] - flux[i * NSPECIES + 6]) / dx;
    wdot[i] -= (flux[(i + 1) * NSPECIES + 7] - flux[i * NSPECIES + 7]) / dx;
  }

  return 0;
}

// Implicit ODE RHS function
int fIrhs(sunrealtype t, N_Vector y, N_Vector f, void* user_data)
{
  // Access problem data
  EulerData* udata = (EulerData*)user_data;

  // initialize output to zeros
  N_VConst(ZERO, f);

  // Access data arrays
  sunrealtype* rho = N_VGetSubvectorArrayPointer_ManyVector(y, 0);
  if (check_ptr(rho, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* mx = N_VGetSubvectorArrayPointer_ManyVector(y, 1);
  if (check_ptr(mx, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* my = N_VGetSubvectorArrayPointer_ManyVector(y, 2);
  if (check_ptr(my, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* mz = N_VGetSubvectorArrayPointer_ManyVector(y, 3);
  if (check_ptr(mz, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* et = N_VGetSubvectorArrayPointer_ManyVector(y, 4);
  if (check_ptr(et, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* u = N_VGetSubvectorArrayPointer_ManyVector(y, 5);
  if (check_ptr(u, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* v = N_VGetSubvectorArrayPointer_ManyVector(y, 6);
  if (check_ptr(v, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* w = N_VGetSubvectorArrayPointer_ManyVector(y, 7);
  if (check_ptr(w, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }

  sunrealtype* udot = N_VGetSubvectorArrayPointer_ManyVector(f, 5);
  if (check_ptr(udot, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* vdot = N_VGetSubvectorArrayPointer_ManyVector(f, 6);
  if (check_ptr(vdot, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* wdot = N_VGetSubvectorArrayPointer_ManyVector(f, 7);
  if (check_ptr(wdot, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }

  // Set shortcut variables
  const long int nx = udata->nx;
  const sunrealtype a = udata->a;
  const sunrealtype b = udata->b;
  const sunrealtype ep = udata->ep;

  // compute reaction terms over domain, filling the RHS
  for (long int i = 0; i < nx; i++)
  {
    udot[i] = a - (w[i] + ONE) * u[i] + v[i] * u[i] * u[i];
    vdot[i] = w[i] * u[i] - v[i] * u[i] * u[i];
    wdot[i] = (b - w[i]) / ep - w[i] * u[i];
  }

  return 0;
}

// Cell-wise Newton system matrix: A = I - gamma * J, where J is the Jacobian
// of the implicit ODE RHS function
int localJ(const sunrealtype gamma, const sunrealtype u, const sunrealtype v,
           const sunrealtype w, SUNMatrix Aloc, void* user_data)
{
  // Access problem data
  EulerData* udata = (EulerData*)user_data;

  // initialize output to zeros
  SUNMatZero_Dense(Aloc);

  // Access matrix data
  sunrealtype** A = SUNDenseMatrix_Cols(Aloc);
  if (check_ptr(A, "SUNDenseMatrix_Cols")) { return -1; }

  // Set shortcut variables
  const sunrealtype a = udata->a;
  const sunrealtype b = udata->b;
  const sunrealtype ep = udata->ep;

  /* Jacobian wrt u */
  A[0][0] = ONE - gamma * (SUN_RCONST(2.0) * u * v - (w + SUN_RCONST(1.0)));
  A[0][1] = -gamma * (w - SUN_RCONST(2.0) * u * v);
  A[0][2] = gamma * w;

  /* Jacobian wrt v */
  A[1][0] = -gamma * (u * u);
  A[1][1] = ONE + gamma * u * u;

  /* Jacobian wrt w */
  A[2][0] = gamma * u;
  A[2][1] = - gamma * u;
  A[2][2] = ONE - gamma * (-SUN_RCONST(1.0) / ep - u);

  return 0;
}

// given a 6-point stencil of solution values,
//   w(x_{j-3}) w(x_{j-2}) w(x_{j-1}), w(x_j), w(x_{j+1}), w(x_{j+2}),
// compute the face-centered flux (flux) at the center of the stencil, x_{j-1/2}.
//
// This precisely follows the recipe laid out in:
// Chi-Wang Shu (2003) "High-order Finite Difference and Finite Volume WENO
// Schemes and Discontinuous Galerkin Methods for CFD," International Journal of
// Computational Fluid Dynamics, 17:2, 107-118, DOI: 10.1080/1061856031000104851
// with the only change that since this is 1D, we manually set the y- and
// z-velocities, v and w, to zero.
void face_flux(sunrealtype (&w1d)[STSIZE][NSPECIES], sunrealtype* f_face,
               const EulerData& udata)
{
  // local data
  int i, j;
  sunrealtype rhosqrL, rhosqrR, rhosqrbar, u, v, w, H, qsq, csnd, cinv, gamm,
    alpha, beta1, beta2, beta3, w1, w2, w3, f1, f2, f3;
  sunrealtype RV[5][5], LV[5][5], p[STSIZE], flux[STSIZE][NSPECIES],
    fproj[5][NSPECIES], fs[5][NSPECIES], ff[NSPECIES];
  const sunrealtype bc =
    SUN_RCONST(1.083333333333333333333333333333333333333); // 13/12
  const sunrealtype epsilon = SUN_RCONST(1.0e-6);

  // compute pressures over stencil
  for (i = 0; i < STSIZE; i++)
  {
    p[i] = udata.eos(w1d[i][0], w1d[i][1], w1d[i][2], w1d[i][3], w1d[i][4]);
  }

  // compute Roe-average state at face:
  //   wbar = [sqrt(rho), sqrt(rho)*vx, sqrt(rho)*vy, sqrt(rho)*vz, (e+p)/sqrt(rho)]
  //          [sqrt(rho), mx/sqrt(rho), my/sqrt(rho), mz/sqrt(rho), (e+p)/sqrt(rho)]
  //   u = wbar_2 / wbar_1
  //   v = wbar_3 / wbar_1
  //   w = wbar_4 / wbar_1
  //   H = wbar_5 / wbar_1
  rhosqrL   = sqrt(w1d[2][0]);
  rhosqrR   = sqrt(w1d[3][0]);
  rhosqrbar = HALF * (rhosqrL + rhosqrR);
  u         = HALF * (w1d[2][1] / rhosqrL + w1d[3][1] / rhosqrR) / rhosqrbar;
  v         = HALF * (w1d[2][2] / rhosqrL + w1d[3][2] / rhosqrR) / rhosqrbar;
  w         = HALF * (w1d[2][3] / rhosqrL + w1d[3][3] / rhosqrR) / rhosqrbar;
  H = HALF * ((p[2] + w1d[2][4]) / rhosqrL + (p[3] + w1d[3][4]) / rhosqrR) /
      rhosqrbar;

  // compute eigenvectors at face (note: eigenvectors for tracers are just identity)
  qsq  = u * u + v * v + w * w;
  gamm = udata.gamma - ONE;
  csnd = gamm * (H - HALF * qsq);
  cinv = ONE / csnd;
  for (i = 0; i < 5; i++)
  {
    for (j = 0; j < 5; j++)
    {
      RV[i][j] = ZERO;
      LV[i][j] = ZERO;
    }
  }

  RV[0][0] = ONE;
  RV[0][3] = ONE;
  RV[0][4] = ONE;

  RV[1][0] = u - csnd;
  RV[1][3] = u;
  RV[1][4] = u + csnd;

  RV[2][0] = v;
  RV[2][1] = ONE;
  RV[2][3] = v;
  RV[2][4] = v;

  RV[3][0] = w;
  RV[3][2] = ONE;
  RV[3][3] = w;
  RV[3][4] = w;

  RV[4][0] = H - u * csnd;
  RV[4][1] = v;
  RV[4][2] = w;
  RV[4][3] = HALF * qsq;
  RV[4][4] = H + u * csnd;

  LV[0][0] = HALF * cinv * (u + HALF * gamm * qsq);
  LV[0][1] = -HALF * cinv * (gamm * u + ONE);
  LV[0][2] = -HALF * v * gamm * cinv;
  LV[0][3] = -HALF * w * gamm * cinv;
  LV[0][4] = HALF * gamm * cinv;

  LV[1][0] = -v;
  LV[1][2] = ONE;

  LV[2][0] = -w;
  LV[2][3] = ONE;

  LV[3][0] = -gamm * cinv * (qsq - H);
  LV[3][1] = u * gamm * cinv;
  LV[3][2] = v * gamm * cinv;
  LV[3][3] = w * gamm * cinv;
  LV[3][4] = -gamm * cinv;

  LV[4][0] = -HALF * cinv * (u - HALF * gamm * qsq);
  LV[4][1] = -HALF * cinv * (gamm * u - ONE);
  LV[4][2] = -HALF * v * gamm * cinv;
  LV[4][3] = -HALF * w * gamm * cinv;
  LV[4][4] = HALF * gamm * cinv;

  // compute fluxes and max wave speed over stencil
  alpha = ZERO;
  for (j = 0; j < STSIZE; j++)
  {
    u          = w1d[j][1] / w1d[j][0];  // ux = vx = mx/rho
    flux[j][0] = w1d[j][1];              // f_rho = rho*ux = mx
    flux[j][1] = u * w1d[j][1] + p[j];   // f_mx = rho*ux*ux + p = mx*u + p
    flux[j][2] = u * w1d[j][2];          // f_my = rho*vx*ux = my*ux
    flux[j][3] = u * w1d[j][3];          // f_mz = rho*wx*ux = mz*ux
    flux[j][4] = u * (w1d[j][4] + p[j]); // f_et = ux*(et + p)
    flux[j][5] = u * w1d[j][5];          // f_u = u*ux
    flux[j][6] = u * w1d[j][6];          // f_v = v*ux
    flux[j][7] = u * w1d[j][7];          // f_w = w*ux
    csnd  = sqrt(udata.gamma * p[j] / w1d[j][0]); // csnd = sqrt(gamma*p/rho)
    alpha = max(alpha, abs(u) + csnd);
  }

  // compute flux from right side of face at x_{i+1/2}:

  //   compute right-shifted Lax-Friedrichs flux over left portion of patch
  for (j = 0; j < 5; j++)
  {
    for (i = 0; i < NSPECIES; i++)
    {
      fs[j][i] = HALF * (flux[j][i] + alpha * w1d[j][i]);
    }
  }

  // compute projected flux for fluid fields (copy reactants)
  for (j = 0; j < 5; j++)
  {
    for (i = 0; i < 5; i++)
    {
      fproj[j][i] = LV[i][0] * fs[j][0] + LV[i][1] * fs[j][1] +
                    LV[i][2] * fs[j][2] + LV[i][3] * fs[j][3] +
                    LV[i][4] * fs[j][4];
    }
    fproj[j][5] = fs[j][5];
    fproj[j][6] = fs[j][6];
    fproj[j][7] = fs[j][7];
  }

  //   compute WENO signed flux
  for (i = 0; i < NSPECIES; i++)
  {
    // smoothness indicators
    beta1 = bc * pow(fproj[2][i] - SUN_RCONST(2.0) * fproj[3][i] + fproj[4][i], 2) +
            FOURTH * pow(SUN_RCONST(3.0) * fproj[2][i] -
                           SUN_RCONST(4.0) * fproj[3][i] + fproj[4][i], 2);
    beta2 = bc * pow(fproj[1][i] - SUN_RCONST(2.0) * fproj[2][i] + fproj[3][i], 2) +
            FOURTH * pow(fproj[1][i] - fproj[3][i], 2);
    beta3 = bc * pow(fproj[0][i] - SUN_RCONST(2.0) * fproj[1][i] + fproj[2][i], 2) +
            FOURTH * pow(fproj[0][i] - SUN_RCONST(4.0) * fproj[1][i] +
                           SUN_RCONST(3.0) * fproj[2][i], 2);
    // nonlinear weights
    w1 = SUN_RCONST(0.3) / ((epsilon + beta1) * (epsilon + beta1));
    w2 = SUN_RCONST(0.6) / ((epsilon + beta2) * (epsilon + beta2));
    w3 = SUN_RCONST(0.1) / ((epsilon + beta3) * (epsilon + beta3));
    // flux stencils
    f1 = SUN_RCONST(0.3333333333333333333333333333333333333333) * fproj[2][i] +
         SUN_RCONST(0.8333333333333333333333333333333333333333) * fproj[3][i] -
         SUN_RCONST(0.1666666666666666666666666666666666666667) * fproj[4][i];
    f2 = -SUN_RCONST(0.1666666666666666666666666666666666666667) * fproj[1][i] +
         SUN_RCONST(0.8333333333333333333333333333333333333333) * fproj[2][i] +
         SUN_RCONST(0.3333333333333333333333333333333333333333) * fproj[3][i];
    f3 = SUN_RCONST(0.3333333333333333333333333333333333333333) * fproj[0][i] -
         SUN_RCONST(1.166666666666666666666666666666666666667) * fproj[1][i] +
         SUN_RCONST(1.833333333333333333333333333333333333333) * fproj[2][i];
    // resulting signed flux at face
    ff[i] = (f1 * w1 + f2 * w2 + f3 * w3) / (w1 + w2 + w3);
  }

  // compute flux from left side of face at x_{i+1/2}:

  //   compute left-shifted Lax-Friedrichs flux over right portion of patch
  for (j = 0; j < 5; j++)
  {
    for (i = 0; i < NSPECIES; i++)
    {
      fs[j][i] = HALF * (flux[j + 1][i] - alpha * w1d[j + 1][i]);
    }
  }

  // compute projected flux for fluid fields (copy reactants)
  for (j = 0; j < 5; j++)
  {
    for (i = 0; i < 5; i++)
    {
      fproj[j][i] = LV[i][0] * fs[j][0] + LV[i][1] * fs[j][1] +
                    LV[i][2] * fs[j][2] + LV[i][3] * fs[j][3] +
                    LV[i][4] * fs[j][4];
    }
    fproj[j][5] = fs[j][5];
    fproj[j][6] = fs[j][6];
    fproj[j][7] = fs[j][7];
  }

  //   compute WENO signed fluxes
  for (i = 0; i < NSPECIES; i++)
  {
    // smoothness indicators
    beta1 = bc * pow(fproj[2][i] - SUN_RCONST(2.0) * fproj[3][i] + fproj[4][i], 2) +
            FOURTH * pow(SUN_RCONST(3.0) * fproj[2][i] -
                           SUN_RCONST(4.0) * fproj[3][i] + fproj[4][i], 2);
    beta2 = bc * pow(fproj[1][i] - SUN_RCONST(2.0) * fproj[2][i] + fproj[3][i], 2) +
            FOURTH * pow(fproj[1][i] - fproj[3][i], 2);
    beta3 = bc * pow(fproj[0][i] - SUN_RCONST(2.0) * fproj[1][i] + fproj[2][i], 2) +
            FOURTH * pow(fproj[0][i] - SUN_RCONST(4.0) * fproj[1][i] +
                           SUN_RCONST(3.0) * fproj[2][i], 2);
    // nonlinear weights
    w1 = SUN_RCONST(0.1) / ((epsilon + beta1) * (epsilon + beta1));
    w2 = SUN_RCONST(0.6) / ((epsilon + beta2) * (epsilon + beta2));
    w3 = SUN_RCONST(0.3) / ((epsilon + beta3) * (epsilon + beta3));
    // flux stencils
    f1 = SUN_RCONST(1.833333333333333333333333333333333333333) * fproj[2][i] -
         SUN_RCONST(1.166666666666666666666666666666666666667) * fproj[3][i] +
         SUN_RCONST(0.3333333333333333333333333333333333333333) * fproj[4][i];
    f2 = SUN_RCONST(0.3333333333333333333333333333333333333333) * fproj[1][i] +
         SUN_RCONST(0.8333333333333333333333333333333333333333) * fproj[2][i] -
         SUN_RCONST(0.1666666666666666666666666666666666666667) * fproj[3][i];
    f3 = -SUN_RCONST(0.1666666666666666666666666666666666666667) * fproj[0][i] +
         SUN_RCONST(0.8333333333333333333333333333333333333333) * fproj[1][i] +
         SUN_RCONST(0.3333333333333333333333333333333333333333) * fproj[2][i];
    // resulting signed flux (add to ff)
    ff[i] += (f1 * w1 + f2 * w2 + f3 * w3) / (w1 + w2 + w3);
  }

  // combine signed fluxes into output, converting back to conserved variables
  for (i = 0; i < 5; i++)
  {
    f_face[i] = RV[i][0] * ff[0] + RV[i][1] * ff[1] + RV[i][2] * ff[2] +
                RV[i][3] * ff[3] + RV[i][4] * ff[4];
  }
  f_face[5] = ff[5];
  f_face[6] = ff[6];
  f_face[7] = ff[7];

  return;
}

// Compute the initial condition
int SetIC(N_Vector y, EulerData& udata)
{
  sunrealtype* rho = N_VGetSubvectorArrayPointer_ManyVector(y, 0);
  if (check_ptr(rho, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* mx = N_VGetSubvectorArrayPointer_ManyVector(y, 1);
  if (check_ptr(mx, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* my = N_VGetSubvectorArrayPointer_ManyVector(y, 2);
  if (check_ptr(my, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* mz = N_VGetSubvectorArrayPointer_ManyVector(y, 3);
  if (check_ptr(mz, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* et = N_VGetSubvectorArrayPointer_ManyVector(y, 4);
  if (check_ptr(et, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* u = N_VGetSubvectorArrayPointer_ManyVector(y, 5);
  if (check_ptr(u, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* v = N_VGetSubvectorArrayPointer_ManyVector(y, 6);
  if (check_ptr(v, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* w = N_VGetSubvectorArrayPointer_ManyVector(y, 7);
  if (check_ptr(w, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }

  if (udata.initial_condition == "Sod_Brusselator")
  {
    for (long int i = 0; i < udata.nx; i++)
    {
      sunrealtype xloc = ((sunrealtype)i + HALF) * udata.dx + udata.xl;
      if (xloc < HALF)
      {
        rho[i] = rhoL;
        et[i]  = udata.eos_inv(rhoL, uL, ZERO, ZERO, pL);
        mx[i]  = rhoL * uL;
      }
      else
      {
        rho[i] = rhoR;
        et[i]  = udata.eos_inv(rhoR, uR, ZERO, ZERO, pR);
        mx[i]  = rhoR * uR;
      }
      my[i] = ZERO;
      mz[i] = ZERO;
      u[i]  = SUN_RCONST(1.2) * rho[i];
      v[i]  = SUN_RCONST(3.1) * rho[i];
      w[i]  = SUN_RCONST(3.0) * rho[i];
    }
  }
  else if (udata.initial_condition == "Brusselator")
  {
    for (long int i = 0; i < udata.nx; i++)
    {
      rho[i] = ONE;
      mx[i] = ZERO;
      my[i] = ZERO;
      mz[i] = ZERO;
      et[i]  = udata.eos_inv(ONE, ZERO, ZERO, ZERO, ONE);
      u[i]  = SUN_RCONST(1.2) * rho[i];
      v[i]  = SUN_RCONST(3.1) * rho[i];
      w[i]  = SUN_RCONST(3.0) * rho[i];
    }
  }
  else if (udata.initial_condition == "bubble_Brusselator")
  {
    for (long int i = 0; i < udata.nx; i++)
    {
      sunrealtype xloc = ((sunrealtype)i + HALF) * udata.dx + udata.xl;
      rho[i] = ONE + 0.5 * exp(-100.0 * (xloc - HALF) * (xloc - HALF));
      const sunrealtype ux = ONE;
      et[i]  = udata.eos_inv(rho[i], ux, ZERO, ZERO, ONE);
      mx[i]  = ux * rho[i];
      my[i]  = ZERO;
      mz[i]  = ZERO;
      u[i]   = SUN_RCONST(1.2) * rho[i];
      v[i]   = SUN_RCONST(3.1) * rho[i];
      w[i]   = SUN_RCONST(3.0) * rho[i];
    }
  }
  else
  {
    fprintf(stderr, "Error: unrecognized initial condition '%s'\n",
            udata.initial_condition.c_str());
    return -1;
  }
  return 0;
}

// Custom linear solver data structure, accessor macros, and routines
SUNLinearSolver MatrixEmbeddedLS(void* arkode_mem, SUNContext ctx)
{
  // Create an empty linear solver
  SUNLinearSolver LS = SUNLinSolNewEmpty(ctx);
  if (LS == NULL) { return NULL; }

  // Attach operations
  LS->ops->gettype = MatrixEmbeddedLSType;
  LS->ops->solve   = MatrixEmbeddedLSSolve;
  LS->ops->free    = MatrixEmbeddedLSFree;

  // Set content pointer to ARKODE memory
  LS->content = arkode_mem;

  // Return solver
  return (LS);
}

SUNLinearSolver_Type MatrixEmbeddedLSType(SUNLinearSolver LS)
{
  return (SUNLINEARSOLVER_MATRIX_EMBEDDED);
}

int MatrixEmbeddedLSSolve(SUNLinearSolver LS, SUNMatrix A, N_Vector x,
                          N_Vector b, sunrealtype tol)
{
  // temporary variables
  N_Vector z, zpred, Fi, sdata;
  sunrealtype tcur, gamma;
  void* user_data;

  // retrieve implicit system data from ARKODE
  int flag = ARKodeGetNonlinearSystemData(LS->content, &tcur, &zpred, &z, &Fi,
                                          &gamma, &sdata, &user_data);
  if (check_flag(flag, "ARKodeGetNonlinearSystemData")) { return 1; }

  // access EulerData structure
  EulerData* udata = (EulerData*)user_data;

  // set shortcut variables
  SUNMatrix Aloc = udata->Aloc;
  N_Vector xloc = udata->xloc;
  N_Vector bloc = udata->bloc;
  SUNLinearSolver LSloc = udata->LSloc;
  sunrealtype *xlocdata = N_VGetArrayPointer(xloc);
  if (check_ptr(xlocdata, "N_VGetArrayPointer")) { return -1; }
  sunrealtype *blocdata = N_VGetArrayPointer(bloc);
  if (check_ptr(blocdata, "N_VGetArrayPointer")) { return -1; }

  // access data arrays
  sunrealtype* zpred_u = N_VGetSubvectorArrayPointer_ManyVector(zpred, 5);
  if (check_ptr(zpred_u, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* zpred_v = N_VGetSubvectorArrayPointer_ManyVector(zpred, 6);
  if (check_ptr(zpred_v, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* zpred_w = N_VGetSubvectorArrayPointer_ManyVector(zpred, 7);
  if (check_ptr(zpred_w, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* z_u = N_VGetSubvectorArrayPointer_ManyVector(z, 5);
  if (check_ptr(z_u, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* z_v = N_VGetSubvectorArrayPointer_ManyVector(z, 6);
  if (check_ptr(z_v, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* z_w = N_VGetSubvectorArrayPointer_ManyVector(z, 7);
  if (check_ptr(z_w, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* x_u = N_VGetSubvectorArrayPointer_ManyVector(x, 5);
  if (check_ptr(x_u, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* x_v = N_VGetSubvectorArrayPointer_ManyVector(x, 6);
  if (check_ptr(x_v, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* x_w = N_VGetSubvectorArrayPointer_ManyVector(x, 7);
  if (check_ptr(x_w, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* b_u = N_VGetSubvectorArrayPointer_ManyVector(b, 5);
  if (check_ptr(b_u, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* b_v = N_VGetSubvectorArrayPointer_ManyVector(b, 6);
  if (check_ptr(b_v, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }
  sunrealtype* b_w = N_VGetSubvectorArrayPointer_ManyVector(b, 7);
  if (check_ptr(b_w, "N_VGetSubvectorArrayPointer_ManyVector")) { return -1; }

  // iterate over domain, performing cell-wise linear solve
  for (long int i = 0; i < udata->nx; i++)
  {
    // assemble current state for cell
    const sunrealtype u = zpred_u[i] + z_u[i];
    const sunrealtype v = zpred_v[i] + z_v[i];
    const sunrealtype w = zpred_w[i] + z_w[i];

    // construct local Jacobian, A = I - gamma * Jloc
    flag = localJ(gamma, u, v, w, Aloc, user_data);
    if (check_flag(flag, "localJ")) { return -1; }

    // extract local RHS
    blocdata[0] = b_u[i];
    blocdata[1] = b_v[i];
    blocdata[2] = b_w[i];

    // perform local linear solve
    flag = SUNLinSolSetup(LSloc, Aloc);
    if (check_flag(flag, "SUNLinSolSetup")) { return -1; }
    flag = SUNLinSolSolve(LSloc, Aloc, xloc, bloc, tol);
    if (check_flag(flag, "SUNLinSolSolve")) { return -1; }

    // store result in x
    x_u[i] = xlocdata[0];
    x_v[i] = xlocdata[1];
    x_w[i] = xlocdata[2];
  }

  /* return with success */
  return (SUN_SUCCESS);
}

SUNErrCode MatrixEmbeddedLSFree(SUNLinearSolver LS)
{
  if (LS == NULL) { return (SUN_SUCCESS); }
  LS->content = NULL;
  SUNLinSolFreeEmpty(LS);
  return (SUN_SUCCESS);
}

//---- end of file ----
