/*---------------------------------------------------------------
 * Programmer(s): Sylvia Amihere @ UMBC
 *---------------------------------------------------------------
 * SUNDIALS Copyright Start
 * Copyright (c) 2002-2024, Lawrence Livermore National Security
 * and Southern Methodist University.
 * All rights reserved.
 *
 * See the top-level LICENSE and NOTICE files for details.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * SUNDIALS Copyright End
 *---------------------------------------------------------------
 * Example problem:
 *
 * The following test simulates a 1D Linear Advection Reaction Problem,
 *    u_t + alpha_1*u_x = -k_1*u + k_2*v + s_1
 *    v_t + alpha_2*v_x =  k_1*u - k_2*v + s_2
 * for t in [0, 1], x in [0, 1] with parameters alpha_1 = 1, alapha_2 = 0
 * k_1 = 1e6, k_2 = 2*k_1, s_1 = 0, s_2 = 1. The initial conditions are
 * u(x,0) = 1 + s_2*x , v(x,0) = (k_1/k_2)*u(x,0) + (1/k_2)*s_2. 
 * The boundary condition is u(0,t) = gamma_1(t) = 1.
 *
 * The spatial derivatives are computed using upwind spatial discretization 
 * (backward finite difference), with the data distributed over N = 100 points
 * on a uniform spatial grid.
 *
 * This program solves the problem with an ARK method. 
 * For the DIRK method, we use a Newton iteration with
 * the SUNLinSol_PCG linear solver, and a user-supplied Jacobian-vector
 * product routine.
 *
 * 100 outputs are printed at equal intervals, and run statistics
 * are printed at the end.
 *---------------------------------------------------------------*/

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>


/* Header files */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <nvector/nvector_serial.h> /* serial N_Vector types, fcts., macros */
#include "nvector/nvector_manyvector.h"
#include <arkode/arkode_arkstep.h> 
#include "sundials/sundials_core.hpp"
#include <sundials/sundials_types.h> /* defs. of sunrealtype, sunindextype, etc */
#include <sunlinsol/sunlinsol_spgmr.h>/* access to GMRES SUNLinearSolver */
#include <sunlinsol/sunlinsol_spbcgs.h> /* access to SPBCGS SUNLinearSolver            */
#include <sundials/sundials_logger.h>


using namespace std;

#if defined(SUNDIALS_EXTENDED_PRECISION)
#define GSYM "Lg"
#define ESYM "Le"
#define FSYM "Lf"
#else
#define GSYM "g"
#define ESYM "e"
#define FSYM "f"
#endif

#define ZERO SUN_RCONST(0.0)

  class UserData
  {
  public:
    sunindextype N;              /* number of intervals   */
    sunrealtype dx;              /* mesh spacing          */
    sunrealtype alpha1, alpha2;  /* advection coefficients */
    sunrealtype k1, k2;          /* reaction coefficients */
    sunrealtype s1, s2; 
    sunrealtype xstart;          /* left endpoint on spatial grid */
    sunrealtype xend;            /* right endpoint on spatial grid */
    string swap_type;            /* Swapping or Non-Swapping of b-vectors of the method and its embedding*/ 

  // constructor (with default values)
  UserData()
  : N(100),
    alpha1(1.0),
    alpha2(0.0),
    k1(1e6),
    k2(2e6),
    s1(0.0),
    s2(1.0),
    xstart(ZERO),
    xend(1.0),
    dx(ZERO),
    swap_type("nonswap"){};
  };

class ARKODEParameters
{
public:
   // Integration method
   std::string IMintegrator;
   std::string EXintegrator;
 
   // Relative and absolute tolerances
   sunrealtype rtol;
   sunrealtype atol;
 
   // Step size selection (ZERO = adaptive steps)
   sunrealtype fixed_h;
 
   // Maximum number of time steps between outputs
   int maxsteps;

   // Time Parameters
   sunrealtype T0;           // initial time
   sunrealtype Tf;           // end time
  //  int Nt;                // number of output times
   
   // Output-related information
   int output;         // 0 = none, 1 = stats, 2 = disk, 3 = disk with tstop
   std::ofstream uout; // output file stream
 
   // constructor (with default values)
   ARKODEParameters()
    : IMintegrator("ARKODE_SSP_SDIRK_2_1_2"),
      EXintegrator("ARKODE_SSP_ERK_2_1_2"),
      rtol(SUN_RCONST(1.e-4)),
      atol(SUN_RCONST(1.e-10)),
      fixed_h(ZERO),
      maxsteps(10000),
      output(1),
      // Nt(10),
      T0(ZERO),
      Tf(1.0){};
 
  }; // end ARKODEParameters

/* User-supplied Functions Called by the Solver */
static int fe(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data); //Explicit RHS
static int f(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data); //Implicit RHS
static int Jac(N_Vector v, N_Vector Jv, sunrealtype t, N_Vector y, N_Vector fy, void* user_data, N_Vector tmp);
static int ReadInputs(std::vector<std::string>& args, UserData& udata, ARKODEParameters& uopts, SUNContext ctx);
static void InputHelp();
static int PrintSetup(UserData& udata,ARKODEParameters& uopts);
static int trueSol(sunrealtype t, N_Vector tSol, void* user_data); //Exact solution

/* Private function to check function return values */
static int check_flag(void* flagvalue, const char* funcname, int opt);

/* Main Program */
int main(int argc, char* argv[])
{
  // int maxl           = 100;

  // SUNDIALS context object for this simulation
  sundials::Context ctx;

  UserData udata;
  ARKODEParameters uopts;

  vector<string> args(argv + 1, argv + argc);

  int flag = ReadInputs(args, udata, uopts, ctx);
  if (check_flag(&flag, "ReadInputs", 1)) { return 1; }
  if (flag > 0) { return 0; }

  N_Vector tSol = NULL;

  /* Initial problem output */
  printf("\n1D Linear Advection Reaction problem:\n");
  printf("  N = %li\n", (long int)udata.N);

  flag = PrintSetup(udata, uopts);
  if (check_flag(&flag, "PrintSetup", 1)) { return 1;}

  /* Initialize data structures */
  N_Vector y = N_VNew_Serial(2*udata.N, ctx); /* Create serial vector for solution */
  if (check_flag((void*)y, "N_VNew_Serial", 0)) { return 1; }

  /* compute the true solution */
  tSol = N_VClone(y);
  flag = trueSol(0.0, tSol, &udata);
  if (check_flag(&flag, "trueSol", 1)) { return 1;}
  
  /* Set initial conditions for u and v */
  sunrealtype* y_data = N_VGetArrayPointer(y);  
  y_data[0]           = 1.0; //boundary on the left for u
  y_data[udata.N]     = (udata.k1/udata.k2)*1.0 + (1.0/udata.k2)*udata.s2; 
  for (int i = 1; i < udata.N; i++){
    sunrealtype xi = udata.xstart + i * udata.dx;
    sunrealtype u0 = 1.0 + udata.s2 * xi;
    sunrealtype v0 = (udata.k1/udata.k2)*u0 + (1.0/udata.k2)*udata.s2;

    y_data[i]           = u0;
    y_data[udata.N + i] = v0;
  }

  /* Call ARKStepCreate to initialize the ARK timestepper module and
     specify the right-hand side function in y'=f(t,y), the initial time
     T0, and the initial dependent variable vector y. */
  void* arkode_mem = ARKStepCreate(fe, f, uopts.T0, y, ctx);
  if (check_flag((void*)arkode_mem, "ARKStepCreate", 0)) { return 1; }

  /* Set routines */
  flag = ARKodeSetUserData(arkode_mem, &udata);
  if (check_flag(&flag, "ARKodeSetUserData", 1)) { return 1; }
  flag = ARKodeSetMaxNumSteps(arkode_mem, uopts.maxsteps);
  if (check_flag(&flag, "ARKodeSetMaxNumSteps", 1)) { return 1; }
  flag = ARKodeSStolerances(arkode_mem, uopts.rtol, uopts.atol);
  if (check_flag(&flag, "ARKodeSStolerances", 1)) { return 1; }

  /*Keep original butcher tableau or swap b-vectors of method and its embedding*/
  if (udata.swap_type == "nonswap"){
    flag = ARKStepSetTableName(arkode_mem, uopts.IMintegrator.c_str(), uopts.EXintegrator.c_str()); 
    if (check_flag(&flag, "ARKStepSetTableName", 1)) { return 1; } 
  }
  else if (udata.swap_type == "swap") {
    ARKodeButcherTable Be = ARKodeButcherTable_LoadERKByName(uopts.EXintegrator.c_str());
    ARKodeButcherTable Bi = ARKodeButcherTable_LoadDIRKByName(uopts.IMintegrator.c_str());

    int s = Bi->stages;
    sunrealtype* A_new = (sunrealtype*) malloc(s * s * sizeof(sunrealtype));
    for (int i = 0; i < s; i++) {
      for (int j = 0; j < s; j++) { A_new[i * s + j] = Bi->A[i][j]; }
    }
    ARKodeButcherTable Bi_swap = ARKodeButcherTable_Create(Bi->stages, Bi->p, Bi->q, Bi->c, A_new, Bi->d, Bi->b);
    if (check_flag((void*)Bi_swap, "ARKodeButcherTable_Create", 0)) { return 1; }

    for (int i = 0; i < s; i++) {
      for (int j = 0; j < s; j++) { A_new[i * s + j] = Be->A[i][j]; }
    }
    ARKodeButcherTable Be_swap = ARKodeButcherTable_Create(Be->stages, Be->p, Be->q, Be->c, A_new, Be->d, Be->b);
    if (check_flag((void*)Be_swap, "ARKodeButcherTable_Create", 0)) { return 1; }
    free(A_new);
    flag = ARKStepSetTables(arkode_mem, Bi_swap->q, Bi_swap->p, Bi_swap, Be_swap);
    if (check_flag(&flag, "ARKStepSetTables", 1)) { return 1; }
    ARKodeButcherTable_Free(Be);
    ARKodeButcherTable_Free(Bi); 
    ARKodeButcherTable_Free(Be_swap);
    ARKodeButcherTable_Free(Bi_swap);
  }

  if (uopts.fixed_h > ZERO)
  {
    flag = ARKodeSetFixedStep(arkode_mem, uopts.fixed_h);
    if (check_flag(&flag, "ARKodeSetFixedStep", 1)) { return 1; }
  }

  flag = ARKodeSetStopTime(arkode_mem, uopts.Tf);
  if (check_flag(&flag, "ARKodeSetStopTime", 1)) { return 1; }

  /* Initialize GMRES solver -- no preconditioning, with up to 2*N iterations  */
  SUNLinearSolver LS = SUNLinSol_SPGMR(y, SUN_PREC_NONE, 2*udata.N, ctx);
  // SUNLinearSolver LS = SUNLinSol_SPBCGS(y, SUN_PREC_NONE, 2*udata.N, ctx);
  // SUNLinearSolver LS = SUNLinSol_SPGMR(y, maxl, 2*udata.N, ctx);
  if (check_flag((void*)LS, "SUNLinSol_SPGMR", 0)) { return 1; }

  /* Linear solver interface */
  flag = ARKodeSetLinearSolver(arkode_mem, LS, NULL);
  if (check_flag(&flag, "ARKodeSetLinearSolver", 1)) { return 1; }

  /* output mesh to disk */
  FILE* FID = fopen("linear_adv_rec_mesh.txt", "w");
  for (int i = 0; i < udata.N; i++) { fprintf(FID, "  %.16" ESYM "\n", udata.dx * i); }
  fclose(FID);

  /* Open output stream for results, access data array */
  FILE* UFID = fopen("linear_adv_rec.txt", "w");
  fprintf(UFID, "Title: Linear Advection Reaction Problem \n");
  fprintf(UFID, "Initial Time %f \n", uopts.T0);
  fprintf(UFID, "Final Time %f \n", uopts.Tf);
  fprintf(UFID, "Spatial Dimension %d \n", udata.N);
  fprintf(UFID, "Left endpoint %f \n", udata.xstart);
  fprintf(UFID, "Right endpoint %f \n", udata.xend);
  sunrealtype* data = N_VGetArrayPointer(y);
  sunrealtype* final_data = N_VGetArrayPointer(y); // solution at final time step
  sunrealtype* true_data = N_VGetArrayPointer(tSol); // true solution 

  /* output initial condition (u and v) to disk */
  for (int i = 0; i < udata.N; i++) { fprintf(UFID, " %.16" ESYM " %.16" ESYM, data[i], data[udata.N + i]); }
  fprintf(UFID, "\n");

  /* Main time-stepping loop: calls ARKodeEvolve to perform the integration, then
     prints results.  Stops when the final time has been reached */
  sunrealtype t = uopts.T0;

  while (t < uopts.Tf)
  {
    flag = ARKodeEvolve(arkode_mem, uopts.Tf, y, &t, ARK_ONE_STEP); /* call integrator */
    if (check_flag(&flag, "ARKodeEvolve", 1)) { break; }
    if (flag < 0)
    { /* unsuccessful solve: break */
      fprintf(stderr, "Solver failure, stopping integration\n");
      break;
    }

    /* output results to disk */
    fprintf(UFID, "Time step: %.2" FSYM "\n", t); 
    for (int i = 0; i < udata.N; i++) { fprintf(UFID, " %.16" ESYM " %.16" ESYM, data[i], data[udata.N + i]); }
    fprintf(UFID, "\n \n");
  }

  /* find the L1 norm */
  sunrealtype sum_error = 0.0;
  for (int i = udata.N; i < 2*udata.N; i++)
  {
    sum_error += SUNRabs(true_data[i]-data[i]);
  }
  printf(" L1-norm = %.16e\n", sum_error / udata.N);

  long int nsteps; //use the number of steps taken in the python plot
  ARKodeGetNumSteps(arkode_mem, &nsteps);
  fprintf(UFID, "Number of Time Steps Taken: %ld \n", nsteps);

  printf(" ---------------------------------\n \n");
  fclose(UFID);

  /* Print final statistics */
  flag = ARKodePrintAllStats(arkode_mem, stdout, SUN_OUTPUTFORMAT_TABLE);
  if (check_flag(&flag, "ARKodePrintAllStats", 1)) { return 1; }

  /* Clean up and return with successful completion */
  N_VDestroy(y);           /* Free vectors */
  // free(udata);             /* Free user data */
  N_VDestroy(tSol);
  ARKodeFree(&arkode_mem); /* Free integrator memory */
  SUNLinSolFree(LS);       /* Free linear solver */

  return 0;
}

/*--------------------------------
 * Functions called by the solver
 *--------------------------------*/

/* f routine to compute the ODE explicitRHS function f(t,y). */
static int fe(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data)
{
  UserData* udata = (UserData*)user_data; /* access problem data */
  sunrealtype *Y = NULL, *Ydot = NULL;
  Y = N_VGetArrayPointer(y); /* access data arrays */
  if (check_flag((void*)Y, "N_VGetArrayPointer", 0)) { return 1; }
  Ydot = N_VGetArrayPointer(ydot);
  if (check_flag((void*)Ydot, "N_VGetArrayPointer", 0)) { return 1; }
  N_VConst(0.0, ydot); /* Initialize ydot to zero */

  /* set parameters */
  const sunindextype N     = udata->N;
  const sunrealtype dx     = udata->dx;
  const sunrealtype alpha1 = udata->alpha1;
  const sunrealtype alpha2 = udata->alpha2;

  sunrealtype* u = Y; //the first N entries for vector u
  sunrealtype* v = Y + N; //the next N entries for vector v
  sunrealtype* udot = Ydot;
  sunrealtype* vdot = Ydot + N;

  //boundary conditions
  udot[0]   = 0.0; 
  vdot[0]   = 0.0; 

  //interior points
  for (int i = 1; i < N; i++){
    udot[i] = -alpha1*(u[i] - u[i-1])/(dx);
    vdot[i] = -alpha2*(v[i] - v[i-1])/(dx);
  }

  return 0; /* Return with success */
}


/* f routine to compute the ODE implicit RHS function f(t,y). */
static int f(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data)
{
  UserData* udata = (UserData*)user_data; /* access problem data */
  sunrealtype *Y = NULL, *Ydot = NULL;
  Y = N_VGetArrayPointer(y); /* access data arrays */
  if (check_flag((void*)Y, "N_VGetArrayPointer", 0)) { return 1; }
  Ydot = N_VGetArrayPointer(ydot);
  if (check_flag((void*)Ydot, "N_VGetArrayPointer", 0)) { return 1; }
  N_VConst(0.0, ydot); /* Initialize ydot to zero */

  /* set parameters */
  const sunindextype N  = udata->N;
  const sunrealtype dx  = udata->dx;
  const sunrealtype k1  = udata->k1;
  const sunrealtype k2  = udata->k2;
  const sunrealtype s1  = udata->s1;
  const sunrealtype s2  = udata->s2;
  sunrealtype* u = Y; //the first N entries for vector u
  sunrealtype* v = Y + N; //the next N entries for vector v
  sunrealtype* udot = Ydot;
  sunrealtype* vdot = Ydot + N;

  //boundary conditions
  udot[0]   = 0.0;

  //interior points
  for (int i = 1; i < N; i++){
    udot[i] = -k1*u[i] + k2*v[i] + s1;
  }

  for (int i = 0; i < N; i++){
    vdot[i] =  k1*u[i] - k2*v[i] + s2; 
  }

  return 0; /* Return with success */
}

/* function to return the exact solution*/
static int trueSol(sunrealtype t, N_Vector tSol, void* user_data)
{
  UserData* udata = (UserData*)user_data; /* access problem data */
  sunrealtype *TSol = NULL;
  TSol = N_VGetArrayPointer(tSol);
  if (check_flag((void*)TSol, "N_VGetArrayPointer", 0)) { return 1; }
  N_VConst(0.0, tSol); /* Initialize tSol to zero */

  /* set parameters */
  const sunindextype N  = udata->N;
  const sunrealtype dx  = udata->dx;
  const sunrealtype k1  = udata->k1;
  const sunrealtype k2  = udata->k2;
  const sunrealtype s1  = udata->s1;
  const sunrealtype s2  = udata->s2;

  sunrealtype* uSol = TSol;
  sunrealtype* vSol = TSol + N;

  for (int i = 0; i < N; i++)
  {
    uSol[i] = i * dx + 1.0;
    vSol[i] = (k1 * (i * dx + 1.0) + 1.0)/k2; //2k1 = k2
  }

  return 0;
}


inline void find_arg(std::vector<std::string>& args, const std::string key,
  sunrealtype& dest)
{
  auto it = find(args.begin(), args.end(), key);
  if (it != args.end())
  {
#if defined(SUNDIALS_SINGLE_PRECISION)
    dest = stof(*(it + 1));
#elif defined(SUNDIALS_DOUBLE_PRECISION)
    dest = stod(*(it + 1));
#elif defined(SUNDIALS_EXTENDED_PRECISION)
    dest = stold(*(it + 1));
#endif
    args.erase(it, it + 2);
  }
}


inline void find_arg(std::vector<std::string>& args, const std::string key,
  long long& dest)
{
auto it = find(args.begin(), args.end(), key);
if (it != args.end())
{
dest = stoll(*(it + 1));
args.erase(it, it + 2);
}
}

inline void find_arg(std::vector<std::string>& args, const std::string key,
  long int& dest)
{
auto it = find(args.begin(), args.end(), key);
if (it != args.end())
{
dest = stoll(*(it + 1));
args.erase(it, it + 2);
}
}

inline void find_arg(std::vector<std::string>& args, const std::string key,
  int& dest)
{
  auto it = find(args.begin(), args.end(), key);
  if (it != args.end())
  {
   dest = stoi(*(it + 1));
   args.erase(it, it + 2);
  }
}

inline void find_arg(std::vector<std::string>& args, const std::string key,
  std::string& dest)
{
  auto it = find(args.begin(), args.end(), key);
  if (it != args.end())
  {
    dest = *(it + 1);
    args.erase(it, it + 2);
  }
}

inline void find_arg(std::vector<std::string>& args, const std::string key,
  bool& dest, bool store = true)
{
  auto it = find(args.begin(), args.end(), key);
  if (it != args.end())
  {
    dest = store;
    args.erase(it);
  }
}


static int ReadInputs(std::vector<std::string>& args, UserData& udata,
  ARKODEParameters& uopts, SUNContext ctx)
{
  if (find(args.begin(), args.end(), "--help") != args.end())
  {
  InputHelp();
  return 1;
  }

// Problem parameters
 find_arg(args, "--N", udata.N);
 find_arg(args, "--xstart", udata.xstart);
 find_arg(args, "--xend", udata.xend);
 find_arg(args, "--swap_type", udata.swap_type);
 find_arg(args, "--k1", udata.k1);
 find_arg(args, "--k2", udata.k2);


// Integrator options
 find_arg(args, "--IMintegrator", uopts.IMintegrator);
 find_arg(args, "--EXintegrator", uopts.EXintegrator);
 find_arg(args, "--rtol", uopts.rtol);
 find_arg(args, "--atol", uopts.atol);
 find_arg(args,  "--fixed_h", uopts.fixed_h);
 find_arg(args,  "--maxsteps", uopts.maxsteps);
 find_arg(args,  "--output", uopts.output);
 find_arg(args,  "--T0", uopts.T0);
 find_arg(args,  "--Tf", uopts.Tf);
//  find_arg(args,  "--Nt", uopts.Nt);

 // Recompute mesh spacing and [re]allocate flux array
 udata.dx = (udata.xend - udata.xstart) / (udata.N);

return 0;
}


static void InputHelp()
 {
   std::cout << std::endl;
   std::cout << "Command line options:" << std::endl;
   std::cout << "  --IMintegrator <str> : method (ARKODE_SSP_SDIRK_2_1_2, "
                "ARKODE_SSP_DIRK_3_1_2, " 
                "ARKODE_SSP_LSPUM_SDIRK_3_1_2, or ARKODE_SSP_ESDIRK_4_2_3)\n";
   std::cout << "  --EXintegrator <str> : method (ARKODE_SSP_ERK_2_1_2, "
                "ARKODE_SSP_ERK_3_1_2, " 
                "ARKODE_SSP_LSPUM_ERK_3_1_2, or ARKODE_SSP_ERK_4_2_3)\n";
   std::cout << "  --swap_type <str> : swap, nonswap  \n";
   std::cout << "  --N <int>         : dimension\n";
   std::cout << "  --rtol <real>     : relative tolerance\n";
   std::cout << "  --atol <real>     : absolute tolerance\n";
   std::cout << "  --fixed_h <real>  : fixed step size\n";
   std::cout << "  --k1 <real>       : stiffness param k1 \n";
   std::cout << "  --k2 <real>       : stiffness param k2 \n";
   std::cout << "  --maxsteps <int>  : max steps between outputs\n";
   std::cout << "  --output <int>    : output level\n";
   std::cout << "  --xstart <real>   : left spatial end point  \n";
   std::cout << "  --xend <real>     : right spatial end point  \n";
   std::cout << "  --T0 <real>       : initial time \n";
   std::cout << "  --Tf <real>       : end time\n";
  //  std::cout << "  --Nt <int>        : number of outputs\n";
   std::cout << "  --help            : print options and exit\n";
 }


 // Print user data
static int PrintSetup(UserData& udata, ARKODEParameters& uopts)
{
  std::cout << std::endl;
  std::cout << "Problem parameters and options:" << std::endl;
  std::cout << " --------------------------------- " << std::endl;
  std::cout << "  N            = " << udata.N << std::endl;
  std::cout << "  dx           = " << udata.dx << std::endl;
  std::cout << "  xstart       = " << udata.xstart << std::endl;
  std::cout << "  xend         = " << udata.xend << std::endl;
  std::cout << "  swap_type    = " << udata.swap_type << std::endl;
  std::cout << "  k1           = " << udata.k1 << std::endl;
  std::cout << "  k2           = " << udata.k2 << std::endl;
  std::cout << " --------------------------------- " << std::endl;
  std::cout << "  IMintegrator = " << uopts.IMintegrator << std::endl;
  std::cout << "  EXintegrator = " << uopts.EXintegrator << std::endl;
  std::cout << "  rtol         = " << uopts.rtol << std::endl;
  std::cout << "  atol         = " << uopts.atol << std::endl;
  std::cout << "  fixed h      = " << uopts.fixed_h << std::endl;
  std::cout << "  maxsteps     = " << uopts.maxsteps << std::endl;
  std::cout << "  T0           = " << uopts.T0 << std::endl;
  std::cout << "  Tf           = " << uopts.Tf << std::endl;
  // std::cout << "  Nt           = " << uopts.Nt << std::endl;
  std::cout << " --------------------------------- " << std::endl;
  std::cout << "  output       = " << uopts.output << std::endl;
  std::cout << " --------------------------------- " << std::endl;
  // std::cout << std::endl;

  return 0;
}


 


/*-------------------------------
 * Private helper functions
 *-------------------------------*/

/* Check function return value...
    opt == 0 means SUNDIALS function allocates memory so check if
             returned NULL pointer
    opt == 1 means SUNDIALS function returns a flag so check if
             flag >= 0
    opt == 2 means function allocates memory so check if returned
             NULL pointer
*/
static int check_flag(void* flagvalue, const char* funcname, int opt)
{
  int* errflag;

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
  if (opt == 0 && flagvalue == NULL)
  {
    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return 1;
  }

  /* Check if flag < 0 */
  else if (opt == 1)
  {
    errflag = (int*)flagvalue;
    if (*errflag < 0)
    {
      fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
              funcname, *errflag);
      return 1;
    }
  }

  /* Check if function returned NULL pointer - no memory allocated */
  else if (opt == 2 && flagvalue == NULL)
  {
    fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return 1;
  }

  return 0;
}

/*---- end of file ----*/
