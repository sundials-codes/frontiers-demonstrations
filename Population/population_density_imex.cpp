/*---------------------------------------------------------------
 * Programmer(s): Daniel R. Reynolds @ SMU
 * Modified by Sylvia Amihere @ SMU
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
 * The following test simulates a simple 1D Population Density equation,
 *    P_t = f + b(x,P) - r_d*P + kP_xx
 * for t in [0, 10], x in [0, 1], with initial conditions
 *    P(0,x) =  0
 * Periodic boundary conditions, and a point-source term,
 *    f(0,x) is a random value in [0.8 1.2]
 *    f(t,x) = 0, for all t>0
 * r_d = 1, b = r_b*(e/(e+P)), e = 0.005
 * r_b = 1*(1 + alpha*t), for 0<=x<=0.5
 * r_b = 100*(1 + alpha*t), for 0.5<x<=1
 * alpha = 0.001 (negative for declining birth rate)
 * k = 0.02, 0.04 (with diffusion) or 0(without diffusion)
 *
 * The spatial derivatives are computed using second-order
 * centered differences, with the data distributed over N points
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
#include <arkode/arkode_arkstep.h> /* prototypes for ARKStep fcts., consts */
// #include <arkode/arkode_erkstep.h> //Sylvia
// #include <arkode/arkode_butcher_erk.h> //Sylvia : ERK butcher tables
// #include <arkode/arkode_butcher_dirk.h> //Sylvia : DIRK butcher tables 
#include <math.h>
#include <nvector/nvector_serial.h> /* serial N_Vector types, fcts., macros */
#include <stdio.h>
#include <stdlib.h>
#include "sundials/sundials_core.hpp"
#include <sundials/sundials_types.h> /* defs. of sunrealtype, sunindextype, etc */
#include <sunlinsol/sunlinsol_pcg.h> /* access to PCG SUNLinearSolver        */
#include <time.h>
// #include "population_density_imex.hpp"
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

/* user data structure */
typedef struct
{
  sunindextype N; /* number of intervals   */
  sunrealtype dx; /* mesh spacing          */
  sunrealtype k;  /* diffusion coefficient */
}* UserData;

 class ARKODEParameters
 {
 public:
   // Integration method
   std::string Iintegrator;
   std::string Eintegrator;
 
   // Relative and absolute tolerances
   sunrealtype rtol;
   sunrealtype atol;
 
   // Step size selection (ZERO = adaptive steps)
   sunrealtype fixed_h;
 
   // Maximum number of time steps between outputs
   int maxsteps;
 
   // Output-related information
   int output;         // 0 = none, 1 = stats, 2 = disk, 3 = disk with tstop
   int nout;           // number of output times
   std::ofstream uout; // output file stream
 
   // constructor (with default values)
   ARKODEParameters()
     : Iintegrator("ARKODE_LSRK_SSP_S_2"),
       Eintegrator("ARKODE_LSRK_SSP_S_2"),
       rtol(SUN_RCONST(1.e-4)),
       atol(SUN_RCONST(1.e-11)),
       fixed_h(ZERO),
       maxsteps(10000),
       output(1),
       nout(10){};
 
 }; // end ARKODEParameters

/* User-supplied Functions Called by the Solver */
static int fe(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data); //Explicit RHS
static int f(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data); //Implicit RHS
static int Jac(N_Vector v, N_Vector Jv, sunrealtype t, N_Vector y, N_Vector fy,
               void* user_data, N_Vector tmp);
static int ReadInputs(std::vector<std::string>& args, UserData& udata,
  ARKODEParameters& uopts, SUNContext ctx);
static void InputHelp();

/* Private function to check function return values */
static int check_flag(void* flagvalue, const char* funcname, int opt);

/* Main Program */
// int main(void)
int main(int argc, char* argv[])
{

  // SUNDIALS context object for this simulation
  sundials::Context ctx;

  UserData udata;
  ARKODEParameters uopts;

  vector<string> args(argv + 1, argv + argc);

  int flag = ReadInputs(args, udata, uopts, ctx);
  if (check_flag(&flag, "ReadInputs", 1)) { return 1; }
  if (flag > 0) { return 0; }

  /* general problem parameters */
  const sunrealtype T0 = SUN_RCONST(0.0); /* initial time */
  const sunrealtype Tf = SUN_RCONST(1.0); /* final time */
  const int Nt         = 10;              /* total number of output times */
  // const sunrealtype rtol = 1.e-6;         /* relative tolerance */
  // const sunrealtype atol = 1.e-10;        /* absolute tolerance */
  const sunindextype N = 201;             /* spatial mesh size */
  const sunrealtype k  = 0.02; // d = 0.02, 0.04 or 0 /* diffusion coefficient */

  /* fill udata structure */
  udata->N  = N;
  udata->k  = k;
  udata->dx = SUN_RCONST(1.0) / (N - 1); /* mesh spacing */

  /* Initial problem output */
  printf("\n1D Population Density test problem:\n");
  printf("  N = %li\n", (long int)udata->N);
  printf("  diffusion coefficient:  k = %" GSYM "\n", udata->k);

  /* Initialize data structures */
  N_Vector y = N_VNew_Serial(N, ctx); /* Create serial vector for solution */
  if (check_flag((void*)y, "N_VNew_Serial", 0)) { return 1; }
  N_VConst(0.0, y); /* Set initial conditions */

  /* Call ARKStepCreate to initialize the ARK timestepper module and
     specify the right-hand side function in y'=f(t,y), the initial time
     T0, and the initial dependent variable vector y. */
  void* arkode_mem = ARKStepCreate(fe, f, T0, y, ctx);
  if (check_flag((void*)arkode_mem, "ARKStepCreate", 0)) { return 1; }

  /* Set routines */
  flag = ARKodeSetUserData(arkode_mem, (void*)udata);
  if (check_flag(&flag, "ARKodeSetUserData", 1)) { return 1; }
  flag = ARKodeSetMaxNumSteps(arkode_mem, 10000);
  if (check_flag(&flag, "ARKodeSetMaxNumSteps", 1)) { return 1; }
  flag = ARKodeSStolerances(arkode_mem, uopts.rtol, uopts.atol);
  if (check_flag(&flag, "ARKodeSStolerances", 1)) { return 1; }
  flag = ARKStepSetTableName(arkode_mem, "ARKODE_SSP_ESDIRK_4_2_3", "ARKODE_SSP_ERK_4_2_3"); //Sylvia: new embedded imex-ssp methods
  if (check_flag(&flag, "ARKStepSetTableName", 1)) { return 1; } //Sylvia
  // flag = ARKStepWriteParameters(arkode_mem, stdout); //Sylvia
  // if (check_flag(&flag, "ARKStepWriteParameters", 1)) { return 1; } //Sylvia

  /* Initialize PCG solver -- no preconditioning, with up to N iterations  */
  SUNLinearSolver LS = SUNLinSol_PCG(y, 0, (int)N, ctx);
  if (check_flag((void*)LS, "SUNLinSol_PCG", 0)) { return 1; }

  /* Linear solver interface */
  flag = ARKodeSetLinearSolver(arkode_mem, LS, NULL);
  if (check_flag(&flag, "ARKodeSetLinearSolver", 1)) { return 1; }

  /* output mesh to disk */
  FILE* FID = fopen("population_mesh.txt", "w");
  for (int i = 0; i < N; i++) { fprintf(FID, "  %.16" ESYM "\n", udata->dx * i); }
  fclose(FID);

  /* Open output stream for results, access data array */
  FILE* UFID = fopen("population.txt", "w");
  sunrealtype* data = N_VGetArrayPointer(y);

  /* output initial condition to disk */
  for (int i = 0; i < N; i++) { fprintf(UFID, " %.16" ESYM "", data[i]); }
  fprintf(UFID, "\n");

  /* Main time-stepping loop: calls ARKodeEvolve to perform the integration, then
     prints results.  Stops when the final time has been reached */
  sunrealtype t = T0;
  sunrealtype dTout = (Tf - T0) / Nt;
  sunrealtype tout  = T0 + dTout;
  printf("        t      ||u||_rms\n");
  printf("   -------------------------\n");
  printf("  %10.6" FSYM "  %10.6f\n", t, sqrt(N_VDotProd(y, y) / N));
  for (int iout = 0; iout < Nt; iout++)
  {
    flag = ARKodeEvolve(arkode_mem, tout, y, &t, ARK_NORMAL); /* call integrator */
    if (check_flag(&flag, "ARKodeEvolve", 1)) { break; }
    printf("  %10.6" FSYM "  %10.6f\n", t,
           sqrt(N_VDotProd(y, y) / N)); /* print solution stats */
    if (flag >= 0)
    { /* successful solve: update output time */
      tout += dTout;
      tout = (tout > Tf) ? Tf : tout;
    }
    else
    { /* unsuccessful solve: break */
      fprintf(stderr, "Solver failure, stopping integration\n");
      break;
    }

    /* output results to disk */
    for (int i = 0; i < N; i++) { fprintf(UFID, " %.16" ESYM "", data[i]); }
    fprintf(UFID, "\n");
  }
  printf("   -------------------------\n");
  fclose(UFID);

  /* Print final statistics */
  flag = ARKodePrintAllStats(arkode_mem, stdout, SUN_OUTPUTFORMAT_TABLE);
  if (check_flag(&flag, "ARKodePrintAllStats", 1)) { return 1; }

  /* Clean up and return with successful completion */
  N_VDestroy(y);           /* Free vectors */
  free(udata);             /* Free user data */
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
  UserData udata = (UserData)user_data; /* access problem data */
  sunrealtype *Y = NULL, *Ydot = NULL;
  Y = N_VGetArrayPointer(y); /* access data arrays */
  if (check_flag((void*)Y, "N_VGetArrayPointer", 0)) { return 1; }
  Ydot = N_VGetArrayPointer(ydot);
  if (check_flag((void*)Ydot, "N_VGetArrayPointer", 0)) { return 1; }
  N_VConst(0.0, ydot); /* Initialize ydot to zero */

  /* set parameters */
  const sunindextype N = udata->N;
  const sunrealtype dx = udata->dx;
  const sunrealtype rd = 1.0;
  const sunrealtype epsb = 0.005;
  const sunrealtype bRate = 0.001; // neg for decline and pos for increase in birth rate
  const sunrealtype bRateE = 1.0 + bRate * t;

  /* update random seed */
  srand(time(NULL));

  /* iterate over domain, computing all equations */
  if (t == 0.0) {
    for (int i = 0; i < N; i++) {
      sunrealtype rb = (dx*i <= 0.5) ? bRateE : 100.0*bRateE;
      sunrealtype rand_num = (float)random() /(float)RAND_MAX;
      sunrealtype fsource = rand_num * 0.4 + 0.8;
      Ydot[i] = rb * (epsb / (epsb + Y[i])) - rd * Y[i] + fsource;
    }
  } else {
    for (int i = 0; i < N; i++) {
      sunrealtype rb = (dx*i <= 0.5) ? bRateE : 100.0*bRateE;
      Ydot[i] = rb * (epsb / (epsb + Y[i])) - rd * Y[i];
    }
  }

  return 0; /* Return with success */
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
//  find_arg(args, "--gamma", udata.gamma);
//  find_arg(args, "--tf", udata.tf);
//  find_arg(args, "--xl", udata.xl);
//  find_arg(args, "--xr", udata.xr);
//  find_arg(args, "--nx", udata.nx);

// Integrator options
//  find_arg(args, "--integrator", uopts.integrator);
//  find_arg(args, "--stages", uopts.stages);
  find_arg(args, "--rtol", uopts.rtol);
  find_arg(args, "--atol", uopts.atol);
//  find_arg(args, "--fixed_h", uopts.fixed_h);
//  find_arg(args, "--maxsteps", uopts.maxsteps);
//  find_arg(args, "--output", uopts.output);
//  find_arg(args, "--nout", uopts.nout);

// Recompute mesh spacing and [re]allocate flux array
//  udata.dx = (udata.xr - udata.xl) / ((sunrealtype)udata.nx);
//  if (udata.flux) { delete[] udata.flux; }
//  udata.flux = new sunrealtype[NSPECIES * (udata.nx + 1)];

//  if (uopts.stages < 0)
//  {
//    std::cerr << "ERROR: Invalid number of stages" << std::endl;
//    return -1;
//  }

return 0;
}



static void InputHelp()
 {
   std::cout << std::endl;
   std::cout << "Command line options:" << std::endl;
  //  std::cout << "  --integrator <str> : method (ARKODE_LSRK_SSP_S_2, "
  //               "ARKODE_LSRK_SSP_S_3, "
  //               "ARKODE_LSRK_SSP_10_4, or any valid ARKODE_ERKTableID)\n";
  //  std::cout << "  --stages <int>     : number of stages (ignored for "
  //               "ARKODE_LSRK_SSP_10_4 and ERK)\n";
  //  std::cout << "  --tf <real>        : final time\n";
  //  std::cout << "  --xl <real>        : domain lower boundary\n";
  //  std::cout << "  --xr <real>        : domain upper boundary\n";
  //  std::cout << "  --gamma <real>     : ideal gas constant\n";
  //  std::cout << "  --nx <int>         : number of mesh points\n";
   std::cout << "  --rtol <real>      : relative tolerance\n";
   std::cout << "  --atol <real>      : absolute tolerance\n";
  //  std::cout << "  --fixed_h <real>   : fixed step size\n";
  //  std::cout << "  --maxsteps <int>   : max steps between outputs\n";
  //  std::cout << "  --output <int>     : output level\n";
  //  std::cout << "  --nout <int>       : number of outputs\n";
   std::cout << "  --help             : print options and exit\n";
 }


/* f routine to compute the ODE implicit RHS function f(t,y). */
static int f(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data)
{
  UserData udata = (UserData)user_data; /* access problem data */
  sunrealtype *Y = NULL, *Ydot = NULL;
  Y = N_VGetArrayPointer(y); /* access data arrays */
  if (check_flag((void*)Y, "N_VGetArrayPointer", 0)) { return 1; }
  Ydot = N_VGetArrayPointer(ydot);
  if (check_flag((void*)Ydot, "N_VGetArrayPointer", 0)) { return 1; }
  N_VConst(0.0, ydot); /* Initialize ydot to zero */

  /* set parameters */
  const sunindextype N = udata->N;
  const sunrealtype k  = udata->k;
  const sunrealtype dx = udata->dx;
  const sunrealtype c1 = k / dx / dx;
  const sunrealtype c2 = -2.0 * k / dx / dx;

  /* update random seed */
  srand(time(NULL));

  /* iterate over domain, computing all equations */
  if (t == 0.0) {
    for (int i = 0; i < N; i++) {
      sunrealtype Yleft = (i > 0) ? Y[i - 1] : Y[N - 1];
      sunrealtype Yright = (i < N - 1) ? Y[i + 1] : Y[0];
      Ydot[i] = c1 * Yleft + c2 * Y[i] + c1 * Yright;
    }
  } else {
    for (int i = 0; i < N; i++) {
      // sunrealtype rb = (dx*i <= 0.5) ? bRateE : 100.0*bRateE;
      sunrealtype Yleft = (i > 0) ? Y[i - 1] : Y[N - 1];
      sunrealtype Yright = (i < N - 1) ? Y[i + 1] : Y[0];
      Ydot[i] = c1 * Yleft + c2 * Y[i] + c1 * Yright;
    }
  }

  return 0; /* Return with success */
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
