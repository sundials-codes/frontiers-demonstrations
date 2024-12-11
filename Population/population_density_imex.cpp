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

/* Header files */
#include <arkode/arkode_arkstep.h> /* prototypes for ARKStep fcts., consts */
#include <math.h>
#include <nvector/nvector_serial.h> /* serial N_Vector types, fcts., macros */
#include <stdio.h>
#include <stdlib.h>
#include <sundials/sundials_types.h> /* defs. of sunrealtype, sunindextype, etc */
#include <sunlinsol/sunlinsol_pcg.h> /* access to PCG SUNLinearSolver        */
#include <time.h>

#if defined(SUNDIALS_EXTENDED_PRECISION)
#define GSYM "Lg"
#define ESYM "Le"
#define FSYM "Lf"
#else
#define GSYM "g"
#define ESYM "e"
#define FSYM "f"
#endif

/* user data structure */
typedef struct
{
  sunindextype N; /* number of intervals   */
  sunrealtype dx; /* mesh spacing          */
  sunrealtype k;  /* diffusion coefficient */
}* UserData;

/* User-supplied Functions Called by the Solver */
static int fe(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data); //Explicit RHS
static int f(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data); //Implicit RHS
static int Jac(N_Vector v, N_Vector Jv, sunrealtype t, N_Vector y, N_Vector fy,
               void* user_data, N_Vector tmp);

/* Private function to check function return values */
static int check_flag(void* flagvalue, const char* funcname, int opt);

/* Main Program */
int main(void)
{
  /* general problem parameters */
  const sunrealtype T0 = SUN_RCONST(0.0); /* initial time */
  const sunrealtype Tf = SUN_RCONST(1.0); /* final time */
  const int Nt         = 10;              /* total number of output times */
  const sunrealtype rtol = 1.e-6;         /* relative tolerance */
  const sunrealtype atol = 1.e-10;        /* absolute tolerance */
  const sunindextype N = 201;             /* spatial mesh size */
  const sunrealtype k  = 0.02; // d = 0.02, 0.04 or 0 /* diffusion coefficient */
  int flag;                              /* reusable error-checking flag */

  /* Create the SUNDIALS context object for this simulation */
  SUNContext ctx;
  flag = SUNContext_Create(SUN_COMM_NULL, &ctx);
  if (check_flag(&flag, "SUNContext_Create", 1)) { return 1; }

  /* allocate and fill udata structure */
  UserData udata = (UserData)malloc(sizeof(*udata));
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
  flag = ARKodeSStolerances(arkode_mem, rtol, atol);
  if (check_flag(&flag, "ARKodeSStolerances", 1)) { return 1; }

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
  SUNContext_Free(&ctx);   /* Free context */

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
