/* -----------------------------------------------------------------------------
 * Programmer(s): Generated with Claude
 * -----------------------------------------------------------------------------
 * In-repository Butcher tables for the embedded IMEX SSP methods
 *
 *   SSP212  : ARKODE_SSP_SDIRK_2_1_2       + ARKODE_SSP_ERK_2_1_2
 *   SSP312  : ARKODE_SSP_DIRK_3_1_2        + ARKODE_SSP_ERK_3_1_2
 *   SSPL312 : ARKODE_SSP_LSPUM_SDIRK_3_1_2 + ARKODE_SSP_LSPUM_ERK_3_1_2
 *   SSP423  : ARKODE_SSP_ESDIRK_4_2_3      + ARKODE_SSP_ERK_4_2_3
 *   SSP923  : ARKODE_SSP_ESDIRK_9_2_3      + ARKODE_SSP_ERK_9_2_3
 *
 * Sources of the coefficients:
 *   - SSP212, SSP923 : SUNDIALS branch feature/ssptables (9fdf95dfa)
 *   - SSP312, SSP423 : identical in feature/ssptables and
 *                      feature/lspum-embedding (in the latter the SSP423
 *                      implicit table is named ARKODE_ESDIRK_4_2_3)
 *   - SSPL312        : UPDATED embedding from feature/lspum-embedding
 *                      (9d7ed81e7, "Updated embedding coefficients for
 *                      ARKODE_SSP_LSPUM tables")
 *
 * HOW THIS FILE IS USED
 * ---------------------
 * The top-level CMakeLists.txt of the demonstrations repository (option
 * USE_IMEX_SSP_TABLES, ON by default) compiles every example
 * (population.cpp, linear_adv_rec.cpp, hyperbolic_relaxation.cpp, ...) with
 * this header force-included ("-include imex_ssp_tables.hpp"). A macro at
 * the bottom of the file then redirects
 *
 *   ARKStepSetTableName(mem, dirk_name, erk_name)
 *
 * to the wrapper below. When a name belongs to one of the five methods above,
 * the tables are built HERE and handed to SUNDIALS with ARKStepSetTables();
 * any other name (e.g. ARKODE_ARK548L2SA_DIRK_8_4_5 used by the reference
 * solutions) is passed straight through to SUNDIALS unchanged.
 *
 * Consequently the results do not depend on which SUNDIALS branch is
 * installed, and SUNDIALS never has to be patched or switched.
 *
 * Note: no special error bias is used for SSPL312. (feature/ssptables
 * hard-codes a bias of 2.0 for the OLD SSPL312 embedding, but that code keys
 * on the SUNDIALS table IDs, so it is never triggered by the tables supplied
 * here, whichever SUNDIALS branch is installed.)
 *
 * ---------------------------------------------------------------------------*/

#ifndef IMEX_SSP_TABLES_HPP
#define IMEX_SSP_TABLES_HPP

/* Pull in the real SUNDIALS declarations FIRST, before the redirect macros
   at the bottom of this file are defined. Their include guards make any later
   #include of these headers (in the demonstration sources) a no-op. */
#include <arkode/arkode.h>
#include <arkode/arkode_arkstep.h>
#include <arkode/arkode_butcher.h>
#include <arkode/arkode_butcher_dirk.h>
#include <arkode/arkode_butcher_erk.h>
#include <sundials/sundials_math.h>
#include <sundials/sundials_types.h>

#include <cstdio>
#include <cstring>

namespace imexssp {

/* ---------------------------------------------------------------------------
 * Explicit tables
 * -------------------------------------------------------------------------*/

/* https://doi.org/10.1016/j.cam.2022.114325 */
inline ARKodeButcherTable SSP_ERK_2_1_2()
{
  ARKodeButcherTable B = ARKodeButcherTable_Alloc(2, SUNTRUE);
  if (B == NULL) { return NULL; }
  B->q       = 2;
  B->p       = 1;
  B->A[1][0] = SUN_RCONST(1.0);

  B->b[0] = SUN_RCONST(0.5);
  B->b[1] = SUN_RCONST(0.5);

  B->d[0] = SUN_RCONST(0.694021459207626);
  B->d[1] = SUN_RCONST(1.0) - SUN_RCONST(0.694021459207626);

  B->c[1] = SUN_RCONST(1.0);
  return B;
}

/* https://doi.org/10.1016/j.cam.2022.114325 */
inline ARKodeButcherTable SSP_ERK_3_1_2()
{
  ARKodeButcherTable B = ARKodeButcherTable_Alloc(3, SUNTRUE);
  if (B == NULL) { return NULL; }
  B->q       = 2;
  B->p       = 1;
  B->A[1][0] = SUN_RCONST(0.5);
  B->A[2][0] = SUN_RCONST(0.5);
  B->A[2][1] = SUN_RCONST(0.5);

  B->b[0] = SUN_RCONST(1.0) / SUN_RCONST(3.0);
  B->b[1] = SUN_RCONST(1.0) / SUN_RCONST(3.0);
  B->b[2] = SUN_RCONST(1.0) / SUN_RCONST(3.0);

  B->d[0] = SUN_RCONST(4.0) / SUN_RCONST(9.0);
  B->d[1] = SUN_RCONST(1.0) / SUN_RCONST(3.0);
  B->d[2] = SUN_RCONST(2.0) / SUN_RCONST(9.0);

  B->c[1] = SUN_RCONST(0.5);
  B->c[2] = SUN_RCONST(1.0);
  return B;
}

/* Method from https://doi.org/10.1016/j.cam.2014.05.011
   UPDATED embedding (feature/lspum-embedding) */
inline ARKodeButcherTable SSP_LSPUM_ERK_3_1_2()
{
  ARKodeButcherTable B = ARKodeButcherTable_Alloc(3, SUNTRUE);
  if (B == NULL) { return NULL; }
  B->q = 2;
  B->p = 1;

  B->A[1][0] = SUN_RCONST(5.0) / SUN_RCONST(6.0);
  B->A[2][0] = SUN_RCONST(11.0) / SUN_RCONST(24.0);
  B->A[2][1] = SUN_RCONST(11.0) / SUN_RCONST(24.0);

  B->b[0] = SUN_RCONST(24.0) / SUN_RCONST(55.0);
  B->b[1] = SUN_RCONST(1.0) / SUN_RCONST(5.0);
  B->b[2] = SUN_RCONST(4.0) / SUN_RCONST(11.0);

  B->d[0] = SUN_RCONST(73467.0) / SUN_RCONST(180400.0);
  B->d[1] = SUN_RCONST(32969.0) / SUN_RCONST(180400.0);
  B->d[2] = SUN_RCONST(41.0) / SUN_RCONST(100.0);

  B->c[1] = SUN_RCONST(5.0) / SUN_RCONST(6.0);
  B->c[2] = SUN_RCONST(11.0) / SUN_RCONST(12.0);
  return B;
}

/* https://doi.org/10.1016/j.cam.2022.114325 */
inline ARKodeButcherTable SSP_ERK_4_2_3()
{
  ARKodeButcherTable B = ARKodeButcherTable_Alloc(4, SUNTRUE);
  if (B == NULL) { return NULL; }
  B->q       = 3;
  B->p       = 2;
  B->A[1][0] = SUN_RCONST(0.5);
  B->A[2][0] = B->A[2][1] = SUN_RCONST(0.5);
  B->A[3][0] = B->A[3][1] = B->A[3][2] = SUN_RCONST(1.0) / SUN_RCONST(6.0);

  B->b[0] = SUN_RCONST(1.0) / SUN_RCONST(6.0);
  B->b[1] = SUN_RCONST(1.0) / SUN_RCONST(6.0);
  B->b[2] = SUN_RCONST(1.0) / SUN_RCONST(6.0);
  B->b[3] = SUN_RCONST(0.5);

  B->d[0] = B->d[1] = B->d[2] = B->d[3] = SUN_RCONST(0.25);

  B->c[1] = SUN_RCONST(0.5);
  B->c[2] = SUN_RCONST(1.0);
  B->c[3] = SUN_RCONST(0.5);
  return B;
}

/* https://doi.org/10.1016/j.cam.2022.114325 */
inline ARKodeButcherTable SSP_ERK_9_2_3()
{
  ARKodeButcherTable B = ARKodeButcherTable_Alloc(9, SUNTRUE);
  if (B == NULL) { return NULL; }

  /* 1/6, 1/15, 1/9 */
  const sunrealtype t1 = SUN_RCONST(1.0) / SUN_RCONST(6.0);
  const sunrealtype t2 = SUN_RCONST(1.0) / SUN_RCONST(15.0);
  const sunrealtype t3 = SUN_RCONST(1.0) / SUN_RCONST(9.0);

  B->q       = 3;
  B->p       = 2;
  B->A[1][0] = t1;
  B->A[2][0] = B->A[2][1] = t1;
  B->A[3][0] = B->A[3][1] = B->A[3][2] = t1;
  B->A[4][0] = B->A[4][1] = B->A[4][2] = B->A[4][3] = t1;
  B->A[5][0] = B->A[5][1] = B->A[5][2] = B->A[5][3] = B->A[5][4] = t1;
  B->A[6][0]                                                     = t1;
  B->A[6][1] = B->A[6][2] = B->A[6][3] = B->A[6][4] = B->A[6][5] = t2;
  B->A[7][0] = B->A[7][6]                                        = t1;
  B->A[7][1] = B->A[7][2] = B->A[7][3] = B->A[7][4] = B->A[7][5] = t2;
  B->A[8][0] = B->A[8][6] = B->A[8][7]                           = t1;
  B->A[8][1] = B->A[8][2] = B->A[8][3] = B->A[8][4] = B->A[8][5] = t2;

  B->b[0] = B->b[6] = B->b[7] = B->b[8] = t1;
  B->b[1] = B->b[2] = B->b[3] = B->b[4] = B->b[5] = t2;

  for (int i = 0; i < 9; i++) { B->d[i] = t3; }

  B->c[1] = t1;
  B->c[2] = SUN_RCONST(2.0) * t1;
  B->c[3] = SUN_RCONST(3.0) * t1;
  B->c[4] = SUN_RCONST(4.0) * t1;
  B->c[5] = SUN_RCONST(5.0) * t1;
  B->c[6] = SUN_RCONST(5.0) * t2 + SUN_RCONST(1.0) * t1;
  B->c[7] = SUN_RCONST(5.0) * t2 + SUN_RCONST(2.0) * t1;
  B->c[8] = SUN_RCONST(5.0) * t2 + SUN_RCONST(3.0) * t1;
  return B;
}

/* ---------------------------------------------------------------------------
 * Implicit tables
 * -------------------------------------------------------------------------*/

/* https://doi.org/10.1007/BF02728986 (custom embedding) */
inline ARKodeButcherTable SSP_SDIRK_2_1_2()
{
  ARKodeButcherTable B = ARKodeButcherTable_Alloc(2, SUNTRUE);
  if (B == NULL) { return NULL; }

  /* 1 - 1 / sqrt(2) */
  const sunrealtype gamma = SUN_RCONST(1.0) -
                            SUN_RCONST(1.0) / SUNRsqrt(SUN_RCONST(2.0));

  B->q = 2;
  B->p = 1;

  B->A[0][0] = gamma;
  B->A[1][0] = SUN_RCONST(1.0) - SUN_RCONST(2.0) * gamma;
  B->A[1][1] = gamma;

  B->b[0] = SUN_RCONST(0.5);
  B->b[1] = SUN_RCONST(0.5);

  B->d[0] = SUN_RCONST(5.0) / SUN_RCONST(12.0);
  B->d[1] = SUN_RCONST(7.0) / SUN_RCONST(12.0);

  B->c[0] = gamma;
  B->c[1] = SUN_RCONST(1.0) - gamma;
  return B;
}

/* https://doi.org/10.1007/BF02728986 (custom embedding) */
inline ARKodeButcherTable SSP_DIRK_3_1_2()
{
  ARKodeButcherTable B = ARKodeButcherTable_Alloc(3, SUNTRUE);
  if (B == NULL) { return NULL; }

  B->q = 2;
  B->p = 1;

  B->A[0][0] = SUN_RCONST(0.25);
  B->A[1][1] = SUN_RCONST(0.25);
  B->A[2][0] = SUN_RCONST(1.0) / SUN_RCONST(3.0);
  B->A[2][1] = SUN_RCONST(1.0) / SUN_RCONST(3.0);
  B->A[2][2] = SUN_RCONST(1.0) / SUN_RCONST(3.0);

  B->b[0] = SUN_RCONST(1.0) / SUN_RCONST(3.0);
  B->b[1] = SUN_RCONST(1.0) / SUN_RCONST(3.0);
  B->b[2] = SUN_RCONST(1.0) / SUN_RCONST(3.0);

  /* 7/30 = 1/3 - 0.1 and 13/30 = 1/3 + 0.1 (feature/ssptables writes them
     the second way; the values agree to round-off) */
  B->d[0] = SUN_RCONST(7.0) / SUN_RCONST(30.0);
  B->d[1] = SUN_RCONST(1.0) / SUN_RCONST(3.0);
  B->d[2] = SUN_RCONST(13.0) / SUN_RCONST(30.0);

  B->c[0] = SUN_RCONST(0.25);
  B->c[1] = SUN_RCONST(0.25);
  B->c[2] = SUN_RCONST(1.0);
  return B;
}

/* Method from https://doi.org/10.1016/j.cam.2014.05.011
   UPDATED embedding (feature/lspum-embedding) */
inline ARKodeButcherTable SSP_LSPUM_SDIRK_3_1_2()
{
  ARKodeButcherTable B = ARKodeButcherTable_Alloc(3, SUNTRUE);
  if (B == NULL) { return NULL; }
  B->q = 2;
  B->p = 1;

  B->A[0][0] = SUN_RCONST(2.0) / SUN_RCONST(11.0);
  B->A[1][0] = SUN_RCONST(205.0) / SUN_RCONST(462.0);
  B->A[1][1] = SUN_RCONST(2.0) / SUN_RCONST(11.0);
  B->A[2][0] = SUN_RCONST(2033.0) / SUN_RCONST(4620.0);
  B->A[2][1] = SUN_RCONST(21.0) / SUN_RCONST(110.0);
  B->A[2][2] = SUN_RCONST(2.0) / SUN_RCONST(11.0);

  B->b[0] = SUN_RCONST(24.0) / SUN_RCONST(55.0);
  B->b[1] = SUN_RCONST(1.0) / SUN_RCONST(5.0);
  B->b[2] = SUN_RCONST(4.0) / SUN_RCONST(11.0);

  B->d[0] = SUN_RCONST(73467.0) / SUN_RCONST(180400.0);
  B->d[1] = SUN_RCONST(32969.0) / SUN_RCONST(180400.0);
  B->d[2] = SUN_RCONST(41.0) / SUN_RCONST(100.0);

  B->c[0] = SUN_RCONST(2.0) / SUN_RCONST(11.0);
  B->c[1] = SUN_RCONST(289.0) / SUN_RCONST(462.0);
  B->c[2] = SUN_RCONST(751.0) / SUN_RCONST(924.0);
  return B;
}

/* custom (named ARKODE_ESDIRK_4_2_3 in feature/lspum-embedding) */
inline ARKodeButcherTable SSP_ESDIRK_4_2_3()
{
  ARKodeButcherTable B = ARKodeButcherTable_Alloc(4, SUNTRUE);
  if (B == NULL) { return NULL; }
  const sunrealtype gam = SUN_RCONST(0.43586652150845899941601945119355684);

  B->q       = 3;
  B->p       = 2;
  B->A[1][0] = SUN_RCONST(0.064133478491541000583980548806443157);
  B->A[1][1] = gam;
  B->A[2][0] = SUN_RCONST(0.39641166139746771164632267806938966);
  B->A[2][1] = SUN_RCONST(0.16772181709407328893765787073705349);
  B->A[2][2] = gam;
  B->A[3][0] = -SUN_RCONST(0.49838612606484063319986117825862669);
  B->A[3][1] = SUN_RCONST(1.3860128578277056960456083134296045);
  B->A[3][2] = -SUN_RCONST(0.8234932532713240622617665863645347);
  B->A[3][3] = gam;

  B->b[0] = SUN_RCONST(1.0) / SUN_RCONST(6.0);
  B->b[1] = SUN_RCONST(1.0) / SUN_RCONST(6.0);
  B->b[2] = SUN_RCONST(1.0) / SUN_RCONST(6.0);
  B->b[3] = SUN_RCONST(1.0) / SUN_RCONST(2.0);

  B->d[0] = SUN_RCONST(1.0) / SUN_RCONST(4.0);
  B->d[1] = SUN_RCONST(1.0) / SUN_RCONST(4.0);
  B->d[2] = SUN_RCONST(1.0) / SUN_RCONST(4.0);
  B->d[3] = SUN_RCONST(1.0) / SUN_RCONST(4.0);

  B->c[1] = SUN_RCONST(0.5);
  B->c[2] = SUN_RCONST(1.0);
  B->c[3] = SUN_RCONST(0.5);
  return B;
}

inline ARKodeButcherTable SSP_ESDIRK_9_2_3()
{
  ARKodeButcherTable B = ARKodeButcherTable_Alloc(9, SUNTRUE);
  if (B == NULL) { return NULL; }

  B->q = 3;
  B->p = 2;

  B->A[1][0] = -SUN_RCONST(2.0) / SUN_RCONST(15.0);
  B->A[1][1] = SUN_RCONST(3.0) / SUN_RCONST(10.0);
  B->A[2][0] = -SUN_RCONST(1.0) / SUN_RCONST(6.0);
  B->A[2][1] = SUN_RCONST(1.0) / SUN_RCONST(2.0);
  B->A[3][1] = SUN_RCONST(1.0) / SUN_RCONST(2.0);
  B->A[4][0] = -SUN_RCONST(2.0) / SUN_RCONST(15.0);
  B->A[4][1] = SUN_RCONST(1.0) / SUN_RCONST(2.0);
  B->A[4][4] = SUN_RCONST(3.0) / SUN_RCONST(10.0);
  B->A[5][0] = SUN_RCONST(10750067225.0) / SUN_RCONST(15104505504.0);
  B->A[5][1] = SUN_RCONST(1.0) / SUN_RCONST(2.0);
  B->A[5][4] = -SUN_RCONST(5715232057.0) / SUN_RCONST(15104505504.0);
  B->A[6][0] = SUN_RCONST(211.0) / SUN_RCONST(2178.0);
  B->A[6][1] = SUN_RCONST(17.0) / SUN_RCONST(5445.0);
  B->A[6][4] = SUN_RCONST(1.0) / SUN_RCONST(10.0);
  B->A[6][6] = SUN_RCONST(3.0) / SUN_RCONST(10.0);
  B->A[7][0] = SUN_RCONST(16853104507.0) / SUN_RCONST(56641895640.0);
  B->A[7][1] = SUN_RCONST(17.0) / SUN_RCONST(5445.0);
  B->A[7][4] = SUN_RCONST(1.0) / SUN_RCONST(10.0);
  B->A[7][6] = SUN_RCONST(2767149.0) / SUN_RCONST(10402552.0);
  B->A[8][0] = SUN_RCONST(1860170351.0) / SUN_RCONST(11328379128.0);
  B->A[8][1] = SUN_RCONST(17.0) / SUN_RCONST(5445.0);
  B->A[8][4] = SUN_RCONST(1.0) / SUN_RCONST(10.0);
  B->A[8][6] = SUN_RCONST(2767149.0) / SUN_RCONST(10402552.0);
  B->A[8][8] = SUN_RCONST(3.0) / SUN_RCONST(10.0);

  B->b[0] = SUN_RCONST(3130013.0) / SUN_RCONST(1058750.0);
  B->b[1] = -SUN_RCONST(46374583.0) / SUN_RCONST(9528750.0);
  B->b[4] = SUN_RCONST(3281263.0) / SUN_RCONST(1361250.0);
  B->b[6] = SUN_RCONST(6789401.0) / SUN_RCONST(3176250.0);
  B->b[8] = -SUN_RCONST(2600638.0) / SUN_RCONST(1588125.0);

  B->d[0] = SUN_RCONST(2941.0) / SUN_RCONST(30250.0);
  B->d[1] = SUN_RCONST(7251.0) / SUN_RCONST(30250.0);
  B->d[4] = SUN_RCONST(933.0) / SUN_RCONST(1210.0);
  B->d[6] = -SUN_RCONST(27.0) / SUN_RCONST(250.0);

  B->c[1] = SUN_RCONST(1.0) / SUN_RCONST(6.0);
  B->c[2] = SUN_RCONST(1.0) / SUN_RCONST(3.0);
  B->c[3] = SUN_RCONST(1.0) / SUN_RCONST(2.0);
  B->c[4] = SUN_RCONST(2.0) / SUN_RCONST(3.0);
  B->c[5] = SUN_RCONST(5.0) / SUN_RCONST(6.0);
  B->c[6] = SUN_RCONST(1.0) / SUN_RCONST(2.0);
  B->c[7] = SUN_RCONST(2.0) / SUN_RCONST(3.0);
  B->c[8] = SUN_RCONST(5.0) / SUN_RCONST(6.0);
  return B;
}

/* ---------------------------------------------------------------------------
 * Registry: method label, implicit table name(s), explicit table name
 * -------------------------------------------------------------------------*/

struct Method
{
  const char* label;     /* short label used in the runtests scripts      */
  const char* dirk_name; /* implicit table name accepted on command line  */
  const char* dirk_alt;  /* alternative implicit name (or NULL)           */
  const char* erk_name;  /* explicit table name accepted on command line  */
  ARKodeButcherTable (*make_dirk)();
  ARKodeButcherTable (*make_erk)();
};

inline const Method* Methods(int* n)
{
  static const Method m[] = {
    {"SSP212", "ARKODE_SSP_SDIRK_2_1_2", NULL, "ARKODE_SSP_ERK_2_1_2",
     SSP_SDIRK_2_1_2, SSP_ERK_2_1_2},
    {"SSP312", "ARKODE_SSP_DIRK_3_1_2", NULL, "ARKODE_SSP_ERK_3_1_2",
     SSP_DIRK_3_1_2, SSP_ERK_3_1_2},
    {"SSPL312", "ARKODE_SSP_LSPUM_SDIRK_3_1_2", NULL,
     "ARKODE_SSP_LSPUM_ERK_3_1_2", SSP_LSPUM_SDIRK_3_1_2, SSP_LSPUM_ERK_3_1_2},
    {"SSP423", "ARKODE_SSP_ESDIRK_4_2_3", "ARKODE_ESDIRK_4_2_3",
     "ARKODE_SSP_ERK_4_2_3", SSP_ESDIRK_4_2_3, SSP_ERK_4_2_3},
    {"SSP923", "ARKODE_SSP_ESDIRK_9_2_3", NULL, "ARKODE_SSP_ERK_9_2_3",
     SSP_ESDIRK_9_2_3, SSP_ERK_9_2_3}};
  *n = (int)(sizeof(m) / sizeof(m[0]));
  return m;
}

inline bool Same(const char* a, const char* b)
{
  return (a != NULL) && (b != NULL) && (std::strcmp(a, b) == 0);
}

inline const Method* FindByDIRK(const char* name)
{
  int n;
  const Method* m = Methods(&n);
  for (int i = 0; i < n; i++)
  {
    if (Same(name, m[i].dirk_name) || Same(name, m[i].dirk_alt)) { return &m[i]; }
  }
  return NULL;
}

inline const Method* FindByERK(const char* name)
{
  int n;
  const Method* m = Methods(&n);
  for (int i = 0; i < n; i++)
  {
    if (Same(name, m[i].erk_name)) { return &m[i]; }
  }
  return NULL;
}

/* ---------------------------------------------------------------------------
 * Internal helpers: build one of the tables above by name, or fall back to the
 * SUNDIALS built-in table for any other name (e.g. ARK548L2SA).
 * -------------------------------------------------------------------------*/

inline ARKodeButcherTable LoadERKByName(const char* name)
{
  const Method* m = FindByERK(name);
  if (m != NULL) { return m->make_erk(); }
  return ARKodeButcherTable_LoadERKByName(name);
}

inline ARKodeButcherTable LoadDIRKByName(const char* name)
{
  const Method* m = FindByDIRK(name);
  if (m != NULL) { return m->make_dirk(); }
  return ARKodeButcherTable_LoadDIRKByName(name);
}

inline int SetTableName(void* arkode_mem, const char* itable, const char* etable)
{
  const Method* mi = FindByDIRK(itable);
  const Method* me = FindByERK(etable);

  /* Neither table is one of ours: let SUNDIALS handle it unchanged */
  if ((mi == NULL) && (me == NULL))
  {
    return ARKStepSetTableName(arkode_mem, itable, etable);
  }

  const bool inone = (itable == NULL) || Same(itable, "ARKODE_DIRK_NONE");
  const bool enone = (etable == NULL) || Same(etable, "ARKODE_ERK_NONE");

  /* Mirror SUNDIALS: an ImEx pair must be a matching pair */
  if (!inone && !enone && (mi != me))
  {
    std::fprintf(stderr,
                 "[imex_ssp_tables] ERROR: incompatible Butcher tables for an "
                 "ARK method: dirk = %s, erk = %s\n",
                 itable, etable);
    return ARK_ILL_INPUT;
  }

  ARKodeButcherTable Bi = inone ? NULL : LoadDIRKByName(itable);
  ARKodeButcherTable Be = enone ? NULL : LoadERKByName(etable);
  if ((!inone && Bi == NULL) || (!enone && Be == NULL))
  {
    std::fprintf(stderr, "[imex_ssp_tables] ERROR: could not build table(s) %s / %s\n",
                 itable ? itable : "(null)", etable ? etable : "(null)");
    ARKodeButcherTable_Free(Bi);
    ARKodeButcherTable_Free(Be);
    return ARK_ILL_INPUT;
  }

  ARKodeButcherTable Bref = (Bi != NULL) ? Bi : Be;
  int flag = ARKStepSetTables(arkode_mem, Bref->q, Bref->p, Bi, Be);

  ARKodeButcherTable_Free(Bi);
  ARKodeButcherTable_Free(Be);
  return flag;
}

} // namespace imexssp

/* ---------------------------------------------------------------------------
 * Redirect the demonstration codes' ARKStepSetTableName calls to the wrapper.
 * -------------------------------------------------------------------------*/
#ifndef IMEXSSP_NO_REDIRECT
#define ARKStepSetTableName imexssp::SetTableName
#endif

#endif /* IMEX_SSP_TABLES_HPP */
