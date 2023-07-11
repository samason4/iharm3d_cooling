/******************************************************************************
 *                                                                            *
 * PROBLEM.C                                                                  *
 *                                                                            *
 * INITIAL CONDITIONS FOR ENTROPY WAVE                                        *
 *                                                                            *
 ******************************************************************************/

#include "decs.h"
#include <complex.h>
#include "hdf5_utils.h"

static double tau;

void set_problem_params()
{
  set_param("tau", &tau);
}

void save_problem_data(hid_t string_type){
        hdf5_write_single_val("flat_space", "PROB", string_type);
        hdf5_write_single_val(&tau, "tau", H5T_STD_I32LE);
}

void init(struct GridGeom *G, struct FluidState *S)
{
  // Mean state
  double rho0 = 1.;
  double u0 = 1.e-2;
  double U10 = 0.;
  double U20 = 0.;
  double U30 = 0.;
  double B10 = 0.;
  double B20 = 0.;
  double B30 = 0.;


  set_grid(G);

  LOG("Set grid");

  ZLOOP {
    S->P[RHO][k][j][i] = rho0;
    S->P[UU][k][j][i] = u0;
    S->P[U1][k][j][i] = U10;
    S->P[U2][k][j][i] = U20;
    S->P[U3][k][j][i] = U30;
    S->P[B1][k][j][i] = B10;
    S->P[B2][k][j][i] = B20;
    S->P[B3][k][j][i] = B30;

  } // ZLOOP

  init_electrons(G, S);

  //Enforce boundary conditions
  fixup(G, S);
  set_bounds(G, S);
}
