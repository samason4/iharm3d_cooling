/******************************************************************************
 *                                                                            *
 * ELECTRONS.C                                                                *
 *                                                                            *
 * ELECTRON THERMODYNAMICS                                                    *
 *                                                                            *
 ******************************************************************************/

#include "decs.h"

#if ELECTRONS

void fixup_electrons_1zone(struct FluidState *S, int i, int j, int k);

// TODO put these in options with a default in decs.h
// Defined as in decs.h, CONSTANT not included in ALLMODELS version
// KAWAZURA is run by default if ALLMODELS=0 
#define KAWAZURA  9
#define WERNER    10
#define ROWAN     11
#define SHARMA    12
#define CONSTANT 5 //tbh, this is never considered 

#if HEATING
void heat_electrons_1zone(struct GridGeom *G, struct FluidState *Sh, struct FluidState *S, int i, int j, int k);
double get_fels(struct GridGeom *G, struct FluidState *S, int i, int j, int k, int model);
#endif

#if COOLING
void cool_electrons_1zone(struct GridGeom *G, struct FluidState *S, int i, int j, int k);
#endif
#if TESTCOOLING
void test_cool_electrons_1zone(struct GridGeom *G, struct FluidState *S, int i, int j, int k);
#endif

void init_electrons(struct GridGeom *G, struct FluidState *S)
{
  ZLOOPALL {
    // Set electron internal energy to constant fraction of internal energy
    double uel = fel0*S->P[UU][k][j][i];

    // Initialize entropies
    S->P[KTOT][k][j][i] = (gam-1.)*S->P[UU][k][j][i]*pow(S->P[RHO][k][j][i],-gam);

    // Initialize model entropy(ies)
    for (int idx = KEL0; idx < NVAR ; idx++) {
      S->P[idx][k][j][i] = (game-1.)*uel*pow(S->P[RHO][k][j][i],-game);
    }
  }

  // Necessary?  Usually called right afterward
  set_bounds(G, S);
}

#if HEATING
// TODO merge these
void heat_electrons(struct GridGeom *G, struct FluidState *Ss, struct FluidState *Sf)
{
  timer_start(TIMER_ELECTRON_HEAT);

#pragma omp parallel for collapse(3)
  ZLOOP {
    heat_electrons_1zone(G, Ss, Sf, i, j, k);
  }

  timer_stop(TIMER_ELECTRON_HEAT);
}

inline void heat_electrons_1zone(struct GridGeom *G, struct FluidState *Ss, struct FluidState *Sf, int i, int j, int k)
{
  // Actual entropy at final time
  double kHarm = (gam-1.)*Sf->P[UU][k][j][i]/pow(Sf->P[RHO][k][j][i],gam);

  //double uel = 1./(game-1.)*S->P[KEL][k][j][i]*pow(S->P[RHO][k][j][i],game);

  // Evolve model entropy(ies)
  for (int idx = KEL0; idx < NVAR ; idx++) {
    double fel = get_fels(G, Ss, i, j, k, idx);
    Sf->P[idx][k][j][i] += (game-1.)/(gam-1.)*pow(Ss->P[RHO][k][j][i],gam-game)*fel*(kHarm - Sf->P[KTOT][k][j][i]);
  }

  // TODO bhlight calculates Qvisc here instead of this
  //double ugHat = S->P[KTOT][k][j][i]*pow(S->P[RHO][k][j][i],gam)/(gam-1.);
  //double ugHarm = S->P[UU][k][j][i];

  // Update electron internal energy
  //uel += fel*(ugHarm - ugHat)*pow(Sh->P[RHO][k][j][i]/S->P[RHO][k][j][i],gam-game);

  // Convert back to electron entropy
  //S->P[KEL][k][j][i] = uel*(game-1.)*pow(S->P[RHO][k][j][i],-game);

  // Reset total entropy
  Sf->P[KTOT][k][j][i] = kHarm;
}

// New function for ALLMODELS runs.
inline double get_fels(struct GridGeom *G, struct FluidState *S, int i, int j, int k, int model)
{
  get_state(G, S, i, j, k, CENT);
  double bsq = bsq_calc(S, i, j, k);
  double fel = 0.0;
if (model == KAWAZURA) {
	// Equation (2) in http://www.pnas.org/lookup/doi/10.1073/pnas.1812491116
  double Tpr = (gamp-1.)*S->P[UU][k][j][i]/S->P[RHO][k][j][i];
  double uel = 1./(game-1.)*S->P[model][k][j][i]*pow(S->P[RHO][k][j][i],game);
  double Tel = (game-1.)*uel/S->P[RHO][k][j][i];
  if(Tel <= 0.) Tel = SMALL;
  if(Tpr <= 0.) Tpr = SMALL;

  double Trat = fabs(Tpr/Tel);
  double pres = S->P[RHO][k][j][i]*Tpr; // Proton pressure
  double beta = pres/bsq*2;
  if(beta > 1.e20) beta = 1.e20;
  
  double QiQe = 35./(1. + pow(beta/15.,-1.4)*exp(-0.1/Trat));
  fel = 1./(1. + QiQe);
} else if (model == WERNER) {
	// Equation (3) in http://academic.oup.com/mnras/article/473/4/4840/4265350
  double sigma = bsq/S->P[RHO][k][j][i];
  fel = 0.25*(1+pow(((sigma/5.)/(2+(sigma/5.))), .5));
} else if (model == ROWAN) {
	// Equation (34) in https://iopscience.iop.org/article/10.3847/1538-4357/aa9380
  double pres = (gamp-1.)*S->P[UU][k][j][i]; // Proton pressure
  double pg = (gam-1)*S->P[UU][k][j][i];
  double beta = pres/bsq*2;
  double sigma = bsq/(S->P[RHO][k][j][i]+S->P[UU][k][j][i]+pg);
  double betamax = 0.25/sigma;
  fel = 0.5*exp(-pow(1-beta/betamax, 3.3)/(1+1.2*pow(sigma, 0.7)));
} else if (model == SHARMA) {
	// Equation for \delta on  pg. 719 (Section 4) in https://iopscience.iop.org/article/10.1086/520800
  double Tpr = (gamp-1.)*S->P[UU][k][j][i]/S->P[RHO][k][j][i];
  double uel = 1./(game-1.)*S->P[model][k][j][i]*pow(S->P[RHO][k][j][i],game);
  double Tel = (game-1.)*uel/S->P[RHO][k][j][i];
  if(Tel <= 0.) Tel = SMALL;
  if(Tpr <= 0.) Tpr = SMALL;

  double Trat_inv = fabs(Tel/Tpr); //Inverse of the temperature ratio in KAWAZURA
  double QeQi = 0.33 * pow(Trat_inv, 0.5);
	fel = 1./(1.+1./QeQi);
}

#if SUPPRESS_HIGHB_HEAT
  if(bsq/S->P[RHO][k][j][i] > 1.) fel = 0;
#endif

  return fel;
}
#endif // HEATING

#if COOLING
void cool_electrons(struct GridGeom *G, struct FluidState *S)
{
  //looping through gridzones:
  #pragma omp parallel for collapse(2)
  ZLOOP {
    cool_electrons_1zone(G, S, i, j, k);
  }
}

inline void cool_electrons_1zone(struct GridGeom *G, struct FluidState *S, int i, int j, int k)
{
  //setting the convertion stuff:
  double CL = 2.99792458e10; // Speed of light
  double GNEWT = 6.6742e-8; // Gravitational constant
  double MSUN = 1.989e33; // grams per solar mass
  double Kbol = 1.380649e-16; // boltzmann constant
  double M_bh_cgs = M_bh * MSUN;
  double L_unit = GNEWT*M_bh_cgs/pow(CL, 2.);
  double T_unit = L_unit/CL;
  double RHO_unit = M_unit*pow(L_unit, -3.);
  double U_unit = RHO_unit*CL*CL;
  double B_unit = CL*sqrt(4.*M_PI*RHO_unit);
  double Ne_unit = RHO_unit/(MP + ME);
  double Thetae_unit = MP/ME;

  for (int idx = KEL0; idx < NVAR ; idx++) {
    //to fing uel and rho and such in cgs:
    double uel = (pow(S->P[RHO][k][j][i], game)*S->P[idx][k][j][i]/(game-1))*U_unit;
    double n_e = (S->P[RHO][k][j][i])*Ne_unit;
    double Tel = (game-1.)*uel/(n_e*Kbol); // this is in kelvin
    double theta_e = Tel/5.92986e9; // therefore this is unitless
    double B_mag = pow(bsq_calc(S, i, j, k), 0.5)*B_unit;

    //update the internal energy of the electrons at (i,j):
    uel = uel*exp(-dt*0.5*1.28567e-14*B_mag*B_mag*n_e*theta_e*theta_e);

    //convert back to code units:
    uel = uel/U_unit;

    //update the entropy with the new internal energy
    S->P[idx][k][j][i] = uel/pow(S->P[RHO][k][j][i], game)*(game-1);

    get_state(G, S, i, j, k, CENT);
    prim_to_flux(G, S, i, j, k, 0, CENT, S->U);
  }
}
#endif // COOLING

#if TESTCOOLING
void test_cool_electrons(struct GridGeom *G, struct FluidState *S)
{
  #pragma omp parallel for collapse(2)
  ZLOOP {
    test_cool_electrons_1zone(G, S, i, j, k);
  }
}

inline void test_cool_electrons_1zone(struct GridGeom *G, struct FluidState *S, int i, int j, int k)
{
//Have to initialize tau here for now because I can't figure out how to initialize it in prob/flat_space.
//I wanted to initialize it in decs.h as "extern int tau;" and then set it equal to 5 in param.dat, but iharm
//didn't like when I had "static int tau;" in problem.c so I just hardcoded it here instead
  double tau = 5.;

 //to fing uel:
  double uel = pow(S->P[RHO][k][j][i], game)*S->P[KEL0][k][j][i]/(game-1);

  //update the internal energy of the electrons at (i,j):
  uel = uel*exp(-dt*0.5/(tau));

  //update the entropy with the new internal energy
  S->P[KEL0][k][j][i] = uel/pow(S->P[RHO][k][j][i], game)*(game-1);

  get_state(G, S, i, j, k, CENT);
  prim_to_flux(G, S, i, j, k, 0, CENT, S->U);
}
#endif // TESTCOOLING

void fixup_electrons(struct FluidState *S)
{
  timer_start(TIMER_ELECTRON_FIXUP);

#pragma omp parallel for collapse(3)
  ZLOOP {
    fixup_electrons_1zone(S, i, j, k);
  }

  timer_stop(TIMER_ELECTRON_FIXUP);
}

inline void fixup_electrons_1zone(struct FluidState *S, int i, int j, int k)
{
  double kelmax = S->P[KTOT][k][j][i]*pow(S->P[RHO][k][j][i],gam-game)/(tptemin*(gam-1.)/(gamp-1.) + (gam-1.)/(game-1.));
  double kelmin = S->P[KTOT][k][j][i]*pow(S->P[RHO][k][j][i],gam-game)/(tptemax*(gam-1.)/(gamp-1.) + (gam-1.)/(game-1.));

  // Replace NANs with cold electrons
  for (int idx = KEL0; idx < NVAR ; idx++) {
    if (isnan(S->P[idx][k][j][i])) S->P[idx][k][j][i] = kelmin;
	// Enforce maximum Tp/Te
    S->P[idx][k][j][i] = MY_MAX(S->P[idx][k][j][i], kelmin);
	// Enforce minimum Tp/Te
    S->P[idx][k][j][i] = MY_MIN(S->P[idx][k][j][i], kelmax);
  }
}
#endif // ELECTRONS

