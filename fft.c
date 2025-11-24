////////////////////////////////////////////////////////////////////////////////
// fft.c --- TP6 :  transform�e de Fourier multiple r�elle-complexe 3D
//
// Auteur          : Jeremie Gaidamour (CNRS/IDRIS) <gaidamou@idris.fr>
// Cr�� le         : Mon Aug 26 15:51:03 2013
// Dern. mod. par  : Jeremie Gaidamour (CNRS/IDRIS) <gaidamou@idris.fr>
// Dern. mod. le   : Mon Aug 26 15:51:03 2013
////////////////////////////////////////////////////////////////////////////////
/// fft.c --- Proyecto SCCA
/// transformada de Fourier multiple 3D
/// Autor          : Pon tu nombre
/// Creado         : Invierno 2025

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <sys/time.h>

#ifdef _OPENMP
#include <omp.h>
#endif // _OPENMP

// Interfaz de rutinas de Fortran de JMFFTW
extern void scfftm_(int* isign, int* n, int* lot, double* scale, double* x,         int* ldx, double complex* y, int* ldy, double* table, double* work, int* isys);
extern void csfftm_(int* isign, int* n, int* lot, double* scale, double complex* x, int* ldx, double* y,         int* ldy, double* table, double* work, int* isys);

// Inicializaci�n aleatoria de una matriz
void random_number(double* array, int size) {
  for(int i=0; i<size; i++) {
    // Generaci�n de un n�mero de un intervalo de [0,1]
    double r = (double)rand() / (double)(RAND_MAX - 1);
    array[i] = r;
  }
}

int main() {

  int nx=128, ny=200, nz=5000, ldx=nx+1, ldy=nx/2+1;
  double x[ldx*ny*nz], xx[ldx*ny*nz];
  double table[100+2*nx];

#ifdef _OPENMP
  int nb_taches; // nb_taches -> num_tareas
#pragma omp parallel
  { nb_taches = omp_get_num_threads(); }
  fprintf(stdout, "\n\n   Ejecucion de fft en paralelo con %d hilos\n", nb_taches);
#endif // _OPENMP

  // Inicializaci�n de la matriz x.
  random_number(&x[0], ldx*ny*nz);
  memcpy(xx, x, ldx*ny*nz*sizeof(double)); // xx = x

  // Tiempo CPU de c�lculo inicial.
  clock_t t_cpu_0 = clock();

  // Tiempo elapsado de referencia.
  struct timeval t_elapsed_0;
  gettimeofday(&t_elapsed_0, NULL);

  // Cálculo de la FFT (si es en paralelo, se procesa una rebanada de elementos por tarea).
  #pragma omp parallel
  {
    int z_debut, z_fin, z_tranche;

#ifdef _OPENMP
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();
#else
    int tid = 0;
    int nthreads = 1;
#endif

    int base = nz / nthreads;
    int rem  = nz % nthreads;

    if (tid < rem) {
      z_tranche = base + 1;
      z_debut   = tid * z_tranche;
    } else {
      z_tranche = base;
      z_debut   = tid * z_tranche + rem;
    }
    z_fin = (z_tranche > 0) ? (z_debut + z_tranche - 1) : (z_debut - 1);

    if (z_tranche <= 0) {
      // Nada que hacer para este hilo
    } else {
      // Alocación dinámica de memoria de matrices de trabajo y temporales (por hilo)
      double*         work   = (double*)        malloc(((2*nx+4)*ny*z_tranche)* sizeof(double));
      double*         temp_x = (double*)        malloc((ldx*ny*z_tranche)     * sizeof(double));
      double complex* temp_y = (double complex*)malloc((ldy*ny*z_tranche)     * sizeof(double complex));
      double*         table  = (double*)        malloc((100+2*nx)             * sizeof(double));

      if (!work || !temp_x || !temp_y || !table) {
        fprintf(stderr, "Error: asignación de memoria falló en tid=%d\n", tid);
        if (work) free(work);
        if (temp_x) free(temp_x);
        if (temp_y) free(temp_y);
        if (table) free(table);
        // No podemos salir del hilo fácilmente; simplemente saltamos.
      } else {
        int code = 0;

        // Definición de constantes para llamar a las rutinas de Fortran
        int    zero = 0, one  = 1, minus_one = -1;
        double d_one = 1.0;
        double inv_nx = 1.0/nx;
        int    lot = ny * z_tranche;

        // Inicialización de la tabla de coeficientes trigonométricos para esta tarea (por hilo)
        scfftm_(&zero, &nx, &lot, &d_one, &x[z_debut*ldx*ny], &ldx, (double complex*)&x[z_debut*ldx*ny], &ldy, table, work, &code);

        // Cálculo de la FFT directo en paralelo (trabajando sobre la rebancha local)
        memcpy(temp_x, &x[z_debut*ldx*ny], (ldx*ny*z_tranche)*sizeof(double)); // temp_x(:,:,:)=x(:,:,z_debut:z_fin)
        scfftm_(&one, &nx, &lot, &d_one, temp_x, &ldx, temp_y, &ldy, table, work, &code);

        // Cálculo de la FFT inversa en paralelo
        csfftm_(&minus_one, &nx, &lot, &inv_nx, temp_y, &ldy, temp_x, &ldx, table, work, &code);
        memcpy(&x[z_debut*ldx*ny], temp_x, (ldx*ny*z_tranche)*sizeof(double)); // x(:,:,z_debut:z_fin)=temp_x(:,:,:);

        // Liberación de memoria alocada para las matrices
        free(work);
        free(temp_x);
        free(temp_y);
        free(table);
      }
    }

  } // omp end parallel

  // Tiempo transcurrido final
  struct timeval t_elapsed_1;
  gettimeofday(&t_elapsed_1, NULL);
  double t_elapsed = (t_elapsed_1.tv_sec - t_elapsed_0.tv_sec) + (t_elapsed_1.tv_usec - t_elapsed_0.tv_usec) / (double)1000000;

  // Tiempo CPU final
  clock_t t_cpu_1 = clock();
  double t_cpu = (t_cpu_1 - t_cpu_0) / (double)CLOCKS_PER_SEC;

  // Verificaci�n de resultados.
  // ecart = maxval(abs(x(1:nx,1:ny,1:nz) - xx(1:nx,1:ny,1:nz)))/real(nx*ny*nz,kind=8)
  // ecart -> error
  double ecart = 0;
  for(int k=0; k<ldx*ny*nz; k++) {
    double curr = fabs(x[k] - xx[k]);
    if (curr > ecart)
      ecart = curr;
  }
  ecart /= nx*ny*nz;

  // Impresi�n de resultados
  fprintf(stdout, "\n\n"
	  "   Error FFT |Directo - Inversa| : %10.3E\n"
	  "   Tiempo transcurrido        : %10.3E seg.\n"
	  "   Tiempo de CPU              : %10.3E seg.\n",
	  ecart, t_elapsed, t_cpu
	  );

  return EXIT_SUCCESS;

}
