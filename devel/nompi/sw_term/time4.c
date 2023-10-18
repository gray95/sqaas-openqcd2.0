
/*******************************************************************************
*
* File time4.c
*
* Copyright (C) 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Timing of sw_exp() and sw_dexp().
*
*******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "utils.h"
#include "random.h"
#include "sw_term.h"

#if (defined AVX)
static pauli_dble Aw[4] ALIGNED32;
#else
static pauli_dble Aw[4] ALIGNED16;
#endif


static void random_pauli(pauli_dble *A)
{
   int k;
   double tr,nrm,*u;

   u=(*A).u;
   gauss_dble(u,36);
   tr=tr0_pauli_mat(A)/6.0;

   for (k=0;k<6;k++)
      u[k]-=tr;

   nrm=tr1_pauli_mat(A,A)/2.0;
   nrm=1.0/sqrt(nrm);

   for (k=0;k<36;k++)
      u[k]*=nrm;
}


int main(void)
{
   int n,count;
   double t1,t2,dt;
   double q[2][36];

   printf("\n");
   printf("Timing of sw_exp() and sw_dexp()\n");
   printf("--------------------------------\n\n");

#if (defined AVX)
   printf("Using AVX instructions\n\n");
#elif (defined x64)
   printf("Using SSE3 instructions and up to 16 xmm registers\n\n");
#endif

   printf("Measurement made with all data in cache\n\n");

   random_pauli(Aw);
   random_pauli(Aw+1);
   sw_exp(18,0,Aw,1.2,Aw+2);
   sw_exp(18,1,Aw+1,0.9,Aw+3);

   n=(int)(1.0e6);
   dt=0.0;

   while (dt<2.0)
   {
      t1=(double)clock();
      for (count=0;count<n;count++)
      {
         sw_exp(18,0,Aw,1.2,Aw+2);
         sw_exp(18,1,Aw+1,0.9,Aw+3);
      }
      t2=(double)clock();
      dt=(t2-t1)/(double)(CLOCKS_PER_SEC);
      n*=2;
   }

   dt*=(1.0e6/(double)(n));
   printf("Time per call of sw_exp() = %.4f usec\n\n",dt);

   n=(int)(1.0e6);
   dt=0.0;

   while (dt<2.0)
   {
      t1=(double)clock();
      for (count=0;count<n;count++)
      {
         sw_dexp(18,Aw,0.912,q[1]);
         sw_dexp(18,Aw+1,1.23,q[2]);
      }
      t2=(double)clock();
      dt=(t2-t1)/(double)(CLOCKS_PER_SEC);
      n*=2;
   }

   dt*=(1.0e6/(double)(n));
   printf("Time per call of sw_dexp() = %.4f usec\n\n",dt);

   exit(0);
}
