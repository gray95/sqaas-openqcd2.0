
/*******************************************************************************
*
* File check3.c
*
* Copyright (C) 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of the programs in the module unorm.c.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "mpi.h"
#include "su3.h"
#include "random.h"
#include "utils.h"
#include "lattice.h"
#include "sflds.h"
#include "global.h"

#define NFLDS 4

typedef union
{
   spinor s;
   float r[24];
} spin_t;

typedef union
{
   spinor_dble s;
   double r[24];
} spin_dble_t;


static float loc_nrm(spinor *s)
{
   int i;
   float sm;
   spin_t *sp;

   sm=0.0f;
   sp=(spin_t*)(s);

   for (i=0;i<24;i++)
      sm+=(*sp).r[i]*(*sp).r[i];

   return (float)(sqrt((double)(sm)));
}


static double loc_nrm_dble(spinor_dble *sd)
{
   int i;
   double sm;
   spin_dble_t *sp;

   sm=0.0;
   sp=(spin_dble_t*)(sd);

   for (i=0;i<24;i++)
      sm+=(*sp).r[i]*(*sp).r[i];

   return sqrt(sm);
}


static int chk_nrm(int icom,spinor *s)
{
   float nrm,ns,dist,dmax,tol;
   spinor *sm;

   nrm=unorm(VOLUME,icom,s);
   dist=nrm;
   dmax=0.0f;
   sm=s+VOLUME;

   for (;s<sm;s++)
   {
      ns=loc_nrm(s);

      if (ns<=nrm)
      {
         ns=nrm-ns;
         if (ns<dist)
            dist=ns;
      }
      else
      {
         ns=ns-nrm;
         if (ns<dist)
            dist=ns;
         if (ns>dmax)
            dmax=ns;
      }
   }

   tol=16.0f*nrm*FLT_EPSILON;

   if ((NPROC>1)&&(icom==1))
   {
      ns=dist;
      MPI_Reduce(&ns,&dist,1,MPI_FLOAT,MPI_MIN,0,MPI_COMM_WORLD);
      MPI_Bcast(&dist,1,MPI_FLOAT,0,MPI_COMM_WORLD);

      ns=dmax;
      MPI_Reduce(&ns,&dmax,1,MPI_FLOAT,MPI_MAX,0,MPI_COMM_WORLD);
      MPI_Bcast(&dmax,1,MPI_FLOAT,0,MPI_COMM_WORLD);
   }

   return ((dist<=tol)&&(dmax<=tol));
}


static int chk_nrm_dble(int icom,spinor_dble *sd)
{
   double nrm,ns,dist,dmax,tol;
   spinor_dble *sm;

   nrm=unorm_dble(VOLUME,icom,sd);
   dist=nrm;
   dmax=0.0;
   sm=sd+VOLUME;

   for (;sd<sm;sd++)
   {
      ns=loc_nrm_dble(sd);

      if (ns<=nrm)
      {
         ns=nrm-ns;
         if (ns<dist)
            dist=ns;
      }
      else
      {
         ns=ns-nrm;
         if (ns<dist)
            dist=ns;
         if (ns>dmax)
            dmax=ns;
      }
   }

   tol=16.0*nrm*DBL_EPSILON;

   if ((NPROC>1)&&(icom==1))
   {
      ns=dist;
      MPI_Reduce(&ns,&dist,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
      MPI_Bcast(&dist,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

      ns=dmax;
      MPI_Reduce(&ns,&dmax,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
      MPI_Bcast(&dmax,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   }

   return ((dist<=tol)&&(dmax<=tol));
}


int main(int argc,char *argv[])
{
   int my_rank,k,ie;
   spinor **ps;
   spinor_dble **psd;
   FILE *flog=NULL;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check3.log","w",stdout);
      printf("\n");
      printf("Check of the programs in the module unorm.c\n");
      printf("-------------------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);
   }

   start_ranlux(0,12345);
   geometry();
   alloc_ws(NFLDS);
   alloc_wsd(NFLDS);
   ps=reserve_ws(NFLDS);
   psd=reserve_wsd(NFLDS);

   for (k=0;k<NFLDS;k++)
   {
      random_s(VOLUME,ps[k],1.0f+(float)(k));
      random_sd(VOLUME,psd[k],1.0+(double)(k));

      ie=chk_nrm(k&0x1,ps[k]);
      error(ie!=1,1,"main [check3.c]",
            "Unexpected result of unorm() (icom=%d)",k&0x1);

      ie=chk_nrm_dble(k&0x1,psd[k]);
      error(ie!=1,1,"main [check3.c]",
            "Unexpected result of unorm_dble() (icom=%d)",k&0x1);
   }

   if (my_rank==0)
   {
      printf("No errors discovered -- all programs work correctly\n");
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
