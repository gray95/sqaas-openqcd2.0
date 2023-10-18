
/*******************************************************************************
*
* File time2.c
*
* Copyright (C) 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Timing of set_Awhat().
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "random.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "sflds.h"
#include "vflds.h"
#include "linalg.h"
#include "dirac.h"
#include "dfl.h"
#include "little.h"
#include "global.h"


static void random_basis(int Ns)
{
   int i;
   spinor **ws;

   ws=reserve_ws(Ns);

   for (i=0;i<Ns;i++)
   {
      random_s(VOLUME,ws[i],1.0f);
      bnd_s2zero(ALL_PTS,ws[i]);
   }

   dfl_subspace(ws);
   release_ws();
}


int main(int argc,char *argv[])
{
   int my_rank,bc,count,nt,ifail;
   int Ns,bs[4];
   double phi[2],phi_prime[2],theta[3];
   double mu[2],wt1,wt2,wdt;
   FILE *fin=NULL,*flog=NULL;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("time2.log","w",stdout);
      fin=freopen("check3.in","r",stdin);

      printf("\n");
      printf("Timing of set_Awhat()\n");
      printf("---------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);

      read_line("bs","%d %d %d %d",&bs[0],&bs[1],&bs[2],&bs[3]);
      read_line("Ns","%d",&Ns);
      fclose(fin);

      printf("bs = %d %d %d %d\n",bs[0],bs[1],bs[2],bs[3]);
      printf("Ns = %d\n\n",Ns);

      bc=find_opt(argc,argv,"-bc");

      if (bc!=0)
         error_root(sscanf(argv[bc+1],"%d",&bc)!=1,1,"main [time2.c]",
                    "Syntax: time2 [-bc <type>]");
   }

   set_lat_parms(5.5,1.0,0,NULL,0,1.978);
   print_lat_parms();

   MPI_Bcast(bs,4,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&Ns,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&bc,1,MPI_INT,0,MPI_COMM_WORLD);

   phi[0]=0.123;
   phi[1]=-0.534;
   phi_prime[0]=0.912;
   phi_prime[1]=0.078;
   theta[0]=0.38;
   theta[1]=-1.25;
   theta[2]=0.54;
   set_bc_parms(bc,1.0,1.0,0.9012,1.2034,phi,phi_prime,theta);
   print_bc_parms(2);

   set_sw_parms(-0.0123);
   set_dfl_parms(bs,Ns);

   start_ranlux(0,123456);
   geometry();

   alloc_ws(Ns);
   alloc_wvd(2);
   random_ud();
   set_ud_phase();
   random_basis(Ns);

   if (my_rank==0)
   {
      printf("Number of points = %d\n",VOLUME);
      printf("Number of blocks = %d\n",VOLUME/(bs[0]*bs[1]*bs[2]*bs[3]));
      printf("Number of points/block = %d\n\n",bs[0]*bs[1]*bs[2]*bs[3]);
      fflush(flog);
   }

   mu[0]=0.0376;
   mu[1]=0.0456;
   ifail=set_Awhat(mu[0]);
   ifail|=set_Awhat(mu[1]);
   error(ifail!=0,1,"main [time2.c]","Inversion of Aee or Aoo failed");

   nt=(int)(1.0e5/(double)(VOLUME));
   if (nt<1)
      nt=1;
   wdt=0.0;

   while (wdt<5.0)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();
      for (count=0;count<nt;count++)
      {
         set_flags(ERASED_AW);
         set_flags(ERASED_AWHAT);
         set_Awhat(mu[0]);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      wt2=MPI_Wtime();

      wdt=wt2-wt1;
      nt*=2;
   }

   wdt=2.0e3*wdt/(double)(nt);

   if (my_rank==0)
   {
      printf("Average time per first call of set_Awhat():\n");
      printf("Total:     %4.3f msec\n",wdt);
      printf("Per point: %4.3f usec\n\n",1.0e3*wdt/(double)(VOLUME));
      fflush(flog);
   }

   nt=(int)(0.5e5/(double)(VOLUME));
   if (nt<1)
      nt=1;
   wdt=0.0;
   set_flags(ERASED_AW);
   set_flags(ERASED_AWHAT);

   while (wdt<5.0)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();
      for (count=0;count<nt;count++)
      {
         set_Awhat(mu[0]);
         set_Awhat(mu[1]);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      wt2=MPI_Wtime();

      wdt=wt2-wt1;
      nt*=2;
   }

   wdt=1.0e3*wdt/(double)(nt);

   if (my_rank==0)
   {
      printf("Average time per mu-shift:\n");
      printf("Total:     %4.3f msec\n",wdt);
      printf("Per point: %4.3f usec\n\n",1.0e3*wdt/(double)(VOLUME));
      fflush(flog);
   }

   if (my_rank==0)
      fclose(flog);

   MPI_Finalize();
   exit(0);
}
