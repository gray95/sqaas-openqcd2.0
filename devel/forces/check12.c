
/*******************************************************************************
*
* File check12.c
*
* Copyright (C) 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of the numerical accuracy of the calculated actions and forces.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "mpi.h"
#include "su3.h"
#include "random.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "archive.h"
#include "uflds.h"
#include "sflds.h"
#include "mdflds.h"
#include "linalg.h"
#include "dfl.h"
#include "sw_term.h"
#include "forces.h"
#include "update.h"
#include "auxfcts.h"
#include "global.h"

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)

static struct
{
   int type,nio_nodes,nio_streams;
   int nb,ib;
   char cnfg_dir[NAME_SIZE];
} iodat;

static struct
{
   array_t *act,*frc,*phi,*dphi;
   array_t *dact0,*dact1,*dfrc;
} arrays;

static int my_rank,first,last,step,nsolv;
static char line[NAME_SIZE];
static char nbase[NAME_SIZE],cnfg_file[NAME_SIZE];
static smd_parms_t smd;
static FILE *flog=NULL,*fin=NULL;


static void read_iodat(void)
{
   int type,nion,nios;

   if (my_rank==0)
   {
      find_section("Run name");
      read_line("name","%s",nbase);

      find_section("Configurations");
      read_line("type","%s",line);

      if (strchr(line,'e')!=NULL)
         type=0x1;
      else if (strchr(line,'b')!=NULL)
         type=0x2;
      else if (strchr(line,'l')!=NULL)
         type=0x4;
      else
         type=0x0;

      error_root((strlen(line)!=1)||(type==0x0),1,"read_iodat [check12.c]",
                 "Improper configuration storage type");

      read_line("cnfg_dir","%s",line);

      if (type&0x6)
      {
         read_line("nio_nodes","%d",&nion);
         read_line("nio_streams","%d",&nios);
      }
      else
      {
         nion=1;
         nios=0;
      }

      read_line("first","%d",&first);
      read_line("last","%d",&last);
      read_line("step","%d",&step);

      error_root((first<1)||(last<first)||(step<1)||((last-first)%step!=0),1,
                 "read_iodat [check12.c]","Improper configuration range");
   }

   MPI_Bcast(nbase,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(line,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(&type,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nion,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nios,1,MPI_INT,0,MPI_COMM_WORLD);

   MPI_Bcast(&first,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&last,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&step,1,MPI_INT,0,MPI_COMM_WORLD);

   iodat.type=type;
   strcpy(iodat.cnfg_dir,line);
   iodat.nio_nodes=nion;
   iodat.nio_streams=nios;
}


static void read_lat_parms(void)
{
   int nk,isw;
   double beta,c0,csw,*kappa;

   if (my_rank==0)
   {
      find_section("Lattice parameters");
      read_line("beta","%lf",&beta);
      read_line("c0","%lf",&c0);
      nk=count_tokens("kappa");
      read_line("isw","%d",&isw);
      read_line("csw","%lf",&csw);
   }

   MPI_Bcast(&beta,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&c0,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&nk,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&isw,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&csw,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

   if (nk>0)
   {
      kappa=malloc(nk*sizeof(*kappa));
      error(kappa==NULL,1,"read_lat_parms [check12.c]",
            "Unable to allocate parameter array");
      if (my_rank==0)
         read_dprms("kappa",nk,kappa);
      MPI_Bcast(kappa,nk,MPI_DOUBLE,0,MPI_COMM_WORLD);
   }
   else
      kappa=NULL;

   set_lat_parms(beta,c0,nk,kappa,isw,csw);

   if (nk>0)
      free(kappa);
}


static void read_bc_parms(void)
{
   int bc;
   double phi[2],phi_prime[2],theta[3];
   double cG,cGp,cF,cFp;

   if (my_rank==0)
   {
      find_section("Boundary conditions");
      read_line("type","%d",&bc);

      cG=1.0;
      cGp=1.0;
      cF=1.0;
      cFp=1.0;
      phi[0]=0.0;
      phi[1]=0.0;
      phi_prime[0]=0.0;
      phi_prime[1]=0.0;

      if (bc==1)
         read_dprms("phi",2,phi);

      if ((bc==1)||(bc==2))
         read_dprms("phi'",2,phi_prime);

      if (bc<3)
      {
         read_line("cG","%lf",&cG);
         read_line("cF","%lf",&cF);

         if (bc==2)
         {
            read_line("cG'","%lf",&cGp);
            read_line("cF'","%lf",&cFp);
         }
         else
         {
            cGp=cG;
            cFp=cF;
         }
      }

      read_dprms("theta",3,theta);
   }

   MPI_Bcast(&bc,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(phi,2,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(phi_prime,2,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&cG,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&cGp,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&cF,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&cFp,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(theta,3,MPI_DOUBLE,0,MPI_COMM_WORLD);

   set_bc_parms(bc,cG,cGp,cF,cFp,phi,phi_prime,theta);
}


static void read_actions(void)
{
   int i,k,l,npf,nmu,nact,*iact;
   double *mu;
   action_parms_t ap;
   rat_parms_t rp;

   if (my_rank==0)
   {
      find_section("Actions");
      nact=count_tokens("actions");
      read_line("npf","%d",&npf);
   }

   MPI_Bcast(&nact,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&npf,1,MPI_INT,0,MPI_COMM_WORLD);

   if (nact>0)
   {
      iact=malloc(nact*sizeof(*iact));
      error(iact==NULL,1,"read_actions [check12.c]",
            "Unable to allocate temporary array");
      if (my_rank==0)
         read_iprms("actions",nact,iact);
      MPI_Bcast(iact,nact,MPI_INT,0,MPI_COMM_WORLD);
   }
   else
      iact=NULL;

   nmu=0;

   for (i=0;i<nact;i++)
   {
      k=iact[i];
      ap=action_parms(k);

      if (ap.action==ACTIONS)
      {
         read_action_parms(k);
         ap=action_parms(k);
      }

      if ((ap.action==ACF_RAT)||(ap.action==ACF_RAT_SDET))
      {
         l=ap.irat[0];
         rp=rat_parms(l);

         if (rp.degree==0)
            read_rat_parms(l);
      }
      else if ((nmu==0)&&((ap.action==ACF_TM1)||
                          (ap.action==ACF_TM1_EO)||
                          (ap.action==ACF_TM1_EO_SDET)||
                          (ap.action==ACF_TM2)||
                          (ap.action==ACF_TM2_EO)))
      {
         if (my_rank==0)
         {
            find_section("Actions");
            nmu=count_tokens("mu");
         }

         MPI_Bcast(&nmu,1,MPI_INT,0,MPI_COMM_WORLD);
      }
   }

   if (nmu>0)
   {
      mu=malloc(nmu*sizeof(*mu));
      error(mu==NULL,1,"read_actions [check12.c]",
            "Unable to allocate temporary array");

      if (my_rank==0)
      {
         find_section("Actions");
         read_dprms("mu",nmu,mu);
      }

      MPI_Bcast(mu,nmu,MPI_DOUBLE,0,MPI_COMM_WORLD);
   }
   else
      mu=NULL;

   smd=set_smd_parms(nact,iact,npf,nmu,mu,1,1.0,1.0,1);
   set_mdint_parms(0,LPFR,1.0,1,nact,iact);

   if (nact>0)
      free(iact);
   if (nmu>0)
      free(mu);
}


static void read_forces(void)
{
   int i,k,nact,*iact;
   force_parms_t fp;

   nact=smd.nact;
   iact=smd.iact;

   for (i=0;i<nact;i++)
   {
      k=iact[i];
      fp=force_parms(k);

      if (fp.force==FORCES)
      {
         read_force_parms2(k);
         fp=force_parms(k);
      }
   }
}


static void read_sap_parms(void)
{
   int bs[4];

   if (my_rank==0)
   {
      find_section("SAP");
      read_line("bs","%d %d %d %d",bs,bs+1,bs+2,bs+3);
   }

   MPI_Bcast(bs,4,MPI_INT,0,MPI_COMM_WORLD);
   set_sap_parms(bs,1,4,5);
}


static void read_dfl_parms(void)
{
   int bs[4],Ns;
   int ninv,nmr,ncy,nkv,nmx;
   double kappa,mu,res;

   if (my_rank==0)
   {
      find_section("Deflation subspace");
      read_line("bs","%d %d %d %d",bs,bs+1,bs+2,bs+3);
      read_line("Ns","%d",&Ns);
   }

   MPI_Bcast(bs,4,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&Ns,1,MPI_INT,0,MPI_COMM_WORLD);
   set_dfl_parms(bs,Ns);

   if (my_rank==0)
   {
      find_section("Deflation subspace generation");
      read_line("kappa","%lf",&kappa);
      read_line("mu","%lf",&mu);
      read_line("ninv","%d",&ninv);
      read_line("nmr","%d",&nmr);
      read_line("ncy","%d",&ncy);
   }

   MPI_Bcast(&kappa,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&mu,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&ninv,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nmr,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&ncy,1,MPI_INT,0,MPI_COMM_WORLD);
   set_dfl_gen_parms(kappa,mu,ninv,nmr,ncy);

   if (my_rank==0)
   {
      find_section("Deflation projection");
      read_line("nkv","%d",&nkv);
      read_line("nmx","%d",&nmx);
      read_line("res","%lf",&res);
   }

   MPI_Bcast(&nkv,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nmx,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&res,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   set_dfl_pro_parms(nkv,nmx,res);
}


static void read_solvers(void)
{
   int nact,*iact,i,j,k;
   int nsp,isap,idfl;
   action_parms_t ap;
   force_parms_t fp;
   solver_parms_t sp;

   nact=smd.nact;
   iact=smd.iact;
   isap=0;
   idfl=0;
   nsolv=0;

   for (i=0;i<nact;i++)
   {
      ap=action_parms(iact[i]);

      if ((ap.action==ACF_TM1)||
          (ap.action==ACF_TM1_EO)||
          (ap.action==ACF_TM1_EO_SDET)||
          (ap.action==ACF_TM2)||
          (ap.action==ACF_TM2_EO)||
          (ap.action==ACF_RAT)||
          (ap.action==ACF_RAT_SDET))
      {
         if ((ap.action==ACF_TM2)||(ap.action==ACF_TM2_EO)||
             (ap.action==ACF_RAT)||(ap.action==ACF_RAT_SDET))
            nsp=2;
         else
            nsp=1;

         for (k=0;k<nsp;k++)
         {
            j=ap.isp[k];
            sp=solver_parms(j);

            if (sp.solver==SOLVERS)
            {
               read_solver_parms(j);
               sp=solver_parms(j);

               if (sp.solver==SAP_GCR)
                  isap=1;
               else if (sp.solver==DFL_SAP_GCR)
               {
                  isap=1;
                  idfl=1;
               }
            }

            if (j>nsolv)
               nsolv=j;
         }
      }
   }

   for (i=0;i<nact;i++)
   {
      fp=force_parms(iact[i]);

      if ((fp.force==FRF_TM1)||
          (fp.force==FRF_TM1_EO)||
          (fp.force==FRF_TM1_EO_SDET)||
          (fp.force==FRF_TM2)||
          (fp.force==FRF_TM2_EO)||
          (fp.force==FRF_RAT)||
          (fp.force==FRF_RAT_SDET))
      {
         k=fp.isp[0];
         sp=solver_parms(k);

         if (sp.solver==SOLVERS)
         {
            read_solver_parms(k);
            sp=solver_parms(k);

            if (sp.solver==SAP_GCR)
               isap=1;
            else if (sp.solver==DFL_SAP_GCR)
            {
               isap=1;
               idfl=1;
            }
         }

         if (k>nsolv)
            nsolv=k;
      }
   }

   if (isap)
      read_sap_parms();

   if (idfl)
      read_dfl_parms();

   nsolv+=1;

   for (i=0;i<nact;i++)
   {
      ap=action_parms(iact[i]);

      if ((ap.action==ACF_TM1)||
          (ap.action==ACF_TM1_EO)||
          (ap.action==ACF_TM1_EO_SDET)||
          (ap.action==ACF_TM2)||
          (ap.action==ACF_TM2_EO)||
          (ap.action==ACF_RAT)||
          (ap.action==ACF_RAT_SDET))
      {
         if ((ap.action==ACF_TM2)||(ap.action==ACF_TM2_EO)||
             (ap.action==ACF_RAT)||(ap.action==ACF_RAT_SDET))
            nsp=2;
         else
            nsp=1;

         for (k=0;k<nsp;k++)
         {
            j=ap.isp[k];
            sp=solver_parms(j+nsolv);

            if (sp.solver==SOLVERS)
            {
               sp=solver_parms(j);
               set_solver_parms(j+nsolv,sp.solver,sp.nkv,sp.isolv,
                                sp.nmr,sp.ncy,sp.nmx,sp.istop,0.05*sp.res);
            }
         }
      }
   }

   for (i=0;i<nact;i++)
   {
      fp=force_parms(iact[i]);

      if ((fp.force==FRF_TM1)||
          (fp.force==FRF_TM1_EO)||
          (fp.force==FRF_TM1_EO_SDET)||
          (fp.force==FRF_TM2)||
          (fp.force==FRF_TM2_EO)||
          (fp.force==FRF_RAT)||
          (fp.force==FRF_RAT_SDET))
      {
         k=fp.isp[0];
         sp=solver_parms(k+nsolv);

         if (sp.solver==SOLVERS)
         {
            sp=solver_parms(k);
            set_solver_parms(k+nsolv,sp.solver,sp.nkv,sp.isolv,
                             sp.nmr,sp.ncy,sp.nmx,sp.istop,0.05*sp.res);
         }
      }
   }
}


static void check_files(void)
{
   int type,nion,nb,ib,n;
   int ns[4],bs[4];
   char *cnfg_dir;

   type=iodat.type;
   cnfg_dir=iodat.cnfg_dir;
   iodat.nb=0;
   iodat.ib=NPROC;

   if (type&0x1)
   {
      error(name_size("%s/%sn%d",cnfg_dir,nbase,last)>=NAME_SIZE,1,
            "check_files [check12.c]","cnfg_dir name is too long");
      check_dir_root(cnfg_dir);

      sprintf(line,"%s/%sn%d",cnfg_dir,nbase,first);
      lat_sizes(line,ns);
      error_root((ns[0]!=N0)||(ns[1]!=N1)||(ns[2]!=N2)||(ns[3]!=N3),1,
                 "check_files [check12.c]","Lattice size mismatch");
   }
   else if (type&0x2)
   {
      error(name_size("%s/0/0/%sn%d_b0",cnfg_dir,nbase,first)>=NAME_SIZE,1,
            "check_files [check8.c]","cnfg_dir name is too long");
      sprintf(line,"%s/0/0",cnfg_dir);
      if ((cpr[0]==0)&&(cpr[1]==0)&&(cpr[2]==0)&&(cpr[3]==0))
         check_dir(line);

      sprintf(line,"%s/0/0/%sn%d_b0",iodat.cnfg_dir,nbase,first);
      blk_sizes(line,ns,bs);
      error_root((ns[0]!=N0)||(ns[1]!=N1)||(ns[2]!=N2)||(ns[3]!=N3),1,
                 "check_files [check12.c]","Lattice size mismatch");

      ib=blk_index(ns,bs,&nb);
      nion=iodat.nio_nodes;
      n=nb/nion;
      error_root(nb%nion!=0,1,"check_files [check12.c]",
                 "Number of blocks is not a multiple of nio_nodes");
      error(name_size("%s/%d/%d/%sn%d_b%d",cnfg_dir,nion-1,n-1,
                      nbase,last,nb-1)>=NAME_SIZE,1,
            "check_files [check12.c]","cnfg_dir name is too long");
      sprintf(line,"%s/%d/%d",cnfg_dir,ib/n,ib%n);
      strcpy(cnfg_dir,line);
      if (((cpr[0]*L0)%bs[0]==0)&&((cpr[1]*L1)%bs[1]==0)&&
          ((cpr[2]*L2)%bs[2]==0)&&((cpr[3]*L3)%bs[3]==0))
         check_dir(cnfg_dir);

      iodat.nb=nb;
      iodat.ib=ib;
   }
   else
   {
      nion=iodat.nio_nodes;
      n=NPROC/nion;
      error_root(NPROC%nion!=0,1,"check_files [check12.c]",
                 "Number of processes is not a multiple of nio_nodes");
      error(name_size("%s/%d/%d/%sn%d_%d",cnfg_dir,nion-1,n-1,
                      nbase,last,NPROC-1)>=NAME_SIZE,1,
            "check_files [check12.c]","cnfg_dir name is too long");
      sprintf(line,"%s/%d/%d",cnfg_dir,my_rank/n,my_rank%n);
      strcpy(cnfg_dir,line);
      check_dir(cnfg_dir);
   }
}


static void read_ud(int icnfg)
{
   int type;
   double wt1,wt2;
   char *cnfg_dir;

   MPI_Barrier(MPI_COMM_WORLD);
   wt1=MPI_Wtime();

   type=iodat.type;
   cnfg_dir=iodat.cnfg_dir;

   if (type&0x1)
   {
      sprintf(cnfg_file,"%s/%sn%d",cnfg_dir,nbase,icnfg);
      import_cnfg(cnfg_file,0x0);
   }
   else if (type&0x2)
   {
      set_nio_streams(iodat.nio_streams);
      sprintf(cnfg_file,"%s/%sn%d_b%d",cnfg_dir,nbase,icnfg,iodat.ib);
      blk_import_cnfg(cnfg_file,0x0);
   }
   else
   {
      set_nio_streams(iodat.nio_streams);
      sprintf(cnfg_file,"%s/%sn%d_%d",cnfg_dir,nbase,icnfg,my_rank);
      read_cnfg(cnfg_file);
   }

   MPI_Barrier(MPI_COMM_WORLD);
   wt2=MPI_Wtime();

   if (my_rank==0)
   {
      printf("Gauge field read from disk in %.2e sec\n\n",wt2-wt1);
      fflush(flog);
   }
}


static qflt set_pf(int iact,int irs,int *status,double *nrm)
{
   int k,ipf,vol;
   double *mu;
   qflt act;
   spinor_dble *pf,**wsd;
   mdflds_t *mdfs;
   action_parms_t ap;

   ap=action_parms(iact);
   set_sw_parms(sea_quark_mass(ap.im0));
   ipf=ap.ipf;

   mdfs=mdflds();
   pf=(*mdfs).pf[ipf];
   if ((*mdfs).eo[ipf])
      vol=VOLUME/2;
   else
      vol=VOLUME;

   mu=smd.mu;
   act.q[0]=0.0;
   act.q[1]=0.0;
   status[0]=0;
   status[1]=0;
   status[2]=0;

   wsd=reserve_wsd(1);

   if (irs==0)
      assign_sd2sd(vol,pf,wsd[0]);

   if (ap.action==ACF_TM1)
      act=setpf1(mu[ap.imu[0]],ipf,1);
   else if (ap.action==ACF_TM1_EO)
      act=setpf4(mu[ap.imu[0]],ipf,0,1);
   else if (ap.action==ACF_TM1_EO_SDET)
      act=setpf4(mu[ap.imu[0]],ipf,1,1);
   else
   {
      k=ap.isp[1]+irs*nsolv;

      if (ap.action==ACF_TM2)
         act=setpf2(mu[ap.imu[0]],mu[ap.imu[1]],ipf,k,1,status);
      else if (ap.action==ACF_TM2_EO)
         act=setpf5(mu[ap.imu[0]],mu[ap.imu[1]],ipf,k,1,status);
      else if (ap.action==ACF_RAT)
         act=setpf3(ap.irat,ipf,0,k,1,status);
      else if (ap.action==ACF_RAT_SDET)
         act=setpf3(ap.irat,ipf,1,k,1,status);
   }

   if (irs==0)
   {
      mulr_spinor_add_dble(vol,wsd[0],pf,-1.0);
      (*nrm)=unorm_dble(vol,1,wsd[0]);
   }
   else
      (*nrm)=unorm_dble(vol,1,pf);

   release_wsd();

   return act;
}


static qflt get_action(int iact,int irs,int *status)
{
   int k,ipf;
   double *mu;
   qflt act;
   action_parms_t ap;

   ap=action_parms(iact);
   set_sw_parms(sea_quark_mass(ap.im0));
   ipf=ap.ipf;
   k=ap.isp[0]+irs*nsolv;

   mu=smd.mu;
   act.q[0]=0.0;
   act.q[1]=0.0;

   if (ap.action==ACF_TM1)
      act=action1(mu[ap.imu[0]],ipf,k,1,0,status);
   else if (ap.action==ACF_TM1_EO)
      act=action4(mu[ap.imu[0]],ipf,0,k,1,0,status);
   else if (ap.action==ACF_TM1_EO_SDET)
      act=action4(mu[ap.imu[0]],ipf,1,k,1,0,status);
   else if (ap.action==ACF_TM2)
      act=action2(mu[ap.imu[0]],mu[ap.imu[1]],ipf,k,1,0,status);
   else if (ap.action==ACF_TM2_EO)
      act=action5(mu[ap.imu[0]],mu[ap.imu[1]],ipf,k,1,0,status);
   else if (ap.action==ACF_RAT)
      act=action3(ap.irat,ipf,0,k,1,status);
   else if (ap.action==ACF_RAT_SDET)
      act=action3(ap.irat,ipf,1,k,1,status);

   return act;
}


static void set_force(int ifr,int irs,int *status)
{
   int k,ipf;
   double *mu;
   force_parms_t fp;

   fp=force_parms(ifr);
   set_sw_parms(sea_quark_mass(fp.im0));
   ipf=fp.ipf;
   k=fp.isp[0]+irs*nsolv;

   mu=smd.mu;
   set_frc2zero();

   if (fp.force==FRF_TM1)
      force1(mu[fp.imu[0]],ipf,k,0,1.0,status);
   else if (fp.force==FRF_TM1_EO)
      force4(mu[fp.imu[0]],ipf,0,k,0,1.0,status);
   else if (fp.force==FRF_TM1_EO_SDET)
      force4(mu[fp.imu[0]],ipf,1,k,0,1.0,status);
   else if (fp.force==FRF_TM2)
      force2(mu[fp.imu[0]],mu[fp.imu[1]],ipf,k,0,1.0,status);
   else if (fp.force==FRF_TM2_EO)
      force5(mu[fp.imu[0]],mu[fp.imu[1]],ipf,k,0,1.0,status);
   else if (fp.force==FRF_RAT)
      force3(fp.irat,ipf,0,k,1.0,status);
   else if (fp.force==FRF_RAT_SDET)
      force3(fp.irat,ipf,1,k,1.0,status);
}


static void print_status_act(int iact,int *status,double wdt)
{
   action_parms_t ap;
   solver_parms_t sp;

   ap=action_parms(iact);
   sp=solver_parms(ap.isp[0]);
   printf("status = ");

   if ((sp.solver==CGNE)||(sp.solver==MSCG)||(sp.solver==SAP_GCR))
      printf("%d",status[0]);
   else if (sp.solver==DFL_SAP_GCR)
      printf("%d,%d,%d",status[0],status[1],status[2]);

   printf(" (time = %.2e sec)\n",wdt);
}


static void print_status_frc(int ifr,int *status,double wdt)
{
   force_parms_t fp;
   solver_parms_t sp;

   fp=force_parms(ifr);
   sp=solver_parms(fp.isp[0]);
   printf("status = ");

   if ((sp.solver==CGNE)||(sp.solver==MSCG))
      printf("%d",status[0]);
   else if (sp.solver==SAP_GCR)
      printf("%d,%d",status[0],status[1]);
   else if (sp.solver==DFL_SAP_GCR)
      printf("%d,%d,%d; %d,%d,%d",status[0],status[1],status[2],
             status[3],status[4],status[5]);

   printf(" (time = %.2e sec)\n",wdt);
}


static void alloc_arrays(void)
{
   size_t n[3];

   n[0]=smd.nact+1;
   n[1]=3;
   arrays.act=alloc_array(2,n,sizeof(double),0);
   arrays.frc=alloc_array(2,n,sizeof(double),0);
   arrays.phi=alloc_array(2,n,sizeof(double),0);
   arrays.dphi=alloc_array(2,n,sizeof(double),0);
   arrays.dact0=alloc_array(2,n,sizeof(double),0);

   n[1]=2;
   n[2]=3;
   arrays.dact1=alloc_array(3,n,sizeof(double),0);
   arrays.dfrc=alloc_array(3,n,sizeof(double),0);
}


static void set_val(double v,double *a)
{
   a[0]=v;
   a[1]=v;
   a[2]=v;
}


static void add_val(double v,double *a)
{
   if (v<a[0])
      a[0]=v;
   if (v>a[1])
      a[1]=v;
   a[2]+=v;
}


int main(int argc,char *argv[])
{
   int icnfg,ncnfg,status[6];
   int n,i,k,nact,*iact;
   int nwud,nwfd,nws,nwv,nwvd;
   int isap,idfl;
   double nrm0,nrm1,dev,atot,datot,wt1,wt2;
   double **act,**frc,**phi,**dphi,**dact0;
   double ***dact1,***dfrc;
   qflt qact0,qact1,qact2;
   su3_alg_dble **wfd;
   mdflds_t *mdfs;
   dfl_parms_t dfl;
   action_parms_t ap;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check12.log","w",stdout);
      fin=freopen("check12.in","r",stdin);

      printf("\n");
      printf("Numerical precision of the calculated actions and forces\n");
      printf("--------------------------------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);
   }

   read_iodat();
   read_lat_parms();
   read_bc_parms();
   read_actions();
   read_forces();
   read_solvers();

   if (my_rank==0)
      fclose(fin);

   check_machine();

   if (my_rank==0)
   {
      print_lat_parms();
      print_bc_parms(3);

      printf("actions =");
      for (i=0;i<smd.nact;i++)
         printf(" %d",smd.iact[i]);
      printf("\n");
      printf("npf = %d\n",smd.npf);

      if (smd.nmu>0)
      {
         printf("mu =");

         for (i=0;i<smd.nmu;i++)
         {
            n=fdigits(smd.mu[i]);
            printf(" %.*f",IMAX(n,1),smd.mu[i]);
         }
         printf("\n");
      }

      printf("\n");
      print_action_parms();
      print_rat_parms();
      print_mdint_parms();
      print_force_parms2();
      print_solver_parms(&isap,&idfl);

      if (isap)
         print_sap_parms(0);

      if (idfl)
         print_dfl_parms(1);

      printf("Configuration storage type = ");

      if (iodat.type&0x1)
         printf("exported\n");
      else if (iodat.type&0x2)
         printf("block-exported\n");
      else
         printf("local\n");

      if (iodat.type&0x6)
         printf("Parallel configuration input: "
                "nio_nodes = %d, nio_streams = %d\n",
                iodat.nio_nodes,iodat.nio_streams);
      printf("\n");
      printf("Configurations %sn%d -> %sn%d in steps of %d\n\n",
             nbase,first,nbase,last,step);
      fflush(flog);
   }

   geometry();
   check_files();

   smd_wsize(&nwud,&nwfd,&nws,&nwv,&nwvd);
   alloc_wud(nwud);
   alloc_wfd(nwfd+2);
   alloc_ws(nws+2);
   wsd_uses_ws();
   alloc_wv(nwv);
   alloc_wvd(nwvd);

   wfd=reserve_wfd(2);

   mdfs=mdflds();
   dfl=dfl_parms();
   nact=smd.nact;
   iact=smd.iact;

   alloc_arrays();
   act=(double**)(arrays.act[0].a);
   frc=(double**)(arrays.frc[0].a);
   phi=(double**)(arrays.phi[0].a);
   dphi=(double**)(arrays.dphi[0].a);
   dact0=(double**)(arrays.dact0[0].a);
   dact1=(double***)(arrays.dact1[0].a);
   dfrc=(double***)(arrays.dfrc[0].a);

   for (icnfg=first;icnfg<=last;icnfg+=step)
   {
      if (my_rank==0)
      {
         printf("Configuration no %d\n\n",icnfg);
         fflush(flog);
      }

      read_ud(icnfg);
      set_ud_phase();

      if (dfl.Ns)
      {
         dfl_modes(status);

         if (my_rank==0)
         {
            printf("Deflation subspace generation: status = %d\n\n",status[0]);

            error_root(status[0]<0,1,"main [check12.c]",
                       "Deflation subspace generation failed");
         }
      }

      random_mom();
      qact0=momentum_action(1);
      qact1=action0(1);
      atot=qact0.q[0]+qact1.q[0];
      datot=0.0;

      force0(1.0);
      assign_alg2alg(4*VOLUME,(*mdfs).frc,wfd[0]);
      nrm0=unorm_alg(4*VOLUME,1,wfd[0]);
      set_alg2zero(4*VOLUME,wfd[1]);

      if (my_rank==0)
      {
         printf("Momentum action:\n");
         printf("act = %.2e\n",qact0.q[0]);
         printf("Gauge action:\n");
         printf("act = %.2e\n",qact1.q[0]);
         printf("Gauge force:\n");
         printf("|frc|_oo = %.2e\n\n",nrm0);
         fflush(flog);
      }

      for (i=0;i<nact;i++)
      {
         k=iact[i];
         ap=action_parms(k);

         if (ap.action!=ACG)
         {
            if ((ap.action!=ACF_TM1)&&(ap.action!=ACF_TM1_EO)&&
                (ap.action!=ACF_TM1_EO_SDET))
               save_ranlux();

            MPI_Barrier(MPI_COMM_WORLD);
            wt1=MPI_Wtime();
            qact0=set_pf(k,1,status,&nrm0);
            MPI_Barrier(MPI_COMM_WORLD);
            wt2=MPI_Wtime();

            if (my_rank==0)
            {
               printf("Generation of the pseudo-fermion (action no %d):\n",k);

               if ((ap.action!=ACF_TM1)&&(ap.action!=ACF_TM1_EO)&&
                   (ap.action!=ACF_TM1_EO_SDET))
               {
                  printf("Precise solve: ");
                  print_status_act(k,status,wt2-wt1);
               }
            }

            if (icnfg==first)
               set_val(nrm0,phi[i]);
            else
               add_val(nrm0,phi[i]);

            MPI_Barrier(MPI_COMM_WORLD);
            wt1=MPI_Wtime();
            qact1=get_action(k,1,status);
            MPI_Barrier(MPI_COMM_WORLD);
            wt2=MPI_Wtime();

            if (my_rank==0)
            {
               printf("Action no %d:\n",k);
               printf("Precise solve: ");
               print_status_act(k,status,wt2-wt1);
            }

            if (icnfg==first)
               set_val(qact1.q[0],act[i]);
            else
               add_val(qact1.q[0],act[i]);

            MPI_Barrier(MPI_COMM_WORLD);
            wt1=MPI_Wtime();
            set_force(k,1,status);
            MPI_Barrier(MPI_COMM_WORLD);
            wt2=MPI_Wtime();

            if (my_rank==0)
            {
               printf("Force no %d:\n",k);
               printf("Precise solve: ");
               print_status_frc(k,status,wt2-wt1);
            }

            muladd_assign_alg(4*VOLUME,1.0,(*mdfs).frc,wfd[0]);
            assign_alg2alg(4*VOLUME,(*mdfs).frc,(*mdfs).mom);
            nrm1=unorm_alg(4*VOLUME,1,(*mdfs).frc);

            if (icnfg==first)
               set_val(nrm1,frc[i]);
            else
               add_val(nrm1,frc[i]);

            MPI_Barrier(MPI_COMM_WORLD);
            wt1=MPI_Wtime();
            qact2=get_action(k,0,status);
            MPI_Barrier(MPI_COMM_WORLD);
            wt2=MPI_Wtime();

            qact2.q[0]=-qact2.q[0];
            qact2.q[1]=-qact2.q[1];
            add_qflt(qact1.q,qact2.q,qact2.q);
            dev=fabs(qact2.q[0]);
            atot+=qact1.q[0];
            datot+=dev;

            if (my_rank==0)
            {
               printf("Action no %d:\n",k);
               printf("Less precise solve: ");
               print_status_act(k,status,wt2-wt1);
               printf("act1 = %.2e, |dact1| = %.2e, |dact1|/act1 = %.2e\n\n",
                      qact1.q[0],dev,dev/qact1.q[0]);
               fflush(flog);
            }

            if (icnfg==first)
               set_val(dev,dact1[i][0]);
            else
               add_val(dev,dact1[i][0]);

            MPI_Barrier(MPI_COMM_WORLD);
            wt1=MPI_Wtime();
            set_force(k,0,status);
            MPI_Barrier(MPI_COMM_WORLD);
            wt2=MPI_Wtime();

            muladd_assign_alg(4*VOLUME,-1.0,(*mdfs).mom,(*mdfs).frc);
            muladd_assign_alg(4*VOLUME,1.0,(*mdfs).frc,wfd[1]);
            dev=unorm_alg(4*VOLUME,1,(*mdfs).frc);

            if (my_rank==0)
            {
               printf("Force no %d:\n",k);
               printf("Less precise solve: ");
               print_status_frc(k,status,wt2-wt1);
               printf("|frc|_oo = %.2e, |dfrc|_oo = %.2e, "
                      "|dfrc|_oo/|frc|_oo = %.2e\n\n",nrm1,dev,dev/nrm1);
               fflush(flog);
            }

            if (icnfg==first)
               set_val(dev,dfrc[i][0]);
            else
               add_val(dev,dfrc[i][0]);

            if ((ap.action!=ACF_TM1)&&(ap.action!=ACF_TM1_EO)&&
                (ap.action!=ACF_TM1_EO_SDET))
            {
               restore_ranlux();

               MPI_Barrier(MPI_COMM_WORLD);
               wt1=MPI_Wtime();
               qact2=set_pf(k,0,status,&dev);
               MPI_Barrier(MPI_COMM_WORLD);
               wt2=MPI_Wtime();

               if (my_rank==0)
               {
                  printf("Generation of the pseudo-fermion (action no %d):\n",k);
                  printf("Less precise solve: ");
                  print_status_act(k,status,wt2-wt1);
                  printf("|phi|_oo = %.2e, |dphi|_oo = %.2e, "
                         "|dphi|_oo/|phi|_oo = %.2e\n",nrm0,dev,dev/nrm0);
                  fflush(flog);
               }

               if (icnfg==first)
                  set_val(dev,dphi[i]);
               else
                  add_val(dev,dphi[i]);

               qact2.q[0]=-qact2.q[0];
               qact2.q[1]=-qact2.q[1];
               add_qflt(qact0.q,qact2.q,qact2.q);
               dev=fabs(qact2.q[0]);

               if (my_rank==0)
               {
                  printf("act0 = %.2e, |dact0| = %.2e, |dact0|/act0 = %.2e\n\n",
                         qact0.q[0],dev,dev/qact0.q[0]);
                  fflush(flog);
               }

               if (icnfg==first)
                  set_val(dev,dact0[i]);
               else
                  add_val(dev,dact0[i]);

               MPI_Barrier(MPI_COMM_WORLD);
               wt1=MPI_Wtime();
               qact2=get_action(k,1,status);
               MPI_Barrier(MPI_COMM_WORLD);
               wt2=MPI_Wtime();

               qact2.q[0]=-qact2.q[0];
               qact2.q[1]=-qact2.q[1];
               add_qflt(qact1.q,qact2.q,qact2.q);
               dev=fabs(qact2.q[0]);

               if (my_rank==0)
               {
                  printf("Action no %d:\n",k);
                  printf("Precise solve, less precise phi: ");
                  print_status_act(k,status,wt2-wt1);
                  printf("act1 = %.2e, |dact1| = %.2e, |dact1|/act1 = %.2e\n\n",
                         qact1.q[0],dev,dev/qact1.q[0]);
                  fflush(flog);
               }

               if (icnfg==first)
                  set_val(dev,dact1[i][1]);
               else
                  add_val(dev,dact1[i][1]);

               MPI_Barrier(MPI_COMM_WORLD);
               wt1=MPI_Wtime();
               set_force(k,1,status);
               MPI_Barrier(MPI_COMM_WORLD);
               wt2=MPI_Wtime();

               muladd_assign_alg(4*VOLUME,-1.0,(*mdfs).mom,(*mdfs).frc);
               dev=unorm_alg(4*VOLUME,1,(*mdfs).frc);

               if (my_rank==0)
               {
                  printf("Force no %d:\n",k);
                  printf("Precise solve, less precise phi: ");
                  print_status_frc(k,status,wt2-wt1);
                  printf("|frc|_oo = %.2e, |dfrc|_oo = %.2e, "
                         "|dfrc|_oo/|frc|_oo = %.2e\n\n",nrm1,dev,dev/nrm1);
                  fflush(flog);
               }

               if (icnfg==first)
                  set_val(dev,dfrc[i][1]);
               else
                  add_val(dev,dfrc[i][1]);
            }
         }
      }

      qact0=scalar_prod_alg(4*VOLUME,1,wfd[0],wfd[1]);
      dev=fabs(qact0.q[0]);

      if (icnfg==first)
      {
         set_val(atot,act[nact]);
         set_val(datot,dact1[nact][0]);
         set_val(dev,dfrc[nact][0]);
      }
      else
      {
         add_val(atot,act[nact]);
         add_val(datot,dact1[nact][0]);
         add_val(dev,dfrc[nact][0]);
      }
   }

   if (my_rank==0)
   {
      printf("Test summary\n");
      printf("------------\n\n");

      if (last==first)
         ncnfg=1;
      else
         ncnfg=(last-first)/step+1;

      printf("Processed %d configurations\n",ncnfg);
      printf("The measured minimal, maximal and average values are:\n\n");

      for (i=0;i<nact;i++)
      {
         k=iact[i];
         ap=action_parms(k);

         if (ap.action!=ACG)
         {
            printf("Pseudo-fermion (action %2d): %.2e, %.2e, %.2e; ",
                   k,phi[i][0],phi[i][1],phi[i][2]/(double)(ncnfg));

            if ((ap.action!=ACF_TM1)&&(ap.action!=ACF_TM1_EO)&&
                (ap.action!=ACF_TM1_EO_SDET))
            {
               printf("Deviations: %.1e, %.1e, %.1e (field); ",
                      dphi[i][0],dphi[i][1],dphi[i][2]/(double)(ncnfg));
               printf("%.1e, %.1e, %.1e (action)\n",
                      dact0[i][0],dact0[i][1],dact0[i][2]/(double)(ncnfg));
            }
            else
               printf("\n");
         }
      }

      printf("\n");

      for (i=0;i<nact;i++)
      {
         k=iact[i];
         ap=action_parms(k);

         if (ap.action!=ACG)
         {
            printf("Action %2d: %.2e, %.2e, %.2e; ",
                   k,act[i][0],act[i][1],act[i][2]/(double)(ncnfg));
            printf("Deviations: %.1e, %.1e, %.1e (solver); ",dact1[i][0][0],
                   dact1[i][0][1],dact1[i][0][2]/(double)(ncnfg));

            if ((ap.action!=ACF_TM1)&&(ap.action!=ACF_TM1_EO)&&
                (ap.action!=ACF_TM1_EO_SDET))
            {
               printf("%.1e, %.1e, %.1e (phi)\n",dact1[i][1][0],dact1[i][1][1],
                      dact1[i][1][2]/(double)(ncnfg));
            }
            else
               printf("\n");
         }
      }

      printf("\n");

      for (i=0;i<nact;i++)
      {
         k=iact[i];
         ap=action_parms(k);

         if (ap.action!=ACG)
         {
            printf("Force %2d: %.2e, %.2e, %.2e; ",
                   k,frc[i][0],frc[i][1],frc[i][2]/(double)(ncnfg));
            printf("Deviation: %.1e, %.1e, %.1e (solver); ",
                   dfrc[i][0][0],dfrc[i][0][1],dfrc[i][0][2]/(double)(ncnfg));

            if ((ap.action!=ACF_TM1)&&(ap.action!=ACF_TM1_EO)&&
                (ap.action!=ACF_TM1_EO_SDET))
            {
               printf("Deviation: %.1e, %.1e, %.1e (phi)\n",dfrc[i][1][0],
                      dfrc[i][1][1],dfrc[i][1][2]/(double)(ncnfg));
            }
            else
               printf("\n");
         }
      }

      printf("\n");
      printf("Total action: %.12e, %.12e, %.12e\n",
             act[nact][0],act[nact][1],act[nact][2]/(double)(ncnfg));
      printf("Total absolute deviation: %.2e, %.2e, %.2e\n",
             dact1[nact][0][0],dact1[nact][0][1],
             dact1[nact][0][2]/(double)(ncnfg));
      printf("Accumulated field-induced deviation |(frc,dfrc)|: "
             "%.2e, %.2e, %.2e\n\n",
             dfrc[nact][0][0],dfrc[nact][0][1],
             dfrc[nact][0][2]/(double)(ncnfg));
      fclose(flog);
   }

   release_wfd();

   MPI_Finalize();
   exit(0);
}
