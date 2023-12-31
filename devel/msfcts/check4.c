
/*******************************************************************************
*
* File check4.c
*
* Copyright (C) 2017, 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Computation of the topological susceptibility.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "mpi.h"
#include "flags.h"
#include "random.h"
#include "su3fcts.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "archive.h"
#include "tcharge.h"
#include "wflow.h"
#include "msfcts.h"
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

static int my_rank,first,last,step,rule;
static int ntm,i3d,rmin,rmax,dmax;
static double eps,*tms;
static double *ws,**av,**ava,**avs,*Q;
static double ***sm,***sma,***sms;
static double **cf,**cfa,**cfs;
static double **f;
static complex_dble **rf;
static u3_alg_dble **ft;
static char line[NAME_SIZE],nbase[NAME_SIZE];
static char cnfg_file[NAME_SIZE],end_file[NAME_SIZE];
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

      error_root((strlen(line)!=1)||(type==0x0),1,"read_iodat [check4.c]",
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
                 "read_iodat [check4.c]","Improper configuration range");
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


static void read_wflow_parms(void)
{
   int i,ie;

   if (my_rank==0)
   {
      find_section("Wilson flow");
      ntm=count_tokens("tm");
      read_line("eps","%lf\n",&eps);
      read_line("rule","%d",&rule);
   }

   error_root(ntm==0,1,"read_wflow_parms [check4.c]",
              "No flow times specified");
   error_root(eps<=0.0,1,"read_wflow_parms [check4.c]",
              "Step size eps must be positive");
   error_root((rule<0)||(rule>3),1,"read_wflow_parms [check4.c]",
              "rule must be 1,2 or 3");

   MPI_Bcast(&ntm,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&rule,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&eps,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

   tms=malloc(ntm*sizeof(*tms));
   error(tms==NULL,1,"read_wflow_parms [check4.c]",
         "Unable to allocate times array");

   if (my_rank==0)
   {
      read_dprms("tm",ntm,tms);
      ie=(tms[0]<0.0);

      for (i=1;i<ntm;i++)
         ie|=(tms[i]<=tms[i-1]);

      error_root(ie!=0,1,"read_wflow_parms [check4.c]","Flow times must be "
                 "non-negative and monotonically increasing");
   }

   MPI_Bcast(tms,ntm,MPI_DOUBLE,0,MPI_COMM_WORLD);
}


static void read_obs_parms(void)
{
   int n;
   double zero[3];

   if (my_rank==0)
   {
      find_section("Correlation function");
      read_line("i3d","%d",&i3d);
      read_line("radius","%d %d",&rmin,&rmax);
      read_line("dmax","%d",&dmax);
   }

   error_root((i3d<0)||(i3d>1),1,"read_obs_parms [check4.c]",
              "Dimension flag i3d must be 0 or 1");
   error_root((rmin<0)||(rmax<rmin),1,"read_obs_parms [check4.c]",
              "Improper radius range");
   error_root(dmax<0,1,"read_obs_parms [check4.c]",
              "Maximal distance dmax must be non-negative");

   MPI_Bcast(&i3d,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&rmin,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&rmax,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&dmax,1,MPI_INT,0,MPI_COMM_WORLD);

   n=0;
   zero[0]=0.0;
   zero[1]=0.0;
   zero[2]=0.0;
   (void)(set_hmc_parms(1,&n,0,0,NULL,1,1.0));
   (void)(set_bc_parms(3,1.0,1.0,1.0,1.0,zero,zero,zero));
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
            "check_files [check4.c]","cnfg_dir name is too long");
      check_dir_root(cnfg_dir);

      sprintf(line,"%s/%sn%d",cnfg_dir,nbase,first);
      lat_sizes(line,ns);
      error_root((ns[0]!=N0)||(ns[1]!=N1)||(ns[2]!=N2)||(ns[3]!=N3),1,
                 "check_files [check4.c]","Lattice size mismatch");
   }
   else if (type&0x2)
   {
      error(name_size("%s/0/0/%sn%d_b0",cnfg_dir,nbase,first)>=NAME_SIZE,1,
            "check_files [check4.c]","cnfg_dir name is too long");
      sprintf(line,"%s/0/0",cnfg_dir);
      if ((cpr[0]==0)&&(cpr[1]==0)&&(cpr[2]==0)&&(cpr[3]==0))
         check_dir(line);

      sprintf(line,"%s/0/0/%sn%d_b0",cnfg_dir,nbase,first);
      blk_sizes(line,ns,bs);
      error_root((ns[0]!=N0)||(ns[1]!=N1)||(ns[2]!=N2)||(ns[3]!=N3),1,
                 "check_files [check4.c]","Lattice size mismatch");

      ib=blk_index(ns,bs,&nb);
      nion=iodat.nio_nodes;
      n=nb/nion;
      error_root(nb%nion!=0,1,"check_files [check4.c]",
                 "Number of blocks is not a multiple of nio_nodes");
      error(name_size("%s/%d/%d/%sn%d_b%d",cnfg_dir,nion-1,n-1,
                      nbase,last,nb-1)>=NAME_SIZE,1,
            "check_files [check4.c]","cnfg_dir name is too long");
      sprintf(line,"%s/%d/%d",cnfg_dir,ib/n,ib%n);
      strcpy(cnfg_dir,line);
      if (((cpr[0]*L0)%bs[0]==0)&&((cpr[1]*L1)%bs[1]==0)&&
          ((cpr[2]*L2)%bs[2]==0)&&((cpr[3]*L3)%bs[3]==0))
         check_dir(cnfg_dir);

      iodat.nb=nb;
      iodat.ib=ib;
   }
   else if (type&0x4)
   {
      nion=iodat.nio_nodes;
      n=NPROC/nion;
      error_root(NPROC%nion!=0,1,"check_files [check4.c]",
                 "Number of processes is not a multiple of nio_nodes");
      error(name_size("%s/%d/%d/%sn%d_%d",cnfg_dir,nion-1,n-1,
                      nbase,last,NPROC-1)>=NAME_SIZE,1,
            "check_files [check4.c]","cnfg_dir name is too long");
      sprintf(line,"%s/%d/%d",cnfg_dir,my_rank/n,my_rank%n);
      strcpy(cnfg_dir,line);
      check_dir(cnfg_dir);
   }
}


static void set_fld(int icnfg)
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
      printf("Gauge field read from disk in %.2e sec\n\n",
             wt2-wt1);
      fflush(flog);
   }
}


static void alloc_flds(int nf,int nrf)
{
   int i;
   double *f1,**f2;
   complex_dble *rf1,**rf2;

   f1=malloc(nf*VOLUME*sizeof(*f1));
   f2=malloc(nf*sizeof(*f2));

   rf1=malloc(nrf*VOLUME*sizeof(*rf1));
   rf2=malloc(nrf*sizeof(*rf2));

   error((f1==NULL)||(f2==NULL)||(rf1==NULL)||(rf2==NULL),1,
         "alloc_flds [check1.c]","Unable to allocate fields");

   f=f2;
   rf=rf2;

   for (i=0;i<nf;i++)
   {
      f2[0]=f1;
      f1+=VOLUME;
      f2+=1;
   }

   for (i=0;i<nrf;i++)
   {
      rf2[0]=rf1;
      rf1+=VOLUME;
      rf2+=1;
   }
}


static void alloc_av(void)
{
   int i,rmx,dmx;
   double *p1,**p2;

   rmx=rmax-rmin+1;
   dmx=dmax+1;

   p2=malloc(3*ntm*sizeof(*p2));
   p1=malloc(3*ntm*rmx*sizeof(*p1));
   error((p1==NULL)||(p2==NULL),1,"alloc_av [check4.c]",
         "Unable to allocate auxiliary arrays");

   av=p2;
   ava=av+ntm;
   avs=ava+ntm;

   for (i=0;i<(3*ntm*rmx);i++)
      p1[i]=0.0;

   for (i=0;i<(3*ntm);i++)
   {
      p2[i]=p1;
      p1+=rmx;
   }

   p1=malloc((ntm+dmx)*sizeof(*p1));
   error(p1==NULL,1,"alloc_av [check4.c]",
         "Unable to allocate auxiliary arrays");

   Q=p1;
   ws=Q+ntm;
}


static void alloc_sm(void)
{
   int i,rmx,dmx;
   double *p1,**p2,***p3;

   rmx=rmax-rmin+1;
   dmx=dmax+1;

   p3=malloc(3*ntm*sizeof(*p3));
   p2=malloc(3*ntm*rmx*sizeof(*p2));
   p1=malloc(3*ntm*rmx*dmx*sizeof(*p1));

   error((p1==NULL)||(p2==NULL)||(p3==NULL),1,"alloc_sm [check4.c]",
         "Unable to allocate auxiliary arrays");

   sm=p3;
   sma=sm+ntm;
   sms=sma+ntm;

   for (i=0;i<(3*ntm*rmx*dmx);i++)
      p1[i]=0.0;

   for (i=0;i<(3*ntm*rmx);i++)
   {
      p2[i]=p1;
      p1+=dmx;
   }

   for (i=0;i<(3*ntm);i++)
   {
      p3[i]=p2;
      p2+=rmx;
   }

   p2=malloc(3*ntm*sizeof(*p2));
   p1=malloc(3*ntm*dmx*sizeof(*p1));

   error((p1==NULL)||(p2==NULL),1,"alloc_sm [check4.c]",
         "Unable to allocate auxiliary arrays");

   cf=p2;
   cfa=cf+ntm;
   cfs=cfa+ntm;

   for (i=0;i<(3*ntm*dmx);i++)
      p1[i]=0.0;

   for (i=0;i<(3*ntm);i++)
   {
      p2[i]=p1;
      p1+=dmx;
   }
}


static double prodXY(u3_alg_dble *X,u3_alg_dble *Y)
{
   double p;

   p=(-2.0/3.0)*((*X).c1+(*X).c2+(*X).c3)*((*Y).c1+(*Y).c2+(*Y).c3)+
      2.0*((*X).c1*(*Y).c1+(*X).c2*(*Y).c2+(*X).c3*(*Y).c3)+
      4.0*((*X).c4*(*Y).c4+(*X).c5*(*Y).c5+(*X).c6*(*Y).c6+
           (*X).c7*(*Y).c7+(*X).c8*(*Y).c8+(*X).c9*(*Y).c9);

   return p;
}


static double density(int ix)
{
   double dn;

   dn=prodXY(ft[0]+ix,ft[3]+ix)+
      prodXY(ft[1]+ix,ft[4]+ix)+
      prodXY(ft[2]+ix,ft[5]+ix);

   return dn;
}


static double set_f0(double t)
{
   int ix;
   double pi,r,s;

   ft=ftensor();
   pi=4.0*atan(1.0);
   r=1.0/(8.0*pi*pi);

   for (ix=0;ix<VOLUME;ix++)
      f[0][ix]=r*density(ix);

   s=center_fld(f[0]);

   return s*(double)(N0*N1)*(double)(N2*N3);
}


static void set_cf(int i)
{
   int ip,ix,x[4];
   int ifc,j;

   x[0]=0;
   x[1]=0;
   x[2]=0;
   x[3]=0;

   for (j=0;j<=dmax;j++)
   {
      ws[j]=0.0;

      for (ifc=0;ifc<8;ifc++)
      {
         if (ifc&0x1)
            x[ifc/2]=j;
         else
            x[ifc/2]=-j;

         ipt_global(x,&ip,&ix);

         if (my_rank==ip)
            ws[j]+=0.125*f[1][ix];

         x[ifc/2]=0;
      }
   }

   if (NPROC>1)
   {
      MPI_Reduce(ws,cf[i],dmax+1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
      MPI_Bcast(cf[i],dmax+1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   }
   else
   {
      for (j=0;j<=dmax;j++)
         cf[i][j]=ws[j];
   }
}


static void acc_sm(void)
{
   int i,j,k;
   double r,rv;

   rv=1.0/((double)(N0*N1)*(double)(N2*N3));

   for (i=0;i<ntm;i++)
   {
      for (j=0;j<=(rmax-rmin);j++)
      {
         ava[i][j]+=av[i][j];
         avs[i][j]+=av[i][j]*av[i][j];

         for (k=0;k<=dmax;k++)
         {
            sm[i][j][k]*=rv;
            r=sm[i][j][k];
            sma[i][j][k]+=r;
            sms[i][j][k]+=r*r;
         }
      }

      for (j=0;j<=dmax;j++)
      {
         r=cf[i][j];
         cfa[i][j]+=r;
         cfs[i][j]+=r*r;
      }
   }
}


static int check_end(void)
{
   int iend;
   FILE *end;

   if (my_rank==0)
   {
      iend=0;
      end=fopen(end_file,"r");

      if (end!=NULL)
      {
         fclose(end);
         remove(end_file);
         iend=1;
         printf("End flag set, run stopped\n\n");
      }
   }

   MPI_Bcast(&iend,1,MPI_INT,0,MPI_COMM_WORLD);

   return iend;
}


int main(int argc,char *argv[])
{
   int icnfg,ncnfg;
   int n,i,j,k;
   double dt,rv,r;
   double wt1,wt2,wtavg;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check4.log","w",stdout);
      fin=freopen("check4.in","r",stdin);

      printf("\n");
      printf("Computation of the topological susceptibility\n");
      printf("---------------------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);
   }

   read_iodat();
   read_wflow_parms();
   read_obs_parms();

   if (my_rank==0)
      fclose(fin);

   check_machine();
   sprintf(end_file,"check4.end");

   if (my_rank==0)
   {
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

      printf("Wilson-flow parameters:\n");
      printf("tm = ");

      for (i=0;i<ntm;i++)
      {
         n=fdigits(tms[i]);

         if (i==0)
            printf("%.*f",IMAX(n,1),tms[i]);
         else
            printf(",%.*f",IMAX(n,1),tms[i]);
      }

      printf("\n");
      n=fdigits(eps);
      printf("eps = %.*f\n",IMAX(n,1),eps);

      if (rule==1)
         printf("Using the Euler integrator\n");
      else if (rule==2)
         printf("Using the 2nd order RK integrator\n");
      else
         printf("Using the 3rd order RK integrator\n");

      printf("\n");
      printf("i3d = %d\n",i3d);
      printf("radius = %d,..,%d\n",rmin,rmax);
      printf("dmax = %d\n\n",dmax);

      printf("Configurations %sn%d -> %sn%d in steps of %d\n\n",
             nbase,first,nbase,last,step);
      fflush(flog);
   }

   geometry();
   start_ranlux(0,12345);
   check_files();

   if (rule>1)
      alloc_wfd(1);
   alloc_flds(2,2);
   alloc_av();
   alloc_sm();

   wtavg=0.0;
   rv=1.0/((double)(N0*N1)*(double)(N2*N3));

   for (icnfg=first;icnfg<=last;icnfg+=step)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();

      if (my_rank==0)
      {
         printf("Configuration no %d\n\n",icnfg);
         fflush(flog);
      }

      set_fld(icnfg);

      for (i=0;i<ntm;i++)
      {
         if (i==0)
         {
            n=(int)(tms[0]/eps);
            dt=tms[0]-(double)(n)*eps;
         }
         else
         {
            n=(int)((tms[i]-tms[i-1])/eps);
            dt=tms[i]-tms[i-1]-(double)(n)*eps;
         }

         if (rule==1)
         {
            if (n>0)
               fwd_euler(n,eps);
            if (dt>0.0)
               fwd_euler(1,dt);
         }
         else if (rule==2)
         {
            if (n>0)
               fwd_rk2(n,eps);
            if (dt>0.0)
               fwd_rk2(1,dt);
         }
         else
         {
            if (n>0)
               fwd_rk3(n,eps);
            if (dt>0.0)
               fwd_rk3(1,dt);
         }

         Q[i]=set_f0(tms[i]);
         convolute_flds(NULL,f[0],f[0],rf[0],rf[0],f[1]);
         set_cf(i);

         for (j=0;j<=(rmax-rmin);j++)
         {
            if (i3d)
               sphere3d_fld(rmin+j,f[1]);
            else
               sphere_fld(rmin+j,f[1]);

            convolute_flds(NULL,f[0],f[1],rf[0],rf[1],f[1]);
            mul_flds(f[1],f[0]);
            av[i][j]=center_fld(f[1]);
            convolute_flds(NULL,f[1],f[1],rf[0],rf[0],f[1]);
            mulr_fld(rv,f[1]);

            if (i3d)
               sphere3d_sum(dmax,f[1],sm[i][j]);
            else
               sphere_sum(dmax,f[1],sm[i][j]);
         }
      }

      acc_sm();
      MPI_Barrier(MPI_COMM_WORLD);
      wt2=MPI_Wtime();
      wtavg+=(wt2-wt1);

      if (my_rank==0)
      {
         for (i=0;i<ntm;i++)
         {
            n=fdigits(tms[i]);
            printf("t = %.*f, Q = %.6e\n",IMAX(n,1),tms[i],Q[i]);

            for (j=0;j<=dmax;j++)
               printf(" cor[%3d]  ",j);
            printf("\n");
            for (j=0;j<=dmax;j++)
               printf("% .2e  ",cf[i][j]);
            printf("\n");

            for (j=0;j<=(rmax-rmin);j++)
            {
               printf(" r = %d, chi_t = %.3e\n",rmin+j,av[i][j]);

               for (k=0;k<=dmax;k++)
                  printf(" var[%3d]  ",k);
               printf("\n");
               for (k=0;k<=dmax;k++)
                  printf("% .2e  ",sm[i][j][k]);
               printf("\n");
            }

            printf("\n");
         }

         printf("Configuration no %d fully processed in %.2e sec ",
                icnfg,wt2-wt1);
         printf("(average = %.2e sec)\n\n",
                wtavg/(double)((icnfg-first)/step+1));
         fflush(flog);
      }

      if (check_end())
         break;
   }

   if (my_rank==0)
   {
      ncnfg=(last-first)/step+1;
      r=1.0/(double)(ncnfg);

      for (i=0;i<ntm;i++)
      {
         for (j=0;j<=(rmax-rmin);j++)
         {
            ava[i][j]*=r;
            avs[i][j]*=r;
            avs[i][j]-=ava[i][j]*ava[i][j];
            avs[i][j]=sqrt(fabs(avs[i][j]));

            for (k=0;k<=dmax;k++)
            {
               sma[i][j][k]*=r;
               sms[i][j][k]*=r;
               sms[i][j][k]-=sma[i][j][k]*sma[i][j][k];
               sms[i][j][k]=sqrt(fabs(sms[i][j][k]));
            }
         }

         for (j=0;j<=dmax;j++)
         {
            cfa[i][j]*=r;
            cfs[i][j]*=r;
            cfs[i][j]-=cfa[i][j]*cfa[i][j];
            cfs[i][j]=sqrt(fabs(cfs[i][j]));
         }
      }

      printf("\n");
      printf("Test summary\n");
      printf("------------\n\n");

      printf("Processed %d configurations\n\n",ncnfg);

      for (i=0;i<ntm;i++)
      {
         n=fdigits(tms[i]);
         printf("t = %.*f\n",IMAX(n,1),tms[i]);

         for (j=0;j<=dmax;j++)
            printf(" cor[%3d]            ",j);
         printf("\n");
         for (j=0;j<=dmax;j++)
            printf("% .2e (%.1e)  ",cfa[i][j],cfs[i][j]);
         printf("\n");

         for (j=0;j<=(rmax-rmin);j++)
         {
            printf(" r = %d, chi_t = %.3e (%.1e)\n",
                   rmin+j,ava[i][j],avs[i][j]);

            for (k=0;k<=dmax;k++)
               printf(" var[%3d]            ",k);
            printf("\n");
            for (k=0;k<=dmax;k++)
               printf("% .2e (%.1e)  ",sma[i][j][k],sms[i][j][k]);
            printf("\n");
         }

         printf("\n");
      }

      printf("\n");
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
