
/*******************************************************************************
*
* File valg_dble.c
*
* Copyright (C) 2007-2016, 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Generic linear algebra routines for double-precision complex fields.
*
*   complex_qflt vprod_dble(int n,int icom,complex_dble *v,complex_dble *w)
*     Returns the scalar product of the n-vectors v and w.
*
*   qflt vnorm_square_dble(int n,int icom,complex_dble *v)
*     Returns the square of the norm of the n-vector v.
*
*   void mulc_vadd_dble(int n,complex_dble *v,complex_dble *w,complex_dble z)
*     Replaces the n-vector v by v+z*w.
*
*   void vproject_dble(int n,int icom,complex_dble *v,complex_dble *w)
*     Replaces the n-vector v by v-(w,v)*w.
*
*   void vscale_dble(int n,double r,complex_dble *v)
*     Replaces the n-vector v by r*v.
*
*   double vnormalize_dble(int n,int icom,complex_dble *v)
*     Normalizes the n-vector v to unity and returns the norm of the
*     input vector.
*
*   void vrotate_dble(int n,int nv,complex_dble **pv,complex_dble *a)
*     Replaces the n-vectors vk=pv[k], k=0,..,nv-1, by the linear
*     combinations sum_{j=0}^{nv-1} vj*a[n*j+k].
*
* The quadruple-precision types qflt and complex_qflt are defined in su3.h.
* See doc/qsum.pdf for further explanations.
*
* All these programs operate on complex n-vectors whose base addresses are
* passed through the arguments. The length n of the arrays is specified by
* the parameter n. Scalar products are globally summed if the parameter
* icom is equal to 1. In this case the calculated values are guaranteed to
* be exactly the same on all processes.
*
* The programs perform no communications except in the case of the programs
* with communication flag icom set to 1.
*
*******************************************************************************/

#define VALG_DBLE_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "utils.h"
#include "linalg.h"
#include "vflds.h"
#include "global.h"

static int nrot=0;
static double smx ALIGNED8;
static complex_dble smz ALIGNED16;
static complex_dble *psi;


static void alloc_wrotate(int n)
{
   if (nrot>0)
      afree(psi);

   psi=amalloc(n*sizeof(*psi),ALIGN);
   error_loc(psi==NULL,1,"alloc_wrotate [valg_dble.c]",
             "Unable to allocate workspace");
   set_vd2zero(n,psi);
   nrot=n;
}


complex_qflt vprod_dble(int n,int icom,complex_dble *v,complex_dble *w)
{
   double *qsm[2];
   complex_qflt cqsm;
   complex_dble *vm,*vb;

   qsm[0]=cqsm.re.q;
   qsm[1]=cqsm.im.q;

   qsm[0][0]=0.0;
   qsm[0][1]=0.0;
   qsm[1][0]=0.0;
   qsm[1][1]=0.0;
   vm=v+n;

   for (vb=v;vb<vm;)
   {
      vb+=32;
      if (vb>vm)
         vb=vm;
      smz.re=0.0;
      smz.im=0.0;

      for (;v<vb;v++)
      {
         smz.re+=((*v).re*(*w).re+(*v).im*(*w).im);
         smz.im+=((*v).re*(*w).im-(*v).im*(*w).re);
         w+=1;
      }

      acc_qflt(smz.re,qsm[0]);
      acc_qflt(smz.im,qsm[1]);
   }

   if ((icom==1)&&(NPROC>1))
      global_qsum(2,qsm,qsm);

   return cqsm;
}


qflt vnorm_square_dble(int n,int icom,complex_dble *v)
{
   double *qsm[1];
   qflt rqsm;
   complex_dble *vm,*vb;

   qsm[0]=rqsm.q;

   qsm[0][0]=0.0;
   qsm[0][1]=0.0;
   vm=v+n;

   for (vb=v;vb<vm;)
   {
      vb+=32;
      if (vb>vm)
         vb=vm;
      smx=0.0;

      for (;v<vb;v++)
         smx+=((*v).re*(*v).re+(*v).im*(*v).im);

      acc_qflt(smx,qsm[0]);
   }

   if ((icom==1)&&(NPROC>1))
      global_qsum(1,qsm,qsm);

   return rqsm;
}


void mulc_vadd_dble(int n,complex_dble *v,complex_dble *w,complex_dble z)
{
   complex_dble *vm;

   vm=v+n;

   for (;v<vm;v++)
   {
      (*v).re+=(z.re*(*w).re-z.im*(*w).im);
      (*v).im+=(z.re*(*w).im+z.im*(*w).re);
      w+=1;
   }
}


void vproject_dble(int n,int icom,complex_dble *v,complex_dble *w)
{
   complex_dble z;
   complex_qflt qz;

   qz=vprod_dble(n,icom,w,v);
   z.re=-qz.re.q[0];
   z.im=-qz.im.q[0];
   mulc_vadd_dble(n,v,w,z);
}


void vscale_dble(int n,double r,complex_dble *v)
{
   complex_dble *vm;

   vm=v+n;

   for (;v<vm;v++)
   {
      (*v).re*=r;
      (*v).im*=r;
   }
}


double vnormalize_dble(int n,int icom,complex_dble *v)
{
   double r;
   qflt qr;

   qr=vnorm_square_dble(n,icom,v);
   r=sqrt(qr.q[0]);

   if (r!=0.0)
      vscale_dble(n,1.0/r,v);
   else
      error_loc(1,1,"vnormalize_dble [valg_dble.c]",
                "Vector field has vanishing norm");

   return r;
}


void vrotate_dble(int n,int nv,complex_dble **pv,complex_dble *a)
{
   int i,k,j;
   complex_dble s,*z,*vj;

   if (nv>nrot)
      alloc_wrotate(nv);

   for (i=0;i<n;i++)
   {
      for (k=0;k<nv;k++)
      {
         z=a+k;
         s.re=0.0;
         s.im=0.0;

         for (j=0;j<nv;j++)
         {
            vj=pv[j]+i;
            s.re+=((*z).re*(*vj).re-(*z).im*(*vj).im);
            s.im+=((*z).re*(*vj).im+(*z).im*(*vj).re);
            z+=nv;
         }

         psi[k].re=s.re;
         psi[k].im=s.im;
      }

      for (k=0;k<nv;k++)
      {
         pv[k][i].re=psi[k].re;
         pv[k][i].im=psi[k].im;
      }
   }
}
