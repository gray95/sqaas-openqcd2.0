/*******************************************************************************
*
* File check3.c
*
* Copyright (C) 2015 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of the program dft_shuf().
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "utils.h"
#include "random.h"
#include "dft.h"

static int ns=0;
static complex_dble *f[5];


static void alloc_arrays(int v)
{
   int i;

   if (v>ns)
   {
      if (ns!=0)
         afree(f[0]);

      f[0]=amalloc(5*v*sizeof(*f[0]),4);
      error(f[0]==NULL,1,"alloc_arrays [check3.c]",
            "Unable to allocate field arrays");
      ns=v;
   }

   for (i=1;i<5;i++)
      f[i]=f[i-1]+v;

   gauss_dble((double*)(f[0]),10*v);
}


static void cmpf(int mu,int csize,int *l)
{
   int nu,L[4],ie;
   int x0,x1,x2,x3,s,ix,iy;

   for (nu=0;nu<4;nu++)
      L[nu]=l[(mu+nu)%4];

   ie=0;

   for (x0=0;x0<L[0];x0++)
   {
      for (x1=0;x1<L[1];x1++)
      {
         for (x2=0;x2<L[2];x2++)
         {
            for (x3=0;x3<L[3];x3++)
            {
               ix=x3+L[3]*x2+L[2]*L[3]*x1+L[1]*L[2]*L[3]*x0;
               iy=x0+L[0]*x3+L[3]*L[0]*x2+L[2]*L[3]*L[0]*x1;

               for (s=0;s<csize;s++)
               {
                  ie|=(f[mu+1][s+csize*iy].re!=f[mu][s+csize*ix].re);
                  ie|=(f[mu+1][s+csize*iy].im!=f[mu][s+csize*ix].im);

                  if (ie!=0)
                     error(1,1,"cmpf [check3.c]",
                           "Mismatch of field values");
               }
            }
         }
      }
   }
}


int main(void)
{
   int csize,v,mu,l[4];

   printf("\n");
   printf("Check of the program dft_shuf()\n");
   printf("-------------------------------\n\n");

   while(1)
   {
      printf("Select local lattice sizes l0,l1,l2,l3: ");
      (void)scanf("%d %d %d %d",l,l+1,l+2,l+3);
      error((l[0]<1)||(l[1]<1)||(l[2]<1)||(l[3]<1),1,"main [check3.c]",
            "Improper choice of the lattice sizes");

      printf("Select data size: ");
      (void)scanf("%d",&csize);
      error(csize<1,1,"main [check3.c]",
            "Improper choice of the data size");
      printf("\n");

      v=l[0]*l[1]*l[2]*l[3];
      alloc_arrays(csize*v);

      for (mu=0;mu<4;mu++)
      {
         printf("mu = %d, l[mu] = %d\n",mu,l[mu]);
         dft_shuf(l[mu],v/l[mu],csize,f[mu],f[mu+1]);
         cmpf(mu,csize,l);
      }

      printf("No errors discovered\n\n");
   }

   exit(0);
}
