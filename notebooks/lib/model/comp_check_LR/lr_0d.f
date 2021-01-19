c  This program integrates the L-R I model 
C     in a 2D strip geometry
c this utilizes a look-up table for all currents
c and the channels are calculated by using the explicity formula
c cross gradient code a la XU and Guevara

      program br  
      implicit real*8(a-h,o-z)
      character*80 filename
      common/pts/npts
      common/space/dx
      common/block1/xinf1,xtau1
      common/blockm/xinfm,xtaum
      common/blockh/xinfh,xtauh
      common/blockj/xinfj,xtauj
      common/blockd/xinfd,xtaud
      common/blockf/xinff,xtauf
      common/xtblock/xttab,xktab
      common/blocke/e1,ej,em,ed,eh,ef
      parameter (ndimtab=2500)
      real*8 xinf1(ndimtab),xtau1(ndimtab)
      real*8 xinfm(ndimtab),xtaum(ndimtab)
      real*8 xinfh(ndimtab),xtauh(ndimtab)
      real*8 xinfj(ndimtab),xtauj(ndimtab)
      real*8 xinfd(ndimtab),xtaud(ndimtab)
      real*8 xinff(ndimtab),xtauf(ndimtab)
      real*8 xttab(ndimtab)
      real*8 xktab(ndimtab)
      real*8 e1(ndimtab),ej(ndimtab)
      real*8 em(ndimtab),ed(ndimtab)
      real*8 eh(ndimtab),ef(ndimtab)

        rtoverf=0.02650
        xna0=140.
        xnai=18.
        vna=1000.*rtoverf*log(xna0/xnai)
        vna=54.40
c       gna=23.
c modified to:
        gna=16.
c gca modified to:
        gca=0.052

c init. conditions
      open(unit=40,file='init_cond_rest',status='unknown')
      read(40,*)u,ca,x1,xm,xh,xj,xd,xf

C-------------------------- I. C. ----------------------------
c     open(unit=21,file='in_cont',status='unknown')
c     read(21,*)tmax,tmod
c     read(21,*)dt
 
      tmax=500.
      tmod=1.
      dt=0.1

      niter=nint(tmax/dt)
      imod=nint(tmod/dt)

      open(unit=40,file='table_qu_non',status='unknown')
      do i=1,ndimtab
       read(40,*)xinf1(i),xtau1(i),xinfm(i),xtaum(i),xinfh(i),xtauh(i),
     1           xinfj(i),xtauj(i),xinfd(i),xtaud(i),xinff(i),xtauf(i),
     1           xttab(i),xktab(i),e1(i),em(i),eh(i),ej(i),ed(i),ef(i)
      enddo

      xstim=10.
      tstim=5.
      nstim=nint(tstim/dt)

      t=0.

      do iter=1,niter

        t=t+dt

        vca=-82.3d0-13.0287d0*dlog(ca)
        xica=gca*xd*xf*(u-vca)
        xna=(gna*xm**3*xh*xj)*(u-vna)

        cao=-1.d-7*xica +.07d0*(1.d-7-ca)

        call waarden(u,xi1,xt1,xim,xtm,xih,xth,
     1 xij,xtj,xid,xtd,xif,xtf,xt,xk,ex1,exm,exh,exj,exd,exf)

        curx1=xk*x1

        ca=ca+dt*cao

        x1=xi1-(xi1-x1)*ex1
        xm=xim-(xim-xm)*exm
        xh=xih-(xih-xh)*exh
        xj=xij-(xij-xj)*exj
        xd=xid-(xid-xd)*exd
        xf=xif-(xif-xf)*exf

        if (iter.le.nstim) then
            u=u+dt*(-(xt+xica+xna+curx1) + xstim)
        else
            u=u+dt*(-(xt+xica+xna+curx1))
        endif

        if (mod(iter,imod).eq.0) write(1,*)t,u,x1,ca

C End of main iteration loop
      enddo

      write(10,*)u,ca,x1,xm,xh,xj,xd,xf

9     format(10(d12.6,1x))
92      format (i4,500(i4,1x))
      end

      subroutine waarden(v,xi1,xt1,xim,xtm,xih,xth,
     1 xij,xtj,xid,xtd,xif,xtf,xt,xk,ex1,exm,exh,exj,exd,exf)
      implicit real*8(a-h,o-z)
      common/block1/xinf1,xtau1
      common/blockm/xinfm,xtaum
      common/blockh/xinfh,xtauh
      common/blockj/xinfj,xtauj
      common/blockd/xinfd,xtaud
      common/blockf/xinff,xtauf
      common/xtblock/xttab,xktab
      common/blocke/e1,ej,em,ed,eh,ef
      parameter (ndimtab=2500)
      real*8 xinf1(ndimtab),xtau1(ndimtab)
      real*8 xinfm(ndimtab),xtaum(ndimtab)
      real*8 xinfh(ndimtab),xtauh(ndimtab)
      real*8 xinfj(ndimtab),xtauj(ndimtab)
      real*8 xinfd(ndimtab),xtaud(ndimtab)
      real*8 xinff(ndimtab),xtauf(ndimtab)
      real*8 xttab(ndimtab)
      real*8 xktab(ndimtab)
      real*8 e1(ndimtab),ej(ndimtab)
      real*8 em(ndimtab),ed(ndimtab)
      real*8 eh(ndimtab),ef(ndimtab)

        temp=(v+100.d0)/.1d0
        index=int(temp)
        fac=temp-index
        xi1=xinf1(index)+fac*(xinf1(index+1)-xinf1(index))
        xim=xinfm(index)+fac*(xinfm(index+1)-xinfm(index))
        xih=xinfh(index)+fac*(xinfh(index+1)-xinfh(index))
        xij=xinfj(index)+fac*(xinfj(index+1)-xinfj(index))
        xid=xinfd(index)+fac*(xinfd(index+1)-xinfd(index))
        xif=xinff(index)+fac*(xinff(index+1)-xinff(index))
        xt1=xtau1(index)+fac*(xtau1(index+1)-xtau1(index))
        xtm=xtaum(index)+fac*(xtaum(index+1)-xtaum(index))
        xth=xtauh(index)+fac*(xtauh(index+1)-xtauh(index))
        xtj=xtauj(index)+fac*(xtauj(index+1)-xtauj(index))
        xtd=xtaud(index)+fac*(xtaud(index+1)-xtaud(index))
        xtf=xtauf(index)+fac*(xtauf(index+1)-xtauf(index))
        xt=xttab(index)+fac*(xttab(index+1)-xttab(index))
        xk=xktab(index)+fac*(xktab(index+1)-xktab(index))
        ex1=e1(index)+fac*(e1(index+1)-e1(index))
        exm=em(index)+fac*(em(index+1)-em(index))
        exh=eh(index)+fac*(eh(index+1)-eh(index))
        exj=ej(index)+fac*(ej(index+1)-ej(index))
        exd=ed(index)+fac*(ed(index+1)-ed(index))
        exf=ef(index)+fac*(ef(index+1)-ef(index))
        return
      end
