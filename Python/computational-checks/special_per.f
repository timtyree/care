      program laplace
      implicit real*8 (a-h,o-z)
      integer nx, ny
      parameter(nx=200,ny=200)
      parameter(mx=600,my=600)
      real*8 uu(0:mx+1,0:my+1),vv(mx,my),ww(mx,my)
      real*8 u(0:nx+1,0:ny+1),v(nx,ny),
     1       w(nx,ny),ut(0:nx+1,0:ny+1)
      integer iam,ifac, PEs, i, j, ioff
      integer myshare, iter, niter, part1d,imod
      common/scale/th_del
      common/iscale/iskip
      character*80 filename
      character*80 FILE
C Define parameters  values
      data tvp,tv1m,tv2m,
     %           twp,twm,td,
     %           uc,uoo,to,
     %           tr,tsi,xk,
     %           ucsi,uv,um/
     %            13.03,19.6,1250.,
     %            800.,40.,0.45,
     %            .13,0,12.5,
     %            33.25,29.0,10,
     %            .85,.04,1/

        threshold=0.4d0
        threshold_stop=0.2d0

        pi=4.*atan(1.)

      endtime=250.0
      writevoltageevery=50.0

      open(98,file='in_cont',status='unknown')
      read(98,*)tmax,tmod,tmod2
      read(98,*)dt
      read(98,*)
      read(98,*)nabl,nfile0

      open(97,file='in',status='unknown')
      read(97,*)ktimes

      nfile=nfile0
        diffcons=0.0005
        dx=0.025
        dx2=dx**2/(diffcons*dt)

      niter=nint(tmax/dt)
      imod=nint(tmod/dt)
      imod2=nint(tmod2/dt)

      imodw=10
      imodwrite=nint(10./dt)
      nfile_end=nfile0+niter/imod
      th_del=16.
      ico=1
      icop=1

      ntime=0

c         write(unit=filename, fmt=111)
c    1               '../../../../d=0.0005/600x600/ic_600x600.',ktimes
c111       format(a40,i3.3)
      open(unit=2,file='ic_600x600.001',status='unknown')
         do i=1,mx
         do j=1,my
              read(2,*)uu(i,j),vv(i,j),ww(i,j)
         enddo
         enddo

      do lx=0,0
      do ly=0,0

         idelx=lx*200
         idely=ly*200

      open(1,file='input_disorder',status='unknown')
         do i=1,nx
         do j=1,ny
           u(i,j)=uu(i+idelx,j+idely)
           v(i,j)=vv(i+idelx,j+idely)
           w(i,j)=ww(i+idelx,j+idely)
           write(1,*)i,j,u(i,j),v(i,j),w(i,j)
         enddo
         enddo

c boundary conditions
      do j = 1, ny
        u(0,j)=u(nx,j)
        u(nx+1,j)=u(1,j)
      enddo
      do i = 1, nx
        u(i,0)=u(i,ny)
        u(i,ny+1)=u(i,1)
      enddo
      u(nx+1,0)=u(1,ny)
      u(nx+1,ny+1)=u(1,1)
      u(0,0)=u(nx,ny)
      u(0,ny+1)=u(nx,1)


C Main iteration loop
      t=0.

      do iter = 1, niter

      t=t+dt

C Update the solution for this iteration

           do j = 1, ny
              do i = 1, nx

                xlap = (u(i+1,j)+u(i-1,j)+u(i,j+1)+
     1                       u(i,j-1)-4.d0*u(i,j))/dx2

              p=0
              q=0
              if(u(i,j).gt.uc)p=1
              if(u(i,j).gt.uv)q=1
              dv=(1-p)*(1-v(i,j))/((1-q)*tv2m+tv1m*q)-p*v(i,j)/tvp
              dw=(1-p)*(1-w(i,j))/twm-p*w(i,j)/twp
              xfi=-v(i,j)*p*(u(i,j)-uc)*(um-u(i,j))/td
              xso=(u(i,j)-uoo)*(1-p)/to+p/tr
              xsi=-w(i,j)*(1+tanh(xk*(u(i,j)-ucsi)))/(2*tsi)
              v(i,j)=v(i,j)+dt*dv
              w(i,j)=w(i,j)+dt*dw
              ut(i,j)=u(i,j)+xlap-dt*(xfi+xso+xsi)

            enddo
          enddo

       if (mod(iter,imod).eq.0) then
c          nfile=nfile+1
c          call write_file(nx,ny,u,nfile)
           write(6,*)t,u(1,1)
       endif

c calculation of tip position
       if (mod(iter,imod2).eq.0) then
           do j = 1, ny
              ut(0,j)=ut(nx,j)
              ut(nx+1,j)=ut(1,j)
           enddo
           do i = 1, nx
              ut(i,0)=ut(i,ny)
              ut(i,ny+1)=ut(i,1)
           enddo
           ut(nx+1,0)=ut(1,ny)
           ut(nx+1,ny+1)=ut(1,1)
           ut(0,0)=ut(nx,ny)
           ut(0,ny+1)=ut(nx,1)
           call mytip(nx,ny,u,ut,threshold,t,ico,icop,nfile)
       endif

           if (ico.eq.0.and.icop.ne.0) then
             ntime=ntime+1
             write(15,*)ntime,t
c            do i=1,nx
c            do j=1,ny
c               if (u(i,j).gt.threshold_stop) go to 778
c            enddo
c            enddo
             go to 777
778          continue
           endif

         do i=1,nx
         do j=1,ny
           u(i,j)=ut(i,j)
         enddo
         enddo

c boundary conditions
      do j = 1, ny
        u(0,j)=u(nx,j)
        u(nx+1,j)=u(1,j)
      enddo
      do i = 1, nx
        u(i,0)=u(i,ny)
        u(i,ny+1)=u(i,1)
      enddo
      u(nx+1,0)=u(1,ny)
      u(nx+1,ny+1)=u(1,1)
      u(0,0)=u(nx,ny)
      u(0,ny+1)=u(nx,1)

C End of main iteration loop
      enddo

777   continue
c     write(6,*)ktimes,lx,ly,ntime,t
      ico=10
      icop=10
      enddo
      enddo

      open(2,file='output_disorder',status='unknown')
         do i=1,nx
         do j=1,ny
           write(2,*)i,j,u(i,j),v(i,j),w(i,j)
         enddo
         enddo



      end

        subroutine write_file(nx,ny,u,nfile)
        implicit real*8(a-h,o-z)
        integer nx,ny,nfile
        real*8 u(0:nx+1,0:ny+1)
        real*8 v(nx,ny)
        character*80 filename
        write(unit=filename, fmt=111)'ufield.',nfile
 111    format(a7,i4.4)
        write(*,*)'writing to ',filename,u(50,50)


        open(unit=30,file=filename,status='unknown')
        do i=1,nx
        do j=1,ny
          write(30,*)u(i,j)
        enddo
        enddo

        return
        end


c  -------------------------------------------------------------------
      subroutine mytip(nx,ny,ux,ut,cutt,t,ico,icop,nfile)
c  -------------------------------------------------------------------
      implicit real*8(a-h,o-z)
      real*8 ux(0:nx+1,0:ny+1)
      real*8 ut(0:nx+1,0:ny+1)
      real*8 cutt,t
      real*8 xt(5000)
      real*8 yt(5000)
      real*8 xtp(5000)
      real*8 ytp(5000)
      common/scale/th_del
        character*80 filename,filename2,filename3
        open(unit=63,file='tip_tim_per_diff',status='unknown')
        open(unit=85,file='tippos_per',status='unknown')

      icop=ico
      ico=0
      kmin=1
      kmax=nx
      lmin=1
      lmax=ny
      dx=0.025
      viso=cutt

      do 23064 k=kmin,kmax
      do 23066 l=lmin,lmax
      i=k
      j=l
      x1=ut(i,j)
      y1=ux(i,j)
      i=k+1
      j=l
      x2=ut(i,j)
      y2=ux(i,j)
      i=k+1
      j=l+1
      x3=ut(i,j)
      y3=ux(i,j)
      i=k
      j=l+1
      x4=ut(i,j)
      y4=ux(i,j)
      d1=x1-x4
      d2=x1-x2+x3-x4
      d3=d2-d1
      aa=-d1*y2+d1*y3+d3*(y4-y1)
      bb=(y4-y1)*(x2-viso)+(y3-y2)*(viso-x1)+d1*y2-viso*d2+d3*y1
      cc=y1*(x2-viso)+y2*(viso-x1)+viso*(x1-x2)
      disc=bb**2-4.d0*aa*cc
      if (disc.lt.0.d0) go to 23072
      ytipp=(-bb+dsqrt(disc))/(2.d0*aa)
      xtipp=(viso-x1+d1*ytipp)/(x2-x1+d2*ytipp)
      ytipm=(-bb-dsqrt(disc))/(2.d0*aa)
      xtipm=(viso-x1+d1*ytipm)/(x2-x1+d2*ytipm)
c     write(51,*)k,l,xtipp,ytipp,xtipm,ytipm
      if(.not.((xtipp.lt.1.0.and.xtipp.gt.0.0).and.
     1     (ytipp.lt.1.0.and.ytipp.gt.0.0)))goto 23071
      ico=ico+1
      xt(ico)=xtipp+float(k)
      yt(ico)=ytipp+float(l)
      goto 23072
23071 continue
      ytipm=(-bb-dsqrt(disc))/(2.d0*aa)
      xtipm=(viso-x1+d1*ytipm)/(x2-x1+d2*ytipm)
      if(.not.((xtipm.lt.1.0.and.xtipm.gt.0.0).and.
     1     (ytipm.lt.1.0.and.ytipm.gt.0.0)))goto 23072

      ico=ico+1
      xt(ico)=xtipm+float(k)
      yt(ico)=ytipm+float(l)

23072 continue
23066 continue
23064 continue

      if (ico.ne.icop) write(63,*)t,ico,icop

          write(85,97)t,ico,icop
          write(85,96)(xt(k),k=1,ico)
          write(85,96)(yt(k),k=1,ico)

c     endif

9     format(12(f12.6,1x))
96     format(100(f5.1,1x))
97     format(f8.1,1x,i3,1x,i3)
      return
      end

