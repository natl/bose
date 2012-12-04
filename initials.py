 
#######################################################################
#
# Program:      Initial state creation for trap simulation
#
# Author:       Nathanael Lampe, School of Physics, Monash University
#
# Created:      December 4, 2012
#
# Changelog:    December 4, 2012: Created from twodtrap.py
#           
#               
#
# Purpose:      Contain functions that create initial states in twod.py
#               
########################################################################

def lastpsi(x,psi):
  return psi

def twoblobs(x,y,npt, **kwargs):
  X,Y = np.meshgrid(x,y)
  xo1 = -2.
  xo2 = +2.
  sig1 = 1.
  pi = np.pi
  sig2 = 1.
  psi1 = 1.0/np.sqrt(2*pi*sig1**2) * np.exp(-((X-xo1)**2 + Y**2.) / (2*sig1**2)) 
  psi2 = 1.0/np.sqrt(2*pi*sig2**2) * np.exp(-((X-xo2)**2 + Y**2.) / (2*sig2**2))
  psi = psi1 + psi2
  norm = sum(sum(psi*psi.conjugate())) * abs(x[1]-x[0]) * abs(y[1]-y[0])
  psi = psi / np.sqrt(norm)
  return psi
  
def threeblobs(x        ,
               y        ,
               npt      ,
               **kwargs ):
  X,Y = np.meshgrid(x,y)
  xo1 = 0.
  yo1 = 4.
  xo2 = 2. * np.sqrt(3.)
  yo2 = -2.
  xo3 = -2. * np.sqrt(3.)
  yo3 = -2.
  sig1 = 1.
  sig2 = 1.
  sig3 = 1.
  pi = np.pi
  psi1 = 1.0 / np.sqrt(2*pi*sig1**2) * np.exp(
                           -((X-xo1)**2 + (Y-yo1)**2.) / (2*sig1**2)) 
  psi2 = 1.0 / np.sqrt(2*pi*sig2**2) * np.exp(
                           -((X-xo2)**2 + (Y-yo2)**2.) / (2*sig2**2)) 
  psi3 = 1.0 / np.sqrt(2*pi*sig3**2) * np.exp(
                           -((X-xo3)**2 + (Y-yo3)**2.) / (2*sig3**2)) 
  psi = psi1 + psi2 + psi3
  norm = sum(sum(psi*psi.conjugate())) * abs(x[1]-x[0]) * abs(y[1]-y[0])
  psi = psi / np.sqrt(norm)
  return psi
  
def gauss( x              ,
           y              ,
           npt            ,
           means = [0.,0.],
           sig   = [1.,1.],
           corr  = 0.     ):
  '''
  Define an initial wavefunction (Gaussian)
  '''
  if abs( corr ) >= 1:
    print '<twod.gauss> corr must be strictly between -1 and 1'
    
  wave = np.zeros([npt,npt], dtype = 'complex' )
  
  #Gaussian Parameters and preliminary calculations for speed
  xsig = sig[0]
  ysig = sig[1]
  xycorr = np.array( [ [ xsig**2.,           corr * xsig * ysig ],
                      [ corr * xsig * ysig, ysig**2.           ] ] )
  
  corrdet = det(xycorr)
  
  xycorrinv = 1. / corrdet * np.array( [ [  xycorr[1,1], -xycorr[0,1]  ],
                                        [ -xycorr[1,0],  xycorr[0,0]  ] ] )
              
  corrdetroot = np.sqrt(corrdet)
  def phase( X, Y , X0, Y0):
    return 1j*np.arctan2( ( Y - Y0) , ( X - X0) )
  
  print '<twod.gauss> combobulating initial wavefunction matrix'
  for jj in range( 0, npt ):
    for kk in range( 0, npt ):
      pos = np.array( [ x[jj], y[kk] ] ) - means
      wave[jj,kk] = 1. / (2. * np.pi * corrdetroot ) * (
         np.exp( -0.5 * np.dot( np.dot( pos.transpose(), xycorrinv ), pos ) ) )
  print '<twod.gauss> completed'
  return wave

  
def vortexgauss( x                 ,  #xvals
                 y                 ,  #yvals
                 npt               ,  #no. gridpoints
                 vort  = [0, 0, 0] ,  #X by 2. array of vortex locations
                 means = [0.,0.]   ,  #[x,y] centre of gaussian
                 sig   = [1.,1.]   ,  #[stddev_x, stddev_y] for gaussian
                 corr  = 0.        ): #xy correlation
  '''
  Define an initial wavefunction (Gaussian) with vortices!
  SYNTAX:
  def vortexgauss( x              ,  #xvals
                   y              ,  #yvals
                   npt            ,  #no. gridpoints
                   vort           ,  #X by 3. array of vortex locations
                   means = [0.,0.],  #[x,y] centre of gaussian
                   sig   = [1.,1.],  #[stddev_x, stddev_y] for gaussian
                   corr  = 0.     ): #xy correlation
                   
  vort should look like:
  [ [-1,  0, +1],
    [-1, -1, -2],
    [+2, +3, +1],
    [ x,  y, ax] ] where ax specifies a vortex or antivortex and strength
  
  '''
  
  if abs( corr ) >= 1:
    print '<twod.vortexgauss> corr must be strictly between -1 and 1'
    
  wave = np.zeros([npt,npt], dtype = 'complex' )
  
  #Gaussian Parameters and preliminary calculations for speed
  xsig = sig[0]
  ysig = sig[1]
  xycorr = np.array( [ [ xsig**2.,           corr * xsig * ysig ],
                      [ corr * xsig * ysig, ysig**2.           ] ] )
  
  corrdet = det(xycorr)
  
  xycorrinv = 1. / corrdet * np.array( [ [  xycorr[1,1], -xycorr[0,1]  ],
                                         [ -xycorr[1,0],  xycorr[0,0]  ] ] )
              
  corrdetroot = np.sqrt(corrdet)
  
  print '<twod.vortexgauss> combobulating initial wavefunction matrix'
  
  #Caclulate phase matrix
  X, Y = np.meshgrid( x, y )
  theta = np.zeros( [npt,npt], dtype = complex )
  for X0, Y0, ax in vort: theta += ax * 1j * np.arctan2( ( Y - Y0) , ( X - X0) )
  
  plt.figure("Initial Phase")
  plt.contourf(x,y,np.angle(np.exp(theta)),20)
  plt.xlabel(r'x ($a_0$)')
  plt.ylabel(r'y ($a_0$)')
  plt.show()
  
  
  #Now do the whole wavefunction
  for jj in range( 0, npt ):
    for kk in range( 0, npt ):
      pos = np.array( [ x[jj], y[kk] ] ) - means
      wave[jj,kk] = 1. / (2. * np.pi * corrdetroot ) * (
         np.exp( -0.5 * np.dot( np.dot( pos.transpose(), xycorrinv ), pos ) 
                ) * np.exp( theta[jj,kk] ) )
  print '<twod.vortexgauss> completed'
  
  return wave