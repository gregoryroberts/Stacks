import matplotlib.pyplot as plt
import numpy as np



degrees_to_radians = np.pi / 180.
angle_min = 0.0 * degrees_to_radians
angle_max = 30.0 * degrees_to_radians

num_angles = 12
num_wavelengths = 3
num_pol = 2
num_optimization_goals = num_wavelengths * num_angles * num_pol

lambda_min_um = 0.4
lambda_max_um = 0.7

angles = np.linspace( angle_min, angle_max, num_angles )
wavelengths_um = np.linspace( lambda_min_um, lambda_max_um, num_wavelengths )

transmit_angle_low = 0.0 * degrees_to_radians
transmit_angle_high = 8.0 * degrees_to_radians

transmission_map = np.zeros( ( num_wavelengths, num_angles ) )

for wl_idx in range( 0, num_wavelengths ):
	for ang_idx in range(0, num_angles ):
		if ( angles[ ang_idx ] >= transmit_angle_low ) and ( angles[ ang_idx ] <= transmit_angle_high ):
			transmission_map[ wl_idx, ang_idx ] = 1.0


num_iterations = 100

fom = np.zeros( ( num_iterations, num_wavelengths, num_angles, num_pol ) )

fom = np.load( 'current_fom.npy' )

choose_wl = 0
choose_angle = 1
choose_spol = 0
choose_ppol = 1

print( num_optimization_goals )

wl_colors = [ 'b', 'g', 'r' ]

plot_fom_spol = [ np.zeros( ( num_iterations, 1 ) ) for i in range( 0, num_wavelengths ) ]
plot_fom_ppol = [ np.zeros( ( num_iterations, 1 ) ) for i in range( 0, num_wavelengths ) ]

for ang_idx in range( 0, num_angles ):
	plt.subplot( 3, 4, 1 + ang_idx )
	for wl_idx in range( 0, num_wavelengths ):
		for iter_idx in range( 0, num_iterations ):
			plot_fom_spol[ wl_idx ][ iter_idx ] = fom[ iter_idx, wl_idx, ang_idx, choose_spol ]
			plot_fom_ppol[ wl_idx ][ iter_idx ] = fom[ iter_idx, wl_idx, ang_idx, choose_ppol ]
		
		plt.plot( plot_fom_spol[ wl_idx ], linewidth=3, color=wl_colors[ wl_idx ] )
		plt.plot( -plot_fom_ppol[ wl_idx ], linewidth=3, linestyle='--', color=wl_colors[ wl_idx ] )
	plt.title( str( int( 10 * angles[ ang_idx ] / degrees_to_radians ) / 10. ) + '$^\circ$' )
	plt.ylim( [ -1, 1 ] )
	plt.gca().get_xaxis().set_ticks( [] )

plt.show()

