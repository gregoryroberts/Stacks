import numpy as np
import matplotlib

matplotlib.use('tkagg')

import matplotlib.pyplot as plt

from tmm.tmm_core import (coh_tmm, coh_tmm_reverse, unpolarized_RT, ellips,
                       position_resolved, find_in_structure_with_inf)

import sys
import time

lambda_min_um = 0.4
lambda_max_um = 0.7

grid_resolution_um = 0.003

device_depth_um = 30 * lambda_max_um;
device_depth_voxels = int( device_depth_um / grid_resolution_um )

density = np.ones( device_depth_voxels )

min_index = 1.5
max_index = 2.25
mid_index = 0.5 * (min_index + max_index)

if len( sys.argv ) < 2:
	print( "Please specify a device name prefix for this optimization" )
	sys.exit()

device_filepath = "devices/"
device_prefix = sys.argv[ 1 ] + "_"

def compute_electric_field_s_pol(device, grid_resolution_um, device_depth_um, device_depth_voxels, angle_radians, lambda_vac_um, reverse=False):
	d_list = [ np.inf, 1, *( grid_resolution_um * np.ones( device_depth_voxels ) ), 1, np.inf ]
	n_list = [ 1, 1, *device, 1, 1 ]

	if reverse:
		coh_tmm_data = coh_tmm_reverse('s', n_list, d_list, angle_radians, lambda_vac_um)
	else:
		coh_tmm_data = coh_tmm('s', n_list, d_list, angle_radians, lambda_vac_um)
	
	get_Ey_pts = np.linspace(1, 1 + device_depth_um, device_depth_voxels)
	Ey_data = np.zeros(device_depth_voxels, dtype=np.complex)

	for d_idx in range( 0, device_depth_voxels ):
		d = get_Ey_pts[d_idx]

		layer = 2 + d_idx
		d_in_layer = 0.5 * grid_resolution_um 

		data = position_resolved( layer, d_in_layer, coh_tmm_data )

		Ey_data[d_idx] = data['Ey']

	adjoint_meausre_point_um = 1 + device_depth_um + 1
	layer, d_in_layer = find_in_structure_with_inf( d_list, adjoint_meausre_point_um )
	data_adjoint_measure = position_resolved( layer, d_in_layer, coh_tmm_data )
	Ey_adjoint_measure = data_adjoint_measure['Ey']

	if reverse:
		Ey_data = np.flip( Ey_data )

	return Ey_data, coh_tmm_data['T'], Ey_adjoint_measure

def compute_electric_field_p_pol(device, grid_resolution_um, device_depth_um, device_depth_voxels, angle_radians, lambda_vac_um, reverse=False):
	d_list = [ np.inf, 1, *( grid_resolution_um * np.ones( device_depth_voxels ) ), 1, np.inf ]
	n_list = [ 1, 1, *device, 1, 1 ]

	if reverse:
		coh_tmm_data = coh_tmm_reverse('p', n_list, d_list, angle_radians, lambda_vac_um)
	else:
		coh_tmm_data = coh_tmm('p', n_list, d_list, angle_radians, lambda_vac_um)

	get_E_pts = np.linspace(1, 1 + device_depth_um, device_depth_voxels)
	Ex_data = np.zeros(device_depth_voxels, dtype=np.complex)
	Ez_data = np.zeros(device_depth_voxels, dtype=np.complex)

	for d_idx in range( 0, device_depth_voxels ):
		d = get_E_pts[d_idx]

		layer = 2 + d_idx
		d_in_layer = 0.5 * grid_resolution_um 

		data = position_resolved( layer, d_in_layer, coh_tmm_data )

		Ex_data[d_idx] = data['Ex']
		Ez_data[d_idx] = data['Ez']

	adjoint_meausre_point_um = 1 + device_depth_um + 1
	layer, d_in_layer = find_in_structure_with_inf( d_list, adjoint_meausre_point_um )
	data_adjoint_measure = position_resolved( layer, d_in_layer, coh_tmm_data )
	Ex_adjoint_measure = data_adjoint_measure['Ex']
	Ez_adjoint_measure = data_adjoint_measure['Ez']
	E_adjoint_measure = Ex_adjoint_measure

	if reverse:
		Ex_data = np.flip( Ex_data )
		Ez_data = np.flip( Ez_data )

	return Ex_data, Ez_data, coh_tmm_data['T'], E_adjoint_measure

def write_info_file(
	device_filepath, device_filename,
	wl_min_um, wl_max_um,
	angle_min_deg, angle_max_deg,
	transmission_angle_min_deg, transission_angle_max_deg,
	num_opt_wl, num_opt_angle, num_opt_pol,
	device_size_um,
	normalized_step_size, num_iterations, step_size_taper ):

	info_file = open( device_filepath + "/" + device_filename, 'w' )

	info_file.write( "wavelength min (um): " + str( wl_min_um ) + "\n" )
	info_file.write( "wavelength max (um): " + str( wl_max_um ) + "\n" )

	info_file.write( "angle min (degrees): " + str( angle_min_deg ) + "\n" )
	info_file.write( "angle max (degrees): " + str( angle_max_deg ) + "\n" )

	info_file.write( "transmission angle min (degrees): " + str( transmission_angle_min_deg ) + "\n" )
	info_file.write( "transmission angle max (degrees): " + str( transission_angle_max_deg ) + "\n" )

	info_file.write( "number optimization wavelengths: " + str( num_opt_wl ) + "\n" )
	info_file.write( "number optimization angles: " + str( num_opt_angle ) + "\n" )
	info_file.write( "number optimization polarizations: " + str( num_opt_pol ) + "\n" )

	info_file.write( "device size (um): " + str( device_size_um ) + "\n" )

	info_file.write( "normalized step size: " + str( normalized_step_size ) + "\n" )
	info_file.write( "number iterations: " + str( num_iterations ) + "\n" )
	info_file.write( "step size taper: " + str(step_size_taper ) + "\n" )

	info_file.close()
	
degrees_to_radians = np.pi / 180.
angle_min_degrees = 0.0
angle_max_degrees = 30.0
angle_min = angle_min_degrees * degrees_to_radians
angle_max = angle_max_degrees * degrees_to_radians

num_angles = 6
num_wavelengths = 3
num_pol = 2
num_optimization_goals = num_wavelengths * num_angles * num_pol

angles = np.linspace( angle_min, angle_max, num_angles )
wavelengths_um = np.linspace( lambda_min_um, lambda_max_um, num_wavelengths )

transmit_angle_low_degrees = 0.0
transmit_angle_high_degrees = 8.0
transmit_angle_low = transmit_angle_low_degrees * degrees_to_radians
transmit_angle_high = transmit_angle_high_degrees * degrees_to_radians

transmission_map = np.zeros( ( num_wavelengths, num_angles ) )

for wl_idx in range( 0, num_wavelengths ):
	for ang_idx in range(0, num_angles ):
		if ( angles[ ang_idx ] >= transmit_angle_low ) and ( angles[ ang_idx ] <= transmit_angle_high ):
			transmission_map[ wl_idx, ang_idx ] = 1.0


max_fom = np.zeros( ( num_wavelengths, num_angles, num_pol ) )

for wl_idx in range( 0, num_wavelengths ):
	wavelength_um = wavelengths_um[ wl_idx ]
	
	for ang_idx in range( 0, num_angles ):

		no_device = np.ones( device_depth_voxels )

		angle = angles[ ang_idx ]

		Ey_fwd_s, transmission_fwd_s, Ey_adjoint_measure_s = compute_electric_field_s_pol( no_device, grid_resolution_um, device_depth_um, device_depth_voxels, angle, wavelength_um, False )
		fom_s = np.abs( Ey_adjoint_measure_s )**2
		Ex_fwd_p, Ez_fwd_p, transmission_fwd_p, E_adjoint_measure_p = compute_electric_field_p_pol( no_device, grid_resolution_um, device_depth_um, device_depth_voxels, angle, wavelength_um, False )
		fom_p = np.abs( E_adjoint_measure_p / np.cos( angle ) )**2

		max_fom[ wl_idx, ang_idx, 0 ] = fom_s
		max_fom[ wl_idx, ang_idx, 1 ] = fom_p


num_iterations = 50
normalized_step_size = 0.004
step_size_taper = 10

write_info_file(
	device_filepath, device_prefix + "info.txt",
	lambda_min_um, lambda_max_um,
	angle_min_degrees, angle_max_degrees,
	transmit_angle_low_degrees, transmit_angle_high_degrees,
	num_wavelengths, num_angles, num_pol,
	device_depth_um,
	normalized_step_size, num_iterations, step_size_taper )


fom = np.zeros( ( num_iterations, num_wavelengths, num_angles, num_pol ) )
flat_fom = np.zeros( ( num_iterations, num_optimization_goals ) )
full_weights = np.zeros( ( num_iterations, num_optimization_goals ) )

device = mid_index * np.ones( device_depth_voxels )

init_transmission_s = np.zeros( ( num_wavelengths, num_angles ) )
init_transmission_p = np.zeros( ( num_wavelengths, num_angles ) )

for wl_idx in range( 0, num_wavelengths ):
	wavelength_um = wavelengths_um[ wl_idx ]
	
	for ang_idx in range( 0, num_angles ):

		angle = angles[ ang_idx ]

		Ey_fwd_s, transmission_fwd_s, Ey_adjoint_measure_s = compute_electric_field_s_pol( device, grid_resolution_um, device_depth_um, device_depth_voxels, angle, wavelength_um, False )
		Ex_fwd_p, Ez_fwd_p, transmission_fwd_p, E_adjoint_measure_p = compute_electric_field_p_pol( device, grid_resolution_um, device_depth_um, device_depth_voxels, angle, wavelength_um, False )

		init_transmission_s[ wl_idx, ang_idx ] = transmission_fwd_s
		init_transmission_p[ wl_idx, ang_idx ] = transmission_fwd_p

for iter_idx in range( 0, num_iterations ):
	start_time = time.time()
	print( "Working on iteration " + str( iter_idx ) + " out of " + str(num_iterations) )
	gradients = np.zeros( ( num_wavelengths * num_angles * num_pol, device_depth_voxels ) )
	fom_iter = np.zeros( num_wavelengths * num_angles * num_pol )

	for wl_idx in range( 0, num_wavelengths ):
		wavelength_um = wavelengths_um[ wl_idx ]
		
		for ang_idx in range( 0, num_angles ):

			angle = angles[ ang_idx ]

			Ey_fwd_s, transmission_fwd_s, Ey_adjoint_measure_s = compute_electric_field_s_pol( device, grid_resolution_um, device_depth_um, device_depth_voxels, angle, wavelength_um, False )
			Ey_adj_s, transmission_adj_s, ignore = compute_electric_field_s_pol( device, grid_resolution_um, device_depth_um, device_depth_voxels, angle, wavelength_um, True )
			gradient_s = -2 * np.real( Ey_fwd_s * Ey_adj_s * np.conj( Ey_adjoint_measure_s ) / 1j )

			fom_s = np.abs( Ey_adjoint_measure_s )**2
			fom_s_normalized = fom_s / max_fom[ wl_idx, ang_idx, 0 ]

			Ex_fwd_p, Ez_fwd_p, transmission_fwd_p, E_adjoint_measure_p = compute_electric_field_p_pol( device, grid_resolution_um, device_depth_um, device_depth_voxels, angle, wavelength_um, False )
			Ex_adj_p, Ez_adj_p, transmission_adj_p, ignore = compute_electric_field_p_pol( device, grid_resolution_um, device_depth_um, device_depth_voxels, angle, wavelength_um, True )

			dot_E_p = Ex_fwd_p * Ex_adj_p + Ez_fwd_p * Ez_adj_p
			phase_weighting_p = np.conj( E_adjoint_measure_p / np.cos( angle ) )
			gradient_p = -2 * np.real( dot_E_p * phase_weighting_p / 1j )

			fom_p = np.abs( E_adjoint_measure_p / np.cos( angle ) )**2
			fom_p_normalized = fom_p / max_fom[ wl_idx, ang_idx, 1 ]

			if transmission_map[ wl_idx, ang_idx ] == 1:
				gradients[ wl_idx * num_angles * num_pol + ang_idx * num_pol + 0, : ] = gradient_s
				gradients[ wl_idx * num_angles * num_pol + ang_idx * num_pol + 1, : ] = gradient_p

				fom_iter[ wl_idx * num_angles * num_pol + ang_idx * num_pol + 0 ] = fom_s_normalized
				fom_iter[ wl_idx * num_angles * num_pol + ang_idx * num_pol + 1 ] = fom_p_normalized

				fom[ iter_idx, wl_idx, ang_idx, 0 ] = fom_s_normalized
				fom[ iter_idx, wl_idx, ang_idx, 1 ] = fom_p_normalized
			else:
				gradients[ wl_idx * num_angles * num_pol + ang_idx * num_pol + 0, : ] = -gradient_s
				gradients[ wl_idx * num_angles * num_pol + ang_idx * num_pol + 1, : ] = -gradient_p

				fom_iter[ wl_idx * num_angles * num_pol + ang_idx * num_pol + 0 ] = 1 - fom_s_normalized
				fom_iter[ wl_idx * num_angles * num_pol + ang_idx * num_pol + 1 ] = 1 - fom_p_normalized

				fom[ iter_idx, wl_idx, ang_idx, 0 ] = 1 - fom_s_normalized
				fom[ iter_idx, wl_idx, ang_idx, 1 ] = 1 - fom_p_normalized


	grad_weightings = ( 2. / num_optimization_goals ) - ( np.power( fom_iter, 2 ) / np.sum( np.power( fom_iter, 2 ) ) )

	if np.min( grad_weightings ) < 0:
		grad_weightings -= np.min( grad_weightings )
		grad_weightings /= np.sum( grad_weightings )

	flat_fom[ iter_idx, : ] = fom_iter
	full_weights[ iter_idx, : ] = grad_weightings

	for opt_goal in range( 0, num_optimization_goals ):
		if ( opt_goal % 2 ) == 0:
			plt.plot( flat_fom[ 0 : ( iter_idx + 1 ), opt_goal ], linewidth=2, color='g' )
		else:
			plt.plot( flat_fom[ 0 : ( iter_idx + 1 ), opt_goal ], linewidth=2, linestyle='--', color='g' )
	plt.ylabel( "Figure of Merit", fontsize=20  )
	plt.xlabel( "Iteration", fontsize=20 )
	
	plt.savefig( device_filepath + "/" + device_prefix + "live_fom_plot.png", bbox_inches='tight' )
	plt.clf()

	for opt_goal in range( 0, num_optimization_goals ):
		if ( opt_goal % 2 ) == 0:
			plt.plot( full_weights[ 0 : ( iter_idx + 1 ), opt_goal ], linewidth=2, color='m' )
		else:
			plt.plot( full_weights[ 0 : ( iter_idx + 1 ), opt_goal ], linewidth=2, linestyle='--', color='m' )
	plt.ylabel( "Gradient Weights", fontsize=20  )
	plt.xlabel( "Iteration", fontsize=20 )

	plt.savefig( device_filepath + "/" + device_prefix + "live_weights_plot.png", bbox_inches='tight' )
	plt.clf()

	weighted_gradient = np.zeros( device_depth_voxels )

	for goal_idx in range( 0, num_optimization_goals ):
		weighted_gradient += grad_weightings[ goal_idx ] * gradients[ goal_idx, : ]

	np.save( device_filepath + "/" + device_prefix + "device_index.npy", device )
	np.save( device_filepath + "/" + device_prefix + "figure_of_merit.npy", fom )

	num_iterations_divisor = num_iterations - 1
	
	step_size = normalized_step_size
	if num_iterations > 1:
		step_size = normalized_step_size / np.power( step_size_taper, iter_idx / ( num_iterations - 1 ) )
	
	device += step_size * weighted_gradient / np.max( np.abs( weighted_gradient ) )
	device = np.maximum( np.minimum( device, max_index ), min_index )

	elapsed = time.time() - start_time
	print("A single iteration took " + str(elapsed / 60.) + " minutes")

measured_transmission_s = np.zeros( ( num_wavelengths, num_angles ) )
measured_transmission_p = np.zeros( ( num_wavelengths, num_angles ) )

for wl_idx in range( 0, num_wavelengths ):
	wavelength_um = wavelengths_um[ wl_idx ]
	
	for ang_idx in range( 0, num_angles ):

		angle = angles[ ang_idx ]

		Ey_fwd_s, transmission_fwd_s, Ey_adjoint_measure_s = compute_electric_field_s_pol( device, grid_resolution_um, device_depth_um, device_depth_voxels, angle, wavelength_um, False )
		Ex_fwd_p, Ez_fwd_p, transmission_fwd_p, E_adjoint_measure_p = compute_electric_field_p_pol( device, grid_resolution_um, device_depth_um, device_depth_voxels, angle, wavelength_um, False )

		measured_transmission_s[ wl_idx, ang_idx ] = transmission_fwd_s
		measured_transmission_p[ wl_idx, ang_idx ] = transmission_fwd_p


plt.subplot( 2, 3, 1 )
plt.imshow( transmission_map )
plt.clim(0, 1)
plt.title('Transmission Goal')
plt.colorbar()
plt.subplot( 2, 3, 2 )
plt.imshow( init_transmission_s )
plt.clim(0, 1)
plt.title('Initial S Transmission')
plt.colorbar()
plt.subplot( 2, 3, 3 )
plt.imshow( init_transmission_p )
plt.clim(0, 1)
plt.title('Initial P Transmission')
plt.colorbar()
plt.subplot( 2, 3, 4 )
plt.imshow( measured_transmission_s )
plt.clim(0, 1)
plt.title('Measured S Transmission')
plt.colorbar()
plt.subplot( 2, 3, 5 )
plt.imshow( measured_transmission_p )
plt.clim(0, 1)
plt.title('Measured P Transmission')
plt.colorbar()
plt.subplot( 2, 3, 6 )
plt.plot( device, linewidth=2, color='b' )
plt.title('Optimized Device Index')

fig = plt.gcf()
fig.set_size_inches(15, 8)
plt.savefig( device_filepath + "/" + device_prefix + "transmission_plots.png", bbox_inches='tight' )

plt.show()
