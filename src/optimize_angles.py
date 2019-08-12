import numpy as np
import matplotlib.pyplot as plt

from tmm.tmm_core import (coh_tmm, coh_tmm_reverse, unpolarized_RT, ellips,
                       position_resolved, find_in_structure_with_inf)

import time


lambda_min_um = 0.4
lambda_max_um = 0.7

# grid_resolution_um = 0.001
grid_resolution_um = 0.003

device_depth_um = 20 * lambda_max_um;
device_depth_voxels = int( device_depth_um / grid_resolution_um )

density = np.ones( device_depth_voxels )


min_index = 1.5#1.2
max_index = 2.25#1.6
mid_index = 0.5 * (min_index + max_index)


np.random.seed(234234)
device = min_index + ( max_index - min_index ) * np.random.random( device_depth_voxels )


def compute_electric_field_s_pol(device, grid_resolution_um, device_depth_um, device_depth_voxels, angle_radians, lambda_vac_um, reverse=False):

	d_list = [ np.inf, 1, *( grid_resolution_um * np.ones( device_depth_voxels ) ), 1, np.inf ]
	n_list = [ 1, 1, *device, 1, 1 ]

	# stack_data_start = time.time()
	
	if reverse:
		coh_tmm_data = coh_tmm_reverse('s', n_list, d_list, angle_radians, lambda_vac_um)
	else:
		coh_tmm_data = coh_tmm('s', n_list, d_list, angle_radians, lambda_vac_um)
	
	# stack_data_elapsed = time.time() - stack_data_start
	# collect_data_start = time.time()

	get_Ey_pts = np.linspace(1, 1 + device_depth_um, device_depth_voxels)
	Ey_data = np.zeros(device_depth_voxels, dtype=np.complex)

	for d_idx in range( 0, device_depth_voxels ):
		d = get_Ey_pts[d_idx]

		# layer = d_idx
		# d_in_layer = 0.5 * grid_resolution_um 

		layer = 2 + d_idx
		d_in_layer = 0.5 * grid_resolution_um 

		# layer, d_in_layer = find_in_structure_with_inf( d_list, d )
		data = position_resolved( layer, d_in_layer, coh_tmm_data )

		Ey_data[d_idx] = data['Ey']

	# collect_data_elapsed = time.time() - collect_data_start

	# print(
	# 	'For the TMM calculation, it took ' +
	# 	str( stack_data_elapsed ) +
	# 	' seconds to propagate the stack and ' +
	# 	str( collect_data_elapsed ) +
	# 	' seconds to collect all the data' )

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

	# stack_data_start = time.time()

	if reverse:
		coh_tmm_data = coh_tmm_reverse('p', n_list, d_list, angle_radians, lambda_vac_um)
	else:
		coh_tmm_data = coh_tmm('p', n_list, d_list, angle_radians, lambda_vac_um)

	# stack_data_elapsed = time.time() - stack_data_start
	# collect_data_start = time.time()

	get_E_pts = np.linspace(1, 1 + device_depth_um, device_depth_voxels)
	Ex_data = np.zeros(device_depth_voxels, dtype=np.complex)
	Ez_data = np.zeros(device_depth_voxels, dtype=np.complex)

	for d_idx in range( 0, device_depth_voxels ):
		d = get_E_pts[d_idx]

		layer = 2 + d_idx
		d_in_layer = 0.5 * grid_resolution_um 

		# layer, d_in_layer = find_in_structure_with_inf( d_list, d )
		data = position_resolved( layer, d_in_layer, coh_tmm_data )

		Ex_data[d_idx] = data['Ex']
		Ez_data[d_idx] = data['Ez']

	# collect_data_elapsed = time.time() - collect_data_start

	# print(
	# 	'For the TMM calculation, it took ' +
	# 	str( stack_data_elapsed ) +
	# 	' seconds to propagate the stack and ' +
	# 	str( collect_data_elapsed ) +
	# 	' seconds to collect all the data' )

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

test_angle = 0;

# device = np.load('current_device.npy')
# tg = np.load('test_grad5.npy')
device = mid_index * np.ones( device_depth_voxels )
# device = 1 * np.ones( device_depth_voxels )

Ey_fwd, transmission_fwd, Ey_adjoint_measure = compute_electric_field_s_pol( device, grid_resolution_um, device_depth_um, device_depth_voxels, test_angle, lambda_min_um, False )
Ey_adj, transmission_adj, ignore = compute_electric_field_s_pol( device, grid_resolution_um, device_depth_um, device_depth_voxels, test_angle, lambda_min_um, True )
# plt.subplot(1, 2, 1)
# plt.plot(np.real(Ey_fwd))
# plt.subplot(1, 2, 2)
# plt.plot(np.real(Ey_adj))
# plt.show()
# print(np.conj(Ey_fwd[-1]))
# print(Ey_adjoint_measure)
gradient = -2 * np.real( Ey_fwd * Ey_adj * np.conj( Ey_adjoint_measure ) / 1j )

#
# Note: The gradient seems to be misaligned by maybe one layer! Something to just sort out. Maybe you need to take it in the middle of each layer!
#

# Ex_fwd, Ez_fwd, transmission_fwd, E_adjoint_measure_p = compute_electric_field_p_pol( device, grid_resolution_um, device_depth_um, device_depth_voxels, test_angle, lambda_min_um, False )
# Ex_adj, Ez_adj, transmission_adj, ignore = compute_electric_field_p_pol( device, grid_resolution_um, device_depth_um, device_depth_voxels, test_angle, lambda_min_um, True )

# dot_E = Ex_fwd * Ex_adj + Ez_fwd * Ez_adj
# phase_weighting = np.conj( E_adjoint_measure_p / np.cos( test_angle ) )
# gradient = -2 * np.real( dot_E * phase_weighting / 1j )

fom0 = np.abs( Ey_adjoint_measure )**2
# fom0 = np.abs( E_adjoint_measure_p / np.cos( test_angle ) )**2
num_fd = 0
h = 1e-2
finite_diff = np.zeros(num_fd)
for fd_idx in range(0, num_fd):
	device[fd_idx] += h
	Ey_fd, transmission_fwd, Ey_adjoint_measure = compute_electric_field_s_pol( device, grid_resolution_um, device_depth_um, device_depth_voxels, test_angle, lambda_min_um, False )
	# Ex_fwd, Ez_fwd, transmission_fwd, E_adjoint_measure_p = compute_electric_field_p_pol( device, grid_resolution_um, device_depth_um, device_depth_voxels, test_angle, lambda_min_um, False )

	fom1 = np.abs( Ey_adjoint_measure )**2
	# fom1 = np.abs( E_adjoint_measure_p / np.cos( test_angle ) )**2

	deriv = ( fom1 - fom0 ) / h

	finite_diff[fd_idx] = deriv
	device[fd_idx] -= h

	print("done fd " + str(fd_idx) + " of " + str(num_fd))

# plt.plot( finite_diff / np.max(np.abs(finite_diff)) )#, 'bo' )
# plt.plot( gradient[0:num_fd] / np.max(np.abs(gradient[0:num_fd])), 'r+' )
# plt.show()



# device = device - 0.01 * gradient / np.max(np.abs(gradient))
device += 0.0001 * gradient / np.max( np.abs( gradient ) )
device = np.maximum( np.minimum( device, max_index ), min_index )

# Ex_fwd, Ez_fwd, transmission_fwd2 = compute_electric_field_p_pol( device, grid_resolution_um, device_depth_um, device_depth_voxels, test_angle, 0.5, False )
Ey_fwd2, transmission_fwd2, Ey_adjoint_measure = compute_electric_field_s_pol( device, grid_resolution_um, device_depth_um, device_depth_voxels, test_angle, lambda_min_um, False )

print(transmission_fwd)
print(transmission_fwd2)

# print(np.abs(Ey_fwd[-1])**2)
# print(np.abs(Ey_fwd2[-1])**2)


# plt.plot(gradient)
# plt.plot(tg)
# plt.show()


degrees_to_radians = np.pi / 180.
angle_min = 0.0 * degrees_to_radians
angle_max = 30.0 * degrees_to_radians

num_angles = 12#12
num_wavelengths = 3#10
num_pol = 2
num_optimization_goals = num_wavelengths * num_angles * num_pol

angles = np.linspace( angle_min, angle_max, num_angles )
wavelengths_um = np.linspace( lambda_min_um, lambda_max_um, num_wavelengths )

transmit_angle_low = 0.0 * degrees_to_radians
transmit_angle_high = 8.0 * degrees_to_radians

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
		# USE ADJOINT MEASURE HERE

		max_fom[ wl_idx, ang_idx, 0 ] = fom_s
		max_fom[ wl_idx, ang_idx, 1 ] = fom_p


num_iterations = 100
normalized_step_size = 0.005

fom = np.zeros( ( num_iterations, num_wavelengths, num_angles, num_pol ) )

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

	# if iter_idx >= 5:
		# normalized_step_size /= 10
		# np.save('test_device' + str(iter_idx) + '.npy', device)
		# np.save('test_grad' + str(iter_idx) + '.npy', gradients[0, :])
		# for goal_idx in range(0, num_optimization_goals):
		# 	grad_weightings[goal_idx] = 0
		# grad_weightings[0] = 1.0
		# grad_weightings[1] = 0.5

	print(fom_iter)
	print(grad_weightings)
	# print(fom)

	weighted_gradient = np.zeros( device_depth_voxels )

	for goal_idx in range( 0, num_optimization_goals ):
		weighted_gradient += grad_weightings[ goal_idx ] * gradients[ goal_idx, : ]

	np.save('current_device.npy', device)
	np.save('current_fom.npy', fom)
	step_size = normalized_step_size / np.power( 10, iter_idx / ( num_iterations - 1 ) )
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
plt.title('Transmission Goal')
plt.colorbar()
plt.subplot( 2, 3, 2 )
plt.imshow( init_transmission_s )
plt.title('Initial S Transmission')
plt.colorbar()
plt.subplot( 2, 3, 3 )
plt.imshow( init_transmission_p )
plt.title('Initial P Transmission')
plt.colorbar()
plt.subplot( 2, 3, 4 )
plt.imshow( measured_transmission_s )
plt.title('Measured S Transmission')
plt.colorbar()
plt.subplot( 2, 3, 5 )
plt.imshow( measured_transmission_p )
plt.title('Measured P Transmission')
plt.colorbar()

plt.show()

plt.plot( device )
plt.show()

print(fom)





