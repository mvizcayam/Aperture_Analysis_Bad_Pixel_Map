'''
code originally written by Owen Leonard
modified by Alex Gleason

This file takes in fits files and calculates the size of the smallest circle that includes the given percentage of light, usually 50%. 
It optimizes for the size of the circle and the location of the center of the circle. Optimization includes subdividing pixels into 
smaller chunks so that more accurate measurements are possible.

The basic workflow of the program is as follows, the script loads files and creates separate threads for each file up to the maximum. 
The DOAstarfinder library is used to locate the centroid of the brightest spot in the image which is used as the initial guess.
A double series of minimization algorithms is run. For each location the radius is optimized and then a new location is chosen. 
Eventually this converges to the location with the smallest radius. Run time is often on the order of an hour.
'''

import os
import numpy as np
from astropy.io import fits
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import concurrent
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from astropy.stats import SigmaClip
from photutils.background import Background2D, BiweightLocationBackground, ModeEstimatorBackground, MedianBackground
from photutils.detection import DAOStarFinder
from astropy.visualization import ImageNormalize, SqrtStretch
import math
from numba import jit
import glob
import multiprocessing

# OPTIMIZATION


@jit(nopython=True)
def calculate_light_through_aperture_vectorized(data, center, radius, subpixel_offsets, pixel_centers_x, pixel_centers_y):
    # Define the dimensions of data
    num_rows, num_cols = data.shape
    
    # Initialize total light intensity
    total_light_intensity = 0.0
    
    # Loop over each pixel
    for i in range(num_rows):
        for j in range(num_cols):
            # Calculate distance from pixel center to centroid
            distance_to_centroid = np.sqrt((pixel_centers_x[i, j] - center[0])**2 + (pixel_centers_y[i, j] - center[1])**2)
            
            # Check if pixel is fully within the circle
            if distance_to_centroid <= (radius - 1):
                total_light_intensity += data[i, j]
            elif distance_to_centroid <= (radius + 1):
                # Calculate subpixel centers
                subpixel_intensity = 0.0
                for k in range(subpixel_offsets.shape[0]):
                    for l in range(subpixel_offsets.shape[0]):
                        subpixel_center_x = pixel_centers_x[i, j] + subpixel_offsets[k]
                        subpixel_center_y = pixel_centers_y[i, j] + subpixel_offsets[l]
                        
                        # Calculate distance from subpixel center to centroid
                        subpixel_distance_to_centroid = np.sqrt((subpixel_center_x - center[0])**2 + (subpixel_center_y - center[1])**2)
                        
                        # Check if subpixel is within the circle
                        if subpixel_distance_to_centroid <= radius:
                            subpixel_intensity += 1.0
                
                # Calculate overlap fraction
                overlap_fraction = subpixel_intensity / (subpixel_offsets.shape[0] ** 2)
                
                # Update total light intensity
                total_light_intensity += data[i, j] * overlap_fraction
    
    return total_light_intensity


# Aims to find the radius within which a specified percentage of the total light 
# in the image is contained, for a given center.
def radius_for_given_percentage(center, data, target_percentage, centroid):
# Calculates the difference between the desired light percentage and the actual 
# light percentage within a circle of a given radius. The minimize function 
# from 'scipy.optimize' is used to adjust the radius to minimize this difference, 
# effectively finding the radius that meets the light percentage criterion.

    def get_centers():
        # Setup for vectorization
        subpixel_size = 1.0 / subdivisions
        subpixel_offsets = np.linspace(-0.5 + 0.5 * subpixel_size, 0.5 - 0.5 * subpixel_size, subdivisions)

        # Meshgrid of pixel center coordinates
        pixel_centers_x, pixel_centers_y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))

        return subpixel_offsets, pixel_centers_x, pixel_centers_y

    def objective_radius(radius):
        if radius < 0:
            return np.inf
        light_in_circle = calculate_light_through_aperture_vectorized(data, center, radius, subpixel_offsets, pixel_centers_x, pixel_centers_y)
        current_percentage = (light_in_circle / total_intensity) * 100
        objective_value = abs(current_percentage - target_percentage)
        print(f"Radius: {radius}, Light Percentage: {current_percentage}%, Objective Value: {objective_value}, Center: {center}")
        return objective_value
    
    subpixel_offsets, pixel_centers_x, pixel_centers_y = get_centers()

    #reference intensity
    total_intensity = calculate_light_through_aperture_vectorized(data, center, 75, subpixel_offsets, pixel_centers_x, pixel_centers_y)

    #starts initial guess of radius as distance between center and the centroid
    guessed_radius = math.sqrt((center[0]-centroid[0])**2+(center[1]-centroid[1])**2)
    if guessed_radius <= initial_radius:
        guessed_radius = initial_radius
    #result = minimize(objective_radius, x0=[guessed_radius], method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-4})
    result = minimize_scalar(objective_radius, method='brent', bracket=(1, 10), options={'xtol': 1e-3})
    optimized_radius = result.x
    return optimized_radius



# Optimizes the position of the center such that the smallest possible radius
# contains the target percentage of the total light.
def optimize_center_and_radius(data, target_percentage, background_rms_median, bad_pixel_mask):
    # Setting the bad pixel values in data to the avg value
    avg_image_intensity = np.mean(data) 
    data[bad_pixel_mask] = avg_image_intensity  
    
    # Use DAOStarFinder to find 'stars' in the image
    daofind = DAOStarFinder(fwhm=3, threshold=1.5*background_rms_median)
    sources = daofind(data)

    if sources is None or len(sources) == 0:
        raise ValueError("No sources found. Consider adjusting DAOStarFinder parameters.")

    # Identify brightest source
    brightest_source = sources[np.argmax(sources['peak'])]

    # Coordinates for brightest source
    centroid_x, centroid_y = brightest_source['xcentroid'], brightest_source['ycentroid']
    
    # Working with bad pixels
    bad_pixel_dist = calculate_bad_pixel_dist(bad_pixel_mask, centroid_x, centroid_y)
    closest_bad_pixel_dist = np.min(bad_pixel_dist)
    
    # Finding if closest bad pixel is within the cutoff value so image is not processed 
    # (just return (0,0) for the center coord and a radius = 0)
    if closest_bad_pixel_dist <= bad_pix_dist_cutoff:
        global num_files_not_processed
        with num_files_not_processed.get_lock():
            num_files_not_processed.value += 1
        print("The closest bad pixel is",closest_bad_pixel_dist,"pixels from the source. "
              "This distance is within the bad pixel distance cutoff value of",bad_pix_dist_cutoff,"pixels.")
        return ((0,0),0)


    initial_center = (centroid_x, centroid_y)
    
# Uses the previously defined radius_for_percentage function to determine
# the required radius for a given center. The minimize function is then used to Ss
# adjust the center coordinates to minimize this radius, finding the most
# efficient center position.

    bounds = [(initial_center[0]-50, initial_center[0]+50), (initial_center[1]-50, initial_center[1]+50)]
    args = (data, target_percentage, initial_center)
    result = minimize(radius_for_given_percentage, x0=initial_center, args=args, bounds=bounds, method='L-BFGS-B', options={'ftol': 1e-3, 'gtol': 1e-3, 'eps': 0.005})
    optimized_center = result.x

    optimal_radius = radius_for_given_percentage(optimized_center, data, target_percentage, initial_center)
    print(optimal_radius)
    return optimized_center, optimal_radius

def calculate_bad_pixel_dist(bad_pixel_mask, centroid_x, centroid_y):
    
    x_pixel, y_pixel = np.meshgrid(np.arange(bad_pixel_mask.shape[1]), np.arange(bad_pixel_mask.shape[0]))
    
    #Finding the x & y coord. of the bad pixels
    bad_pixel_x = x_pixel[bad_pixel_mask]
    bad_pixel_y = y_pixel[bad_pixel_mask]
    
    # Finding # of bad pixels
    num_bad_pixels = len(bad_pixel_x)
    print("Based on the chi square cutoff value of",chi_square_cutoff,"and the dead pixel cutoff value of",dead_pixel_cutoff,
          ",", num_bad_pixels,"bad pixels have been detected.")
    
    #Calculating the distance to all the bad pixels
    bad_pixel_dist = np.sqrt((bad_pixel_x - centroid_x)**2 + (bad_pixel_y - centroid_y)**2)
    return bad_pixel_dist
    

# DATA PROCESSING

# Fits handling
def read_fits_data_every_twentieth(folder_path):
    data_list = []
    file_names = sorted([f for f in os.listdir(folder_path) if f.endswith('.fits')])
    for i, file_name in enumerate(file_names):
        if i % 20 == 0:  # Process every 20th file
            file_path = os.path.join(folder_path, file_name)
            with fits.open(file_path) as hdul:
                data = hdul[0].data
                data_list.append(data)
    return data_list, data


def process_data(data, target_percentage, bad_pixel_mask):
    # Background estimation and subtraction
    sigma_clip = SigmaClip(sigma=3)
    bkg_estimator = BiweightLocationBackground()
    bkg = Background2D(data, (50,50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    background_rms_median = bkg.background_rms_median + 1e-4
    data_subtracted = data - bkg.background
    # Clip so no negatives
    if allow_neg == False:
        data_clipped = np.clip(data_subtracted, a_min=0, a_max=None)
    else:
        data_clipped = data_subtracted

    # Processing
    optimal_center, optimal_radius = optimize_center_and_radius(data_clipped, target_percentage, background_rms_median, bad_pixel_mask)
    optimal_center = np.nan_to_num(optimal_center, nan=0.0)
    is_center_valid = not (optimal_center == 0.0).all()

    return optimal_center, optimal_radius, is_center_valid

# Parallel processing
def parallel_optimization(data_list, target_percentage, bad_pixel_mask, max_workers=15):
    optimal_centers = []
    optimal_radii = []
    opt_cenx_f = open(output_path + f'/{target_percentage}perc_{observation_log}_opt_cenx_list.txt', 'w')
    opt_ceny_f = open(output_path + f'/{target_percentage}perc_{observation_log}_opt_ceny_list.txt', 'w')
    opt_radii_f = open(output_path + f'/{target_percentage}perc_{observation_log}_opt_radii_list.txt', 'w')
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_data, data, target_percentage, bad_pixel_mask) for data in data_list]
        for future in concurrent.futures.as_completed(futures):
            optimal_center, optimal_radius, is_center_valid = future.result()
            if is_center_valid:
                optimal_centers.append(optimal_center)
                opt_cenx_f.write(str(optimal_center[0])+'\n')
                opt_ceny_f.write(str(optimal_center[1])+'\n')
                optimal_radii.append(optimal_radius)
                opt_radii_f.write(str(optimal_radius)+'\n')
    opt_cenx_f.close
    opt_ceny_f.close
    opt_radii_f.close
    

    return np.array(optimal_centers), np.array(optimal_radii)


# ANALYSIS


def plt_opt_radii_hist_prog(optimal_radii, observation_log, target_percentage, output_path, place):  
    
    filename=f'{target_percentage}perc_{observation_log}_opt_radii_hist_prog.jpg'  
    std_dev = np.std(optimal_radii)
    data_range = np.ptp(optimal_radii)
    avg_rad = np.mean(optimal_radii)

    plt.figure(figsize=(10, 6))
    plt.hist(optimal_radii, bins=np.arange(0,140,.5), color='red', edgecolor='black')
    plt.yticks(np.arange(0,190,10))
    plt.xticks(np.arange(0,150,10)) 
    plt.title(f'{target_percentage}% Capture Optimal Radii ({place})')
    plt.xlabel('Radius')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.axvline(avg_rad, color='green', linestyle='dashed', linewidth=.75)

    plt.text(0.05, 0.95, f'Standard Deviation: {std_dev:.2f}\nRange: {data_range:.2f}\n Avg.: {avg_rad:.1f}',
            transform=plt.gca().transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
    plt.savefig(os.path.join(output_path, filename), format='jpg', dpi=300)
    plt.close() 


def plt_opt_radii_hist_ind(optimal_radii, observation_log, target_percentage, output_path, place):

    filename=f'{target_percentage}perc_{observation_log}_opt_radii_hist_ind.jpg'
    std_dev = np.std(optimal_radii)
    data_range = np.ptp(optimal_radii)
    avg_rad = np.mean(optimal_radii)

    plt.figure(figsize=(10, 6))
    plt.hist(optimal_radii, bins=np.arange(np.min(optimal_radii),np.max(optimal_radii),.25), color='red', edgecolor='black')
    plt.title(f'{target_percentage}% Capture Optimal Radii ({place})')
    plt.xlabel('Radius')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.axvline(avg_rad, color='green', linestyle='dashed', linewidth=.75)

    plt.text(0.05, 0.95, f'Standard Deviation: {std_dev:.2f}\nRange: {data_range:.2f}\n Avg.: {avg_rad:.1f}',
            transform=plt.gca().transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
    plt.savefig(os.path.join(output_path, filename), format='jpg', dpi=300)
    plt.close() 

def plt_opt_cen_hist_prog(optimal_centers, observation_log, target_percentage, output_path, place):

    filename = f"{target_percentage}perc_{observation_log}_opt_cen_hist_prog.jpg"
    x_centers = optimal_centers[:, 0]  
    y_centers = optimal_centers[:, 1]  

    plt.figure(figsize=(10, 6))
    plt.hist2d(x_centers, y_centers, bins=[np.arange(50,275,.5),np.arange(50,175, .5)], cmap='gist_heat_r', norm = mcolors.Normalize(vmin=0, vmax=17))
    plt.colorbar(label='Frequency')
    plt.title(f'{target_percentage}% Capture Optimal Centers ({place})')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)

    std_dev_x = np.std(x_centers)
    std_dev_y = np.std(y_centers)
    range_x = np.ptp(x_centers)
    range_y = np.ptp(y_centers)
    avg_x = np.mean(x_centers)
    avg_y = np.mean(y_centers)
    
    plt.text(0.05, 0.95, f'X Std Dev: {std_dev_x:.2f}, Y Std Dev: {std_dev_y:.2f}\nX Range: {range_x:.2f}, Y Range: {range_y:.2f}\n Avg: ({avg_x:.1f},{avg_y:.1f})',
             transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
    plt.savefig(os.path.join(output_path, filename), format='jpg', dpi=300)
    plt.close()


def plt_opt_cen_hist_ind(optimal_centers, observation_log, target_percentage, output_path, place):
    filename = f"{target_percentage}perc_{observation_log}_opt_cen_hist_ind.jpg"
    x_centers = optimal_centers[:, 0] - np.mean(optimal_centers[:, 0])
    y_centers = optimal_centers[:, 1] - np.mean(optimal_centers[:, 1])

    # Adjust bin ranges to center around the mean
    bin_range_x = np.max(np.abs(x_centers))
    bin_range_y = np.max(np.abs(y_centers))
    bins_x = np.arange(-bin_range_x, bin_range_x + 0.5, 0.5)
    bins_y = np.arange(-bin_range_y, bin_range_y + 0.5, 0.5)

    plt.figure(figsize=(10, 6))
    plt.hist2d(x_centers, y_centers, bins=[bins_x, bins_y], cmap='gist_heat_r', norm=mcolors.Normalize(vmin=0, vmax=17))
    plt.colorbar(label='Frequency')
    plt.title(f'{target_percentage}% Capture Optimal Centers ({place})')
    plt.xlabel('X (Pixels)')
    plt.ylabel('Y (Pixeels)')
    plt.grid(True)
    plt.axhline(0, color='gray', linestyle='--')  
    plt.axvline(0, color='gray', linestyle='--')  
    plt.axis([-10, 10, -10, 10])

    std_dev_x = np.std(optimal_centers[:, 0])
    std_dev_y = np.std(optimal_centers[:, 1])
    range_x = np.ptp(optimal_centers[:, 0])
    range_y = np.ptp(optimal_centers[:, 1])


    plt.text(0.05, 0.95, f'X Std Dev: {std_dev_x:.2f}, Y Std Dev: {std_dev_y:.2f}\nX Range: {range_x:.2f}, Y Range: {range_y:.2f}',
             transform=plt.gca().transAxes,  
             fontsize=15, verticalalignment='top', bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))

    plt.savefig(os.path.join(output_path, filename), format='jpg', dpi=750)
    plt.close()


def main(folder_path, output_path, chi_square_map_file, dead_pixels_map_file, max_workers=15):
    #data_list, data = read_fits_data_every_twentieth(folder_path)

    data_list = []
    file_names = os.listdir(folder_path)
    for i, file_name in enumerate(file_names):
        if (file_name[-5:] == '.fits'):
            file_path = os.path.join(folder_path, file_name)
            with fits.open(file_path) as hdul:
                data = hdul[0].data
                data_list.append(data)
                
    print("\n")
    print("data_list shape: ", np.shape(data_list))
    print("data shape: ", np.shape(data))
    
    # Getting the pan and tilt from camera settings text file
    cam_settings_txt_list = glob.glob(folder_path + '\\*.CameraSettings.txt')
    cam_settings_txt_file = cam_settings_txt_list[0]
    print("Camera Settings file: ",cam_settings_txt_list) 
    
    # Getting the pan and tilt
    with open(cam_settings_txt_file, 'r') as file:
        lines = [line.strip() for line in file]
        pan = int(lines[3].split('=')[1])
        tilt = int(lines[4].split('=')[1])
    print("Pan: ",pan)
    print("Tilt: ", tilt)
        
    # Load the chi square map from the .npy file   
    chi_square_map_file = np.load(chi_square_map_file)  
    # Getting the chi square map in the image dimensions
    chi_square_map = chi_square_map_file[pan:pan + data.shape[1], tilt:tilt + data.shape[0]] 
    chi_square_map = chi_square_map.T # setting chi_square_map shape to match data shape: (240,320) 
    print("chi_square_map shape: ", np.shape(chi_square_map))
    
    # Load the dead pixels map from the .npy file   
    dead_pixels_map_file = np.load(dead_pixels_map_file)  
    # Getting the chi square map in the image dimensions
    dead_pixels_map = dead_pixels_map_file[pan:pan + data.shape[1], tilt:tilt + data.shape[0]] 
    dead_pixels_map = dead_pixels_map.T # setting dead_pixels_map shape to match data shape: (240,320) 
    print("dead_pixels_map shape: ", np.shape(dead_pixels_map))
    
    # Creating bad pixel mask (boolean array)
    bad_pixel_mask = np.logical_or(chi_square_map >= chi_square_cutoff, dead_pixels_map >= dead_pixel_cutoff)


    optimal_centers, optimal_radii = parallel_optimization(data_list, target_percentage, bad_pixel_mask, max_workers)
    print(num_files_not_processed.value, "files were not processed due to a bad pixel exceeding the cutoff distance of", bad_pix_dist_cutoff,
          "pixels (too close to the source).")
    #optimal_centers, optimal_radii, is_valid = process_data(data_list[0], target_percentage)
    plt_opt_radii_hist_prog(optimal_radii, observation_log, target_percentage, output_path, place)
    plt_opt_cen_hist_prog(optimal_centers, observation_log, target_percentage, output_path, place)
    plt_opt_radii_hist_ind(optimal_radii, observation_log, target_percentage, output_path, place)
    plt_opt_cen_hist_ind(optimal_centers, observation_log, target_percentage, output_path, place)


observation_log = 'Mariana_0824' #name & date of observation
place = 'Vega - P:0.01, I:0.0095, D:0.00001'
target_percentage = 10
initial_radius = 1.5
subdivisions = 200
#controls whether pixel values are allowed to be negative, wierd things start to happen if so but it can help eliminate background total from low SNR images
allow_neg = True
chi_square_cutoff = 1000
dead_pixel_cutoff = 5
bad_pix_dist_cutoff = 10


# Starting counter for # of files not processed
num_files_not_processed = multiprocessing.Value('I', 0)



if __name__ == "__main__":
    folder_path = r"C:\Users\medin\QuantAstro proj\SharpCap Captures\Vega Observations\2024-08-23\Vega_3ms_300gain_320x240_0.01-p_0.0095-i_0.00001-d\23_59_57"
    output_path = r"C:\Users\medin\QuantAstro proj\SharpCap Captures\Vega Observations\2024-08-23\Vega_3ms_300gain_320x240_0.01-p_0.0095-i_0.00001-d\Double Opt (bad pixel map)\10% light, CSC - 1000, BPDC - 10 pix, DPC - 5"
    chi_square_map_file = r"C:\Users\medin\QuantAstro proj\chi_square_gaussian_fit.npy"
    dead_pixels_map_file = r"C:\Users\medin\QuantAstro proj\dead_pixels_hist.npy"
    
    main(folder_path, output_path, chi_square_map_file, dead_pixels_map_file)

'Vega - P:0.06, I:0.03, D:0.0'

"""
Art. Turb. Initial Test Optimal radii guesses:

5% - cc:  1.3         gc: 2.1
10% - cc: 1.7         gc: 3.3
15% - cc: 2.4         gc: 4.7
25% - cc: 3.2         gc: 6.8
35% - cc: 4.6         gc: 9.4
50% - cc: 6.6         gc: 14.5
60% - cc: 8.8         gc: 21.2
75% - cc: 15.1       gc: 75.8
90% - cc: 50          gc: 132
"""
