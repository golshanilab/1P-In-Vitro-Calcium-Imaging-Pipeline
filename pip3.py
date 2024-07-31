import cv2 # Import the OpenCV library to read video files
import os # Import the os library to create directories and handle file paths
import tifffile # Import the tifffile library to save OME-TIFF files
import numpy as np # Import the NumPy library for numerical operations
import matplotlib.pyplot as plt # Import the Matplotlib library for plotting
from suite2p.extraction import dcnv  # Import the dcnv module from Suite2p for deconvolution
from suite2p import run_s2p, default_ops # Import the necessary functions from Suite2p
from scipy.signal import find_peaks, correlate # Import the find_peaks and correlate functions from SciPy for signal processing
import pandas as pd # Import the Pandas library for data manipulation

# Path to your AVI file (replace with path)
avi_path = r"\\?\E:\2024_06_Neuron_wGlials_Astros_tesing\20240620_Astros_Neurons_MFW_Geo_96wells_widefield_YanJ\Mouse_glials\Glial_140\Glial_140_MFW\20240626_Glial140_SW1Neurons300_Day47_MFW_WF5min_96wells_A5_YanJ\20240626_Glial140_SW1Neurons300_Day47_MFW_WF5min_96wells_A5_YanJ.avi"


# Open the video file
cap = cv2.VideoCapture(avi_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Create directories for each quadrant
output_base_dir = os.path.join(os.path.dirname(avi_path), "Quadrants")
quadrants_dirs = {
    "top_left": os.path.join(output_base_dir, "top left"),
    "top_right": os.path.join(output_base_dir, "top right"),
    "bottom_left": os.path.join(output_base_dir, "bottom left"),
    "bottom_right": os.path.join(output_base_dir, "bottom right"),
}
for quadrant_dir in quadrants_dirs.values():
    if not os.path.exists(quadrant_dir):
        os.makedirs(quadrant_dir)

# Initialize a dictionary to hold lists of frames for each quadrant
quadrant_frames = {
    "top_left": [],
    "top_right": [],
    "bottom_left": [],
    "bottom_right": []
}
# Read the frames in a loop (for processing)
frame_index = 0  # Frame counter
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Reached the end of the video

    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Crop 640 pixels from the left and right
    cropped_frame = gray_frame[:, 640:-640]
    # Get the height and width of the cropped frame
    height, width = cropped_frame.shape
    # Divide into 4 quadrants
    quadrants = {
        "top_left": cropped_frame[:height//2, :width//2],
        "top_right": cropped_frame[:height//2, width//2:],
        "bottom_left": cropped_frame[height//2:, :width//2],
        "bottom_right": cropped_frame[height//2:, width//2:]
    }
    for quadrant_name, quad in quadrants.items():
        quadrant_frames[quadrant_name].append(quad)

   # Extract the base name of the AVI file without its extension
avi_base_name = os.path.basename(avi_path)
avi_base_name_without_ext = os.path.splitext(avi_base_name)[0]

# After processing all frames, write each list of quadrants to a separate OME-TIFF file
for quadrant_name, frames in quadrant_frames.items():
    quadrant_dir = quadrants_dirs[quadrant_name]
    # Construct the new file name with the original AVI base name, quadrant name, and new extension
    new_file_name = f"{avi_base_name_without_ext}_{quadrant_name}.ome.tiff"
    quadrant_path = os.path.join(quadrant_dir, new_file_name)
    # Convert list of frames to a 3D numpy array (T, Y, X)
    frames_array = np.stack(frames, axis=0)
    tifffile.imwrite(quadrant_path, frames_array, metadata={'axes': 'TYX'})

# Release the video capture object and close all OpenCV windows
cap.release()


print("processing completed")


# Define manual parameters for suite2p operations
manual_ops = {
    'tau': 5,  # fluo-4 decay time in seconds (1P)
    'fs': 30,  # Sampling frequency
    'smooth_sigma': 5,  # standard deviation in pixels of the gaussian used to smooth the phase correlation
    '1Preg': 1,  # high pass filter/taper for 1P data
    'anatomical_only': 2,  # ROI based on mean image
    'diameter': 0,  # Automatic cell diameter
    'allow_overlap': 0,  # overlap ROIs
    'maxregshiftNR': 8,  # maximum shift in pixels of a block relative to the rigid shift
    'high_pass': 6,  # running mean subtraction across time with window of size
    'use_builtin_classifier' : True, # Use the built-in classifier for ROI detection
    'nonrigid': 0,  # nonrigid registration
}

# Load default operations and update them with manual parameters
ops = default_ops()
ops.update(manual_ops)  # Update default ops with manual parameters

# Print all the parameters used
for key, value in ops.items():
    print(f"{key}: {value}")

# Iterate over each quadrant to process the generated OME-TIFF files
for quadrant_name in quadrant_frames.keys():
    quadrant_dir = quadrants_dirs[quadrant_name]
    new_file_name = f"{avi_base_name_without_ext}_{quadrant_name}.ome.tiff"
    quadrant_path = os.path.join(quadrant_dir, new_file_name)
    
    # Update ops with specific file information
    ops['save_path0'] = quadrant_dir  # Output directory
    ops['tiff_list'] = [quadrant_path]  # Input file
    ops['data_path'] = [quadrant_dir]  # Data path
    
    # Run suite2p with the specified ops
    ops_out = run_s2p(ops=ops)
    print(f"Suite2p processing completed for {quadrant_name}")


    # Define the path to the plane0 directory within the quadrant folder
    plane0_dir = os.path.join(quadrant_dir, 'suite2p' ,'plane0')

    # Load the necessary files
    F = np.load(os.path.join(plane0_dir, 'F.npy'))
    Fneu = np.load(os.path.join(plane0_dir, 'Fneu.npy'))
    iscell = np.load(os.path.join(plane0_dir, 'iscell.npy'))
    stat = np.load(os.path.join(plane0_dir, 'stat.npy'), allow_pickle=True)

    # Filter the ROIs with a probability below 0.05
    probability_threshold = 0.05
    filtered_indices = iscell[:, 1] >= probability_threshold

    # Filter the data
    F_filtered = F[filtered_indices]
    Fneu_filtered = Fneu[filtered_indices]
    iscell_filtered = iscell[filtered_indices]
    stat_filtered = [stat[i] for i in range(len(stat)) if filtered_indices[i]]

    # Replace the first value of each remaining ROI with 1
    iscell_filtered[:, 0] = 1

    # Create a new folder for the filtered data
    filtered_output_dir = os.path.join(quadrant_dir, 'filtered')
    os.makedirs(filtered_output_dir, exist_ok=True)

    # Save the filtered data
    np.save(os.path.join(filtered_output_dir, 'F.npy'), F_filtered)
    np.save(os.path.join(filtered_output_dir, 'Fneu.npy'), Fneu_filtered)
    np.save(os.path.join(filtered_output_dir, 'iscell.npy'), iscell_filtered)
    np.save(os.path.join(filtered_output_dir, 'stat.npy'), stat_filtered)

    print(f"Filtered data for cell prob saved successfully for {quadrant_name}")  
    
    # Load the filtered data
    F = np.load(os.path.join(filtered_output_dir, 'F.npy'))
    Fneu = np.load(os.path.join(filtered_output_dir, 'Fneu.npy'))
    iscell = np.load(os.path.join(filtered_output_dir, 'iscell.npy'))

    # Subtract neuropil signal from raw fluorescence signal
    F_corrected = F - ops['neucoeff'] * Fneu

    # Baseline operation
    F_corrected = dcnv.preprocess(
        F=F_corrected,
        baseline=ops['baseline'],
        win_baseline=ops['win_baseline'],
        sig_baseline=ops['sig_baseline'],
        fs=ops['fs'],
        prctile_baseline=ops['prctile_baseline']
    )

    # Calculate the deconvolved signals using Suite2p's deconvolution function
    deconvolved_signals = dcnv.oasis(F=F_corrected, batch_size=ops['batch_size'], tau=ops['tau'], fs=ops['fs'])

    num_cells = deconvolved_signals.shape[0] 

    print(f"Deconvolution for {quadrant_name}")


    # Print the total number of cells
    num_cells = deconvolved_signals.shape[0]
    print(f"Total number of cells: {num_cells}")
    
    # Load spatial coordinates of neurons from stat.npy
    stat = np.load(os.path.join(filtered_output_dir, 'stat.npy'), allow_pickle=True)

    # Extract neuron positions
    neuron_positions = np.array([neuron['med'] for neuron in stat])
    average_activity_levels = np.mean(deconvolved_signals, axis=1)

    # Create a scatter plot of neuron positions
    plt.figure(figsize=(10, 8))
    plt.scatter(neuron_positions[:, 0], neuron_positions[:, 1], c=average_activity_levels, cmap='hot', s=50)
    plt.colorbar(label='Activity Level')
    plt.title('Spatial Map of Neuron Activity')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    output_file_path = os.path.join(os.path.dirname(filtered_output_dir), 'scatter_plot.png')
    plt.savefig(output_file_path)
    plt.close()


    # Function to calculate mean amplitudes and firing frequencies for each cell
    def calculate_mean_amplitudes_and_frequencies(signals, fs):
        num_cells = signals.shape[0] 
        mean_amplitudes = []
        firing_frequencies = []
    
        for i in range(num_cells):
            # Find local maxima (spikes)
            peaks, _ = find_peaks(signals[i])
            peak_values = signals[i][peaks]
        
            # Calculate mean amplitude of local maxima
            mean_amplitude = np.mean(peak_values)
            mean_amplitudes.append(mean_amplitude)
        
            # Calculate the duration of the signal in seconds
            signal_duration = len(signals[i]) / fs       

            # Calculate firing frequency (spikes per second)
            firing_frequency = len(peaks) / signal_duration
            firing_frequencies.append(firing_frequency)
        
        return mean_amplitudes, firing_frequencies

    # Calculate mean amplitudes and firing frequencies for all cells
    mean_amplitudes, firing_frequencies = calculate_mean_amplitudes_and_frequencies(deconvolved_signals, ops['fs'])

    # Create a DataFrame with the results
    data = {
        'Cell Number': range(1, len(deconvolved_signals) + 1),
        'Mean Amplitude': mean_amplitudes,
        'Firing Frequency': firing_frequencies
    }
    df = pd.DataFrame(data)

    # Save the DataFrame to an Excel file
    excel_file_path = os.path.join(os.path.dirname(filtered_output_dir), 'mean_amplitudes_and_firing_frequencies.xlsx')
    df.to_excel(excel_file_path, index=False)

    num_cells = len(mean_amplitudes)

    # Create a bar graph of the mean amplitudes
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(1, num_cells + 1), mean_amplitudes)
    plt.xlabel('Cell Number')
    plt.ylabel('Mean Amplitude of Local Maxima')
    plt.title('Mean Amplitudes of Local Maxima for Each Cell')

    # Annotate each bar with the mean amplitude value
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

    # Save the plot in the same directory as the input file
    output_file_path = os.path.join(os.path.dirname(filtered_output_dir), 'mean_amplitudes_plot.png')
    plt.savefig(output_file_path)
    plt.close()

    # Create a bar graph of the firing frequencies
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(1, num_cells + 1), firing_frequencies)
    plt.xlabel('Cell Number')
    plt.ylabel('Firing Frequency (Hz)')
    plt.title('Firing Frequencies for Each Cell')

    # Annotate each bar with the firing frequency value
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

    # Save the plot in the same directory as the input file
    output_file_path = os.path.join(os.path.dirname(filtered_output_dir), 'firing_frequencies_plot.png')
    plt.savefig(output_file_path)
    plt.close()

    # Raster plot
    plt.figure(figsize=(10, 6))
    for i, peak in enumerate([find_peaks(deconvolved_signals[i])[0] for i in range(deconvolved_signals.shape[0])]):
        plt.vlines(peak, i + 0.5, i + 1.5)
    plt.xlabel('Frame')
    plt.ylabel('Neuron')
    plt.title('Raster Plot of Neuron Firing')

    # Save the rastermap in the same folder
    rastermap_path = os.path.join(os.path.dirname(filtered_output_dir), 'rastermap.png')
    plt.savefig(rastermap_path)
    plt.close()

    # Cross-correlation matrix
    num_neurons = deconvolved_signals.shape[0]
    correlation_matrix = np.zeros((num_neurons, num_neurons))

    for i in range(num_neurons):
        for j in range(num_neurons):
            if i != j:
                correlation_matrix[i, j] = np.max(correlate(deconvolved_signals[i], deconvolved_signals[j]))

    # Convert the correlation matrix to a pandas DataFrame
    correlation_df = pd.DataFrame(correlation_matrix)

    # Path to the existing Excel file
    excel_file_path = os.path.join(os.path.dirname(filtered_output_dir), 'mean_amplitudes_and_firing_frequencies.xlsx')

    # Save the DataFrame to a new sheet in the same Excel file
    with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='a') as writer:
        correlation_df.to_excel(writer, sheet_name='CrossCorrelationMatrix', index=False)


    plt.figure(figsize=(8, 8))
    plt.imshow(correlation_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Cross-Correlation Matrix')
    plt.xlabel('Neuron')
    plt.ylabel('Neuron')

    # Save the matrix as an image
    image_path = os.path.join(os.path.dirname(filtered_output_dir), 'cross_correlation_matrix.png')
    plt.savefig(image_path)
    plt.close()

    print(f"Data processed for {quadrant_name}")


print("All analysis completed")


