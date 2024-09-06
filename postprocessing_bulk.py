import os
import cv2
import ast
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define necessary functions
def process_series(data_list):
    if not data_list:
        return pd.Series()  # Handle empty data
    df = pd.concat(data_list, axis=1)
    df = df.loc[~df.index.duplicated(keep='first')]
    # df = df.groupby(df.index).mean()  # Handle duplicates by averaging
    df = df.sort_index()
    return df.mean(axis=1)

def average_polar_data(angles, radii):
    x, y = radii * np.cos(angles), radii * np.sin(angles)
    return np.arctan2(np.mean(y), np.mean(x)), np.sqrt(np.mean(x)**2 + np.mean(y)**2)

def plot_length_changes(cluster_label, times, lengths, base_folder):
    plt.figure(figsize=(10, 4))
    plt.plot(times, lengths, 'b-o', label='Length Changes')
    plt.title(f'Neurite Length Changes Over Time (Cluster {cluster_label})')
    plt.xlabel('Time (hours)')
    plt.ylabel('Length Changes (μm)')
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='gray', linewidth=0.8)
    plt.savefig(os.path.join(base_folder, f'Length_Changes_Cluster_{cluster_label}.png'))
    plt.close()

def plot_polar_displacement(cluster_label, angles, radii, base_folder):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 6))
    ax.plot(angles, radii, 'b-', alpha=0.75, label='Displacement')
    ax.set_title(f'Neurite Centroid Displacement (Cluster {cluster_label})')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(base_folder, f'Centroid_Displacement_Cluster_{cluster_label}.png'))
    plt.close()

def zoom_into_centroid(image, grid_mask, centroid, tracing_points, zoom_size=250, padding_color=(0, 0, 0)):
    x, y = int(centroid[0]), int(centroid[1])
    h, w = image.shape[:2]

    # Padding the image to ensure the zoom does not go out of bounds
    padded_image = cv2.copyMakeBorder(image, zoom_size, zoom_size, zoom_size, zoom_size, cv2.BORDER_CONSTANT, value=padding_color)
    padded_grid_mask = cv2.copyMakeBorder(grid_mask, zoom_size, zoom_size, zoom_size, zoom_size, cv2.BORDER_CONSTANT, value=0)  # Assuming grid_mask is binary

    # Adjust centroid position for the padding
    padded_x = x + zoom_size
    padded_y = y + zoom_size

    # Define the ROI in the padded image
    start_x = padded_x - zoom_size
    end_x = padded_x + zoom_size
    start_y = padded_y - zoom_size
    end_y = padded_y + zoom_size

    # Extract the zoomed area
    zoomed_image = padded_image[start_y:end_y, start_x:end_x].copy()
    zoomed_grid_mask = padded_grid_mask[start_y:end_y, start_x:end_x].copy()

    # Adjust tracing points to the new coordinates in the padded image
    adjusted_tracing_points = [(int(point[0] - start_x + zoom_size), int(point[1] - start_y + zoom_size)) for point in tracing_points]
    mask = np.zeros(zoomed_image.shape[:2], dtype=np.uint8)  # Use only the spatial dimensions (height, width)

    # Draw the tracing on the zoomed image
    for i in range(1, len(adjusted_tracing_points)):
        cv2.line(zoomed_image, adjusted_tracing_points[i-1], adjusted_tracing_points[i], (0, 255, 0), 2)
        cv2.line(mask, adjusted_tracing_points[i-1], adjusted_tracing_points[i], 255, 4)

    return zoomed_image, zoomed_grid_mask, adjusted_tracing_points, mask

def determine_interaction(zoomed_image, grid_mask, neurite_mask):
    # Ensure same dimensions and binary thresholds for masks
    _, zoomed_grid_mask = cv2.threshold(grid_mask, 127, 255, cv2.THRESH_BINARY)

    # Calculate the intersection
    intersection = cv2.bitwise_and(zoomed_grid_mask, neurite_mask)

    # Debugging: Check the intersection output
    if np.sum(intersection) == 0:
        print("No intersection detected.")

    interaction_image = cv2.cvtColor(zoomed_image, cv2.COLOR_GRAY2BGR) if len(zoomed_image.shape) == 2 else zoomed_image.copy()
    interaction_image[intersection > 0] = [0, 255, 0]  # Highlight intersections in green
    intersection_area = np.sum(intersection > 0)
    total_neurite_area = np.sum(neurite_mask > 0)

    # Determine the interaction type based on the area of intersection
    if intersection_area > 0:
        if intersection_area >= 0.5 * total_neurite_area:
            return 'on_top', interaction_image
        else:
            return 'partially_touching', interaction_image
    else:
        return 'flat', interaction_image
    

def create_video_for_cluster(cluster_id, df_cluster, frames_folder, output_folder, interaction_dirs, grid_mask, zoom_size=250, frame_rate=2):
    video_writers = {}
    interaction_data = []

    for itype in interaction_dirs:
        video_path = os.path.join(interaction_dirs[itype], f"cluster_{cluster_id}.mp4")
        print(f"Video path for {itype}: {video_path}")  # Debugging output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writers[itype] = cv2.VideoWriter(video_path, fourcc, frame_rate, (zoom_size * 2, zoom_size * 2))

    for _, row in df_cluster.iterrows():
        frame_path = row['frame_path']
        image = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Failed to read image at {frame_path}")
            continue

        centroid = (int(row['centroid_x']), int(row['centroid_y']))
        tracing_points_str = row['tracing_points']

        try:
            tracing_points = ast.literal_eval(tracing_points_str)
            if not isinstance(tracing_points, list) or not all(isinstance(point, list) and len(point) == 2 for point in tracing_points):
                raise ValueError("Invalid tracing points format")
        except ValueError as e:
            print(f"Error parsing tracing points for frame {frame_path}: {e}")
            continue

        zoomed_image, zoomed_grid_mask, adjusted_tracing_points, neurite_mask = zoom_into_centroid(image, grid_mask, centroid, tracing_points)

        interaction_type, interaction_image = determine_interaction(zoomed_image, zoomed_grid_mask, neurite_mask)
        print(f"Interaction for frame {frame_path}: {interaction_type}")  # Debugging output
        video_writers[interaction_type].write(zoomed_image)

        # Collecting data for DataFrame
        interaction_data.append({
            'frame_path': row['frame_path'],
            'cluster_id': cluster_id,
            'interaction_type': interaction_type,
            'shape': 'control' if interaction_type == 'flat' else row['shape'],
            'length': row['length_microns'],
            'velocity': row['velocity'],
            'pitch': row['pitch'],
            'centroid_x': row['centroid_x'],
            'centroid_y': row['centroid_y'],
            'time': row['time_hours']
        })

    for writer in video_writers.values():
        writer.release()

    return pd.DataFrame(interaction_data)

def plot_polar_angle_length_changes(cluster_label, angles, lengths, base_folder):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 6))
    ax.plot(angles, lengths, 'b-', alpha=0.75, label='Length Changes')
    ax.set_title(f'Neurite Angles and Lengths Over Time (Cluster {cluster_label})')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(base_folder, f'Angles_Lengths_Cluster_{cluster_label}.png'))
    plt.close()

def plot_velocity_changes(label, group, plots_dir):
    velocities = group['velocity'].dropna()
    if velocities.empty:
        print(f"No velocities to plot for label {label}.")
        return

    times = group['time_hours'].loc[velocities.index]
    plt.figure(figsize=(10, 4))
    plt.plot(times, velocities, 'b-o', label='Velocity Changes')
    plt.title(f'Velocity Changes Over Time (Cluster {label})')
    plt.xlabel('Time (hours)')
    plt.ylabel('Velocity (μm/hour)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'Velocity_Changes_Cluster_{label}.png'))
    plt.close()

def plot_combined_average_data(avg_length, base_folder, filename):
    plt.figure(figsize=(10, 4))
    plt.plot(avg_length.index, avg_length.values, 'b-o', label='Average Length Changes')
    plt.title('Average Neurite Length Changes Over Time')
    plt.xlabel('Time (hours)')
    plt.ylabel('Length Changes (μm)')
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='gray', linewidth=0.8)
    plt.savefig(os.path.join(base_folder, filename))
    plt.close()

# Function to calculate average velocity
def plot_combined_average_velocity(avg_velocity, base_folder, filename):
    plt.figure(figsize=(10, 4))
    plt.plot(avg_velocity.index, avg_velocity.values, 'b-o', label='Average Velocity Changes')
    plt.title('Average Neurite Velocity Changes Over Time')
    plt.xlabel('Time (hours)')
    plt.ylabel('Velocity (μm/min)')
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='gray', linewidth=0.8)
    plt.savefig(os.path.join(base_folder, filename))
    plt.close()

def plot_combined_polar_data(avg_angles, avg_radii, base_folder, filename):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 6))
    ax.plot(avg_angles, avg_radii, 'b-', alpha=0.75, label='Average Displacement')
    ax.set_title('Average Neurite Directionality Over Time')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(base_folder, filename))
    plt.close()

def preprocess_data(df):
    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Initialize columns for additional metrics
    df['velocity'] = 0
    df['angle_displacement'] = 0
    df['radii_displacement'] = 0

    df['velocity'] = df['velocity'].astype(float)
    df['angle_displacement'] = df['angle_displacement'].astype(float)
    df['radii_displacement'] = df['radii_displacement'].astype(float)

    grouped = df.groupby('super_cluster_label')
    for label, group in grouped:
        if len(group['time_hours']) > 1:
            time_diffs = np.diff(group['time_hours'])
            if np.any(time_diffs == 0):
                continue  # Skip groups with zero time intervals

            velocities = np.diff(group['length_microns']) / time_diffs
            df.loc[group.index[:-1], 'velocity'] = velocities  # Assign to all but last due to diff

            # Angular and radial displacements
            angles = np.arctan2(np.diff(group['centroid_y']), np.diff(group['centroid_x']))
            radii = np.sqrt(np.diff(group['centroid_x'])**2 + np.diff(group['centroid_y'])**2)
            df.loc[group.index[:-1], 'angle_displacement'] = angles
            df.loc[group.index[:-1], 'radii_displacement'] = radii

    return df

def plot_and_capture_neurites(df, frames_folder):
    # Define paths for the output directories
    base_dir = frames_folder
    plots_dir = os.path.join(base_dir, 'neurite_plots')
    polar_plots_dir = os.path.join(base_dir, 'neurite_polar_plots')
    videos_dir = os.path.join(base_dir, 'neurite_videos')

    # Create directories if they don't exist
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(polar_plots_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)

    # Dynamic path for grid mask
    grid_mask_dir = base_dir.replace('pngs', 'mask')
    grid_mask_path = os.path.join(grid_mask_dir, 'grid_mask.png')

    # Print for debugging
    print(f"Using grid mask at: {grid_mask_path}")

    # Loop over each cluster label group
    for label, group in df.groupby('super_cluster_label'):
        if len(group) < 2:
            print(f"Insufficient data for cluster {label}")
            continue  # Skip clusters with insufficient data

        times = group['time_hours']
        lengths = group['length_microns']
        adjusted_lengths = lengths - lengths.iloc[0]
        velocities = group['velocity']
        angles = group['angle_displacement']
        radii = group['radii_displacement']

        # Plot various graphs for each cluster
        plot_length_changes(label, times, adjusted_lengths, plots_dir)
        plot_polar_angle_length_changes(label, angles, adjusted_lengths, polar_plots_dir)
        if not velocities.isna().all():
            plot_velocity_changes(label, group, plots_dir)
        plot_polar_displacement(label, angles, radii, polar_plots_dir)

    # Combine and plot aggregated data if applicable
    if not df.empty:
        avg_length = process_series([g['length_microns'] - g['length_microns'].iloc[0] for _, g in df.groupby('super_cluster_label')])
        avg_velocity = process_series([g['velocity'] for _, g in df.groupby('super_cluster_label') if not g['velocity'].isna().all()])
        avg_angles, avg_radii = average_polar_data(df['angle_displacement'].dropna().values, df['radii_displacement'].dropna().values)

        plot_combined_average_data(avg_length, plots_dir, 'Average_length_changes.png')
        plot_combined_average_velocity(avg_velocity, plots_dir, 'Average_velocity_changes.png')
        plot_combined_polar_data(avg_angles, avg_radii, polar_plots_dir, 'Average_directionality.png')

    # Create videos for all clusters if necessary
    interaction_df = create_videos_for_all_clusters(df, frames_folder, videos_dir, grid_mask_path)
    print("Plots and videos generated successfully.")

    return interaction_df

def create_videos_for_all_clusters(df, frames_folder, output_folder, grid_mask_path):
    interaction_data = []

    grid_mask = cv2.imread(grid_mask_path, cv2.IMREAD_GRAYSCALE)
    if grid_mask is None:
        print(f"Error: Grid mask at {grid_mask_path} not found.")
        return

    interaction_types = ['on_top', 'partially_touching', 'flat']
    interaction_dirs = {itype: os.path.join(output_folder, itype) for itype in interaction_types}
    for path in interaction_dirs.values():
        os.makedirs(path, exist_ok=True)

    for label, group in df.groupby('super_cluster_label'):
        print(f"Creating videos for cluster {label}")
        interaction_df = create_video_for_cluster(label, group, frames_folder, output_folder, interaction_dirs, grid_mask)
        interaction_data.append(interaction_df)
        print("interaction_df\n", interaction_df)

    # Combine all interaction data into a single DataFrame
    all_interaction_data = pd.concat(interaction_data, ignore_index=True)
    print("all_interaction_data\n", all_interaction_data)
    return all_interaction_data

def load_and_combine_data(base_folder):
    """Load all CSV files within the specified base folder and combine into a single DataFrame."""
    csv_files = glob.glob(os.path.join(base_folder, '**', '*.csv'), recursive=True)
    data_frames = []
    for file in csv_files:
        df = pd.read_csv(file)
        folder_name = os.path.basename(os.path.dirname(file))
        
        # Extract shape and pitch from folder name assuming naming convention like 'mushroom_p10'
        parts = folder_name.split('_')
        if len(parts) >= 2:
            df['shape'] = parts[0]
            df['pitch'] = parts[1]
        else:
            df['shape'] = folder_name  # Default to only shape if no underscore
            df['pitch'] = 'unknown'   # Default value if pitch is not specified

        df['source'] = folder_name
        data_frames.append(df)

    combined_df = pd.concat(data_frames, ignore_index=True)
    return combined_df

def preprocess_interaction_data(interaction_df):
    # Map 'on top' and 'partially touching' to a new single category
    interaction_df['interaction_type'] = interaction_df['interaction_type'].replace({
        'on_top': 'on_top',
        'partially_touching': 'on_top'
    })
    return interaction_df

def main_analysis(base_folder):
    # Load and combine all CSV files data
    combined_df = load_and_combine_data(base_folder)
    processed_df = preprocess_data(combined_df)

    # Aggregate interaction data from all clusters
    all_interaction_data = pd.DataFrame()

    csv_files = glob.glob(os.path.join(base_folder, '**', '*.csv'), recursive=True)
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        source_folder = os.path.basename(os.path.dirname(csv_file))
        frames_folder_png = os.path.join(base_folder, source_folder, 'pngs')

        interaction_df = plot_and_capture_neurites(processed_df[processed_df['source'] == source_folder], frames_folder_png)
        all_interaction_data = pd.concat([all_interaction_data, interaction_df], ignore_index=True)

    # Preprocess to merge interaction types
    all_interaction_data = preprocess_interaction_data(all_interaction_data)

    # Save processed data to CSV
    output_csv_path = os.path.join(base_folder, 'all_interaction_data.csv')
    all_interaction_data.to_csv(output_csv_path, index=False)

    print("All analyses and plots generated successfully.")

# Call main_analysis
base_folder = 'neurite_tracking_code_files/live_imaging'
main_analysis(base_folder)
