import os
import numpy as np
import math
from sklearn.cluster import DBSCAN
import pandas as pd
from PIL import Image
import cv2
import glob 

def rename_files(frames_folder):
    files = os.listdir(frames_folder)
    ndf_files = [f for f in files if f.endswith('.ndf')]

    def extract_number(filename):
        parts = filename.split('-')
        if len(parts) > 1:
            try:
                return int(parts[-1].split('.')[0])
            except ValueError:
                return 0
        return 0

    sorted_ndf_files = sorted(ndf_files, key=extract_number)

    for i, filename in enumerate(sorted_ndf_files, start=1):
        new_name = f"frame_{i:04d}.ndf"
        os.rename(os.path.join(frames_folder, filename), os.path.join(frames_folder, new_name))
        
        tif_filename = filename.replace('.ndf', '.tif')
        if os.path.exists(os.path.join(frames_folder, tif_filename)):
            new_tif_name = new_name.replace('.ndf', '.tif')
            os.rename(os.path.join(frames_folder, tif_filename), os.path.join(frames_folder, new_tif_name))

def load_tracing_data(frames_folder):
    tracings_data = {}
    for frame_file in sorted(os.listdir(frames_folder)):
        if frame_file.endswith(".txt"):
            frame_index = int(frame_file.split('_')[1].split('.')[0])  
            txt_path = os.path.join(frames_folder, frame_file)
            tracings_data[frame_index] = parse_ndf_content(txt_path)
    return tracings_data

def ndf_to_txt(ndf_path, txt_path):
    with open(ndf_path, 'r') as ndf_file:
        ndf_content = ndf_file.read()

    with open(txt_path, 'w') as txt_file:
        txt_file.write(ndf_content)

def convert_tif_to_png(source_folder, target_folder):
    # Ensure target folder exists
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Loop through all files in the source folder
    for filename in os.listdir(source_folder):
        if filename.endswith(".tif"): #or filename.endswith(".tiff"):
            # Path to the current file
            file_path = os.path.join(source_folder, filename)
            # Open the TIFF image
            with Image.open(file_path) as img:
                # Convert the file name to PNG
                img = img.convert("RGB")
                # Convert the file name to PNG
                target_file = os.path.splitext(filename)[0] + ".png"
                target_path = os.path.join(target_folder, target_file)
                # Save the image as PNG
                img.save(target_path, "PNG")
                print(f"Converted {filename} to {target_file}")

def parse_ndf_content(txt_path):
    data = {
        'Parameters': [],
        'TypeNamesColors': {},
        'ClusterNames': [],
        'Tracings': {}
    }
    current_tracing = None
    coordinates = []  # Temporary storage for coordinates within the current segment

    with open(txt_path, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        line = line.strip()
        next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""

        if line.startswith('//'):
            if 'Tracing' in line or 'Segment' in line or 'Parameters' in line or 'Type names and colors' in line or 'Cluster names' in line:
                # Check if we need to finalize the current segment for the current tracing
                if coordinates:
                    # Ensure there's an even number of coordinates before creating tuples
                    if len(coordinates) % 2 == 0:
                        segment_tuples = [(coordinates[j], coordinates[j+1]) for j in range(0, len(coordinates), 2)]
                        data['Tracings'].setdefault(current_tracing, []).extend(segment_tuples)
                    coordinates = []  # Reset for a new segment

                if 'Tracing' in line:
                    current_tracing = line.split()[-1]
                elif current_tracing and line.replace(' ', '').isdigit():
                    coordinates.extend(map(int, line.split()))
                elif 'Segment' in line or 'Parameters' in line or 'Type names and colors' in line or 'Cluster names' in line:
                    continue

        elif current_tracing and line.isdigit():
            coordinates.append(int(line))

    # Finalize any remaining segment for the last tracing
    if current_tracing and coordinates:
        if len(coordinates) % 2 == 0:
            segment_tuples = [(coordinates[i], coordinates[i + 1]) for i in range(0, len(coordinates), 2)]
            data['Tracings'][current_tracing] = segment_tuples

    return data

def frames_ndf_to_txt(frames_folder, ndf_folder):
    '''  Converts all .ndf files in the frames folder to .txt format.'''
    for frame_file in os.listdir(frames_folder):
        if frame_file.endswith(".tif"):  # Adjust based on your image file type
            base_filename = os.path.splitext(frame_file)[0]
            ndf_path = os.path.join(ndf_folder, base_filename + '.ndf')
            txt_path = os.path.join(ndf_folder, base_filename + '.txt')
            
            # Convert NDF to TXT
            ndf_to_txt(ndf_path, txt_path)
            
def calculate_path_length(path):
    """Calculate the length of a path given a list of coordinates."""
    length = 0
    for i in range(1, len(path)):
        length += math.sqrt((path[i][0] - path[i - 1][0]) ** 2 + (path[i][1] - path[i - 1][1]) ** 2)
    return length

def calculate_centroid(points):
    """ Calculate the centroid from a list of points """
    return np.mean(points, axis=0)

def cluster_neurites_in_frame(frame_index, tracings, eps=30, min_samples=5):
    """ Cluster neurite positions within a frame using DBSCAN, calculate centroids, and include tracing data. """
    pixel_to_micron = 7.5

    all_points = np.array([point for sublist in tracings.values() for point in sublist])
    if all_points.size == 0:
        return pd.DataFrame()  # Return an empty DataFrame if no points

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(all_points)
    labels = clustering.labels_
    unique_labels = set(labels)
    data = []

    for label in unique_labels:
        if label == -1:
            continue  # Ignore noise
        class_member_mask = (labels == label)
        xy = all_points[class_member_mask]
        centroid = calculate_centroid(xy)
        tracing_length = calculate_path_length(xy) / pixel_to_micron  # Convert pixels to microns
        data.append({
            'frame_index': frame_index,
            'cluster_label': label,
            'centroid_x': centroid[0],
            'centroid_y': centroid[1],
            'length_microns': tracing_length,
            'tracing_points': xy.tolist()  # Store the tracing points
        })

    return pd.DataFrame(data)

def cluster_across_frames(df, eps=30, min_samples=3, time_scale=0.1):
    # Incorporate frame index with a scaling factor to account for temporal proximity
    df['temporal_feature'] = df['frame_index'] * time_scale
    features = df[['centroid_x', 'centroid_y', 'temporal_feature']]

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
    df['super_cluster_label'] = clustering.labels_
    return df

def process_all_frames(all_frame_tracings, frames_folder):
    frames = sorted(all_frame_tracings.keys())
    all_data = []
    time_per_frame = 24 / len(frames)  # Adjusted for the actual number of frames

    for i, frame in enumerate(frames):
        frame_data = cluster_neurites_in_frame(frame, all_frame_tracings[frame]['Tracings'], 30, 5)
        if not frame_data.empty:
            frame_data['time_hours'] = i * time_per_frame  # Assign the time in hours
            frame_data['frame_path'] = os.path.join(frames_folder, f'frame_{frame:04d}.png')
            all_data.append(frame_data)

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

# Load data and process
def process_frames_and_ndf(frames_folder, ndf_folder, output_folder):
    rename_files(frames_folder)
    convert_tif_to_png(frames_folder, output_folder)
    frames_ndf_to_txt(frames_folder, ndf_folder)
    tracing_data = load_tracing_data(ndf_folder)
    return tracing_data

def get_folder_name(path):
    return os.path.basename(os.path.normpath(path))

# Mask generation
# Initialize global variables
drawing = False  # True if mouse is pressed
rectangles = []  # List to store rectangles
ix, iy = -1, -1  # Initial mouse position

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, img, rectangles

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = img.copy()
            for rect in rectangles:
                cv2.rectangle(img_copy, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), 2)
            cv2.rectangle(img_copy, (ix, iy), (x, y), (255, 0, 0), 2)
            cv2.imshow("First Frame", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Normalize the rectangle coordinates to ensure they are stored from top-left to bottom-right
        x_min, x_max = min(ix, x), max(ix, x)
        y_min, y_max = min(iy, y), max(iy, y)
        rectangles.append((x_min, y_min, x_max, y_max))
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.imshow("First Frame", img)

def generate_mask(image_path):
    global img, rectangles
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Failed to read image")
        return None

    cv2.namedWindow("First Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("First Frame", 800, 780)  # Adjust the size as needed

    cv2.setMouseCallback("First Frame", draw_rectangle)

    while True:
        cv2.imshow("First Frame", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    # Create the final mask
    mask = np.zeros_like(img)
    for rect in rectangles:
        # Apply the rectangles to the mask
        mask[rect[1]:rect[3], rect[0]:rect[2]] = 255

    rectangles = []  # Clear the list of rectangles after creating the mask
    return mask

# Main function
def main_processing(base_folder):
    # List all subfolders in the base folder, assuming each subfolder represents a separate dataset
    subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir()]
    
    for folder in subfolders:
        frames_folder = folder
        ndf_folder = folder
        output_folder = os.path.join(folder, 'pngs')
        mask_folder = os.path.join(folder, 'mask')

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        if not os.path.exists(mask_folder):
            os.makedirs(mask_folder)

        # Find the first .png file in the output folder
        png_files = sorted(glob.glob(os.path.join(output_folder, '*.png')))
        if png_files:
            first_image_path = png_files[0]  # Select the first .png file found
        else:
            print(f"No PNG files found in {output_folder}. Skipping folder.")
            continue

        mask_path = os.path.join(mask_folder, 'grid_mask.png')

        if not os.path.isfile(mask_path):  # Generate mask if not already created
            mask = generate_mask(first_image_path)
            if mask is not None:
                cv2.imwrite(mask_path, mask)
            else:
                print(f"Failed to generate mask for {folder}")

        folder_name = get_folder_name(frames_folder)
        
        # Process frames and NDFs
        tracing_data = process_frames_and_ndf(frames_folder, ndf_folder, output_folder)
        if tracing_data:
            df = process_all_frames(tracing_data, output_folder)
            df = cluster_across_frames(df)
            output_file = os.path.join(ndf_folder, f'clustered_data_{folder_name}.csv')
            df.to_csv(output_file, index=False)
            print(f"Processed {folder_name}:")
            print(df['super_cluster_label'].value_counts())

# Main script
base_folder = 'neurite_tracking_code_files/live_imaging'
main_processing(base_folder)
