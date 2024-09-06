# NeuriteTracking
# 

- ## **Pre-processing code** ##
    
    ### Step 1: **`rename_files(frames_folder)`**
    
    - **Purpose:** Renames all `.ndf` and related `.tif` files in the folder to a standard format for easier tracking and processing.
    
    ### Step 2: **`convert_tif_to_png(source_folder, target_folder)`**
    
    - **Purpose:** Converts all `.tif` images in the source folder to `.png` format in a designated target folder.
    
    ### Step 3: **`ndf_to_txt(ndf_path, txt_path)`**
    
    - **Purpose:** Reads data from `.ndf` files and saves it as `.txt` files for further processing.
    
    ### Step 4: **`load_tracing_data(frames_folder)`**
    
    - **Purpose:** Loads and organizes tracing data from `.txt` files, preparing it for analysis.
    
    ### Step 5: **`frames_ndf_to_txt(frames_folder, ndf_folder)`**
    
    - **Purpose:** Converts all `.ndf` files in the frames folder to `.txt` format, preparing them for data extraction.
    
    ### Step 6: **`cluster_neurites_in_frame(frame_index, tracings, eps, min_samples)`**
    
    - **Purpose:** Clusters neurite data within each frame to identify groups based on their spatial proximity.
    
    ### Step 7: **`cluster_across_frames(df, eps, min_samples, time_scale)`**
    
    - **Purpose:** Applies clustering across multiple frames, combining spatial and temporal data to track changes over time.
    
    ### Step 8: **`process_all_frames(all_frame_tracings, frames_folder)`**
    
    - **Purpose:** Processes all frames collectively, applying clustering within and across frames, and compiles the results.
    
    ### Step 9: **`process_frames_and_ndf(frames_folder, ndf_folder, output_folder)`**
    
    - **Purpose:** Coordinates the entire process of renaming files, converting images, and loading data for each dataset in the specified folders.
    
    ### Step 10: **`generate_mask(image_path)`**
    
    - **Purpose:** This function allows the user to interactively draw rectangles on the first frame image to generate a binary mask. Each rectangle is stored and then used to create the final mask image.
    
    ### Step 11: **`main_processing(base_folder)`**
    
    - **Purpose:** Executes the entire processing sequence for each dataset in the base folder, saving results and summarizing the data processed.  The code checks if a mask for the current dataset exists (`grid_mask.png`). If not, the `generate_mask` function is called to allow the user to create one.
          
- ## **Post-processing code** ##
    
    ### 1. **`process_series(data_list)`**
    
    - **Purpose:** Combines multiple series into a single dataframe, removes duplicate indices, sorts, and calculates the mean across columns, which helps in standardizing data for analysis.
    
    ### 2. **`average_polar_data(angles, radii)`**
    
    - **Purpose:** Computes the average angle and radius from arrays of polar coordinates to summarize the overall direction and extent of changes.
    
    ### 3. **`calculate_neurite_angles(x_coords, y_coords)`**
    
    - **Purpose:** Determines the angles of movement based on differences in x and y coordinates, providing insights into the trajectory of neurite growth.
    
    ### 4. **Plotting Functions: `plot_velocity_changes`, `plot_length_changes`, `plot_polar_angle_length_changes`, etc.**
    
    - **Purpose:** These functions visualize various aspects of neurite data such as velocity, length, and directional changes over time, and save the plots to specified directories.
    
    ### 5. **`zoom_into_centroid(image, centroid, tracing_points, zoom_size=250, padding_color=(0, 0, 0))`**
    
    - **Purpose:** Enhances a specific area around a centroid in an image to focus on detailed features, which is useful for close examination of neurite behavior at specific points.
    
    ### 6. **Video Creation: `create_video_for_cluster` and `create_videos_for_all_clusters`**
    
    - **Purpose:** Generates videos from frames that show the progression or changes of neurites over time for individual clusters or across all clusters, aiding in dynamic visualization.
    
    ### 7. **`plot_and_capture_neurites(df, frames_folder)`**
    
    - **Purpose:** Acts as the central function for processing and visualizing neurite data from the frames directory, managing the creation of plots and videos.
    
    ### 8. **Data Management: `load_and_combine_data(base_folder)` and `main_analysis(base_folder)`**
    
    - **Purpose:** Loads all CSV data files within a specified base directory, combines them into a single DataFrame, and then processes this data through plotting and video generation functions.



- ## **Data Analysis** ##
    
    Brief overview of what each function in the code does:
    
    - **`get_image_size(image_path)`**: Retrieves the dimensions (width, height) of the input image.
    - **`classify_by_pitch(df, image_path)`**: Classifies neurites by pitch based on the `y-coordinate` of their centroid relative to the image height.
    - **`load_interaction_data(csv_path, image_path)`**: Loads interaction data from the generated CSV file (from postprocessing code) and classifies it based on the image pitch.
    - **`ensure_plot_directories(base_dir)`**: Creates directories to save the output plots if they don’t exist.
    - **`normalize_data(df, column)`**: Normalizes the values in a column to a 0-100 scale.
    - **`plot_combined_pitch_trends(interaction_df, metric, plots_dir)`**: Generates trend plots for different pitches and saves them as PDFs.
    - **`main(csv_path, output_dir, image_path)`**: The main function that orchestrates data loading, classification, plotting, and saving the updated data.
    
