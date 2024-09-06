import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

def get_image_size(image_path):
    """Get dimensions of the image."""
    with Image.open(image_path) as img:
        return img.size  # returns (width, height)

def classify_by_pitch(df, image_path):
    """Classify 'thin' neurites based on their centroid's y-coordinate related to image size."""
    width, height = get_image_size(image_path)
    threshold_y = height / 2  # Adjust as necessary based on specific grid layout

    # Update the pitch for 'thin' shapes only if initially marked as 'p4'
    def update_pitch(row):
        if row['shape'] == 'thin' and row['pitch'] == 'p4':
            return 'p10' if row['centroid_y'] < threshold_y else 'p4'
        return row['pitch']

    df['pitch'] = df.apply(update_pitch, axis=1)
    return df

def load_interaction_data(csv_path, image_path):
    """Load the interaction data from a CSV file and classify by pitch using a copy."""
    df = pd.read_csv(csv_path)
    df_copy = df.copy()  # Work with a copy to preserve the original data
    df_classified = classify_by_pitch(df_copy, image_path)
    return df_classified
         
def ensure_plot_directories(base_dir):
    """Ensure that the plot directory exists."""
    plots_dir = os.path.join(base_dir, 'velocity_plots_pdf_units')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    return plots_dir

def normalize_data(df, column):
    """ Normalize a dataframe column to a 0-100 scale based on its max value. """
    max_value = df[column].max()
    min_value = df[column].min()
    df[column] = 100 * (df[column] - min_value) / (max_value - min_value)
    return df


def plot_combined_pitch_trends(interaction_df, metric, plots_dir):
    """Generate combined trend plots for each pitch, showing all shapes including a consistent control."""
    # Normalize the metric for consistent scale
    # interaction_df = normalize_data(interaction_df.copy(), metric)

    # Group data by pitch
    pitches = interaction_df['pitch'].unique()

    for pitch in pitches:
        fig, ax = plt.subplots()
        # Filter data for the current pitch
        pitch_data = interaction_df[interaction_df['pitch'] == pitch]

        # Group by shape within this pitch
        shape_groups = pitch_data.groupby('shape')
        colors = plt.cm.viridis(np.linspace(0, 1, len(shape_groups)))

        for (shape, group), color in zip(shape_groups, colors):
            # Calculate the rolling mean to visualize trends
            trend_series = group.groupby('time')[metric].mean().rolling(window=5, center=True).mean()

            # Plot the trend for each shape
            trend_series.plot(ax=ax, marker='o', linestyle='-', label=f'{shape.capitalize()} - Pitch {pitch}', color=color)

        plt.title(f'Combined {metric.capitalize()} Trends Over Time for Pitch {pitch}')
        plt.xlabel('Time (Hours)')
        plt.ylabel(f'{metric.capitalize()} (µm/hour)')
        plt.legend()
        plt.grid(True)

        # Save the plot as a PDF
        plot_filename = f'combined_{metric}_trend_pitch_{pitch}.pdf'
        plot_path = os.path.join(plots_dir, plot_filename)
        with PdfPages(plot_path) as pdf:
            pdf.savefig(fig)
        plt.close()

        print(f"Combined trend plot saved for pitch {pitch}: {plot_path}")


def main(csv_path, output_dir, image_path):
    """Main function to load data and generate plots."""
    interaction_df = load_interaction_data(csv_path, image_path)
    plots_dir = ensure_plot_directories(output_dir)
    plot_combined_pitch_trends(interaction_df, 'velocity', plots_dir)
    plot_combined_pitch_trends(interaction_df, 'length', plots_dir)

    # Specify the path where you want to save the updated CSV
    updated_csv_path = os.path.join(output_dir, 'updated_interaction_data.csv')
    interaction_df.to_csv(updated_csv_path, index=False)
    print(f"Updated interaction data saved to {updated_csv_path}")

# Usage example
csv_path = 'neurite_tracking_code_files/live_imaging/all_interaction_data.csv'
output_dir = 'neurite_tracking_code_files/live_imaging'
image_path = 'neurite_tracking_code_files/live_imaging/thin_p4_p10/pngs/frame_0001.png'
main(csv_path, output_dir, image_path)