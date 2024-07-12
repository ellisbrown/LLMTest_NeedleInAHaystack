import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import json
import glob
import warnings


def generate_plot(folder_path: str, model_name: str, max_ctx_len: int):
    if model_name is None:
        model_name = ""
        warnings.warn("No model name provided. Defaulting to empty string")
    if max_ctx_len is None:
        max_ctx_len = ""
        warnings.warn("No max context length provided. Defaulting to empty string")
    if folder_path is None:
        raise ValueError("No folder path provided")

    print(f"Generating plot for folder: {folder_path}")

    ## 1. Load Data
    # Using glob to find all json files in the directory
    json_files = glob.glob(f"{folder_path}/*.json")

    if len(json_files) == 0:
        json_files = glob.glob(f"{folder_path}/results/*.json")

    # List to hold the data
    data = []

    # Iterating through each file and extract the 3 columns we need
    for file in json_files:
        with open(file, 'r') as f:
            json_data = json.load(f)
            # Extracting the required fields
            document_depth = json_data.get("depth_percent", None)
            context_length = json_data.get("context_length", None)
            score = json_data.get("score", None)
            # Appending to the list
            data.append({
                "Document Depth": document_depth,
                "Context Length": context_length,
                "Score": score
            })

    # Creating a DataFrame
    df = pd.DataFrame(data)

    print (df.sample(5))
    print (f"You have {len(df)} rows")

    ## 2. Pivot Data

    pivot_table = pd.pivot_table(df, values='Score', index=['Document Depth', 'Context Length'], aggfunc='mean').reset_index() # This will aggregate
    pivot_table = pivot_table.pivot(index="Document Depth", columns="Context Length", values="Score") # This will turn into a proper pivot
    pivot_table.iloc[:5, :5]


    ## 3. Make Visualization
    # Create a custom colormap. Go to https://coolors.co/ and pick cool colors
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

    # Create the heatmap with better aesthetics
    plt.figure(figsize=(10, 6))  # Can adjust these dimensions as needed
    sns.heatmap(
        pivot_table,
        # annot=True,
        fmt="g",
        cmap=cmap,
        cbar_kws={'label': 'Score'}
    )

    # More aesthetics
    title = f"Pressure Testing {model_name} {max_ctx_len} Context\nFact Retrieval Across Context Lengths ('Needle In A HayStack')"
    plt.title(title)  # Title
    plt.xlabel('Token Limit')  # X-axis label
    plt.ylabel('Depth Percent')  # Y-axis label
    plt.xticks(rotation=45)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area

    # Save the plot
    plt.savefig(f"{folder_path}/niah.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{folder_path}/niah.pdf", dpi=300, bbox_inches='tight')

    print(f"Plot saved to {folder_path}/niah.png")
    print(f"Plot saved to {folder_path}/niah.pdf")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate a plot from the results of the NeedleInAHaystack test')
    parser.add_argument('--folder_path', type=str, help='The path to the folder containing the results of the test')
    parser.add_argument('--model_name', type=str, help='The name of the model used in the test')
    parser.add_argument('--max_ctx_len', type=int, help='The maximum context length used in the test')
    args = parser.parse_args()

    generate_plot(args.folder_path, args.model_name, args.max_ctx_len)
