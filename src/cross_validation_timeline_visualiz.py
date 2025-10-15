# --- IMPORTS
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

fold_results_data= [
    # Fold 1 Results
    [1, 'Random Forest', 1.6133],
    [1, 'XGBoost', 1.7947],
    [1, 'Linear Regression', 2.0773],

    # Fold 2 Results
    [2, 'Random Forest', 1.6916],
    [2, 'XGBoost', 1.7249],
    [2, 'Linear Regression', 2.0098],
    
    # Fold 3 Results
    [3, 'Random Forest', 1.4458],
    [3, 'XGBoost', 1.5594],
    [3, 'Linear Regression', 1.6819],

    # Fold 4 Results
    [4, 'Random Forest', 1.6980],
    [4, 'XGBoost', 1.7721],
    [4, 'Linear Regression', 2.0074],

    # Fold 5 Results
    [5, 'Random Forest', 1.6845],
    [5, 'XGBoost', 1.9545],
    [5, 'Linear Regression', 2.1030],

    # Fold 6 Results
    [6, 'Random Forest', 1.6493],
    [6, 'XGBoost', 1.7063],
    [6, 'Linear Regression', 2.0406]
]


models = ['Random Forest', 'XGBoost', 'Linear Regression']
num_folds = 6

# --- Data Processing and Visualization ---

# Convert the results into a pandas DataFrame
try:
    results_df = pd.DataFrame(fold_results_data, columns=['Fold', 'Model', 'MAE'])
    print("Successfully created DataFrame from results.")
    print(results_df.head())
except Exception as e:
    print(f"Error creating DataFrame: {e}")
    # Exit if DataFrame creation fails
    exit()


# Set the style for the plot for a professional look
sns.set_style("whitegrid")

# Create the figure and axes for the plot
plt.figure(figsize=(10, 7))

# Create the box plot using seaborn
# A box plot is perfect for comparing distributions across categories (models)
ax = sns.boxplot(
    x='Model',
    y='MAE',
    data=results_df,
    palette='viridis', # Using a pleasant color palette
    width=0.5
)

# Add a title and clear labels for the axes
plt.title('Model Performance Comparison (MAE Across 6 Folds)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Machine Learning Model', fontsize=12, labelpad=15)
plt.ylabel('Mean Absolute Error (MAE)', fontsize=12, labelpad=15)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

# Add a caption for context
plt.figtext(0.5, 0.01, 'Lower MAE indicates better performance. The box shows the interquartile range (IQR),\n'
                      'the line indicates the median, and whiskers show the data range.',
            ha='center', fontsize=9, style='italic', color='gray')


# Final adjustments and saving the plot
plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout to make space for caption

# Save the figure to a file for your report
plt.savefig('model_mae_comparison.png', dpi=300)

# Display the plot
plt.show()
