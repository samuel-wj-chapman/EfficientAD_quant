import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np

# Search for all result files
files = glob.glob('*_results.txt')

# Initialize a list to store data
data = []

# Process each file
for file in files:
    with open(file, 'r') as f:
        content = f.readlines()
        non_quantized_auc = float(content[0].split(': ')[1])
        quantized_auc = float(content[1].split(': ')[1])
        mean_values = eval(content[3].split(': ')[1].strip())
        std_values = eval(content[4].split(': ')[1].strip())
        q_st_start = float(content[5].split(': ')[1])
        q_st_end = float(content[6].split(': ')[1])
        q_ae_start = float(content[7].split(': ')[1])
        q_ae_end = float(content[8].split(': ')[1])

        # Calculate average of mean and std values
        mean_of_means = np.mean(mean_values)
        mean_of_stds = np.mean(std_values)
        
        # Calculate sum of quantization parameters
        quant_params_sum = q_st_start + q_st_end + q_ae_start + q_ae_end
        
        # Append to data list
        data.append({
            'SubDataset': file.split('_')[0],
            'NonQuantizedAUC': non_quantized_auc,
            'QuantizedAUC': quantized_auc,
            'Mean_TeacherMean': mean_of_means,
            'Mean_TeacherStd': mean_of_stds,
            'AUC_Drop': non_quantized_auc - quantized_auc,
            'Q_st_start': q_st_start,
            'Q_st_end': q_st_end,
            'Q_ae_start': q_ae_start,
            'Q_ae_end': q_ae_end,
            'QuantParamsSum': quant_params_sum
        })

# Create DataFrame
df = pd.DataFrame(data)

# Sort DataFrame by AUC Drop in descending order
df_sorted = df.sort_values(by='AUC_Drop', ascending=False)

# Print the sorted list
print("Sub-dataset names ordered by AUC Drop (most to least):")
print(df_sorted[['SubDataset', 'NonQuantizedAUC', 'QuantizedAUC', 'AUC_Drop']])


# Plotting
fig, ax = plt.subplots(3, 2, figsize=(12, 18))  # Adjust subplot grid size

# Scatter plot for Teacher Mean vs. AUC Drop
ax[0, 0].scatter(df['Mean_TeacherMean'], df['AUC_Drop'], color='blue')
ax[0, 0].set_title('AUC Drop vs. Average Teacher Mean')
ax[0, 0].set_xlabel('Average Teacher Mean')
ax[0, 0].set_ylabel('AUC Drop')

# Scatter plot for Teacher Std vs. AUC Drop
ax[0, 1].scatter(df['Mean_TeacherStd'], df['AUC_Drop'], color='red')
ax[0, 1].set_title('AUC Drop vs. Average Teacher Std')
ax[0, 1].set_xlabel('Average Teacher Std')
ax[0, 1].set_ylabel('AUC Drop')

# Scatter plot for Q_st_start vs. AUC Drop
ax[1, 0].scatter(df['Q_st_start'], df['AUC_Drop'], color='green')
ax[1, 0].set_title('AUC Drop vs. Q_st_start')
ax[1, 0].set_xlabel('Q_st_start')
ax[1, 0].set_ylabel('AUC Drop')

# Scatter plot for Q_st_end vs. AUC Drop
ax[1, 1].scatter(df['Q_st_end'], df['AUC_Drop'], color='purple')
ax[1, 1].set_title('AUC Drop vs. Q_st_end')
ax[1, 1].set_xlabel('Q_st_end')
ax[1, 1].set_ylabel('AUC Drop')

# Scatter plot for Q_ae_start vs. AUC Drop
ax[2, 0].scatter(df['Q_ae_start'], df['AUC_Drop'], color='orange')
ax[2, 0].set_title('AUC Drop vs. Q_ae_start')
ax[2, 0].set_xlabel('Q_ae_start')
ax[2, 0].set_ylabel('AUC Drop')

# Scatter plot for Q_ae_end vs. AUC Drop
ax[2, 1].scatter(df['Q_ae_end'], df['AUC_Drop'], color='cyan')
ax[2, 1].set_title('AUC Drop vs. Q_ae_end')
ax[2, 1].set_xlabel('Q_ae_end')
ax[2, 1].set_ylabel('AUC Drop')

# Scatter plot for QuantParamsSum vs. AUC Drop
ax[1, 1].scatter(df['QuantParamsSum'], df['AUC_Drop'], color='magenta')
ax[1, 1].set_title('AUC Drop vs. Sum of Quant Params')
ax[1, 1].set_xlabel('Sum of Quant Params')
ax[1, 1].set_ylabel('AUC Drop')

plt.tight_layout()
plt.savefig('AUC_Drop_Quantization_Full_Analysis.png')  # Save the plot to a file
plt.show()