import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from cycler import cycler
import matplotlib as mpl
import scipy.stats as stats
from scipy.stats import wilcoxon


# CSS to center the elements
st.markdown(
    """
    <style>
    .center {
        display: flex;
        justify-content: center;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Centering the headers
st.markdown("<h1 class='center'>An EsteStyle Streamlit Page</h1>", unsafe_allow_html=True)
st.markdown("<h1 class='center'>Where Python Wiz Meets Data Viz!</h1>", unsafe_allow_html=True)

st.markdown("<img src='https://1drv.ms/i/s!ArWyPNkF5S-foZspwsary83MhqEWiA?embed=1&width=307&height=307' width='300' style='display: block; margin: 0 auto;'>" , unsafe_allow_html=True)

st.title("Study Design")

st.write('Subjects were given 16 days of drinking behavior aquisition. On day 17, vehicle and drug groups were assigned. Liquid drinking amounts were measured throughout '
         'to determine if this behavior responded to the treatment administered.')

st.title("Data analysis")

st.write('All groups were checked for normal distribution of their datasets. Groups were determined as not significantly different from one another upon vehicle or drug assignments.'
         'Since the datasets were previously determined to not follow a normal distribution, a nonparametric test (Wilcoxon Signed-Rank) was used to determine whether to reject the '
         'null hypothesis. As a significant difference was found between treatment groups, data per day was also analyzed for statistically significant differences.')

pre_veh = np.array([[0.3, 3.4, 1.2, 1.4, 3.4, 2.8, 1.5, 0.3],
                [2.2, 9.9, 5.9, 6, 3.9, 3.4, 1.8, 3.1],
                [0, 5.1, 0, 4.5, 3.5, 2.6, 6.1, 3.5],
                [0, 6.9, 3.1, 3.2, 4.8, 4.9, 4, 6.4],
                [5.3, 2.8, 2.5, 3.8, 4.6, 5.7, 4.5,6.7],
                [4.4, 8.6, 1.2, 5.3, 4.4, 6.3, 6.2, 5.3],
                [4.3, 4.8, 2.3, 2.9, 2.5, 7.7, 4.7, 4.7],
                [3.8, 6.5, 1.3, 3.8, 4.3, 6.5, 3.7, 3.9],
                [3.3, 4.7, 2.3, 3, 2.6, 3.8, 5.3, 4.5],
                [5.4, 3.3, 4, 3.2, 4.2, 8.3, 5.9, 4.8],
                [4.5, 4, 5.5, 5, 5, 10, 6.3, 4.2],
                [3.7, 3.7, 2.5, 4.8, 5.9, 3.4, 3.4, 6.4],
                [4.1, 5, 3.9, 4.2, 3, 4, 3.6, 3.8],
                [3.4, 11.6, 5.1, 4.2, 4.5, 2.2, 1.6, 1.5],
                [5.9, 9.2, 4.4, 3.9, 4, 4.2, 3.7, 2.9],
                [7.4, 5.1, 8, 4.5, 2.1, 2.3, 3.6, 3.8]])

pst_veh = np.array([[7.4, 5.1, 8, 4.5, 2.1, 2.3, 3.6, 3.8],
                [4.3, 9.8, 2.3, 5.8, 4.2, 4.8, 4.2, 6.7],
                [5, 8.7, 9.1, 6.2, 2.9, 3, 4.2, 4.8],
                [5.1, 8.4, 5.4, 6.2, 3.6, 3.3, 2.3, 4.8],
                [3.5, 10.1, 8.8, 4.9, 3.1, 3.8, 4, 5.4],
                [5, 5, 6, 6.2, 5.6, 4.4, 1.8, 7.7],
                [4.8, 8.3, 10.7, 7.2, 3.1, 5.1, 3.7, 9.1],
                [2.8, 5.6, 3.7, 8.6, 3.1, 1.7, 4.5, 7.2],
                [3.4, 10.5, 9, 6.7, 4.5, 4, 3.1, 2.3],
                [3.4, 6.5, 4.4, 8.2, 3.2, 3, 4.3, 1.7],
                [3.8, 8.9, 1.3, 6.8, 4.1, 3.1, 4.3, 1.1],
                [4.2, 5.9, 2.8, 4.4, 3.4, 2.8, 5.4, 2.8]])

pre_drg = np.array([[2.5, 0.4, 3.1, 0.1, 5.6, 4.6, 2.6, 2.3],
                [2.2, 1.2, 4.6, 0.4, 6.5, 6.2, 3.7, 4],
                [1.3, 3.7, 5.1, 1.7, 0.5, 6.1, 3.2, 4.1],
                [1.8, 4.9, 7.1, 2.8, 0.3, 6.3, 5.1, 2.2],
                [2.6, 3.6, 4.9, 3.8, 2.5, 6.8, 4.1, 3.8],
                [0.8, 5.4, 5.2, 2, 0.7, 8, 6.7, 0.3],
                [7.2, 4, 5.9, 8.5, 0.1, 8.2, 3.5, 5],
                [0.9, 3.8, 4.6, 6.6, 3.3, 6.1, 2.2, 11.1],
                [6, 3.1, 4.1, 4.4, 5.7, 6, 3.1, 3.1],
                [9, 3.4, 4.1, 7.4, 2.1, 6.9, 4.5, 5.9],
                [1.4, 4.4, 6.4, 4.7, 3.4, 8.3, 5.9, 0.2],
                [1.9, 3.9, 4, 1, 3.4, 5.6, 4, 6.6],
                [6.1, 6.6, 4.3, 1.7, 4.3, 9, 4, 7],
                [2.9, 4.1, 6.1, 6.3, 2.2, 2.3, 3.5, 2.7],
                [2.6, 4, 6.4, 3.8, 2.6, 7, 6.2, 5.1],
                [1.6, 3, 5.3, 3.2, 2.6, 8.1, 6.2, 8.5]])

pst_drg = np.array([[1.6, 3, 5.3, 3.2, 2.6, 8.1, 6.2, 8.5],
                [3.1, 3.5, 7.2,3.4,1.5,2.9,5.8,4],
                [0.4, 2.8, 8.1, 1.6,2, 1.5,3.8,2.6],
                [0.4, 7.2, 7.7, 1.8, 1.8, 0.7, 6,1.5],
                [0.7, 1.8, 6.6, 0.9, 0.3, 2.3, 3.4, 1.6],
                [1.5, 1.5, 6.6, 0.1, 1.2, 2.6, 3.6, 1.3],
                [0.7, 1, 7.9, 1, 1.5, 2.1, 4.4, 2.4],
                [0.7, 8.1, 2.5, 0.9, 0.4, 1.2, 3, 0],
                [0.9, 1.8, 1, 1, 1, 1.5, 2.4, 3],
                [0.4, 0.8, 2.3, 0.4, 0.7, 0.9, 2.4, 0.9],
                [0.1, 1.5, 1.1, 0.2, 0, 1.7, 4.1, 4],
                [0.2, 1, 1.8, 0.5, 0.9, 1.2, 2, 1.4]])

# Flatten all arrays
pre_veh_flat = pre_veh.flatten()
pre_drug_flat = pre_drg.flatten()
pst_veh_flat = pst_veh.flatten()
pst_drug_flat = pst_drg.flatten()

##### Logic for Q-Q & Histo plots #####

# Function to perform Shapiro-Wilk test and display result
def shapiro_test(data, group_name):
    stat, p_value = stats.shapiro(data)
    if p_value > 0.05:
        result = f"{group_name}: Data looks normal (p-value: {p_value:.4f})"
    else:
        result = f"{group_name}: Data is not normal (p-value: {p_value:.4f})"
    return result

def plot_data_distribution(data, group_name, point_color='blue'):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Q-Q plot
    stats.probplot(data, dist="norm", plot=ax[0])
    ax[0].set_title(f"Q-Q Plot for {group_name}")
    ax[0].set_xlabel('Theoretical Quantiles')
    ax[0].set_ylabel('Sample Quantiles')

    # Histogram
    ax[1].hist(data, bins=15, color=point_color, alpha=1)
    ax[1].set_title(f'Histogram for {group_name}')
    ax[1].set_xlabel('Data Values')
    ax[1].set_ylabel('Frequency')

    plt.tight_layout()
    return fig

# Streamlit app
st.title("Q-Q Plot and Shapiro-Wilk Test")

st.write(shapiro_test(pre_veh_flat, 'Pre-Treatment (Vehicle)'))
st.pyplot(plot_data_distribution(pre_veh_flat, 'Pre-Treatment (Vehicle)', point_color='#42A7F5'))

st.write(shapiro_test(pre_drug_flat, 'Pre-Treatment (Drug)'))
st.pyplot(plot_data_distribution(pre_drug_flat, 'Pre-Treatment (Drug)', point_color='#ed7b7b'))

st.write(shapiro_test(pst_veh_flat, 'Post-Treatment (Vehicle)'))
st.pyplot(plot_data_distribution(pst_veh_flat, 'Post-Treatment (Vehicle)', point_color='#1879CE'))

st.write(shapiro_test(pst_drug_flat, 'Post-Treatment (Drug)'))
st.pyplot(plot_data_distribution(pst_drug_flat, 'Post-Treatment (Drug)', point_color='#FC4F30'))


##### Logic for Bar plots #####

pre_veh_av_list = np.average(pre_veh, axis=1)
#print(pre_veh_av_list)

pre_veh_se_list = np.std(pre_veh, axis=1, ddof=1) / np.sqrt(pre_veh.shape[1])
#print(pre_veh_se_list)

pst_veh_av_list = np.average(pst_veh, axis=1)
#print(pst_veh_av_list)

pst_veh_se_list = np.std(pst_veh, axis=1, ddof=1) / np.sqrt(pst_veh.shape[1])
#print(pst_veh_se_list)

pre_drg_av_list = np.average(pre_drg, axis=1)
#print(pre_drg_av_list)

pre_drg_se_list = np.std(pre_drg, axis=1, ddof=1) / np.sqrt(pre_drg.shape[1])
#print(pre_drg_se_list)

pst_drg_av_list = np.average(pst_drg, axis=1)
#print(pst_drg_av_list)

pst_drg_se_list = np.std(pst_drg, axis=1, ddof=1) / np.sqrt(pst_drg.shape[1])
#print(pst_drg_se_list)

all_veh_av_list = list(pre_veh_av_list) + list(pst_veh_av_list)

all_veh_se_list = list(pre_veh_se_list) + list(pst_veh_se_list)

all_drg_av_list = list(pre_drg_av_list) + list(pst_drg_av_list)

all_drg_se_list = list(pre_drg_se_list) + list(pst_drg_se_list)

drg_minus_veh_list = [all_drg_av_list[i] - all_veh_av_list[i] for i in range(28)]

days = range(1, 29)

data = {
    'Days': days,
    'Vehicle Group': all_veh_av_list,
    'VehG Std Err': all_veh_se_list,
    'Drug Group': all_drg_av_list,
    'DrgG Std Err': all_drg_se_list,
    'Veh - Drg Vals': drg_minus_veh_list
}

# Fill missing values with empty strings to align column lengths
df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))

# Replace NaN with empty strings
df = df.fillna('')

# Round all numeric columns to 2 decimal places
df = df.round(2)

# Reset the index so that it does not display as a separate column
df_reset = df.set_index('Days')

# Set the colors for your plots
mpl.rcParams['axes.prop_cycle'] = cycler(color=['#42A7F5', '#ed7b7b', '#1879CE', '#FC4F30', '#C0EBC5'])

# Retrieve colors from Matplotlib
colors = [color for color in mpl.rcParams['axes.prop_cycle'].by_key()['color']]

# Function to apply custom color formatting based on 'Days' index and column name
def color_by_days_and_column(row):
    styles = []
    day = row.name  # Access the index value (Days)

    for col_name in row.index:
        value = row[col_name]
        
        # Apply different color rules based on 'Days' and column name
        if day < 17 and col_name == 'Vehicle Group':
            styles.append(f'color: {colors[0]}')  # Apply blue text for days 0-16, Vehicle Group
        elif day >= 17 and col_name == 'Vehicle Group':
            styles.append(f'color: {colors[2]}')  # Apply red text for days 17 and above, Vehicle Group
        elif col_name == 'Drug Group':
            # Example condition for Drug Group (customize as needed)
            if day < 17:
                styles.append(f'color: {colors[1]}')  # Apply green text for days 0-14, Drug Group
            else:
                styles.append(f'color: {colors[3]}')  # Apply orange text for days 15 and above, Drug Group
        elif col_name == 'Veh - Drg Vals':
            styles.append(f'color: {colors[4]}')
        else:
            styles.append('color: grey')  # Default color for any other cases
        
    return styles

# Apply the color formatting based on 'Days' index and column name
df_styled = df_reset.style.apply(color_by_days_and_column, axis=1)

st.title("Alcohol Consumption Data Table")
st.write("Daily consumption data means")
st.dataframe(df_styled)

# Calculate means and standard errors for bar plot
mean_pre_veh = np.mean(pre_veh_flat)
mean_pre_drg = np.mean(pre_drug_flat)

error_pre_veh = np.std(pre_veh_flat) / np.sqrt(len(pre_veh_flat)) # Standard error
error_pre_drg = np.std(pre_drug_flat) / np.sqrt(len(pre_drug_flat)) # Standard error

# X locations for the bars
x_pos = np.arange(2)

# Mean values for the groups
means = [mean_pre_veh, mean_pre_drg]
errors = [error_pre_veh, error_pre_drg]

plt.style.use('dark_background')

bar_colors = ['#42A7F5', '#ed7b7b']
error_bar_colors = ['#42A7F5', '#ed7b7b']  # Individually specify error bar colors
outline_color = 'black'  # Color for the outline of the error bars

# Plotting the bar chart with error bars
fig, ax = plt.subplots()  # Create a figure and an axis
bars = ax.bar(x_pos, means, yerr=errors, capsize=10, color=bar_colors, alpha=1)

# Adding error bars with outlines
for i, (bar, color, err) in enumerate(zip(bars, error_bar_colors, errors)):
    # Add thicker error bar outline
    ax.errorbar(bar.get_x() + bar.get_width() / 2, means[i], yerr=err, fmt='none', 
                ecolor=outline_color, elinewidth=2, capsize=10, capthick=4)
    
    # Add actual error bars on top of the outline
    ax.errorbar(bar.get_x() + bar.get_width() / 2, means[i], yerr=err, fmt='none', 
                ecolor=color, elinewidth=1, capsize=10, capthick=2)

# Customize the plot
ax.set_xticks(x_pos)
ax.set_xticklabels(['Con group prior Tx', 'Drug group prior Tx'])
ax.set_ylabel('Mean Intake (grams)')
#ax.set_title('Mean Control vs Drug Groups with Std Error')

# Customize the grid lines (set opacity)
ax.set_axisbelow(True)  # This ensures that grid lines are drawn behind bars
ax.grid(True, which='both', axis='y', linestyle='-', alpha=0.2)  # Set opacity of grid lines on y-axis

st.title('Prior Treatment Phase')
st.write('Error bars representing SEM')

# Display the plot in Streamlit
st.pyplot(fig)

# Calculate means and standard errors for bar plot
mean_pst_veh = np.mean(pst_veh_flat)
mean_pst_drg = np.mean(pst_drug_flat)

error_pst_veh = np.std(pst_veh_flat) / np.sqrt(len(pst_veh_flat)) # Standard error
error_pst_drg = np.std(pst_drug_flat) / np.sqrt(len(pst_drug_flat)) # Standard error

# X locations for the bars
x_pos = np.arange(2)

# Mean values for the groups
means = [mean_pst_veh, mean_pst_drg]
errors = [error_pst_veh, error_pst_drg]

plt.style.use('dark_background')

bar_colors = ['#1879CE', '#FC4F30']
error_bar_colors = ['#1879CE', '#FC4F30']  # Individually specify error bar colors
outline_color = 'black'  # Color for the outline of the error bars

# Plotting the bar chart with error bars
fig, ax = plt.subplots()  # Create a figure and an axis
bars = ax.bar(x_pos, means, yerr=errors, capsize=10, color=bar_colors, alpha=1)

# Adding error bars with outlines
for i, (bar, color, err) in enumerate(zip(bars, error_bar_colors, errors)):
    # Add thicker error bar outline
    ax.errorbar(bar.get_x() + bar.get_width() / 2, means[i], yerr=err, fmt='none', 
                ecolor=outline_color, elinewidth=2, capsize=10, capthick=4)
    
    # Add actual error bars on top of the outline
    ax.errorbar(bar.get_x() + bar.get_width() / 2, means[i], yerr=err, fmt='none', 
                ecolor=color, elinewidth=1, capsize=10, capthick=2)

# Customize the plot
ax.set_xticks(x_pos)
ax.set_xticklabels(['Con group after Tx', 'Drug group after Tx'])
ax.set_ylabel('Mean Intake (grams)')
#ax.set_title('Mean Control vs Drug Groups with Std Error')

# Flatten the arrays
pst_veh_flat = pst_veh.flatten()
pst_drg_flat = pst_drg.flatten()

# Perform independent t-test
t_stat, p_value = wilcoxon(pst_veh_flat, pst_drg_flat)

# Customize the grid lines (set opacity)
ax.set_axisbelow(True)  # This ensures that grid lines are drawn behind bars
ax.grid(True, which='both', axis='y', linestyle='-', alpha=0.2)  # Set opacity of grid lines on y-axis
ax.text(x=.78, y=.85, s="""*p < 0.05""", 
        transform=fig.transFigure, ha='left', fontsize=10, alpha=.7)
ax.text(x=.699, y=.55, s="""*""", 
        transform=fig.transFigure, ha='left', fontsize=15, alpha=.7)

st.title('During Treatment Phase')
st.write(f'Error bars representing SEM & Wilcoxon Signed-Rank test for p-val')
st.write(f'p-val = {p_value}')

# Display the plot in Streamlit
st.pyplot(fig)


st.title('Daily Alcohol Intake')
st.write('Error bars representing SEM & Wilcoxon Signed-Rank test for p-val')

pre_days = [*range(1,17,1)]
pst_days = [*range(16,28,1)]

mpl.rcParams['axes.prop_cycle'] = cycler(color=['#42A7F5', '#ed7b7b', '#1879CE', '#FC4F30'])
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.left'] = True
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.bottom'] = True
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.grid'] = True
mpl.rcParams['axes.grid.axis'] = 'y'
mpl.rcParams['grid.linewidth'] = 1
mpl.rcParams['grid.color'] = '#A5A5A5'
mpl.rcParams['axes.axisbelow'] = True
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.facecolor'] = 'white'
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0.2
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['font.size'] = 12
mpl.rcParams['xtick.major.pad'] = 1
mpl.rcParams['ytick.major.pad'] = 12
mpl.rcParams['axes.titlelocation'] = 'center'
mpl.rcParams['axes.titlepad'] = 20
mpl.rcParams['axes.titlesize'] = 15
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.labelpad'] = 20
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.minor.size'] = 1
mpl.rcParams['ytick.major.size'] = 0
mpl.rcParams['ytick.minor.size'] = 0

plt.style.use('dark_background')

fig, ax = plt.subplots(figsize=(14, 10))
ax.text(x=.82, y=.85, s="""*p < 0.05""", 
        transform=fig.transFigure, ha='left', fontsize=13, alpha=.7)
ax.text(x=.6172, y=.75, s="""*""", 
        transform=fig.transFigure, ha='left', fontsize=15, alpha=.7)
ax.text(x=.6710, y=.74, s="""*""", 
        transform=fig.transFigure, ha='left', fontsize=15, alpha=.7)
ax.text(x=.6979, y=.72, s="""*""",
        transform=fig.transFigure, ha='left', fontsize=15, alpha=.7)
ax.text(x=.7248, y=.85, s="""*""", 
        transform=fig.transFigure, ha='left', fontsize=15, alpha=.7)
ax.text(x=.7517, y=.67, s="""*""", 
        transform=fig.transFigure, ha='left', fontsize=15, alpha=.7)
ax.text(x=.7786, y=.74, s="""*""", 
        transform=fig.transFigure, ha='left', fontsize=15, alpha=.7)
ax.text(x=.8055, y=.63, s="""*""", 
        transform=fig.transFigure, ha='left', fontsize=15, alpha=.7)
ax.text(x=.8324, y=.62, s="""*""", 
        transform=fig.transFigure, ha='left', fontsize=15, alpha=.7)
ax.text(x=.8593, y=.6, s="""*""", 
        transform=fig.transFigure, ha='left', fontsize=15, alpha=.7)

plt.title("Liquid Alcohol Consumption")
plt.xlabel("Days of Alcohol Access")
plt.ylabel("Amount Consumed (grams)")

plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27],
           [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27])

plt.errorbar(pre_days, pre_veh_av_list, color='#42A7F5', label = "control group prior to treatment", 
             yerr=pre_veh_se_list,  linestyle = "-", fmt = "o", 
             elinewidth=0.7, capsize=1.5)
plt.errorbar(pre_days, pre_drg_av_list, color='#ed7b7b', label = "drug group prior to treatment",
              yerr=pre_drg_se_list,  linestyle = "-", fmt = "o", 
             elinewidth=0.7, capsize=1.5)
plt.errorbar(pst_days, pst_veh_av_list,color='#1879CE', label = "control group after treatment",
              yerr=pst_veh_se_list,  linestyle = "-", fmt = "o", 
              elinewidth=0.7, capsize=1.5)
plt.errorbar(pst_days, pst_drg_av_list, color='#FC4F30', label = "drug group after treatment",
              yerr=pst_drg_se_list,  linestyle = "-", fmt = "o", 
              elinewidth=0.7, capsize=1.5)
plt.legend(loc='upper left')

# Use st.pyplot to display matplotlib figures
st.pyplot(fig)
