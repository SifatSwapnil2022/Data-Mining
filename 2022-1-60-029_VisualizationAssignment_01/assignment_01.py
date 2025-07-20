# -----------------------------
# Data Visualization Assignment 01
# Student ID: 2022-1-60-029
# MD SIFAT ULLAH SHEIKH
# Date: 2023-10-01
# CSE477 (DATA MINING)
# -----------------------------

import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter


student_id = "2022-1-60-029"
last5_digits = 60029
folder = f"{student_id}_VisualizationAssignment"
os.makedirs(folder, exist_ok=True)

np.random.seed(last5_digits)



data_uniform = np.random.uniform(50, 750, size=25)     
data_skewed = [1.2, 1.5, 1.6, 1.7, 2.0, 1.8, 2.1, 1.9, 2.5, 3.0,
               2.6, 3.1, 2.8, 3.3, 2.7, 3.5, 4.0, 20, 50, 90]     
data_normal = np.random.normal(loc=800, scale=75, size=25)       
data_small = [9999, 10000]                                       
data_large = np.random.randint(250, 10001, size=15000)           

dataset_collection = {
    'uniform': data_uniform,
    'skewed': data_skewed,
    'normal': data_normal,
    'small': data_small,
    'large': data_large
}



def describe_data(arr, tag):
    summary = f"--- {tag.title()} Dataset Summary ---\n"
    summary += f"Min    : {np.min(arr):.2f}\n"
    summary += f"Q1     : {np.percentile(arr, 25):.2f}\n"
    summary += f"Median : {np.median(arr):.2f}\n"
    summary += f"Q3     : {np.percentile(arr, 75):.2f}\n"
    summary += f"Max    : {np.max(arr):.2f}\n\n"
    return summary


summary_out = ""
for key, values in dataset_collection.items():
    summary_out += describe_data(values, key)

with open(f"{folder}/{student_id}_five_number_summary.txt", "w") as f:
    f.write(summary_out)



def plot_box(data, label):
    plt.figure(figsize=(4.5, 6))
    plt.boxplot(data)
    plt.title(f"{label.title()} Box Plot")
    plt.ylabel("Values")
    plt.grid(True, axis='y')
    plt.text(1, max(data) * 0.93, f"ID: {student_id}", ha='center', fontsize=8, alpha=0.5)
    plt.savefig(f"{folder}/{student_id}_box_{label}.png")
    plt.close()



def plot_hist(data, label, bin_count=8):
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=bin_count, edgecolor='black')
    plt.title(f"{label.title()} Histogram")
    plt.xlabel("Value Range")
    plt.ylabel("Frequency")
    plt.grid(True, axis='y')
    plt.text(0.5, 0.92, f"Student ID: {student_id}", transform=plt.gca().transAxes,
             ha='center', fontsize=8, alpha=0.6)
    plt.savefig(f"{folder}/{student_id}_hist_{label}.png")
    plt.close()


for key, values in dataset_collection.items():
    plot_box(values, key)
    bin_count = 60 if key == 'large' else 8
    plot_hist(values, key, bin_count)



fruits = ["Apple"] * 28 + ["Banana"] * 23 + ["Mango"] * 12 + ["Orange"] * 30 + ["Papaya"] * 7
fruit_counts = Counter(fruits)

plt.figure(figsize=(6, 4))
plt.bar(fruit_counts.keys(), fruit_counts.values(), color='lightgreen', edgecolor='black')
plt.title("Fruit Preference Count")
plt.xlabel("Fruit Type")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.text(0.5, 0.95, f"ID: {student_id}", transform=plt.gca().transAxes, ha='center', fontsize=8, alpha=0.5)
plt.savefig(f"{folder}/{student_id}_bar_categorical.png")
plt.close()



interpretation_lines = f"""{student_id}_box_uniform.png: Fairly symmetric spread — consistent uniform sampling.
{student_id}_box_skewed.png: Clear skew to right — presence of high-end outliers.
{student_id}_hist_normal.png: Approximates bell-shaped curve — fits normal distribution.
{student_id}_box_small.png: Flat-line box — only two distant values.
{student_id}_hist_large.png: Very dense distribution — wide value dispersion.
"""

with open(f"{folder}/{student_id}_interpretations.txt", "w") as f:
    f.write(interpretation_lines)



print(f" Completed! Files saved in: {folder}")
