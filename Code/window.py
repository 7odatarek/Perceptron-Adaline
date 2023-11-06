import tkinter as tk
from tkinter import ttk
from main import workFlow


# Function to be called when the 'Submit' button is pressed
def submit_options():
    feature1 = feature1_select.get()
    feature2 = feature2_select.get()
    class1 = class1_select.get()
    class2 = class2_select.get()
    learning_rate = float(eta_entry.get())
    epochs = int(m_entry.get())
    mse_threshold = float(mse_threshold_entry.get())
    add_bias = bias_check_var.get()
    algorithm = algorithm_var.get()

    # Here you can add the code to process these options, like training a model
    print("Features selected:", feature1)
    print("Features selected:", feature2)
    print("Classes selected:", class1)
    print("Classes selected:", class2)
    print("Learning Rate (eta):", learning_rate)
    print("Number of Epochs (m):", epochs)
    print("MSE Threshold:", mse_threshold)
    print("Add Bias:", "Yes" if add_bias else "No")
    print("Algorithm:", algorithm)

    workFlow(feature1,feature2,class1,class2,learning_rate,epochs,mse_threshold,add_bias,algorithm)

# Root window
root = tk.Tk()
root.title("ML Parameter Selector")

# Feature selection
feature_label = ttk.Label(root, text="Select feature 1:")
feature_label.pack()
feature1_select = ttk.Combobox(root, values=["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "roundnes"])
feature1_select.set("Perimeter")
feature1_select.pack()

# Feature selection
feature_label = ttk.Label(root, text="Select feature 2:")
feature_label.pack()
feature2_select = ttk.Combobox(root, values=["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "roundnes"])
feature2_select.set("MinorAxisLength")
feature2_select.pack()

# Class selection
class_label = ttk.Label(root, text="Select two classes:")
class_label.pack()
class1_select = ttk.Combobox(root, values=["BOMBAY", "CALI","SIRA"])
class1_select.set("BOMBAY")
class1_select.pack()

# Class selection
class_label = ttk.Label(root, text="Select two classes:")
class_label.pack()
class2_select = ttk.Combobox(root, values=["BOMBAY", "CALI","SIRA"])
class2_select.set("CALI")
class2_select.pack()

# Learning rate entry
eta_label = ttk.Label(root, text="Enter learning rate (eta):")
eta_label.pack()
eta_entry = ttk.Entry(root)
eta_entry.insert(0,0.01)
eta_entry.pack()

# Number of epochs entry
m_label = ttk.Label(root, text="Enter number of epochs (m):")
m_label.pack()
m_entry = ttk.Entry(root)
m_entry.insert(0,100)
m_entry.pack()

# MSE threshold entry
mse_threshold_label = ttk.Label(root, text="Enter MSE threshold (mse_threshold):")
mse_threshold_label.pack()
mse_threshold_entry = ttk.Entry(root)
mse_threshold_entry.insert(0,0.001)
mse_threshold_entry.pack()

# Bias checkbox
bias_check_var = tk.BooleanVar()
bias_check = ttk.Checkbutton(root, text="Add bias", variable=bias_check_var)
bias_check.pack()

# Algorithm selection
algorithm_var = tk.StringVar()
algorithm_label = ttk.Label(root, text="Choose the used algorithm:")
algorithm_label.pack()
perceptron_radio = ttk.Radiobutton(root, text="Perceptron", variable=algorithm_var, value="Perceptron")
adaline_radio = ttk.Radiobutton(root, text="Adaline", variable=algorithm_var, value="Adaline")
perceptron_radio.pack()
adaline_radio.pack()

# Submit button
submit_button = ttk.Button(root, text="Submit", command=submit_options)
submit_button.pack()

# Start the GUI loop
root.mainloop()
