import tkinter as tk
import subprocess
import os
def launch_form(form_name):
    # Launch the specified form
    if form_name == "Tunnel form Class":
        title = "Tunnel Form"
    elif form_name == "Vanilla form Class":
        title = "Vanilla Form"
    elif form_name == "Himalaya form Class":
        title = "Himalaya Form"
    elif form_name == "Napoleon form Class":
        title = "Napoleon Form"
    form_path = os.path.join(os.path.dirname(__file__), f"{form_name}.py")
    subprocess.Popen(["python", form_path, title])

def display_form():
    # Create the form
    form = tk.Tk()
    form.title("Form Selector")
    form.geometry("400x200")

    # Create the label
    label = tk.Label(form, text="Select a form to launch:")
    label.pack()

    # Create the buttons
    tunnel_button = tk.Button(form, text="Tunnel Form", command=lambda: launch_form("Tunnel form Class"))
    tunnel_button.pack()

    vanilla_button = tk.Button(form, text="Vanilla Form", command=lambda: launch_form("Vanilla form Class"))
    vanilla_button.pack()

    himalaya_button = tk.Button(form, text="Himalaya Form", command=lambda: launch_form("Himalaya form Class"))
    himalaya_button.pack()

    napoleon_button = tk.Button(form, text="Napoleon Form", command=lambda: launch_form("Napoleon form Class"))
    napoleon_button.pack()

    # Run the form
    form.mainloop()

display_form()