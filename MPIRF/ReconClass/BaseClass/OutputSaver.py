# OutputSaver.py
import json
import os
import vtkmodules.all as vtk

'''
def save_simulation_message(message, filename):
    """
    Save the simulation message dictionary to a JSON file.
    """
    try:
        # Ensure the directory exists
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(message, f, indent=4)
        print(f"Simulation message saved to {filename}")
    except Exception as e:
        print(f"Error saving simulation message: {e}")


def save_vtk_screenshot(render_window, filename):
    """
    Capture a screenshot of the given VTK render window and save it as a PNG file.
    """
    try:
        # Ensure the directory exists
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Force a re-render to ensure the window is up-to-date.
        render_window.Render()

        # Capture the window to an image.
        window_to_image = vtk.vtkWindowToImageFilter()
        window_to_image.SetInput(render_window)
        window_to_image.Update()

        # Write the image to a PNG file.
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(filename)
        writer.SetInputData(window_to_image.GetOutput())
        writer.Write()
        print(f"VTK screenshot saved to {filename}")
    except Exception as e:
        print(f"Error saving VTK screenshot: {e}")
'''

# OutputSaver.py
import json
import os
import vtkmodules.all as vtk
import matplotlib.pyplot as plt
import numpy as np


# JSON extension
class JsonDefaultEncoding(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, complex):
            return f"{o.real}+{o.imag}i"
        elif isinstance(o, np.ndarray):
            return o.tolist()
        else:
            # As a fallback, try to convert to string.
            return str(o)



def save_simulation_message(message, filename):
    """
    Save the simulation message dictionary to a JSON file.
    Uses a custom encoder to handle numpy arrays, complex numbers, and other types.
    """
    try:
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(message, f, cls=JsonDefaultEncoding, indent=4)
        print(f"Simulation message saved to {filename}")
    except Exception as e:
        print(f"Error saving simulation message: {e}")



def save_vtk_screenshot(render_window, filename):
    """
    Capture a screenshot of the given VTK render window and save it as a PNG file.
    """
    try:
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        render_window.Render()  # Ensure up-to-date rendering
        window_to_image = vtk.vtkWindowToImageFilter()
        window_to_image.SetInput(render_window)
        window_to_image.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(filename)
        writer.SetInputData(window_to_image.GetOutput())
        writer.Write()
        print(f"VTK screenshot saved to {filename}")
    except Exception as e:
        print(f"Error saving VTK screenshot: {e}")


def save_projection_slice(ImgData, plane_index, slice_index, filename):
    """
    Save a 2D projection (or slice) of the reconstructed image to a file.

    Parameters:
      ImgData (numpy.ndarray): The reconstructed 3D image array.
      plane_index (int): The projection type (0: X-Y max projection, 1: single X-Y slice).
      slice_index (int): If plane_index is 1, the index of the slice to use.
      filename (str): The path to save the image (PNG format).
    """
    try:
        # Create a new figure
        plt.figure()
        l, r, o = np.shape(ImgData)
        if plane_index == 0:
            # Maximum intensity projection in the X-Y plane.
            Ixy = np.zeros((l, r))
            for i in range(l):
                for j in range(r):
                    Ixy[i, j] = np.max(ImgData[i, j, :])
            Ixy = Ixy / np.max(Ixy)
            plt.imshow(Ixy, cmap=plt.get_cmap("binary"))
            plt.title("X-Y Max Projection")
        elif plane_index == 1:
            # Single slice in the X-Y plane.
            Ixy = ImgData[:, :, slice_index]
            Ixy = Ixy / np.max(Ixy)
            plt.imshow(Ixy, cmap=plt.get_cmap("binary"))
            plt.title("X-Y Slice")
        else:
            print(f"Projection for plane index {plane_index} not implemented.")
            plt.close()
            return

        plt.axis("off")
        # Save the figure to file with minimal borders.
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Projection slice saved to {filename}")
    except Exception as e:
        print(f"Error saving projection slice: {e}")
