import os
import matplotlib.pyplot as plt

def show_images(path_directory):

    for directory in os.listdir(path_directory):
        images = []
        if directory == '.DS_Store':
            continue
        for filename in os.listdir(path_directory + '/' + directory):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                images.append(os.path.join(path_directory + '/' + directory, filename))

        
        # create a table with 6 imaged 2 rows and 3 columns
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(directory, fontsize=16)
        for i, ax in enumerate(axs.flat):
            if i < len(images):
                ax.imshow(plt.imread(images[i]))
                ax.axis('off')

        plt.show()

        
            


if __name__ == '__main__':


    path_directory = 'results_subsetADE/results_subsetADE_images'


    show_images(path_directory)