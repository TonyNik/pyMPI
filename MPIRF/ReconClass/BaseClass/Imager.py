# coding=UTF-8
import matplotlib.pyplot as plot

'''
class ImagerClass(object):

    #Save the image data on the disk.
    def WriteImage(ImageData, filename):
        if ImageData.ndim==1:
            plot.bar(range(256), ImageData)
            plot.savefig(filename)
        else:
            plot.gray()
            plot.axis("off")
            plot.imshow(ImageData)
            plot.savefig(filename)

        plot.close()

        return True
'''

########### MPI RF's Version #############

class ImagerClass(object):

    #Save the image data on the disk.
    def WriteImage(ImageData,direction, filename):
        if ImageData.ndim==1:
            plot.bar(range(256), ImageData)
            plot.savefig(direction + filename)
        else:
            plot.gray()
            plot.imshow(ImageData)
            plot.savefig(direction+filename)

        plot.close()

        return True
