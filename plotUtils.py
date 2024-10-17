from matplotlib import pyplot as plt

plt.switch_backend('TkAgg')  # force matplot separate window


def showImage(image, title=None):
    plt.figure(figsize=(6, 9))
    plt.subplot(111)
    plt.imshow(image, cmap='gray')
    plt.title('Untitled' if title is None else title)

    plt.tight_layout()
    plt.show()


def compareToOg(image, original):
    plt.figure(figsize=(10, 9))
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.title('Processed')

    plt.subplot(122)
    plt.imshow(original, cmap='gray')
    plt.title('Original')

    plt.tight_layout()
    plt.show()
