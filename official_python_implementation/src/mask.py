import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def create_mask():
    # Generate a random 28x28 array with binary values (0 or 1)
    random_bits = np.random.randint(2, size=(28, 28), dtype=np.uint8)

    # Convert the binary values to a proper image format (0 or 255)
    random_image = random_bits * 255

    kernel = np.array([[0.0625, 0.125 , 0.0625],
                        [0.125 , 0.25  , 0.125 ],
                        [0.0625, 0.125 , 0.0625]])
    
    for _ in range(0,10):
        random_image = cv.filter2D(random_image, -1, kernel)
    
    return (random_image >= 127.5).astype(np.uint8), (random_image <= 127.5).astype(np.uint8)
    

m1, m2  = create_mask()
# Display the image using matplotlib
plt.imshow(m1, cmap='gray')
plt.title("Random 28x28 Bit Image")
plt.show()
plt.imshow(m2, cmap='gray')
plt.title("Random 28x28 Bit Image2")
plt.show()
plt.imshow(m2+m1, cmap='gray')
plt.title("Random 28x28 Bit Image2")
plt.show()