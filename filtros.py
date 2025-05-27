import cv2 as cv
import matplotlib.pyplot as plt
import os

def process_image(img_path):
    img = cv.imread(img_path)
    if img is None:
        return None, None #Se a imagem não puder ser lida, retorna None
    # Redimensionar para 128x128
    img_resized = cv.resize(img, (128, 128))

    # Filtro gaussiano
    img_blur = cv.GaussianBlur(img_resized, (15, 15), 0)

    # Equalização de histograma (apenas para imagens em escala de cinza)
    img_gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)
    img_eq = cv.equalizeHist(img_gray)
    return img_blur, img_eq

def main():
    input_folder = "imagens"
    blur_imgs = []
    eq_imgs = []
    filenames = []
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            img_blur, img_eq = process_image(input_path)
            if img_blur is not None and img_eq is not None:
                blur_imgs.append(cv.cvtColor(img_blur, cv.COLOR_BGR2RGB))  # Para mostrar em RGB
                eq_imgs.append(img_eq)
                filenames.append(filename)

    # Mostrar imagens com blur
    plt.figure(figsize=(4*len(blur_imgs), 4))
    plt.suptitle("Imagens com Gaussian Blur", fontsize=16)
    for i, img in enumerate(blur_imgs):
        plt.subplot(1, len(blur_imgs), i+1)
        plt.imshow(img)
        plt.title(f'{[i]}')
        plt.axis('off')
    plt.show()

    # Mostrar imagens com equalização de histograma
    plt.figure(figsize=(4*len(eq_imgs), 4))
    plt.suptitle("Imagens com Equalizacao Histograma", fontsize=16)
    for i, img in enumerate(eq_imgs):
        plt.subplot(1, len(eq_imgs), i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f'{[i]}')
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
