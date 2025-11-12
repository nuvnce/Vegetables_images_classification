import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import shutil
import matplotlib.pyplot as plt
import time

# Paramètres configurables
BASE_DIR = 'dataset'  # Dossier contenant les images originales
OUTPUT_DIR = 'data_augmented'  # Dossier où les images augmentées seront sauvegardées
IMG_SIZE = (224, 224)  # Taille cible des images
NUM_AUGMENTED_PER_IMG = 10  # Nombre d'images augmentées par image originale

# Configuration du générateur d'augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]
)


def process_directory(src_dir, dst_dir):
    """Traite un dossier récursivement pour copier les images et générer leurs versions augmentées dans des sous-dossiers."""
    if not os.path.exists(src_dir) or not os.listdir(src_dir):
        raise ValueError(f"Le dossier {src_dir} est vide ou n'existe pas.")

    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir)

    for item in os.listdir(src_dir):
        src_item = os.path.join(src_dir, item)
        dst_item = os.path.join(dst_dir, item)

        if os.path.isdir(src_item):
            process_directory(src_item, dst_item)
        elif item.lower().endswith(('.jpg', '.jpeg', '.png')):
            if not os.path.exists(dst_item):
                os.makedirs(dst_item)
            shutil.copy2(src_item, dst_item)

            try:
                img = Image.open(src_item).convert('RGB').resize(IMG_SIZE)
                img_array = np.array(img)
                img_array = np.expand_dims(img_array, axis=0)

                i = 0
                for batch in datagen.flow(
                        img_array, batch_size=1, save_to_dir=dst_item,
                        save_prefix=f"{os.path.splitext(item)[0]}_aug", save_format='jpg'
                ):
                    i += 1
                    if i >= NUM_AUGMENTED_PER_IMG:
                        break
                print(f"Augmentation réussie: {src_item}")
            except Exception as e:
                print(f"Erreur lors de l'augmentation de {src_item}: {e}")


def visualize_sample(output_dir):
    """Affiche un échantillon de 5 images augmentées pour vérification, avec gestion des erreurs."""
    time.sleep(30)  # Délai pour éviter les erreurs de permission

    try:
        sample_dir = next(os.walk(output_dir))[1][0]  # Premier sous-dossier (bon ou mauvais)
        sub_dir = next(os.walk(os.path.join(output_dir, sample_dir)))[1][0]  # Premier sous-dossier d'image
        sample_files = os.listdir(os.path.join(output_dir, sample_dir, sub_dir))[:5]

        plt.figure(figsize=(15, 3))
        for i, file in enumerate(sample_files):
            try:
                img_path = os.path.join(output_dir, sample_dir, sub_dir, file)
                img = Image.open(img_path)
                plt.subplot(1, 5, i + 1)
                plt.imshow(img)
                plt.axis('off')
            except PermissionError:
                print(f"Permission refusée pour {img_path}, fichier ignoré.")
            except Exception as e:
                print(f"Erreur lors de l'ouverture de {img_path}: {e}")
        plt.savefig(os.path.join(output_dir, "augmentation_sample.png"))
        plt.close()
    except StopIteration:
        print("Aucun sous-dossier trouvé dans dataug, visualisation annulée.")
    except Exception as e:
        print(f"Erreur dans visualize_sample: {e}")


def main():
    """Exécute l'augmentation des données et affiche des statistiques."""
    print("Début de l'augmentation des données...")
    process_directory(BASE_DIR, OUTPUT_DIR)
    visualize_sample(OUTPUT_DIR)

    total_images = sum(len(files) for _, _, files in os.walk(OUTPUT_DIR) if files)
    print(f"Augmentation terminée. Total: {total_images} images dans {OUTPUT_DIR}")


if __name__ == "__main__":
    main()