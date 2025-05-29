import cv2
import numpy as np

################################################################################################################################################
######################################################## Written by EO ZUE NGUEMA PATRICK RENAUD ###############################################
################################################################################################################################################


def gaussian_kernel(sigma: float, size: int) -> np.ndarray:
    """
    Génère un noyau gaussien 2D normalisé de dimension (size x size).

    Args:
        sigma (float): écart-type de la gaussienne.
        size (int): taille du noyau (impaire de préférence).

    Returns:
        np.ndarray: matrice 2D du filtre gaussien dont la somme vaut 1.
    """
    # Création d'un axe centré autour de zéro
    ax = np.linspace(-(size//2), size//2, size)
    xx, yy = np.meshgrid(ax, ax)
    # Calcul de la fonction gaussienne
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    # Normalisation pour que la somme des coefficients soit égale à 1
    return kernel / np.sum(kernel)


def process_video(input_path: str,
                  output_path: str,
                  sigma: float = 0.8,
                  ksize: int = 3):
    """
    Traite une vidéo pour appliquer un flou gaussien sur chaque image en niveaux de gris.

    Paramètres :
    - input_path (str) : chemin vers la vidéo d'entrée.
    - output_path (str) : chemin où enregistrer la vidéo traitée.
    - sigma (float) : écart-type du filtre gaussien.
    - ksize (int) : taille du noyau du filtre (doit être impair).
    """
    # --- Ouverture de la vidéo d'entrée ---------------------------------------
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Impossible d'ouvrir la vidéo : {input_path!r}")

    # --- Lecture des propriétés vidéo ----------------------------------------
    fps = cap.get(cv2.CAP_PROP_FPS)  # Nombre d'images par seconde
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # Largeur en pixels
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Hauteur en pixels

    # --- Préparation du VideoWriter pour la sortie ----------------------------
    # fourcc : codec vidéo (mp4v pour .mp4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # isColor=False car on écrira des images en niveaux de gris (1 canal)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    # --- Création du noyau gaussien -------------------------------------------
    kernel = gaussian_kernel(sigma, ksize)

    print(f"Début du traitement : {input_path} -> {output_path}")
    print(f"  Résolution : {width}x{height}, FPS : {fps:.1f}")
    print(f"  Filtre gaussien : taille={ksize}, sigma={sigma}")

    # --- Boucle principale de traitement image par image ----------------------
    while True:
        ret, frame = cap.read()
        # Si plus d'images, on sort de la boucle
        if not ret:
            break

        # Conversion en niveaux de gris
        # cv2.COLOR_BGR2GRAY : convertit l'image BGR (3 canaux) en gris (1 canal)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Application du filtre gaussien avec bordures par répétition
        # borderType=cv2.BORDER_REPLICATE imite 'replicate' de MATLAB
        blurred = cv2.filter2D(
            src=gray,
            ddepth=-1,         # même profondeur que l'original (uint8)
            kernel=kernel,
            borderType=cv2.BORDER_REPLICATE #remplit les pixels hors-image en répliquant la valeur du pixel le plus proche en bordure. 
        )

        # Écriture de l'image floutée dans le fichier de sortie
        # Le VideoWriter attend une image mono-canal si isColor=False
        out.write(blurred)

        # Affichage en direct (utile pour debug) :
        cv2.imshow('Original (gris)', gray)
        cv2.imshow('Flouté (gaussien)', blurred)

        # Sortir si l'utilisateur presse 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Libération des ressources -------------------------------------------
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Traitement terminé et fichier de sortie enregistré.")


if __name__ == "__main__":
    # Exemple d'utilisation : à adapter selon vos chemins
    video_in = r"C:\Users\rngue\Documents\Projets\PFA\Video_IR\IR_AIRPLANE_035.mp4"                  
    video_out = r"C:\Users\rngue\Documents\Projets\PFA\env\FESTMD ZNPR\video_output_retina_layer.mp4"
    # Paramètres du flou : sigma=1.0, taille du noyau=3 (impair)
    process_video(video_in, video_out, sigma=0.2, ksize=1)



 # BIRD_038 
""" la détection fonctionne uniquement pour 1 cible et alterne entre le 1er oiseau et le second; le multiciblage est donc compliqué
         """
# IR_HELICOPTER_054
""" la détection marche la premiere partie de la vidéo mais à la seconde partie, dès la couche Lamaina, la détection devient mauvaise
    donc le suivi aussi sera mauvais"""