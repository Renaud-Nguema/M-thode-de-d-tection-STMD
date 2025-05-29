
import numpy as np
import cv2
from scipy.signal import convolve2d
import math

################################################################################################################################################
######################################################## Written by EO ZUE NGUEMA PATRICK RENAUD ###############################################
################################################################################################################################################

# --- Paramètres Retina (couche de flou spatial) ---
sigma_retina = 0.2     # Écart-type pour le noyau gaussien
kernel_size_retina =3  # Taille du noyau (doit être impaire)

# --- Paramètres Lamina (filtrage temporel et inhibition latérale) ---
# Ordres et constantes de temps pour 4 noyaux Gamma successifs
n1, Tau1 = 2, 3
n2, Tau2 = 6, 9
n3, Tau3 = 2, 0.5
n4, Tau4 = 1, 0.3
# Dimensions du noyau d'inhibition latérale : (hauteur, largeur, profondeur temporelle)
size_W1 = (25, 25, 11)
# Écarts-types spatiaux pour les deux Gaussiennes de l'inhibition
sigma2 = 1.5
sigma3= 2*sigma2
# Constantes de temps pour la composante temporelle de l'inhibition
lambda1, lambda2 = 3, 9

# --- Fonctions utilitaires ---
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


def generalized_gamma_kernel(order: int, tau: float, width: int) -> np.ndarray:
    """
    Crée un noyau Gamma temporel 1D généralisé, normalisé.

        order (int): ordre du noyau Gamma (nombre d'intégrations).
        tau (float): constante de temps.
        width (int): nombre d'échantillons temporels.
        np.ndarray: vecteur 1D normalisé formant le noyau Gamma.
    """
    # Assure une largeur minimale pour éviter division par zéro
    width = max(width, 2)
    t = np.arange(width)
    # Formule du noyau Gamma généralisé
    kernel = ((order * t / tau)**order * np.exp(-order * t / tau) /
              (math.factorial(order - 1) * (tau**(order + 1))))
    # Normalisation du vecteur
    return kernel / np.sum(kernel)

# --- Étape 1 : Retina layer (flou spatial) ---
def apply_retina_blur(frame: np.ndarray) -> np.ndarray:
    """
    Applique un flou gaussien spatial à une trame couleur.

    1. Conversion BGR -> niveaux de gris
    2. Convolution avec le noyau gaussien

    Args:
        frame (np.ndarray): image BGR d'entrée.

    Returns:
        np.ndarray: image floutée en niveaux de gris (uint8).
    """
    # Conversion en niveaux de gris (1 canal)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Génération du noyau gaussien défini globalement
    kernel = gaussian_kernel(sigma_retina, kernel_size_retina)
    # Convolution 2D avec extension des bords par réplication
    blurred = cv2.filter2D(gray, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    # Conversion en entiers 8 bits pour affichage/écriture
    return blurred.astype(np.uint8)

# --- Étape 2 : Lamina layer (filtre temporel + inhibition latérale) ---
def lamina_layer(buffer: list) -> np.ndarray:
    """
        np.ndarray: résultat de la couche Lamina (float).
    """
    # 1. Somme de la sortie Retina et d'une approximation Lobula (ici même trame)
    stack = np.stack(buffer, axis=-1)  # forme (h, w, T)
    sum_RL = np.sum(stack, axis=-1)    # somme spatiale (h, w)

    # 2. Filtrage temporel par différence de deux noyaux Gamma (band-pass)
    G1 = generalized_gamma_kernel(n1, Tau1, int(3 * Tau1))
    G2 = generalized_gamma_kernel(n2, Tau2, int(3 * Tau2))
    # On applique la convolution temporelle sur l'axe horizontal du plan
    conv1 = convolve2d(sum_RL, G1[np.newaxis, :], mode='same', boundary='symm')
    conv2 = convolve2d(sum_RL, G2[np.newaxis, :], mode='same', boundary='symm')
    gamma_out = conv1 - conv2  # sortie band-pass

    # 3. Inhibition latérale spatiale (DoG)
    half_x, half_y = size_W1[0]//2, size_W1[1]//2
    xs = np.arange(-half_x, half_x + 1)
    ys = np.arange(-half_y, half_y + 1)
    X, Y = np.meshgrid(xs, ys)
    # Deux gaussiennes de variances différentes
    G_sp2 = np.exp(-(X**2 + Y**2) / (2 * sigma2**2))
    G_sp3 = np.exp(-(X**2 + Y**2) / (2 * sigma3**2))
    W_s = G_sp2 - G_sp3  # noyau spatial bipolaire

    # Composante temporelle de l'inhibition (différence exponentielle)
    t = np.arange(size_W1[2])
    W_t1 = np.exp(-t / lambda1) / lambda1
    W_t2 = np.exp(-t / lambda2) / lambda2
    W_t = W_t1 - W_t2

    # Application séquentielle : spatial puis temporal
    spatial = convolve2d(gamma_out, W_s, mode='same', boundary='symm')
    lateral = convolve2d(spatial, W_t[np.newaxis, :], mode='same', boundary='symm')
    return lateral

# --- Pipeline complet sur vidéo ---
def full_pipeline(input_path: str, output_path: str) -> None:
    """
    Traite une vidéo entière selon les couches Retina puis Lamina.

    1. Lit chaque trame de la vidéo d'entrée
    2. Applique le flou Retina, stocke dans un buffer circulaire
    3. Une fois plein, applique lamina_layer sur le buffer
    4. Normalise et écrit la trame de sortie
    En résumé, on génère quatre noyaux Gamma afin de couvrir plusieurs échelles temporelles de filtrage —
      c’est ce qui permet au modèle Lamina d’être à la fois sélectif aux changements lents ET rapides, 
      exactement comme le système visuel insecte.
    Entrée:
        input_path (str): chemin de la vidéo source.
    sortie:
        output_path (str): chemin du fichier vidéo de sortie.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Impossible d'ouvrir la vidéo : {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    buffer = []  # buffer circulaire pour le filtrage temporel
    BUFFER_SIZE = size_W1[2]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Étape Retina : floutage spatial
        retina = apply_retina_blur(frame).astype(float)
        buffer.append(retina)
        if len(buffer) > BUFFER_SIZE:
            buffer.pop(0)

        # Étape Lamina : une fois buffer rempli
        if len(buffer) == BUFFER_SIZE:
            lamina_out = lamina_layer(buffer)
            # Normalisation pour mise à l'échelle sur [0,255]
            norm = (lamina_out - lamina_out.min()) / (lamina_out.ptp() + 1e-6)
            lamina_norm = (norm * 255).astype(np.uint8)
            out.write(lamina_norm)
            cv2.imshow('Output Lamina', lamina_norm)

            # Quitter si 'q' est pressé
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Libération des ressources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    input_video = r"C:\Users\rngue\Documents\Projets\PFA\env\FESTMD ZNPR\video_output_retina_layer.mp4"
    output_video = r"C:\Users\rngue\Documents\Projets\PFA\env\FESTMD ZNPR\video_output_lamina_layer.mp4"
    full_pipeline(input_video, output_video)

