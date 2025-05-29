import numpy as np
import scipy.signal as signal
import cv2
import math
n=5
tau=10
def gamma_kernel(tau, n, size):
    """Génère un noyau Gamma 1D pour la convolution spatiale verticale.
    
    Args:
        tau (float): Paramètre de décroissance exponentielle
        n (int): Ordre du noyau Gamma
        size (int): Taille du noyau
    
    Returns:
        np.ndarray: Noyau Gamma normalisé (forme colonne)
    """
    t = np.linspace(0, size, size)
    kernel = (t ** (n - 1) * np.exp(-t / tau)) / (math.factorial(n - 1) * (tau ** n))
    return (kernel / np.sum(kernel)).reshape(-1, 1)  # Convertit en colonne pour convolution verticale

def apply_lobula_processing(medulla_output, output_path, tau=0.1, n=5, kernel_size=25):
    """Applique le traitement de la couche Lobula selon le modèle MATLAB.
    
    Args:
        medulla_output (str): Chemin vers la vidéo d'entrée (2 canaux: Tm3 et Tm1)
        output_path (str): Chemin de sortie pour la vidéo traitée
        tau (float): Paramètre temporel du noyau Gamma
        n (int): Ordre du noyau Gamma
        kernel_size (int): Taille du noyau Gamma
    
    Returns:
        bool: True si l'opération réussit
    """
    # Initialisation de la capture vidéo
    cap = cv2.VideoCapture(medulla_output)
    if not cap.isOpened():
        print(f"Erreur: Impossible d'ouvrir {medulla_output}")
        return False
    
    # Récupération des propriétés vidéo
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Création du VideoWriter pour la sortie
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), isColor=False)
    
    # Génération du noyau Gamma pour la convolution spatiale verticale
    gamma_k = gamma_kernel(tau, n, kernel_size)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Séparation des canaux Tm3 (B) et Tm1 (G) depuis l'entrée BGR
        tm3 = frame[:, :, 0].astype(float)  # Canal Bleu (Tm3)
        tm1 = frame[:, :, 1].astype(float)  # Canal Vert (Tm1)
        
        # Étape 1: Calcul de Lobula_out = Tm3 .* Tm1 (multiplication élément par élément)
        lobula_out = np.multiply(tm3, tm1)
        
        # Étape 2: Application de la convolution spatiale verticale avec le noyau Gamma
        flobula_out = signal.convolve2d(lobula_out, gamma_k, mode='same', boundary='symm')
        
        # Normalisation et conversion en uint8
        flobula_out = np.clip(flobula_out, 0, 255).astype(np.uint8)
        
        # Écriture de la sortie
        out.write(flobula_out)
        cv2.imshow("Lobula Processing", flobula_out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Nettoyage des ressources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Vidéo Lobula générée : {output_path}")
    return True

# Exemple d'utilisation
def full_pipeline_lobula(input_video):
    
    lobula_output = r"C:\Users\rngue\Documents\Projets\PFA\env\FESTMD ZNPR\video_output_lobula_layer.mp4"
    medulla_output = r"C:\Users\rngue\Documents\Projets\PFA\env\FESTMD ZNPR\video_output_medulla_layer.mp4"
    
   
    
    print("\nÉtape finale: Traitement Lobula...")
    if not apply_lobula_processing(medulla_output, lobula_output):
        return
    
    print("\nTraitement réussi!")
    print(f"Vidéo finale : {lobula_output}")

# Lancer le traitement avec Medulla
input_video = r"C:\Users\rngue\Documents\Projets\PFA\env\FESTMD ZNPR\video_output_medulla_layer.mp4"
full_pipeline_lobula(input_video)