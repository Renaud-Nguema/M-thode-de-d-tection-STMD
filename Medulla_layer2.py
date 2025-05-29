import numpy as np
import scipy.signal as signal
import cv2
import math
from collections import deque

##############################################################################
# PARAMÈTRES OPTIMISÉS POUR LA COUCHE MEDULLA
##############################################################################

# Paramètres temporels (optimisés pour 30 FPS)
BUFFER_DURATION = 0.25  # 250 ms de mémoire temporelle
TAU_S = 1            # Constante de temps pour S⁺/S⁻ (légèrement augmenté)
TAU_M = 2             # Constante de temps pour Tm1 (ajusté empiriquement)

# Paramètres spatiaux
INHIBITION_KERNEL_SIZE = 9  # Taille du noyau d'inhibition (doit être impair)
#SIGMA_CENTER = 2           # Sigma pour la zone excitatrice
#SIGMA_SURROUND = 1       # Sigma pour la zone inhibitrice
"""
Si sigma de la zone excitatrice est supérieure à celle de la zone inhibitrice, alors, on aura un fond noir et la cible en blanc qui aparaitra plus grande ensuite
mais sans grande atténuation d'informations
    """
SIGMA_CENTER = 1.5           # Sigma pour la zone excitatrice
SIGMA_SURROUND = 3       # Sigma pour la zone inhibitrice
# entrainent un fond blanc avec une cible pas très bien rreprésentée

##############################################################################
# FONCTIONS DE BASE AVEC COMMENTAIRES DÉTAILLÉS
##############################################################################

def generate_inhibition_kernel(size=INHIBITION_KERNEL_SIZE):
    """
    Génère un noyau d'inhibition latérale par différence de Gaussiennes (DoG)
    
    Returns:
        np.ndarray: Noyau de convolution normalisé de forme (size, size)
    """
    # Création d'une grille centrée
    half_size = size // 2
    x = np.linspace(-half_size, half_size, size)
    y = np.linspace(-half_size, half_size, size)
    xx, yy = np.meshgrid(x, y)
    
    # Calcul des deux composantes gaussiennes
    g_center = np.exp(-(xx**2 + yy**2) / (2 * SIGMA_CENTER**2))
    g_surround = np.exp(-(xx**2 + yy**2) / (2 * SIGMA_SURROUND**2))
    
    # Différence de Gaussiennes normalisée
    dog_kernel = g_center - 0.9 * g_surround  # Pondération ajustée
    return dog_kernel / np.sum(np.abs(dog_kernel))

def temporal_gamma_filter(tau, n_samples):
    """
    Génère un noyau Gamma 1D pour le filtrage temporel
    
    Args:
        tau (float): Constante de temps du filtre
        n_samples (int): Nombre d'échantillons temporels
        
    Returns:
        np.ndarray: Noyau Gamma normalisé de forme (n_samples,)
    """
    t = np.arange(n_samples)
    kernel = (t**(tau-1) * np.exp(-t/tau)) 
    kernel /= (math.factorial(int(tau)-1) * (tau**tau))  # Conversion explicite en int
    return kernel / kernel.sum()

##############################################################################
# FONCTION PRINCIPALE OPTIMISÉE
##############################################################################

def optimized_medulla_processing(input_path, output_path):
    """
    Pipeline optimisé de traitement de la couche Medulla avec :
    - Buffer circulaire dynamique
    - Filtrage temporel incrémental
    - Normalisation adaptative
    
    Args:
        input_path (str): Chemin de la vidéo d'entrée
        output_path (str): Chemin de sauvegarde de la vidéo traitée
    """
    # Initialisation des flux vidéo
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Impossible d'ouvrir le fichier {input_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    # Calcul dynamique de la taille du buffer
    buffer_size = max(2, int(fps * BUFFER_DURATION))
    
    # Pré-calcul des noyaux
    inhibition_kernel = generate_inhibition_kernel()
    temporal_kernel_s = temporal_gamma_filter(TAU_S, buffer_size)
    temporal_kernel_m = temporal_gamma_filter(TAU_M, buffer_size)
    
    # Initialisation des buffers
    s_plus_buffer = deque(maxlen=buffer_size)
    s_minus_buffer = deque(maxlen=buffer_size)
    
    # Création du writer de sortie
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size, isColor=False)
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Pré-traitement de l'image
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray.astype(float), (5, 5), 1.5)  # Réduction de bruit
            
            # Séparation des voies ON/OFF avec seuil adaptatif
            mean_val = np.mean(gray)
            s_plus = np.maximum(gray - mean_val, 0)
            s_minus = np.maximum(mean_val - gray, 0)
            
            # Inhibition latérale
            s_plus_inhibited = signal.convolve2d(s_plus, inhibition_kernel, 
                                                mode='same', boundary='symm')
            s_minus_inhibited = signal.convolve2d(s_minus, inhibition_kernel,
                                                 mode='same', boundary='symm')
            
            # Mise à jour des buffers
            s_plus_buffer.append(s_plus_inhibited)
            s_minus_buffer.append(s_minus_inhibited)
            
            # Filtrage temporel pondéré
            tm3 = sum(buf * kernel for buf, kernel in 
                     zip(s_plus_buffer, temporal_kernel_s[-len(s_plus_buffer):]))
            tm2 = sum(buf * kernel for buf, kernel in 
                     zip(s_minus_buffer, temporal_kernel_s[-len(s_minus_buffer):]))
            
            # Calcul de Tm1 avec filtrage récursif
            tm1 = sum(tm3 * kernel for kernel in temporal_kernel_m)
            
            # Post-traitement
            output = cv2.normalize(tm1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Écriture et affichage
            out.write(output)
            cv2.imshow('Medulla Output', output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Erreur de traitement: {str(e)}")
        return False
    
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    print(f"Traitement Medulla réussi! Sortie: {output_path}")
    return True

##############################################################################
# PIPELINE COMPLET AVEC GESTION DES ERREURS
##############################################################################

def neurovision_pipeline(input_path, output_path):
    """
    Exécute le pipeline complet de vision bio-inspirée avec :
    - Gestion des dépendances
    - Feedback visuel
    - Chronométrage des opérations
    """
    print("[NeuroVision Pipeline] Démarrage...")
    
    try:
        # Étape Medulla
        print("- Traitement Medulla en cours...")
        if not optimized_medulla_processing(input_path, output_path):
            return False
        
        print("\n[SUCCÈS] Pipeline exécuté avec succès!")
        return True
    
    except Exception as e:
        print(f"\n[ERREUR] Échec du pipeline: {str(e)}")
        return False

# Exemple d'utilisation
if __name__ == "__main__":
    input_video = r"C:\Users\rngue\Documents\Projets\PFA\env\FESTMD ZNPR\video_output_lamina_layer.mp4"
    output_video = r"C:\Users\rngue\Documents\Projets\PFA\env\FESTMD ZNPR\video_output_medulla_layer.mp4"
    
    neurovision_pipeline(input_video, output_video)