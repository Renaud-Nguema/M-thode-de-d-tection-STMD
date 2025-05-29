import cv2
import os

################################################################################################################################################
######################################################## Written by EO ZUE NGUEMA PATRICK RENAUD ###############################################
################################################################################################################################################


# Définition du chemin de la vidéo d'entrée (modifiez cette variable directement)
input_video = r"C:\Users\rngue\Documents\Projets\PFA\Video_IR\IR_HELICOPTER_054.mp4"
if not os.path.isfile(input_video):
    raise FileNotFoundError(f"Le fichier vidéo spécifié n'existe pas : {input_video}")

# Paramètres Retina
sigma_retina = 1     # Écart-type pour le noyau gaussien
kernel_size_retina =3

# Paramètres Lamina
n1, Tau1 = 2, 3
n2, Tau2 = 6, 9
size_W1 = (25, 25, 11)
sigma2 = 1.5
sigma3 = 2 * sigma2
lambda1, lambda2 = 3.0, 9.0

# Paramètres Medulla
BUFFER_DURATION = 0.25
TAU_S = 1
TAU_M = 2
INHIBITION_KERNEL_SIZE = 11
SIGMA_CENTER = 1.5
SIGMA_SURROUND = 3.0

# Paramètres Lobula
tau_lobula = 2.0
n_lobula = 5
kernel_size_lobula = 25

# Préparation des noms de fichiers de sortie
base, ext = os.path.splitext(os.path.basename(input_video))
retina_out  = f"{base}_retina{ext}"
lamina_out  = f"{base}_lamina{ext}"
medulla_out = f"{base}_medulla{ext}"
lobula_out  = f"{base}_lobula{ext}"

# Importe les modules de chaque couche
from retina_layer1 import process_video as retina_layer
from Lamina_layer2 import full_pipeline as lamina_layer
from Medulla_layer2 import optimized_medulla_processing as medulla_layer
from lobula_layer2 import apply_lobula_processing as lobula_layer

# 1) Retina
print("[1/4] Couche Retina : flou gaussien spatial...")
retina_layer(input_video, retina_out, sigma_retina, kernel_size_retina)

# 2) Lamina
print("[2/4] Couche Lamina : filtrage temporel + inhibition latérale...")
lamina_layer(retina_out, lamina_out,
            
             )

# 3) Medulla
print("[3/4] Couche Medulla : voies ON/OFF + filtrage...")
medulla_layer(lamina_out, medulla_out,
              buffer_duration=BUFFER_DURATION,
              tau_s=TAU_S, tau_m=TAU_M,
              inhibition_size=INHIBITION_KERNEL_SIZE,
              sigma_center=SIGMA_CENTER, sigma_surround=SIGMA_SURROUND)

# 4) Lobula
print("[4/4] Couche Lobula : STMD final...")
lobula_layer(medulla_out, lobula_out,
             tau=tau_lobula, n=n_lobula, kernel_size=kernel_size_lobula)

# Lecture et affichage comparatif
print("[Terminé] Affichage comparatif de l'original et du traité...")
cap_in = cv2.VideoCapture(input_video)
cap_out = cv2.VideoCapture(lobula_out)
fps = cap_in.get(cv2.CAP_PROP_FPS)

while True:
    ret_in, frame_in = cap_in.read()
    ret_out, frame_out = cap_out.read()
    if not ret_in or not ret_out:
        break

    cv2.imshow('Original', frame_in)
    cv2.imshow('Traité', frame_out)

    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

cap_in.release()
cap_out.release()
cv2.destroyAllWindows()

