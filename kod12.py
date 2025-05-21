import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read

# === 1. Wczytanie obrazu panoramicznego ===
image_path = "zdjecia_3d/R0010631.JPG"  # Zamień na własną ścieżkę
room_image = cv2.imread(image_path)
room_image = cv2.cvtColor(room_image, cv2.COLOR_BGR2RGB)

height, width, _ = room_image.shape

# === 2. Wczytanie pliku B-Format (W, X, Y, Z) ===
wav_path = "pliki_wave/M0008_S02_R03_AMBIX.wav"  # Zamień na własną ścieżkę
sample_rate, audio_data = read(wav_path)

if audio_data.ndim != 2 or audio_data.shape[1] != 4:
    raise ValueError("Plik WAV musi mieć 4 kanały: W, X, Y, Z")

audio_data = audio_data / np.max(np.abs(audio_data))  # Normalizacja

W = audio_data[:, 0]
X = audio_data[:, 1]
Y = audio_data[:, 2]
Z = audio_data[:, 3]

# === 3. Funkcje pomocnicze ===
def bformat_to_direction(x, y, z):
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
    return np.degrees(azimuth), np.degrees(elevation)

def direction_to_image_coords(azimuth_deg, elevation_deg, width, height):
    x = (azimuth_deg + 180) / 360 * width
    y = (90 - elevation_deg) / 180 * height
    return int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1))

# === 4. Automatyczne znalezienie pozycji mikrofonu ===
max_index = np.argmax(np.abs(W))  # Największy pik
az_mic, el_mic = bformat_to_direction(X[max_index], Y[max_index], Z[max_index])
mic_x, mic_y = direction_to_image_coords(az_mic, el_mic, width, height)
microphone_position = (mic_x, mic_y)

# === 5. Przetwarzanie odbić z danych ambisonicznych ===
reflections = []
step = 500  # co ile próbek analizujemy
for i in range(0, len(W), step):
    az, el = bformat_to_direction(X[i], Y[i], Z[i])
    x_img, y_img = direction_to_image_coords(az, el, width, height)
    amplitude = np.abs(W[i])
    reflections.append((x_img, y_img, amplitude, az, el))

# === 6. Wypisanie parametrów odbić w terminalu ===
print("Odbicia dźwięku:")
for i, (x, y, amplitude, azimuth, elevation) in enumerate(reflections):
    print(f"{i+1:02d}: x={x}, y={y}, amp={amplitude:.3f}, azimuth={azimuth:.1f}°, elevation={elevation:.1f}°")

# === 7. Wizualizacja bez mapy hipsometrycznej ===
plt.figure(figsize=(12, 8))
plt.imshow(room_image, extent=(0, width, height, 0))

# Odbicia (punkty)
for x, y, amplitude, _, _ in reflections:
    plt.scatter(x, y, color='red', alpha=0.9, s=amplitude * 300)

# Pozycja mikrofonu
plt.scatter(*microphone_position, color='blue', s=200, marker='x', label='Pozycja mikrofonu')

plt.title("Odbicia dźwięku (B-Format) w pomieszczeniu")
plt.axis('off')
plt.legend()
plt.show()
