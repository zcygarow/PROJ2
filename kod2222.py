import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.interpolate import griddata

# === 1. Wczytanie obrazu panoramicznego ===
image_path = "zdjecia_3d/R0010624.JPG"
room_image = cv2.imread(image_path)
room_image = cv2.cvtColor(room_image, cv2.COLOR_BGR2RGB)

height, width, _ = room_image.shape
microphone_position = (width // 1.66, height // 2.26)

# === 2. Wczytanie pliku B-Format (W, X, Y, Z) ===
wav_path = "pliki_wave/M0008_S02_R03_AMBIX.wav"
sample_rate, audio_data = read(wav_path)

if audio_data.ndim != 2 or audio_data.shape[1] != 4:
    raise ValueError("Plik WAV musi mieć 4 kanały: W, X, Y, Z")

audio_data = audio_data / np.max(np.abs(audio_data))

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

# === 4. Przetwarzanie odbić z danych ambisonicznych ===
reflections = []
step = 500  # co ile próbek analizujemy
for i in range(0, len(W), step):
    az, el = bformat_to_direction(X[i], Y[i], Z[i])
    x_img, y_img = direction_to_image_coords(az, el, width, height)
    amplitude = np.abs(W[i])
    reflections.append((x_img, y_img, amplitude, az, el))

# === 5. Wypisanie parametrów odbić w terminalu ===
print("Odbicia dźwięku:")
for i, (x, y, amplitude, azimuth, elevation) in enumerate(reflections):
    print(f"{i+1:02d}: x={x}, y={y}, amp={amplitude:.3f}, azimuth={azimuth:.1f}°, elevation={elevation:.1f}°")

# === 6. Interpolacja do mapy hipsometrycznej ===
x_points = np.array([r[0] for r in reflections])
y_points = np.array([r[1] for r in reflections])
amplitudes = np.array([r[2] for r in reflections])

grid_x, grid_y = np.meshgrid(np.arange(0, width), np.arange(0, height))
grid_amplitudes = griddata((x_points, y_points), amplitudes, (grid_x, grid_y), method='cubic', fill_value=0)

# === 7. Wizualizacja odbić i mapy amplitud ===
plt.figure(figsize=(12, 8))
plt.imshow(room_image, extent=(0, width, height, 0))
plt.imshow(grid_amplitudes, extent=(0, width, height, 0), cmap='hot', alpha=0.6)

for x, y, amplitude, _, _ in reflections:
    plt.scatter(x, y, color='red', alpha=0.9, s=amplitude * 300)

plt.colorbar(label='Amplituda (W)')

# === 8. Znalezienie głównego źródła dźwięku ===
peak_index = np.argmax(np.abs(W))
peak_az, peak_el = bformat_to_direction(X[peak_index], Y[peak_index], Z[peak_index])
peak_x_img, peak_y_img = direction_to_image_coords(peak_az, peak_el, width, height)
peak_amp = np.abs(W[peak_index])

print(f"\n>>> Największy peak (źródło dźwięku):")
print(f"x={peak_x_img}, y={peak_y_img}, amp={peak_amp:.3f}, azimuth={peak_az:.1f}°, elevation={peak_el:.1f}°")

# === 9. Zaznaczenie źródła dźwięku na wykresie ===
plt.scatter(peak_x_img, peak_y_img, color='cyan', s=400, edgecolors='black', marker='*', label='Źródło dźwięku')

# === 10. Dodanie legendy i wyświetlenie wykresu ===
plt.title("Odbicia dźwięku (B-Format) w pomieszczeniu")
plt.legend(loc='upper right')
plt.axis('off')
plt.show()
