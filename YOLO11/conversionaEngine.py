from ultralytics import YOLO

# Carga tu modelo actual
model = YOLO('/Users/admin/PycharmProjects/CistemVisionIP/YOLO11/yolo11s.pt')

# Exportalo a TensorRT con precisión media (FP16)
# device=0 usa la GPU. half=True activa FP16 (crucial en Jetson).
model.export(format='engine', device=0, half=True)