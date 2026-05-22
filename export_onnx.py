from ultralytics import YOLO
import sys

def main():
    print("[INFO] Exportando modelo YOLOv8 para ONNX (Otimização Máxima para CPU Local)...")
    try:
        model = YOLO("yolov8n-seg.pt")
        # Exporta para ONNX (ideal para inferência em CPU sem GPU)
        path = model.export(format="onnx", imgsz=640, opset=12)
        print(f"[SUCESSO] Modelo exportado e otimizado em: {path}")
    except Exception as e:
        print(f"[ERRO] Falha na exportação: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
