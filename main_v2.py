import cv2
import time
import threading
import sys
from ultralytics import YOLO

class VideoStream:
    """Classe para captura de vídeo em thread dedicada.
    Maximiza o FPS evitando bloqueios de I/O da câmera."""
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW) # Otimização para Windows
        if not self.stream.isOpened():
            print("[ERRO FATAL] Não foi possível acessar a webcam.")
            sys.exit(1)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.lock:
            if not self.grabbed or self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.stopped = True
        self.stream.release()

def main():
    print("[INFO] Iniciando Sistema de Detecção Inteligente (Fase 4 - Produção)")
    print("[INFO] Carregando o modelo YOLOv8 ONNX (CPU Otimizado)...")
    
    try:
        # Tenta carregar o modelo ONNX otimizado. Se não existir, usa o .pt como fallback seguro.
        model = YOLO("yolov8n-seg.onnx", task="segment")
        print("[SUCESSO] Modelo ONNX carregado. Inferência em velocidade máxima.")
    except Exception as e:
        print(f"[AVISO] ONNX não encontrado ({e}). Usando PyTorch fallback (yolov8n-seg.pt).")
        try:
            model = YOLO("yolov8n-seg.pt")
        except Exception as e:
            print(f"[ERRO FATAL] Falha ao carregar qualquer modelo: {e}")
            sys.exit(1)

    print("[INFO] Acessando câmera...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0) # Tempo para a câmera aquecer e estabilizar

    prev_time = time.time()
    fps = 0.0

    print("[INFO] Sistema Operacional. Pressione 'ESC' ou 'Q' para sair.")
    
    # Classes COCO usadas como Proxy para o projeto: 39 (bottle/tetrapack), 46 (banana), 47 (apple)
    allowed_classes = [39, 46, 47]

    try:
        while True:
            frame = vs.read()
            if frame is None:
                print("[AVISO] Perda de sinal da câmera. Tentando reconectar...")
                time.sleep(0.5)
                continue
                
            # Flip para agir como um espelho
            frame = cv2.flip(frame, 1)

            # Inferência Segura com Tratamento de Erros
            try:
                # classes=allowed_classes filtra para detectar APENAS maçã, banana e garrafas
                results = model.predict(source=frame, imgsz=640, conf=0.55, classes=allowed_classes, verbose=False)
                # Renderiza a imagem com os boxes e máscaras
                annotated_frame = results[0].plot()
            except Exception as e:
                print(f"[ERRO NA INFERÊNCIA] {e}")
                annotated_frame = frame # Continua exibindo o vídeo limpo se a IA falhar

            # Cálculo Seguro de FPS
            current_time = time.time()
            time_diff = current_time - prev_time
            if time_diff > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / time_diff)
            prev_time = current_time

            # HUD Premium
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 30), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 255, 0), 2)
            cv2.putText(annotated_frame, "Detector IA v4 - Otimizado e Seguro", (20, 60), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("DETECCION Y SEGMENTACION (PRODUCAO)", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("[INFO] Interrupção por teclado detectada.")
    except Exception as e:
        print(f"[ERRO INESPERADO] {e}")
    finally:
        # Liberação Garantida de Recursos (Prevenção de Memory Leaks)
        print("[INFO] Encerrando recursos de vídeo e IA de forma segura...")
        vs.stop()
        cv2.destroyAllWindows()
        print("[INFO] Processo finalizado com sucesso.")

if __name__ == "__main__":
    main()
