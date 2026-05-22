import cv2
import time
import threading
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

class VideoStream:
    """Captura de vídeo otimizada em thread dedicada para não travar a IA."""
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
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
    print("[INFO] Fase 5: Iniciando Sistema de Propósito Específico (Dataset Original)")
    
    # Caminhos para os 3 modelos originais já treinados
    model_paths = [
        r"c:\Users\carol\testee\Detector_objetos_AI\DETECTOR\platano.pt",
        r"c:\Users\carol\testee\Detector_objetos_AI\DETECTOR\manzana.pt",
        r"c:\Users\carol\testee\Detector_objetos_AI\DETECTOR\tetra.pt"
    ]
    
    nomes_classes = ["Platano", "Manzana", "Tetrapack"]
    
    print("[INFO] Carregando os 3 modelos originais (Isso pode consumir memória)...")
    models = []
    try:
        for path in model_paths:
            models.append(YOLO(path))
        print("[SUCESSO] Modelos carregados!")
    except Exception as e:
        print(f"[ERRO FATAL] Falha ao carregar modelos: {e}")
        return

    print("[INFO] Acessando câmera...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

    prev_time = time.time()
    fps = 0.0

    print("[INFO] Pressione 'ESC' ou 'Q' para sair.")

    try:
        while True:
            frame = vs.read()
            if frame is None:
                continue
                
            frame = cv2.flip(frame, 1)
            
            # Inicializamos o anotador para desenhar tudo na MESMA imagem 
            # (Corrigindo o bug de escurecimento do projeto original)
            annotator = Annotator(frame, line_width=2)

            # Processa os 3 modelos
            for idx, model in enumerate(models):
                # Usando imgsz=320 para ficar MUITO mais rápido, já que temos 3 modelos simultâneos
                results = model.predict(source=frame, imgsz=320, conf=0.60, verbose=False)
                
                for r in results:
                    if r.boxes:
                        for box in r.boxes:
                            b = box.xyxy[0]  # get box coordinates
                            c = int(box.cls)
                            # Usamos o nome fixo da classe, já que cada modelo só tem 1 classe (0)
                            label = f"{nomes_classes[idx]} {float(box.conf):.2f}"
                            annotator.box_label(b, label, color=colors(idx, True))
                    
                    # Desenhar máscaras de segmentação se existirem
                    if r.masks is not None:
                        annotator.masks(r.masks.data, r.masks.orig_shape, colors=[colors(idx, True)])

            annotated_frame = annotator.result()

            # Cálculo Seguro de FPS
            current_time = time.time()
            time_diff = current_time - prev_time
            if time_diff > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / time_diff)
            prev_time = current_time

            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 30), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, "Fase 5 - Deteccao Customizada", (20, 60), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("DETECCION ORIGINAL", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        print("[INFO] Encerrando...")
        vs.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
