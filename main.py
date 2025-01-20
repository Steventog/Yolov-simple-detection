import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageTk
import torch
from datetime import datetime
from collections import Counter
from sort import Sort

class ObjectDetectionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Détection d'objets - YOLOv8")
        # Définir une taille de fenêtre raisonnable
        self.window.geometry("1280x720")

        try:
            print("Chargement du modèle YOLOv8...")
            self.model = YOLO('yolov8n.pt')
            print("Modèle chargé avec succès!")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {str(e)}")
            messagebox.showerror("Erreur", f"Erreur lors du chargement du modèle YOLOv8: {str(e)}")
            sys.exit(1)
        
        # Variables de contrôle
        self.is_detecting = False
        self.video_source = None
        self.detection_counts = Counter()
        self.current_video_path = None
        self.tracker = None
        
        # Dimensions initiales du canvas
        self.canvas_width = 960  # 75% de la largeur de la fenêtre
        self.canvas_height = 540  # Ratio 16:9
        
        # Créer les widgets
        self.create_widgets()
        
        # Configurer le redimensionnement
        self.window.grid_rowconfigure(1, weight=1)
        self.window.grid_columnconfigure(0, weight=1)
        self.window.bind('<Configure>', self.on_window_resize)

    def on_window_resize(self, event):
        # Ne pas réagir aux événements de redimensionnement des widgets enfants
        if event.widget != self.window:
            return
            
        # Calculer les nouvelles dimensions en conservant le ratio 16:9
        # Utiliser 75% de la largeur de la fenêtre pour le canvas
        new_width = int(event.width * 0.75)
        new_height = int(event.width * 0.75 * 9/16)
        
        # S'assurer que la hauteur ne dépasse pas la hauteur de la fenêtre
        if new_height > event.height * 0.9:  # Garder 10% de marge
            new_height = int(event.height * 0.9)
            new_width = int(new_height * 16/9)
        
        self.canvas_width = new_width
        self.canvas_height = new_height
        
        # Redimensionner le canvas
        self.canvas.config(width=self.canvas_width, height=self.canvas_height)

    def create_widgets(self):
        try:
            # Frame pour les boutons en haut
            button_frame = tk.Frame(self.window)
            button_frame.pack(side=tk.TOP, pady=10)
            
            # Frame pour les boutons de gauche (détection)
            detection_buttons = tk.Frame(button_frame)
            detection_buttons.pack(side=tk.LEFT, padx=20)
            
            # Boutons de détection
            self.realtime_btn = tk.Button(detection_buttons, text="Détection en temps réel", 
                                        command=self.toggle_realtime_detection,
                                        width=25, height=2)
            self.realtime_btn.pack(side=tk.LEFT, padx=5)
            
            self.video_btn = tk.Button(detection_buttons, text="Charger une vidéo", 
                                     command=self.start_video_detection,
                                     width=25, height=2)
            self.video_btn.pack(side=tk.LEFT, padx=5)
            
            # Frame pour les boutons de contrôle vidéo
            self.video_controls = tk.Frame(button_frame)
            self.video_controls.pack(side=tk.LEFT, padx=20)
            
            # Boutons de contrôle vidéo
            self.stop_video_btn = tk.Button(self.video_controls, 
                                          text="Arrêter la vidéo", 
                                          command=self.stop_video,
                                          width=15, height=2,
                                          state=tk.DISABLED)
            self.stop_video_btn.pack(side=tk.LEFT, padx=5)
            
            self.new_video_btn = tk.Button(self.video_controls, 
                                         text="Nouvelle vidéo", 
                                         command=self.load_new_video,
                                         width=15, height=2,
                                         state=tk.DISABLED)
            self.new_video_btn.pack(side=tk.LEFT, padx=5)
            
            # Frame principale qui contiendra le canvas et les compteurs
            main_frame = tk.Frame(self.window)
            main_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)
            
            # Canvas pour l'affichage à gauche
            self.canvas = tk.Canvas(main_frame, width=self.canvas_width, height=self.canvas_height, bg='black')
            self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
            
            # Frame pour les compteurs à droite
            counter_frame = tk.Frame(main_frame, width=300)
            counter_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=20)
            counter_frame.pack_propagate(False)
            
            # Label pour les compteurs
            self.counter_label = tk.Label(counter_frame, 
                                        text="Objets détectés:", 
                                        font=('Arial', 14, 'bold'))
            self.counter_label.pack(pady=10)
            
            # Frame pour la liste des objets détectés
            self.detection_frame = tk.Frame(counter_frame)
            self.detection_frame.pack(pady=5, fill=tk.BOTH, expand=True)
            
            # Bouton de sauvegarde
            self.save_btn = tk.Button(counter_frame, 
                                    text="Sauvegarder les statistiques", 
                                    command=self.save_statistics,
                                    width=25, height=2,
                                    state=tk.DISABLED)
            self.save_btn.pack(pady=20, side=tk.BOTTOM)
            
            print("Interface créée avec succès!")
        except Exception as e:
            print(f"Erreur lors de la création de l'interface: {str(e)}")
            messagebox.showerror("Erreur", f"Erreur lors de la création de l'interface: {str(e)}")
            sys.exit(1)
            
    def update_counter_display(self):
        # Effacer les anciens labels
        for widget in self.detection_frame.winfo_children():
            widget.destroy()
            
        # Créer de nouveaux labels pour chaque type d'objet
        for obj, count in self.detection_counts.most_common():
            label = tk.Label(self.detection_frame, 
                           text=f"{obj}: {count}",
                           font=('Arial', 12))
            label.pack(pady=5)
    
    def save_statistics(self):
        if not self.detection_counts:
            messagebox.showwarning("Attention", "Aucune donnée à sauvegarder!")
            return
            
        try:
            # Créer un nom de fichier avec la date et l'heure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_name = "realtime"
            if self.current_video_path:
                video_name = self.current_video_path.split('/')[-1].split('.')[0]
                
            filename = f"detection_stats_{video_name}_{timestamp}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Statistiques de détection - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Source: {'Temps réel' if not self.current_video_path else self.current_video_path}\n")
                f.write("\nObjets uniques détectés:\n")
                for obj, count in self.detection_counts.most_common():
                    f.write(f"{obj}: {count}\n")
                    
            messagebox.showinfo("Succès", f"Statistiques sauvegardées dans {filename}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la sauvegarde: {str(e)}")
    
    def toggle_realtime_detection(self):
        try:
            if not self.is_detecting:
                self.video_source = cv2.VideoCapture(0)
                if not self.video_source.isOpened():
                    raise Exception("Impossible d'accéder à la webcam")
                self.is_detecting = True
                self.realtime_btn.config(text="Arrêter la détection")
                self.current_video_path = None
                self.detection_counts.clear()
                self.tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.3)
                
                # Désactiver les boutons vidéo pendant la détection en temps réel
                self.video_btn.config(state=tk.DISABLED)
                self.stop_video_btn.config(state=tk.DISABLED)
                self.new_video_btn.config(state=tk.DISABLED)
                
                self.save_btn.config(state=tk.NORMAL)
                print("Démarrage de la détection en temps réel")
                self.detect_realtime()
            else:
                self.is_detecting = False
                self.realtime_btn.config(text="Détection en temps réel")
                if self.video_source:
                    self.video_source.release()
                    
                # Réactiver les boutons vidéo
                self.video_btn.config(state=tk.NORMAL)
                self.stop_video_btn.config(state=tk.DISABLED)
                self.new_video_btn.config(state=tk.DISABLED)
                
                print("Arrêt de la détection en temps réel")
        except Exception as e:
            print(f"Erreur lors de la détection en temps réel: {str(e)}")
            messagebox.showerror("Erreur", f"Erreur lors de la détection en temps réel: {str(e)}")
            self.is_detecting = False
            if self.video_source:
                self.video_source.release()
    
    def process_detections(self, results):
        detections = []
        class_names = []
        
        # Convertir les détections YOLO en format SORT
        for i, box in enumerate(results.boxes.xyxy):
            box = box.cpu().numpy()
            conf = results.boxes.conf[i].item()
            class_id = int(results.boxes.cls[i].item())
            class_name = results.names[class_id]
            
            detection = np.array([box[0], box[1], box[2], box[3], conf])
            detections.append(detection)
            class_names.append(class_name)
            
        if len(detections) > 0:
            detections = np.stack(detections)
        else:
            detections = np.empty((0, 5))
            
        return detections, class_names
    
    def process_frame(self, frame):
        # Calculer le ratio d'aspect du frame
        frame_height, frame_width = frame.shape[:2]
        frame_aspect = frame_width / frame_height
        
        # Calculer les dimensions cibles en conservant le ratio d'aspect
        canvas_aspect = self.canvas_width / self.canvas_height
        
        if frame_aspect > canvas_aspect:
            # Le frame est plus large que le canvas
            target_width = self.canvas_width
            target_height = int(self.canvas_width / frame_aspect)
        else:
            # Le frame est plus haut que le canvas
            target_height = self.canvas_height
            target_width = int(self.canvas_height * frame_aspect)
        
        # Redimensionner le frame
        if frame_width != target_width or frame_height != target_height:
            frame = cv2.resize(frame, (target_width, target_height))
        
        return frame

    def detect_realtime(self):
        try:
            if self.is_detecting and self.video_source:
                ret, frame = self.video_source.read()
                if ret:
                    # Redimensionner le frame
                    frame = self.process_frame(frame)
                    
                    # Faire la détection avec YOLOv8
                    results = self.model(frame)[0]
                    
                    # Préparer les détections pour SORT
                    detections, class_names = self.process_detections(results)
                    
                    # Mettre à jour le tracking et les compteurs
                    counted_objects = self.tracker.update(detections, class_names)
                    self.detection_counts = Counter(counted_objects)
                    
                    # Dessiner les résultats
                    annotated_frame = results.plot()
                    
                    # Convertir l'image pour Tkinter
                    img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
                    img = ImageTk.PhotoImage(image=img)
                    
                    # Mettre à jour le canvas
                    self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
                    self.canvas.image = img
                    
                    # Mettre à jour les compteurs
                    self.update_counter_display()
                    
                    # Rappeler cette fonction après 10ms
                    self.window.after(10, self.detect_realtime)
        except Exception as e:
            print(f"Erreur lors de la détection: {str(e)}")
            messagebox.showerror("Erreur", f"Erreur lors de la détection: {str(e)}")
            self.is_detecting = False
            if self.video_source:
                self.video_source.release()
    
    def start_video_detection(self):
        try:
            if not self.is_detecting:
                # Ouvrir le sélecteur de fichier
                video_path = filedialog.askopenfilename(
                    filetypes=[("Fichiers vidéo", "*.mp4 *.avi *.mov")])
                
                if video_path:
                    self.video_source = cv2.VideoCapture(video_path)
                    if not self.video_source.isOpened():
                        raise Exception("Impossible d'ouvrir la vidéo")
                    
                    self.is_detecting = True
                    self.current_video_path = video_path
                    self.detection_counts.clear()
                    self.tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.3)
                    
                    # Mettre à jour l'état des boutons
                    self.video_btn.config(state=tk.DISABLED)
                    self.stop_video_btn.config(state=tk.NORMAL)
                    self.new_video_btn.config(state=tk.NORMAL)
                    self.realtime_btn.config(state=tk.DISABLED)
                    self.save_btn.config(state=tk.NORMAL)
                    
                    print(f"Démarrage de la détection sur la vidéo: {video_path}")
                    self.detect_video()
        except Exception as e:
            print(f"Erreur lors du chargement de la vidéo: {str(e)}")
            messagebox.showerror("Erreur", f"Erreur lors du chargement de la vidéo: {str(e)}")
            self.is_detecting = False
            if self.video_source:
                self.video_source.release()
    
    def detect_video(self):
        try:
            if self.is_detecting and self.video_source:
                ret, frame = self.video_source.read()
                if ret:
                    # Redimensionner le frame
                    frame = self.process_frame(frame)
                    
                    # Faire la détection avec YOLOv8
                    results = self.model(frame)[0]
                    
                    # Préparer les détections pour SORT
                    detections, class_names = self.process_detections(results)
                    
                    # Mettre à jour le tracking et les compteurs
                    counted_objects = self.tracker.update(detections, class_names)
                    self.detection_counts = Counter(counted_objects)
                    
                    # Dessiner les résultats
                    annotated_frame = results.plot()
                    
                    # Convertir l'image pour Tkinter
                    img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
                    img = ImageTk.PhotoImage(image=img)
                    
                    # Mettre à jour le canvas
                    self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
                    self.canvas.image = img
                    
                    # Mettre à jour les compteurs
                    self.update_counter_display()
                    
                    # Rappeler cette fonction après 10ms
                    self.window.after(10, self.detect_video)
                else:
                    print("Fin de la vidéo")
                    self.is_detecting = False
                    self.video_source.release()
        except Exception as e:
            print(f"Erreur lors de la détection vidéo: {str(e)}")
            messagebox.showerror("Erreur", f"Erreur lors de la détection vidéo: {str(e)}")
            self.is_detecting = False
            if self.video_source:
                self.video_source.release()

    def stop_video(self):
        """Arrête la lecture de la vidéo actuelle"""
        self.is_detecting = False
        if self.video_source:
            self.video_source.release()
        self.video_source = None
        
        # Réinitialiser le canvas avec un fond noir
        self.canvas.delete("all")
        self.canvas.configure(bg='black')
        
        # Mettre à jour l'état des boutons
        self.video_btn.config(state=tk.NORMAL)
        self.stop_video_btn.config(state=tk.DISABLED)
        self.new_video_btn.config(state=tk.NORMAL)
        self.realtime_btn.config(state=tk.NORMAL)
        
        print("Vidéo arrêtée")
    
    def load_new_video(self):
        """Charge une nouvelle vidéo"""
        self.stop_video()  # Arrêter la vidéo actuelle
        self.start_video_detection()  # Démarrer la sélection d'une nouvelle vidéo

if __name__ == "__main__":
    try:
        print("Démarrage de l'application...")
        print(f"Version de Python: {sys.version}")
        print(f"Version de PyTorch: {torch.__version__}")
        print(f"Version d'OpenCV: {cv2.__version__}")
        
        root = tk.Tk()
        app = ObjectDetectionApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Erreur fatale: {str(e)}")
        messagebox.showerror("Erreur fatale", f"L'application n'a pas pu démarrer: {str(e)}")
        sys.exit(1)
