import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QTabWidget, QComboBox, QSlider, QLineEdit, QMessageBox, 
                             QProgressBar, QGroupBox, QFormLayout, QCheckBox, QScrollArea)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

class GoruntIslemeProjesi(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Resim ve Video Editör Uygulaması")
        self.setGeometry(50, 50, 1300, 900)

        # ANA SEKME
        self.main_tabs = QTabWidget()
        self.setCentralWidget(self.main_tabs)

        # SEKME 1: VIDEO MODU
        self.tab_video = QWidget()
        self.init_video_ui()
        self.main_tabs.addTab(self.tab_video, "VİDEO STUDYO")

        # SEKME 2: FOTOĞRAF MODU
        self.tab_photo = QWidget()
        self.init_photo_ui()
        self.main_tabs.addTab(self.tab_photo, "FOTOĞRAF STUDYO")

        # VIDEO DEGISKENLERI
        self.video_path = ""
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video_frame)
        self.is_playing = False
        self.video_speed = 30
        
        # Video Filtreleri
        self.v_gray = False
        self.v_text_active = False
        self.v_text = ""
        self.v_val_r = 1.0; self.v_val_g = 1.0; self.v_val_b = 1.0; self.v_gamma = 1.0
        self.v_lut = self.build_gamma_lut(1.0)
        
        # Video Takip
        self.tracker = None
        self.tracking_method = "None"
        self.meanshift_hist = None
        self.meanshift_window = None
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

        # Optik Akis
        self.v_opt_flow = False
        self.old_gray = None; self.p0 = None; self.mask_opt = None
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # FOTOGRAF DEGISKENLERI  
        self.img_original = None
        self.img_processed = None
        self.p_rotation = 0
        self.p_scale = 100
        self.p_gamma = 1.0
        self.p_val_r = 1.0; self.p_val_g = 1.0; self.p_val_b = 1.0

    # BOLUM 1: VIDEO FONKSIYONLARI VE ARAYUZLERI
    def init_video_ui(self):
        layout = QHBoxLayout()
        
        # SOL PANEL
        control_tabs = QTabWidget()
        control_tabs.setFixedWidth(400)
        
        # Giris
        t1 = QWidget(); l1 = QVBoxLayout()
        btn_load = QPushButton("Video Yükle"); btn_load.clicked.connect(self.load_video)
        self.lbl_v_info = QLabel("Video Seçilmedi")
        btn_play = QPushButton("Oynat / Duraklat"); btn_play.clicked.connect(self.toggle_play)
        l1.addWidget(btn_load); l1.addWidget(self.lbl_v_info); l1.addWidget(btn_play); l1.addStretch()
        t1.setLayout(l1); control_tabs.addTab(t1, "Giriş")

        # Renk/Hiz
        t2 = QWidget(); l2 = QVBoxLayout()
        btn_gray = QPushButton("Gri Tonlama"); btn_gray.setCheckable(True); btn_gray.clicked.connect(self.toggle_v_gray)
        l2.addWidget(btn_gray)
        
        g_rgb = QGroupBox("RGB Kanalları"); form_rgb = QFormLayout()
        self.s_vr = self.create_slider(0, 200, 100, self.update_v_rgb); form_rgb.addRow("R:", self.s_vr)
        self.s_vg = self.create_slider(0, 200, 100, self.update_v_rgb); form_rgb.addRow("G:", self.s_vg)
        self.s_vb = self.create_slider(0, 200, 100, self.update_v_rgb); form_rgb.addRow("B:", self.s_vb)
        g_rgb.setLayout(form_rgb); l2.addWidget(g_rgb)

        self.s_vgamma = self.create_slider(1, 300, 100, self.update_v_gamma)
        l2.addWidget(QLabel("Gama:")); l2.addWidget(self.s_vgamma)

        l2.addWidget(QLabel("Oynatma Hızı (FPS Gecikmesi):"))
        self.s_speed = QSlider(Qt.Horizontal)
        self.s_speed.setMinimum(1); self.s_speed.setMaximum(200); self.s_speed.setValue(30)
        self.s_speed.valueChanged.connect(self.update_video_speed)
        l2.addWidget(self.s_speed)

        self.txt_v_overlay = QLineEdit(); self.txt_v_overlay.setPlaceholderText("Metin...")
        btn_txt = QPushButton("Metni Ekle"); btn_txt.setCheckable(True); btn_txt.clicked.connect(self.toggle_v_text)
        l2.addWidget(self.txt_v_overlay); l2.addWidget(btn_txt)
        l2.addStretch(); t2.setLayout(l2); control_tabs.addTab(t2, "Renk/Hız")

        # Takip
        t3 = QWidget(); l3 = QVBoxLayout()
        btn_opt = QPushButton("Optik Akış (Lucas-Kanade)"); btn_opt.setCheckable(True); btn_opt.clicked.connect(self.toggle_v_opt)
        l3.addWidget(btn_opt)
        
        self.cmb_track = QComboBox(); self.cmb_track.addItems(["KCF", "CSRT", "MeanShift"])
        btn_track = QPushButton("Nesne Seç ve Takip Et"); btn_track.clicked.connect(self.start_v_track)
        l3.addWidget(QLabel("Algoritma:")); l3.addWidget(self.cmb_track); l3.addWidget(btn_track)
        l3.addStretch(); t3.setLayout(l3); control_tabs.addTab(t3, "Takip")

        # Kayit
        t4 = QWidget(); l4 = QVBoxLayout()
        btn_rev = QPushButton("Ters Kaydet"); btn_rev.clicked.connect(lambda: self.process_video("reverse"))
        
        # Multi ROI 
        btn_crop = QPushButton("Çoklu Alan Kırpma (Multi-ROI)"); btn_crop.clicked.connect(self.save_multi_roi)
        l4.addWidget(btn_rev); l4.addWidget(btn_crop)
        
        btn_blur = QPushButton("Alan Sansürle (Blur)"); btn_blur.clicked.connect(self.blur_video_region)
        l4.addWidget(btn_blur)
        
        self.txt_trim_s = QLineEdit("0"); self.txt_trim_e = QLineEdit("5")
        btn_trim = QPushButton("Aralığı Kes"); btn_trim.clicked.connect(lambda: self.process_video("trim"))
        l4.addWidget(QLabel("Başlangıç/Bitiş (sn):")); l4.addWidget(self.txt_trim_s); l4.addWidget(self.txt_trim_e); l4.addWidget(btn_trim)
        l4.addStretch(); t4.setLayout(l4); control_tabs.addTab(t4, "Kayıt")

        # Analiz
        t5 = QWidget(); l5 = QVBoxLayout()
        btn_scene = QPushButton("Sahne Analizi (Grafik)"); btn_scene.clicked.connect(self.analyze_scenes)
        self.cmb_codec = QComboBox(); self.cmb_codec.addItems(["XVID (.avi)", "MP4V (.mp4)", "MJPG (.avi)"])
        btn_conv = QPushButton("Dönüştür"); btn_conv.clicked.connect(lambda: self.process_video("convert"))
        self.pbar_v = QProgressBar() # Video icin progress bar
        l5.addWidget(btn_scene); l5.addWidget(self.cmb_codec); l5.addWidget(btn_conv); l5.addWidget(self.pbar_v)
        l5.addStretch(); t5.setLayout(l5); control_tabs.addTab(t5, "Analiz")

        # SAG EKRAN
        self.lbl_video = QLabel(); self.lbl_video.setAlignment(Qt.AlignCenter); self.lbl_video.setStyleSheet("background: black;")
        
        layout.addWidget(control_tabs); layout.addWidget(self.lbl_video, 1)
        self.tab_video.setLayout(layout)

    # Video icin fonksiyonlar
    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Video", "", "Video (*.mp4 *.avi)")
        if path:
            self.video_path = path; self.cap = cv2.VideoCapture(path)
            self.lbl_v_info.setText(path.split('/')[-1])
            ret, frame = self.cap.read()
            if ret: self.display_image(frame, self.lbl_video); self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def toggle_play(self):
        if not self.cap: return
        if self.is_playing: self.timer.stop(); self.is_playing = False
        else: self.timer.start(self.video_speed); self.is_playing = True

    def update_video_speed(self):
        self.video_speed = self.s_speed.value()
        if self.is_playing:
            self.timer.setInterval(self.video_speed)

    def toggle_v_gray(self): self.v_gray = not self.v_gray
    def toggle_v_text(self): self.v_text_active = not self.v_text_active; self.v_text = self.txt_v_overlay.text()
    def update_v_rgb(self): self.v_val_r = self.s_vr.value()/100; self.v_val_g = self.s_vg.value()/100; self.v_val_b = self.s_vb.value()/100
    def update_v_gamma(self): val = self.s_vgamma.value()/100; self.v_gamma = val if val>0 else 0.01; self.v_lut = self.build_gamma_lut(self.v_gamma)
    
    def toggle_v_opt(self):
        self.v_opt_flow = not self.v_opt_flow
        if self.v_opt_flow and self.cap:
            ret, f = self.cap.read(); 
            if ret: 
                self.old_gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=None, **dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7))
                self.mask_opt = np.zeros_like(f); self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def start_v_track(self):
        if not self.cap: return
        self.timer.stop(); self.is_playing = False; ret, f = self.cap.read()
        if not ret: return
        
        roi = cv2.selectROI("Nesne Sec", f, False); cv2.destroyWindow("Nesne Sec")
        
        chosen_method = self.cmb_track.currentText()
        if chosen_method == "MeanShift":
            x, y, w, h = roi
            roi_crop = f[y:y+h, x:x+w]
            hsv_roi = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
            roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
            self.meanshift_hist = roi_hist
            self.meanshift_window = (x, y, w, h)
            self.tracking_method = "MeanShift"
        else:
            t_map = {"KCF": cv2.TrackerKCF_create, "CSRT": cv2.TrackerCSRT_create}
            self.tracker = t_map[chosen_method]()
            self.tracker.init(f, roi)
            self.tracking_method = "Tracker"

        self.timer.start(self.video_speed); self.is_playing = True

    def update_video_frame(self):
        if not self.cap: return
        ret, frame = self.cap.read()
        if not ret: self.timer.stop(); self.is_playing = False; return
        
        # 1. Gama & RGB
        if self.v_gamma != 1.0: frame = cv2.LUT(frame, self.v_lut)
        if self.v_val_r != 1.0 or self.v_val_g != 1.0 or self.v_val_b != 1.0:
            frame = frame.astype(np.float32)
            frame[:,:,0]*=self.v_val_b; frame[:,:,1]*=self.v_val_g; frame[:,:,2]*=self.v_val_r
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        
        disp = frame.copy()
        
        if self.tracking_method == "Tracker" and self.tracker:
            suc, roi = self.tracker.update(frame)
            if suc: x,y,w,h = map(int, roi); cv2.rectangle(disp, (x,y), (x+w,y+h), (0,255,0), 2)
        elif self.tracking_method == "MeanShift" and self.meanshift_hist is not None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], self.meanshift_hist, [0, 180], 1)
            ret, self.meanshift_window = cv2.meanShift(dst, self.meanshift_window, self.term_crit)
            x, y, w, h = self.meanshift_window
            cv2.rectangle(disp, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(disp, "MeanShift", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        if self.v_opt_flow and self.p0 is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, gray, self.p0, None, **self.lk_params)
            if p1 is not None:
                gn = p1[st==1]; go = self.p0[st==1]
                for n, o in zip(gn, go):
                    a,b = n.ravel(); c,d = o.ravel()
                    self.mask_opt = cv2.line(self.mask_opt, (int(a),int(b)), (int(c),int(d)), (0,255,0), 2)
                    disp = cv2.circle(disp, (int(a),int(b)), 5, (0,0,255), -1)
                disp = cv2.add(disp, self.mask_opt); self.old_gray = gray.copy(); self.p0 = gn.reshape(-1,1,2)
        
        if self.v_gray: disp = cv2.cvtColor(disp, cv2.COLOR_BGR2GRAY); disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
        if self.v_text_active: cv2.putText(disp, self.v_text, (50,50), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 2)
        
        self.display_image(disp, self.lbl_video)

    # Video Kayit Islemleri
    def save_multi_roi(self):
        if not self.video_path: return
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        
        QMessageBox.information(self, "Bilgi", "Açılan pencerede ROI seçin ve SPACE veya ENTER'a basın. İptal için C. Seçimi bitirmek için ESC.")
        rois = cv2.selectROIs("Coklu Secim", frame)
        cv2.destroyWindow("Coklu Secim")
        
        if len(rois) == 0: return

        # VideoWriterlari hazirla
        fps = cap.get(cv2.CAP_PROP_FPS)
        writers = []
        for i, roi in enumerate(rois):
            x, y, w, h = roi
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            vw = cv2.VideoWriter(f"crop_part_{i+1}.mp4", fourcc, fps, (w, h))
            writers.append(vw)

        # Videoyu işle
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.pbar_v.setValue(0)
        curr = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            for i, roi in enumerate(rois):
                x, y, w, h = roi
                crop = frame[y:y+h, x:x+w]
                writers[i].write(crop)
            
            curr += 1
            if curr % 10 == 0:
                self.pbar_v.setValue(int(curr/total_frames*100))
                QApplication.processEvents()

        for vw in writers: vw.release()
        cap.release()
        QMessageBox.information(self, "Bitti", f"{len(writers)} adet kırpılmış video kaydedildi.")

    def blur_video_region(self):
        if not self.video_path: return
        cap = cv2.VideoCapture(self.video_path); ret, f = cap.read()
        roi = cv2.selectROI("Blur", f); cv2.destroyWindow("Blur")
        x,y,w,h = roi; fps=cap.get(5)
        vw = cv2.VideoWriter("blurred.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(cap.get(3)), int(cap.get(4))))
        total=int(cap.get(7)); curr=0
        while True:
            ret, f = cap.read()
            if not ret: break
            f[y:y+h, x:x+w] = cv2.GaussianBlur(f[y:y+h, x:x+w], (51,51), 0)
            vw.write(f); curr+=1; self.pbar_v.setValue(int(curr/total*100)); QApplication.processEvents()
        vw.release(); QMessageBox.information(self,"Bitti","blurred.mp4 hazır")

    def analyze_scenes(self):
        if not self.video_path: return
        cap = cv2.VideoCapture(self.video_path); ret, p = cap.read(); 
        if not ret: return
        ph = cv2.cvtColor(p, cv2.COLOR_BGR2HSV); ds=[]; ts=[]; i=0; fps=cap.get(5)
        while True:
            ret, c = cap.read()
            if not ret: break
            ch = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
            d = np.sum(cv2.absdiff(cv2.calcHist([ph],[0],None,[256],[0,256]), cv2.calcHist([ch],[0],None,[256],[0,256])))
            if d>5000: ds.append(d); ts.append(i/fps)
            ph=ch; i+=1; QApplication.processEvents()
        plt.figure(); plt.plot(ts, ds); plt.title("Sahne Değişimi"); plt.show()

    def process_video(self, mode):
        if not self.video_path: return
        cap = cv2.VideoCapture(self.video_path); fps=cap.get(5); w,h = int(cap.get(3)), int(cap.get(4))
        name = "out.mp4"; fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if mode=="convert":
            t = self.cmb_codec.currentText()
            if "XVID" in t: fourcc=cv2.VideoWriter_fourcc(*'XVID'); name="out.avi"
            elif "MJPG" in t: fourcc=cv2.VideoWriter_fourcc(*'MJPG'); name="out.avi"
        elif mode=="reverse": name="reverse.mp4"
        elif mode=="trim": name="trim.mp4"
        
        vw = cv2.VideoWriter(name, fourcc, fps, (w,h))
        if mode=="reverse":
            fs = []
            while True: 
                r,f = cap.read()
                if not r: break
                fs.append(f)
            for i, f in enumerate(reversed(fs)): vw.write(f); self.pbar_v.setValue(int(i/len(fs)*100)); QApplication.processEvents()
        elif mode=="trim":
            s=int(float(self.txt_trim_s.text())*fps); e=int(float(self.txt_trim_e.text())*fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, s); c=s
            while c<e: 
                r,f = cap.read()
                if not r: break
                vw.write(f); c+=1; QApplication.processEvents()
        else:
            tot=int(cap.get(7)); c=0
            while True:
                r,f=cap.read()
                if not r: break
                vw.write(f); c+=1; self.pbar_v.setValue(int(c/tot*100)); QApplication.processEvents()
        vw.release(); QMessageBox.information(self,"Bitti",f"{name} hazır")

    # BOLUM 2: FOTOGRAF FONKSIYONLARI VE ARAYUZLERI
    def init_photo_ui(self):
        layout = QHBoxLayout()

        # SOL PANEL
        p_scroll = QScrollArea()
        p_controls = QWidget()
        form = QFormLayout()

        # Dosya Islemleri
        btn_load = QPushButton("Fotoğraf Yükle"); btn_load.clicked.connect(self.load_photo)
        btn_save = QPushButton("Fotoğrafı Kaydet"); btn_save.clicked.connect(self.save_photo)
        form.addRow(btn_load, btn_save)
        
        form.addRow(QLabel("<b>--- Temel Ayarlar ---</b>"))
        
        # RGB
        self.sp_r = self.create_slider(0, 200, 100, self.process_photo)
        self.sp_g = self.create_slider(0, 200, 100, self.process_photo)
        self.sp_b = self.create_slider(0, 200, 100, self.process_photo)
        form.addRow("R Kanalı:", self.sp_r)
        form.addRow("G Kanalı:", self.sp_g)
        form.addRow("B Kanalı:", self.sp_b)

        # Gama
        self.sp_gamma = self.create_slider(1, 300, 100, self.process_photo)
        form.addRow("Gama/Parlaklık:", self.sp_gamma)

        form.addRow(QLabel("<b>--- Geometri ---</b>"))
        
        # Rotate & Resize
        self.sp_rot = self.create_slider(-180, 180, 0, self.process_photo)
        form.addRow("Döndür (Derece):", self.sp_rot)
        self.sp_scale = self.create_slider(10, 200, 100, self.process_photo)
        form.addRow("Boyutlandır (%):", self.sp_scale)
        
        self.chk_hflip = QCheckBox("Yatay Aynala (Horizontal Flip)")
        self.chk_hflip.stateChanged.connect(self.process_photo)
        self.chk_vflip = QCheckBox("Dikey Aynala (Vertical Flip)")
        self.chk_vflip.stateChanged.connect(self.process_photo)
        form.addRow(self.chk_hflip)
        form.addRow(self.chk_vflip)

        form.addRow(QLabel("<b>--- Filtreler & Efektler ---</b>"))
        
        # Filtre Secimi
        self.cmb_filter = QComboBox()
        self.cmb_filter.addItems(["Yok", "Gri Tonlama", "Binary (Threshold)", 
                                  "Bulanıklaştırma (Blur)", "Keskinleştirme (Sharpen)", 
                                  "Kenar Tespiti (Canny)", "Kenar Tespiti (Sobel)",
                                  "Morfoloji: Erozyon", "Morfoloji: Genişletme"])
        self.cmb_filter.currentIndexChanged.connect(self.process_photo)
        form.addRow("Efekt Seç:", self.cmb_filter)

        form.addRow(QLabel("<b>--- Analiz ---</b>"))
        btn_hist = QPushButton("RGB Histogramı Göster")
        btn_hist.clicked.connect(self.show_photo_hist)
        form.addRow(btn_hist)

        p_controls.setLayout(form)
        p_scroll.setWidget(p_controls)
        p_scroll.setWidgetResizable(True)
        p_scroll.setFixedWidth(350)

        # SAG PANEL
        self.lbl_photo = QLabel()
        self.lbl_photo.setAlignment(Qt.AlignCenter)
        self.lbl_photo.setStyleSheet("background: #202020;")
        
        layout.addWidget(p_scroll)
        layout.addWidget(self.lbl_photo, 1)
        self.tab_photo.setLayout(layout)

    # FOTOGRAF FONKSIYONLARI
    def load_photo(self):
        path, _ = QFileDialog.getOpenFileName(self, "Resim Seç", "", "Resim (*.jpg *.png *.jpeg *.bmp)")
        if path:
            img = cv2.imread(path)
            if img is not None:
                self.img_original = img
                self.process_photo()

    def save_photo(self):
        if self.img_processed is None: return
        path, _ = QFileDialog.getSaveFileName(self, "Kaydet", "islenmis_foto.jpg", "JPG (*.jpg);;PNG (*.png)")
        if path:
            cv2.imwrite(path, self.img_processed)
            QMessageBox.information(self, "Başarılı", "Fotoğraf kaydedildi.")

    def process_photo(self):
        if self.img_original is None: return
        img = self.img_original.copy()

        # Boyutlandirma
        scale = self.sp_scale.value() / 100.0
        if scale != 1.0:
            h, w = img.shape[:2]
            img = cv2.resize(img, (int(w*scale), int(h*scale)))

        # Renk kanallari
        vr, vg, vb = self.sp_r.value()/100, self.sp_g.value()/100, self.sp_b.value()/100
        if vr != 1.0 or vg != 1.0 or vb != 1.0:
            img = img.astype(np.float32)
            img[:,:,0] *= vb; img[:,:,1] *= vg; img[:,:,2] *= vr
            img = np.clip(img, 0, 255).astype(np.uint8)

        # Gama
        gamma = self.sp_gamma.value() / 100.0
        if gamma <= 0: gamma = 0.01
        if gamma != 1.0:
            table = self.build_gamma_lut(gamma)
            img = cv2.LUT(img, table)

        # Geometri
        if self.chk_hflip.isChecked(): img = cv2.flip(img, 1)
        if self.chk_vflip.isChecked(): img = cv2.flip(img, 0)
        angle = self.sp_rot.value()
        if angle != 0:
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            cos = np.abs(M[0, 0]); sin = np.abs(M[0, 1])
            nW = int((h * sin) + (w * cos)); nH = int((h * cos) + (w * sin))
            M[0, 2] += (nW / 2) - w / 2; M[1, 2] += (nH / 2) - h / 2
            img = cv2.warpAffine(img, M, (nW, nH))

        # Filtreler
        mode = self.cmb_filter.currentText()
        if mode == "Gri Tonlama":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif mode == "Binary (Threshold)":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif mode == "Bulanıklaştırma (Blur)": img = cv2.GaussianBlur(img, (15, 15), 0)
        elif mode == "Keskinleştirme (Sharpen)":
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            img = cv2.filter2D(img, -1, kernel)
        elif mode == "Kenar Tespiti (Canny)":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.Canny(gray, 100, 200); img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif mode == "Kenar Tespiti (Sobel)":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
            img = cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5, cv2.convertScaleAbs(grad_y), 0.5, 0)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif mode == "Morfoloji: Erozyon": img = cv2.erode(img, np.ones((5,5), np.uint8), iterations=1)
        elif mode == "Morfoloji: Genişletme": img = cv2.dilate(img, np.ones((5,5), np.uint8), iterations=1)

        self.img_processed = img
        self.display_image(img, self.lbl_photo)

    def show_photo_hist(self):
        if self.img_processed is None: return
        colors = ('b', 'g', 'r')
        plt.figure("RGB Histogram Analizi")
        plt.title("Renk Dağılımı")
        for i, col in enumerate(colors):
            hist = cv2.calcHist([self.img_processed], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
        plt.show()

    # EK FONKSIYONLAR
    def create_slider(self, min_v, max_v, def_v, func):
        s = QSlider(Qt.Horizontal)
        s.setMinimum(min_v); s.setMaximum(max_v); s.setValue(def_v)
        s.valueChanged.connect(func)
        return s

    def build_gamma_lut(self, gamma):
        inv = 1.0 / gamma
        return np.array([((i / 255.0) ** inv) * 255 for i in np.arange(0, 256)]).astype("uint8")

    def display_image(self, img, label):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(q_img).scaled(label.width(), label.height(), Qt.KeepAspectRatio))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GoruntIslemeProjesi()
    window.show()
    sys.exit(app.exec_())