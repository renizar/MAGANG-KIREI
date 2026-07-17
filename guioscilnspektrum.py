"""
=====================================================
OSILOSKOP SEDERHANA - GUI (Python + PyQt5 + pyqtgraph)
+ TAB SPECTRUM ANALYZER (FFT)
+ TOMBOL RUN/STOP (HOLD)
=====================================================
Versi migrasi dari Tkinter+matplotlib ke PyQt5+pyqtgraph.
Alasan migrasi: pyqtgraph didesain khusus untuk real-time plotting
(dipakai di banyak software SDR / spectrum analyzer beneran), jauh
lebih cepat refresh-nya dibanding matplotlib karena dia pakai OpenGL/
Qt native rendering, bukan redraw seluruh figure tiap update.

Protokol data biner dari Arduino Due SAMA PERSIS dengan versi sebelumnya,
jadi kode Arduino tidak perlu diubah sama sekali.

TAMBAHAN vs versi sebelumnya:
  - Tombol "Run/Stop" (Hold) di baris kontrol atas. Saat di-stop, data yang
    masuk dari serial TETAP dibaca & di-drain dari queue (biar buffer tidak
    numpuk/lag saat di-resume nanti), tapi TIDAK diproses & TIDAK digambar
    ulang -> layar "freeze" di frame terakhir sebelum Stop ditekan. Ini
    persis fungsi tombol "Run/Stop" / "Hold" di osiloskop asli, berguna
    utk mengamati/zoom satu snapshot tanpa keburu ke-overwrite data baru
    tiap 500ms.

Instalasi dependency (sekali saja):
    pip install pyserial numpy pyqtgraph pyqt5

Menjalankan:
    python oscilloscope_gui_pyqtgraph.py
"""

import struct
import threading
import queue
import time

import numpy as np
import serial
import serial.tools.list_ports

from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from scipy.signal import find_peaks

# ================= KONFIGURASI PROTOKOL (harus sama dgn sisi Arduino) =================
SYNC_BYTES = b"\xAA\x55"
ADC_MAX = 4095       # ADC 12-bit
VREF = 3.3           # volt, referensi ADC Due

TIME_DIV_OPTIONS_MS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5,
                       1, 2, 5, 10, 20, 25, 50, 100]  # pilihan time/div (ms)
DIVISIONS = 10                                  # konvensi osiloskop: 10 kotak horizontal

POLL_INTERVAL_MS = 15   # seberapa sering GUI cek antrian data & redraw -> refresh rate tinggi


# ================= THREAD BACA SERIAL (sama seperti versi Tkinter) =================
class SerialReader(threading.Thread):
    """Baca & parsing data serial di thread terpisah biar GUI tidak freeze."""

    def __init__(self, port, baud, data_queue):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.data_queue = data_queue
        self._stop_flag = threading.Event()
        self.ser = None

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
        except Exception as e:
            self.data_queue.put(("error", f"Gagal buka port: {e}"))
            return

        self.data_queue.put(("connected", None))

        buf = bytearray()
        last_data_time = time.time()
        while not self._stop_flag.is_set():
            try:
                chunk = self.ser.read(4096)
                if not chunk:
                    if time.time() - last_data_time > 3:
                        self.data_queue.put(("waiting", None))
                        last_data_time = time.time()
                    continue
                last_data_time = time.time()
                buf.extend(chunk)
                self._try_parse(buf)
            except Exception as e:
                self.data_queue.put(("error", str(e)))
                time.sleep(0.5)

    def _try_parse(self, buf):
        while True:
            sync_idx = buf.find(SYNC_BYTES)
            if sync_idx == -1:
                buf.clear()
                return
            if sync_idx > 0:
                del buf[:sync_idx]

            if len(buf) < 10:
                return

            n, interval_ns = struct.unpack_from("<II", buf, 2)

            if n <= 0 or n > 200000:
                del buf[:2]
                continue

            total_len = 10 + n * 2 + 2
            if len(buf) < total_len:
                return

            samples_bytes = bytes(buf[10:10 + n * 2])
            checksum_recv = struct.unpack_from("<H", buf, 10 + n * 2)[0]

            samples = np.frombuffer(samples_bytes, dtype="<u2")
            checksum_calc = 0
            for v in samples:
                checksum_calc ^= int(v)

            if checksum_calc == checksum_recv:
                self.data_queue.put(("data", samples.copy(), interval_ns))
            else:
                self.data_queue.put(("warn", "Checksum tdk cck"))

            del buf[:total_len]

    def stop(self):
        self._stop_flag.set()
        if self.ser and self.ser.is_open:
            self.ser.close()


# ================= JENDELA UTAMA =================
class OscilloscopeWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Osiloskop Sederhana - Arduino Due (PyQt5 + pyqtgraph)")
        self.resize(1000, 760)

        self.data_queue = queue.Queue()
        self.reader = None

        self._t_ms = None
        self._voltage = None
        self._interval_s = None

        self.running = True  # status Run/Stop: True = jalan normal, False = di-hold/freeze

        pg.setConfigOptions(antialias=False)  # antialias mahal -> matikan demi refresh rate tinggi

        self._build_ui()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._poll_queue)
        self.timer.start(POLL_INTERVAL_MS)

    # ---------------- UI ----------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)

        # --- baris kontrol atas ---
        ctrl_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(ctrl_layout)

        ctrl_layout.addWidget(QtWidgets.QLabel("Port:"))
        self.port_combo = QtWidgets.QComboBox()
        self.port_combo.addItems(self._list_ports())
        self.port_combo.setEditable(True)
        ctrl_layout.addWidget(self.port_combo)

        refresh_btn = QtWidgets.QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_ports)
        ctrl_layout.addWidget(refresh_btn)

        ctrl_layout.addWidget(QtWidgets.QLabel("Baud:"))
        self.baud_edit = QtWidgets.QLineEdit("2000000")
        self.baud_edit.setFixedWidth(90)
        ctrl_layout.addWidget(self.baud_edit)

        self.connect_btn = QtWidgets.QPushButton("Connect")
        self.connect_btn.clicked.connect(self._toggle_connect)
        ctrl_layout.addWidget(self.connect_btn)

        # --- tombol Run/Stop (Hold) ---
        self.run_stop_btn = QtWidgets.QPushButton("Stop")
        self.run_stop_btn.setStyleSheet(
            "font-weight: bold; background-color: #c62828; color: white;"
        )
        self.run_stop_btn.clicked.connect(self._toggle_run_stop)
        ctrl_layout.addWidget(self.run_stop_btn)

        self.status_label = QtWidgets.QLabel("Belum terhubung")
        self.status_label.setStyleSheet("color: gray;")
        ctrl_layout.addWidget(self.status_label, stretch=1)

        ctrl_layout.addWidget(QtWidgets.QLabel("Time/Div:"))
        self.time_div_combo = QtWidgets.QComboBox()
        self.time_div_combo.addItems([str(v) for v in TIME_DIV_OPTIONS_MS])
        self.time_div_combo.setCurrentText("2")
        self.time_div_combo.currentTextChanged.connect(self._redraw_time)
        ctrl_layout.addWidget(self.time_div_combo)
        ctrl_layout.addWidget(QtWidgets.QLabel("ms/div"))

        ctrl_layout.addWidget(QtWidgets.QLabel("Window:"))
        self.fft_window_combo = QtWidgets.QComboBox()
        self.fft_window_combo.addItems(["None", "Hanning", "Hamming", "Blackman"])
        self.fft_window_combo.setCurrentText("Hanning")
        self.fft_window_combo.currentTextChanged.connect(self._redraw_fft)
        ctrl_layout.addWidget(self.fft_window_combo)

        ctrl_layout.addWidget(QtWidgets.QLabel("Skala:"))
        self.fft_scale_combo = QtWidgets.QComboBox()
        self.fft_scale_combo.addItems(["dB", "Linear"])
        self.fft_scale_combo.currentTextChanged.connect(self._redraw_fft)
        ctrl_layout.addWidget(self.fft_scale_combo)

        ctrl_layout.addWidget(QtWidgets.QLabel("Start (Hz):"))
        self.fft_start_edit = QtWidgets.QLineEdit("10")
        self.fft_start_edit.setFixedWidth(70)
        self.fft_start_edit.editingFinished.connect(self._redraw_fft)
        ctrl_layout.addWidget(self.fft_start_edit)

        ctrl_layout.addWidget(QtWidgets.QLabel("Stop (Hz):"))
        self.fft_stop_edit = QtWidgets.QLineEdit("500000")
        self.fft_stop_edit.setFixedWidth(80)
        self.fft_stop_edit.editingFinished.connect(self._redraw_fft)
        ctrl_layout.addWidget(self.fft_stop_edit)

        # --- tab Waveform & Spectrum ---
        self.tabs = QtWidgets.QTabWidget()
        main_layout.addWidget(self.tabs, stretch=1)

        # Tab Waveform
        self.plot_time = pg.PlotWidget()
        self.plot_time.setLabel("bottom", "Waktu", units="ms")
        self.plot_time.setLabel("left", "Tegangan", units="V")
        self.plot_time.showGrid(x=True, y=True, alpha=0.3)
        self.plot_time.setYRange(-0.2, VREF + 0.2)
        self.curve_time = self.plot_time.plot([], [], pen=pg.mkPen("#00c853", width=1.5))
        self.tabs.addTab(self.plot_time, "Waveform (Domain Waktu)")

        # Tab Spectrum
        self.plot_fft = pg.PlotWidget()
        self.plot_fft.setLabel("bottom", "Frekuensi", units="Hz")
        self.plot_fft.setLabel("left", "Magnitude", units="dB")
        self.plot_fft.showGrid(x=True, y=True, alpha=0.3)
        self.plot_fft.setLogMode(x=True, y=False)
        self.curve_fft = self.plot_fft.plot([], [], pen=pg.mkPen("#2979ff", width=1.2))
        # curve khusus tanpa garis, cuma titik bulat -> nandain tiap puncak frekuensi
        # yang lolos deteksi _find_peaks(). Dibuat via .plot() (bukan ScatterPlotItem)
        # supaya otomatis ikut transform sumbu-x log seperti curve_fft.
        self.curve_peaks = self.plot_fft.plot(
            [], [], pen=None, symbol="o", symbolSize=9,
            symbolBrush=pg.mkBrush("#ff5252"), symbolPen=pg.mkPen("#ffffff", width=1),
        )
        self.tabs.addTab(self.plot_fft, "Spectrum (FFT)")

        # --- panel info bawah ---
        info_group = QtWidgets.QGroupBox("Informasi Sinyal")
        info_layout = QtWidgets.QGridLayout(info_group)
        main_layout.addWidget(info_group)

        self.freq_label = self._add_info_field(info_layout, 0, 0, "Frekuensi (zero-crossing)")
        self.peak_fft_label = self._add_info_field(info_layout, 0, 1, "Frekuensi Dominan (FFT)")
        self.vpp_label = self._add_info_field(info_layout, 0, 2, "Vpp")
        self.rate_label = self._add_info_field(info_layout, 0, 3, "Sampling Rate (aktual)")
        self.duration_label = self._add_info_field(info_layout, 1, 0, "Durasi / Waktu Sampling")
        self.dc_label = self._add_info_field(info_layout, 1, 1, "DC Offset (rata-rata)")
        self.update_label = self._add_info_field(info_layout, 1, 2, "Update Terakhir")
        self.fps_label = self._add_info_field(info_layout, 1, 3, "Refresh Rate GUI")

        self._frame_times = []

        # --- panel daftar puncak frekuensi terdeteksi (multi-peak) ---
        peaks_group = QtWidgets.QGroupBox("Puncak Frekuensi Terdeteksi (Spectrum)")
        peaks_layout = QtWidgets.QVBoxLayout(peaks_group)
        main_layout.addWidget(peaks_group)

        self.peaks_list_label = QtWidgets.QLabel("- (belum ada data / cuma 1 puncak dominan)")
        self.peaks_list_label.setStyleSheet("font-family: monospace; font-size: 10pt;")
        self.peaks_list_label.setWordWrap(True)
        peaks_layout.addWidget(self.peaks_list_label)

    def _add_info_field(self, layout, row, col, title):
        box = QtWidgets.QVBoxLayout()
        title_lbl = QtWidgets.QLabel(title + ":")
        title_lbl.setStyleSheet("font-weight: bold; font-size: 9pt;")
        value_lbl = QtWidgets.QLabel("-")
        value_lbl.setStyleSheet("font-size: 12pt;")
        box.addWidget(title_lbl)
        box.addWidget(value_lbl)
        wrapper = QtWidgets.QWidget()
        wrapper.setLayout(box)
        layout.addWidget(wrapper, row, col)
        return value_lbl

    def _list_ports(self):
        return [p.device for p in serial.tools.list_ports.comports()]

    def _refresh_ports(self):
        self.port_combo.clear()
        self.port_combo.addItems(self._list_ports())

    # ---------------- Koneksi ----------------
    def _toggle_connect(self):
        if self.reader is None:
            port = self.port_combo.currentText().strip()
            if not port:
                self.status_label.setText("Pilih port dulu!")
                return
            try:
                baud = int(self.baud_edit.text())
            except ValueError:
                self.status_label.setText("Baud rate tidak valid")
                return

            self.reader = SerialReader(port, baud, self.data_queue)
            self.reader.start()
            self.status_label.setText(f"Menghubungkan ke {port} @ {baud}...")
            self.connect_btn.setText("Disconnect")
        else:
            self.reader.stop()
            self.reader = None
            self.status_label.setText("Terputus")
            self.connect_btn.setText("Connect")

    # ---------------- Run/Stop (Hold) ----------------
    def _toggle_run_stop(self):
        self.running = not self.running
        if self.running:
            self.run_stop_btn.setText("Stop")
            self.run_stop_btn.setStyleSheet(
                "font-weight: bold; background-color: #c62828; color: white;"
            )
        else:
            self.run_stop_btn.setText("Run")
            self.run_stop_btn.setStyleSheet(
                "font-weight: bold; background-color: #2e7d32; color: white;"
            )

    # ---------------- Polling data (dipanggil QTimer, jauh lebih sering drpd versi Tkinter) ----------------
    def _poll_queue(self):
        got_data = False
        try:
            while True:
                item = self.data_queue.get_nowait()
                kind = item[0]
                if kind == "data":
                    _, samples, interval_ns = item
                    # Queue tetap di-drain terus (biar tidak numpuk/lag saat Run lagi),
                    # tapi kalau lagi di-Stop, data baru TIDAK diproses & TIDAK digambar
                    # -> layar freeze di frame terakhir sebelum Stop ditekan.
                    if self.running:
                        self._process_data(samples, interval_ns)
                        self.status_label.setText("Terhubung - menerima data")
                        got_data = True
                    else:
                        self.status_label.setText("DI-HOLD (Stop) - data masuk diabaikan")
                elif kind == "connected":
                    self.status_label.setText("Port terbuka - menunggu data dari Arduino...")
                elif kind == "waiting":
                    self.status_label.setText(
                        "Port terbuka, TAPI belum ada byte masuk sama sekali "
                        "(cek: Due sudah diupload & running? kabel Native USB? power Due?)"
                    )
                elif kind == "error":
                    self.status_label.setText(f"Error: {item[1]}")
                elif kind == "warn":
                    self.status_label.setText(item[1])
        except queue.Empty:
            pass

        if got_data:
            self._frame_times.append(time.perf_counter())
            cutoff = self._frame_times[-1] - 1.0
            self._frame_times = [t for t in self._frame_times if t >= cutoff]
            self.fps_label.setText(f"{len(self._frame_times)} paket/s")

    # ---------------- Pengolahan sinyal ----------------
    def _process_data(self, samples, interval_ns):
        voltage = (samples.astype(np.float64) / ADC_MAX) * VREF
        n = len(voltage)
        interval_s = interval_ns * 1e-9
        t_ms = np.arange(n) * interval_s * 1000.0

        vpp = float(voltage.max() - voltage.min())
        dc_offset = float(voltage.mean())
        freq = self._estimate_frequency(voltage, interval_s)
        duration_ms = n * interval_s * 1000.0
        actual_rate = 1.0 / interval_s if interval_s > 0 else 0

        self.vpp_label.setText(f"{vpp:.3f} Vpp")
        self.dc_label.setText(f"{dc_offset:.3f} V")
        self.duration_label.setText(f"{duration_ms:.2f} ms")
        self.update_label.setText(time.strftime("%H:%M:%S"))

        if freq is not None:
            self.freq_label.setText(f"{freq:.2f} Hz")
        else:
            self.freq_label.setText("N/A (kurang dari 1 siklus penuh)")

        if actual_rate >= 1e6:
            self.rate_label.setText(f"{actual_rate / 1e6:.3f} MSa/s")
        elif actual_rate >= 1e3:
            self.rate_label.setText(f"{actual_rate / 1e3:.3f} kSa/s")
        else:
            self.rate_label.setText(f"{actual_rate:.1f} Sa/s")

        self._t_ms = t_ms
        self._voltage = voltage
        self._interval_s = interval_s
        self._redraw_time()
        self._redraw_fft()

    def _estimate_frequency(self, voltage, interval_s):
        mean_v = voltage.mean()
        centered = voltage - mean_v
        crossings = np.where((centered[:-1] <= 0) & (centered[1:] > 0))[0]
        if len(crossings) < 2:
            return None
        periods_samples = np.diff(crossings)
        avg_period_s = np.mean(periods_samples) * interval_s
        if avg_period_s <= 0:
            return None
        return 1.0 / avg_period_s

    # ---------------- FFT / Spectrum ----------------
    def _compute_fft(self, voltage, interval_s):
        n = len(voltage)
        if n < 4 or interval_s is None or interval_s <= 0:
            return None, None

        sig = voltage - voltage.mean()

        win_name = self.fft_window_combo.currentText()
        if win_name == "Hanning":
            window = np.hanning(n)
        elif win_name == "Hamming":
            window = np.hamming(n)
        elif win_name == "Blackman":
            window = np.blackman(n)
        else:
            window = np.ones(n)

        sig_windowed = sig * window
        window_correction = n / np.sum(window) if np.sum(window) != 0 else 1.0

        fft_vals = np.fft.rfft(sig_windowed)
        freqs = np.fft.rfftfreq(n, d=interval_s)

        magnitude = (np.abs(fft_vals) / n) * 2.0 * window_correction
        magnitude[0] = np.abs(fft_vals[0]) / n

        if self.fft_scale_combo.currentText() == "dB":
            eps = 1e-12
            magnitude = 20 * np.log10(magnitude + eps)

        return freqs, magnitude
    
    def _parabolic_interpolation(self, mag_plot, freqs_plot, peak_idx):
        """Estimasi posisi puncak sebenarnya (sub-bin) pakai interpolasi parabola 
        dari 3 titik di sekitar bin puncak."""
        if peak_idx <= 0 or peak_idx >= len(mag_plot) - 1:
            return freqs_plot[peak_idx]
        
        y1 = mag_plot[peak_idx - 1]
        y2 = mag_plot[peak_idx]
        y3 = mag_plot[peak_idx + 1]
        
        denom = (y1 - 2*y2 + y3)
        if denom == 0:
            return freqs_plot[peak_idx]
        
        delta = 0.5 * (y1 - y3) / denom
        bin_spacing = freqs_plot[peak_idx] - freqs_plot[peak_idx - 1]
        
        return freqs_plot[peak_idx] + delta * bin_spacing

    def _redraw_fft(self, *_):
        if self._voltage is None or self._interval_s is None:
            return

        freqs, magnitude = self._compute_fft(self._voltage, self._interval_s)
        if freqs is None:
            return

        freqs_plot = freqs[1:]
        mag_plot = magnitude[1:]
        if len(freqs_plot) == 0:
            return

        self.curve_fft.setData(freqs_plot, mag_plot)

        ylabel = "Magnitude (dB)" if self.fft_scale_combo.currentText() == "dB" else "Magnitude (V)"
        self.plot_fft.setLabel("left", ylabel)

        # --- deteksi multi-puncak pakai scipy find_peaks ---
        is_db = self.fft_scale_combo.currentText() == "dB"
        threshold = np.max(mag_plot) - 20 if is_db else np.max(mag_plot) * 0.1
        peak_indices, _ = find_peaks(mag_plot, height=threshold, distance=5)

        if len(peak_indices) > 0:
            peak_freqs = freqs_plot[peak_indices]
            peak_mags = mag_plot[peak_indices]
            self.curve_peaks.setData(peak_freqs, peak_mags)

            order = np.argsort(peak_mags)[::-1][:5]
            unit = "dB" if is_db else "V"
            lines = [f"{peak_freqs[i]:.1f} Hz   ({peak_mags[i]:.2f} {unit})" for i in order]
            self.peaks_list_label.setText("\n".join(lines))

            top_idx = order[0]
            interpolated_freq = self._parabolic_interpolation(mag_plot, freqs_plot, peak_indices[top_idx])
            self.peak_fft_label.setText(f"{interpolated_freq:.2f} Hz")
        else:
            self.curve_peaks.setData([], [])
            self.peaks_list_label.setText("- (tidak ada puncak signifikan)")
            self.peak_fft_label.setText("-")

        # --- terapkan Start/Stop frequency range ---
        try:
            f_start = float(self.fft_start_edit.text())
            f_stop = float(self.fft_stop_edit.text())
            if f_start <= 0:
                f_start = 0.1
            if f_stop > f_start:
                self.plot_fft.setXRange(np.log10(f_start), np.log10(f_stop), padding=0)
        except ValueError:
            pass

    # ---------------- Plot waveform ----------------
    def _redraw_time(self, *_):
        if self._t_ms is None:
            return

        t_ms = self._t_ms
        voltage = self._voltage

        div_ms = float(self.time_div_combo.currentText())
        window_ms = div_ms * DIVISIONS

        mask = t_ms <= window_ms
        if mask.sum() < 2:
            mask = np.ones_like(t_ms, dtype=bool)

        self.curve_time.setData(t_ms[mask], voltage[mask])
        self.plot_time.setXRange(0, window_ms, padding=0)

    def closeEvent(self, event):
        if self.reader is not None:
            self.reader.stop()
        event.accept()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = OscilloscopeWindow()
    win.show()
    sys.exit(app.exec_())