"""
=====================================================
OSILOSKOP SEDERHANA - GUI (Python + Tkinter + Matplotlib)
=====================================================
Menerima paket data biner dari Arduino Due lewat serial port,
menampilkan gelombang di tengah + info (frekuensi, Vpp, sampling rate,
waktu/durasi sampling) di bawah, plus pengaturan Time/Div ala osiloskop asli.

Instalasi dependency (sekali saja):
    pip install pyserial numpy matplotlib

Menjalankan:
    python oscilloscope_gui.py
"""

import struct
import threading
import queue
import time

import numpy as np
import serial
import serial.tools.list_ports
import tkinter as tk
from tkinter import ttk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ================= KONFIGURASI PROTOKOL (harus sama dgn sisi Arduino) =================
SYNC_BYTES = b"\xAA\x55"
ADC_MAX = 4095       # ADC 12-bit
VREF = 3.3           # volt, referensi ADC Due

TIME_DIV_OPTIONS_MS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 
                       1, 2, 5, 10, 20, 25, 50, 100]  # pilihan time/div (ms)
DIVISIONS = 10                                  # konvensi osiloskop: 10 kotak horizontal


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
        """Cari & ekstrak paket lengkap dari buffer bytearray (bisa dipanggil berkali-kali
        karena data serial datang per potongan/chunk, belum tentu 1 paket utuh)."""
        while True:
            sync_idx = buf.find(SYNC_BYTES)
            if sync_idx == -1:
                buf.clear()
                return
            if sync_idx > 0:
                del buf[:sync_idx]

            if len(buf) < 10:  # 2 sync + 4 n + 4 interval
                return

            n, interval_ns = struct.unpack_from("<II", buf, 2)
            
            if n <= 0 or n > 200000:
                # nilai n ga masuk akal -> kemungkinan false sync match, buang & cari lagi
                del buf[:2]
                continue

            total_len = 10 + n * 2 + 2  # header + samples + checksum
            if len(buf) < total_len:
                return  # paket belum lengkap, tunggu data berikutnya

            samples_bytes = bytes(buf[10:10 + n * 2])
            checksum_recv = struct.unpack_from("<H", buf, 10 + n * 2)[0]

            samples = np.frombuffer(samples_bytes, dtype="<u2")
            checksum_calc = 0
            for v in samples:
                checksum_calc ^= int(v)

            print("calc =", checksum_calc, "recv =", checksum_recv)

            if checksum_calc == checksum_recv:
                self.data_queue.put(("data", samples.copy(), interval_ns))
            else:
                self.data_queue.put(("warn", "Checksum tdk cck"))

            del buf[:total_len]

    def stop(self):
        self._stop_flag.set()
        if self.ser and self.ser.is_open:
            self.ser.close()


class OscilloscopeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Osiloskop Sederhana - Arduino Due")
        self.geometry("950x700")

        self.data_queue = queue.Queue()
        self.reader = None
        self.time_div_ms = tk.DoubleVar(value=2)

        self._last_samples = None
        self._last_interval_ns = None
        self._t_ms = None
        self._voltage = None

        self._build_ui()
        self._poll_queue()

    # ---------------- UI ----------------
    def _build_ui(self):
        top = ttk.Frame(self, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        left = ttk.Frame(top)
        left.pack(side=tk.LEFT)

        right = ttk.Frame(top)
        right.pack(side=tk.RIGHT)

        # Kontrol koneksi — pakai left
        ttk.Label(left, text="Port:").pack(side=tk.LEFT)
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(
            left, textvariable=self.port_var, width=15, values=self._list_ports()
        )
        self.port_combo.pack(side=tk.LEFT, padx=4)

        ttk.Button(left, text="Refresh", command=self._refresh_ports).pack(side=tk.LEFT)

        ttk.Label(left, text="Baud:").pack(side=tk.LEFT, padx=(12, 0))
        self.baud_var = tk.StringVar(value="2000000")
        ttk.Entry(left, textvariable=self.baud_var, width=10).pack(side=tk.LEFT, padx=4)

        self.connect_btn = ttk.Button(left, text="Connect", command=self._toggle_connect)
        self.connect_btn.pack(side=tk.LEFT, padx=12)

        self.status_var = tk.StringVar(value="Belum terhubung")
        ttk.Label(left, textvariable=self.status_var, foreground="gray").pack(
            side=tk.LEFT, padx=12
        )

        # Time/Div — pakai right, sehingga selalu menempel di kanan
        ttk.Label(right, text="Time/Div:").pack(side=tk.LEFT, padx=(0, 5))

        self.time_div_combo = ttk.Combobox(
            right, textvariable=self.time_div_ms, width=8, state="readonly",
            values=TIME_DIV_OPTIONS_MS
        )

        self.time_div_combo.pack(side=tk.LEFT, padx=4)

        ttk.Label(right, text="ms/div").pack(side=tk.LEFT)    
        self.time_div_combo.bind("<<ComboboxSelected>>", lambda e: self._redraw())

        # --- area gelombang (tengah) ---
        self.fig = Figure(figsize=(9, 4.2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Waktu (ms)")
        self.ax.set_ylabel("Tegangan (V)")
        self.ax.grid(True, linestyle="--", alpha=0.5)
        (self.line,) = self.ax.plot(
            [], [], color="#00c853",
            linewidth=1.2,
            drawstyle="steps-post",
            marker=".",
            markersize=2
        )
        self.ax.set_ylim(-0.2, VREF + 0.2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=4)

        # --- info panel (bawah) ---
        info = ttk.LabelFrame(self, text="Informasi Sinyal", padding=10)
        info.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=8)

        self.freq_var = tk.StringVar(value="-")
        self.vpp_var = tk.StringVar(value="-")
        self.rate_var = tk.StringVar(value="-")
        self.duration_var = tk.StringVar(value="-")
        self.dc_var = tk.StringVar(value="-")
        self.update_var = tk.StringVar(value="-")

        fields = [
            ("Frekuensi", self.freq_var),
            ("Vpp", self.vpp_var),
            ("Sampling Rate (aktual)", self.rate_var),
            ("Durasi / Waktu Sampling", self.duration_var),
            ("DC Offset (rata-rata)", self.dc_var),
            ("Update Terakhir", self.update_var),
        ]
        for i, (label, var) in enumerate(fields):
            r, c = divmod(i, 3)
            cell = ttk.Frame(info)
            cell.grid(row=r, column=c, sticky="w", padx=16, pady=4)
            ttk.Label(cell, text=label + ":", font=("Segoe UI", 9, "bold")).pack(anchor="w")
            ttk.Label(cell, textvariable=var, font=("Segoe UI", 12)).pack(anchor="w")

    def _list_ports(self):
        return [p.device for p in serial.tools.list_ports.comports()]

    def _refresh_ports(self):
        self.port_combo["values"] = self._list_ports()

    # ---------------- Koneksi ----------------
    def _toggle_connect(self):
        if self.reader is None:
            port = self.port_var.get()
            if not port:
                self.status_var.set("Pilih port dulu!")
                return
            try:
                baud = int(self.baud_var.get())
            except ValueError:
                self.status_var.set("Baud rate tidak valid")
                return

            self.reader = SerialReader(port, baud, self.data_queue)
            self.reader.start()
            self.status_var.set(f"Menghubungkan ke {port} @ {baud}...")
            self.connect_btn.config(text="Disconnect")
        else:
            self.reader.stop()
            self.reader = None
            self.status_var.set("Terputus")
            self.connect_btn.config(text="Connect")

    # ---------------- Polling data dari thread serial ----------------
    def _poll_queue(self):
        try:
            while True:
                item = self.data_queue.get_nowait()
                kind = item[0]
                if kind == "data":
                    _, samples, interval_ns = item
                    self._last_samples = samples
                    self._last_interval_ns = interval_ns
                    self._process_and_draw(samples, interval_ns)
                    self.status_var.set("Terhubung - menerima data")
                elif kind == "connected":
                    self.status_var.set("Port terbuka - menunggu data dari Arduino...")
                elif kind == "waiting":
                    self.status_var.set("Port terbuka, TAPI belum ada byte masuk sama sekali "
                                         "(cek: Due sudah diupload & running? kabel Native USB? power Due?)")
                elif kind == "error":
                    self.status_var.set(f"Error: {item[1]}")
                elif kind == "warn":
                    self.status_var.set(item[1])
        except queue.Empty:
            pass
        finally:
            self.after(50, self._poll_queue)


    # ---------------- Pengolahan sinyal ----------------
    def _process_and_draw(self, samples, interval_ns):
        voltage = (samples.astype(np.float64) / ADC_MAX) * VREF
        n = len(voltage)
        interval_s = interval_ns * 1e-9
        t_ms = np.arange(n) * interval_s * 1000.0  # sumbu waktu dalam ms

        vpp = float(voltage.max() - voltage.min())
        dc_offset = float(voltage.mean())
        freq = self._estimate_frequency(voltage, interval_s)
        duration_ms = n * interval_s * 1000.0
        actual_rate = 1.0 / interval_s if interval_s > 0 else 0

        self.vpp_var.set(f"{vpp:.3f} Vpp")
        self.dc_var.set(f"{dc_offset:.3f} V")
        self.duration_var.set(f"{duration_ms:.2f} ms")
        self.update_var.set(time.strftime("%H:%M:%S"))

        if freq is not None:
            self.freq_var.set(f"{freq:.2f} Hz")
        else:
            self.freq_var.set("N/A (kurang dari 1 siklus penuh)")

        if actual_rate >= 1e6:
            self.rate_var.set(f"{actual_rate / 1e6:.3f} MSa/s")
        elif actual_rate >= 1e3:
            self.rate_var.set(f"{actual_rate / 1e3:.3f} kSa/s")
        else:
            self.rate_var.set(f"{actual_rate:.1f} Sa/s")

        self._t_ms = t_ms
        self._voltage = voltage
        self._redraw()

    def _estimate_frequency(self, voltage, interval_s):
        """Deteksi frekuensi pakai metode zero-crossing (naik) di sekitar rata-rata sinyal."""
        mean_v = voltage.mean()
        centered = voltage - mean_v
        # pakai <= 0 di sisi kiri supaya crossing tepat di titik rata-rata (fase 0) tetap terdeteksi
        crossings = np.where((centered[:-1] <= 0) & (centered[1:] > 0))[0]

        if len(crossings) < 2:
            return None

        periods_samples = np.diff(crossings)
        avg_period_s = np.mean(periods_samples) * interval_s
        if avg_period_s <= 0:
            return None
        return 1.0 / avg_period_s

    # ---------------- Plot ----------------
    def _redraw(self):
        if self._t_ms is None:
            return

        t_ms = self._t_ms
        voltage = self._voltage

        div_ms = float(self.time_div_ms.get())
        window_ms = div_ms * DIVISIONS

        mask = t_ms <= window_ms
        if mask.sum() < 2:
            mask = np.ones_like(t_ms, dtype=bool)  # buffer lebih pendek dari window -> tampilkan semua

        self.line.set_data(t_ms[mask], voltage[mask])
        self.ax.set_xlim(0, window_ms)

        self.ax.set_xticks(np.arange(0, window_ms + div_ms, div_ms))
        self.canvas.draw_idle()

    def on_close(self):
        if self.reader is not None:
            self.reader.stop()
        self.destroy()


if __name__ == "__main__":
    app = OscilloscopeApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()