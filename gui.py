import tkinter as tk
from tkinter import ttk, messagebox
import sys
import threading
import subprocess
import os
import time
import re

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


TB_LOG_DIR = "./husky_ppo_logs/"

TB_METRICS = [
    ("rollout/ep_rew_mean",        "Ep Reward"),
    ("rollout/ep_len_mean",        "Ep Length"),
    ("train/policy_gradient_loss", "Policy Loss"),
    ("train/value_loss",           "Value Loss"),
    ("train/entropy_loss",         "Entropy"),
    ("train/explained_variance",   "Explained Var"),
]


class TipBotTrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("TipBot Training Control Panel")
        self.root.geometry("1100x720")
        self.root.resizable(True, True)

        self.training_process = None
        self.demo_process = None
        self.is_training = False
        self.is_demo = False

        self._timesteps_history = []
        self._reward_history = []
        self._pending = {}

        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ── Left: controls ──────────────────────────────────────────────────
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        ttk.Button(control_frame, text="Start Training", command=self.start_training, width=25).pack(pady=5)
        self.stop_train_btn = ttk.Button(
            control_frame, text="Stop Training", command=self.stop_training,
            state="disabled", width=25,
        )
        self.stop_train_btn.pack(pady=5)

        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Button(control_frame, text="Run Demo", command=self.start_demo, width=25).pack(pady=5)
        self.stop_demo_btn = ttk.Button(
            control_frame, text="Stop Demo", command=self.stop_demo,
            state="disabled", width=25,
        )
        self.stop_demo_btn.pack(pady=5)

        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Button(control_frame, text="Open TensorBoard", command=self.open_tensorboard, width=25).pack(pady=5)

        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Label(control_frame, text="Status:", font=(None, 10, "bold")).pack(anchor=tk.W)
        self.status_label = ttk.Label(control_frame, text="Ready", foreground="green")
        self.status_label.pack(anchor=tk.W, pady=(0, 10))

        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(control_frame, text="Live Stats:", font=(None, 10, "bold")).pack(anchor=tk.W)

        self.lbl_timesteps = ttk.Label(control_frame, text="Timesteps:    —")
        self.lbl_timesteps.pack(anchor=tk.W, pady=1)
        self.lbl_reward = ttk.Label(control_frame, text="Mean Reward:  —")
        self.lbl_reward.pack(anchor=tk.W, pady=1)
        self.lbl_ep_len = ttk.Label(control_frame, text="Mean Ep Len:  —")
        self.lbl_ep_len.pack(anchor=tk.W, pady=1)
        self.lbl_fps = ttk.Label(control_frame, text="FPS:          —")
        self.lbl_fps.pack(anchor=tk.W, pady=1)

        # ── Right: log + tabbed charts ───────────────────────────────────────
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        log_frame = ttk.LabelFrame(right_frame, text="Console Log", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(log_frame, wrap=tk.WORD, state=tk.DISABLED, height=15)
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # ── Chart notebook ───────────────────────────────────────────────────
        chart_notebook = ttk.Notebook(right_frame)
        chart_notebook.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        # Tab 1: live reward
        live_tab = ttk.Frame(chart_notebook)
        chart_notebook.add(live_tab, text="  Live Training  ")

        self.fig = Figure(figsize=(5, 2.5), dpi=90, tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Timesteps")
        self.ax.set_ylabel("Reward")
        self.ax.grid(True, linestyle="--", alpha=0.4)
        (self.reward_line,) = self.ax.plot([], [], color="#2196F3", linewidth=1.8)

        self.canvas = FigureCanvasTkAgg(self.fig, master=live_tab)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Tab 2: TensorBoard log viewer
        tb_tab = ttk.Frame(chart_notebook)
        chart_notebook.add(tb_tab, text="  TF Log Viewer  ")

        tb_top = ttk.Frame(tb_tab)
        tb_top.pack(fill=tk.X, padx=6, pady=6)

        ttk.Label(tb_top, text="Run:").pack(side=tk.LEFT, padx=(0, 4))
        self.tb_run_var = tk.StringVar()
        self.tb_run_combo = ttk.Combobox(tb_top, textvariable=self.tb_run_var, state="readonly", width=32)
        self.tb_run_combo.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(tb_top, text="Load", command=self._load_tb_logs).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(tb_top, text="Refresh Runs", command=self._populate_tb_runs).pack(side=tk.LEFT)

        self.tb_fig = Figure(figsize=(7, 3.5), dpi=90, tight_layout=True)
        self.tb_axes = {}
        for i, (tag, title) in enumerate(TB_METRICS):
            ax = self.tb_fig.add_subplot(2, 3, i + 1)
            ax.set_title(title, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, linestyle="--", alpha=0.3)
            self.tb_axes[tag] = ax

        self.tb_canvas = FigureCanvasTkAgg(self.tb_fig, master=tb_tab)
        self.tb_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Tab 3: Sim View
        sim_tab = ttk.Frame(chart_notebook)
        chart_notebook.add(sim_tab, text="  Sim View  ")

        self._sim_photo = None
        self.sim_image_label = tk.Label(sim_tab, text="No preview yet.\nStart training to see the live simulation.",
                                        bg="#1e1e1e", fg="#aaaaaa", font=(None, 11))
        self.sim_image_label.pack(fill=tk.BOTH, expand=True)

        self._populate_tb_runs()
        self._poll_sim_view()
        self.log("Ready")

    # ── TensorBoard log viewer ───────────────────────────────────────────────

    def _populate_tb_runs(self):
        if not os.path.isdir(TB_LOG_DIR):
            return
        runs = sorted(
            [d for d in os.listdir(TB_LOG_DIR) if os.path.isdir(os.path.join(TB_LOG_DIR, d))],
            reverse=True,
        )
        self.tb_run_combo["values"] = runs
        if runs and not self.tb_run_var.get():
            self.tb_run_var.set(runs[0])

    def _load_tb_logs(self):
        run = self.tb_run_var.get()
        if not run:
            messagebox.showwarning("Warning", "No run selected")
            return
        log_path = os.path.join(TB_LOG_DIR, run)
        self.log(f"Loading TF logs from {run}...")
        threading.Thread(target=self._load_tb_thread, args=(log_path,), daemon=True).start()

    def _load_tb_thread(self, log_path):
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            ea = EventAccumulator(log_path)
            ea.Reload()
            available = set(ea.Tags().get("scalars", []))
            data = {}
            for tag, _ in TB_METRICS:
                if tag in available:
                    events = ea.Scalars(tag)
                    data[tag] = ([e.step for e in events], [e.value for e in events])
            self.root.after(0, lambda d=data: self._draw_tb_charts(d))
        except Exception as e:
            self.log(f"TF log load error: {e}")

    def _draw_tb_charts(self, data):
        for tag, title in TB_METRICS:
            ax = self.tb_axes[tag]
            ax.clear()
            ax.set_title(title, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, linestyle="--", alpha=0.3)
            if tag in data:
                steps, values = data[tag]
                ax.plot(steps, values, linewidth=1.4, color="#2196F3")
        self.tb_fig.tight_layout()
        self.tb_canvas.draw_idle()
        self.log("TF logs loaded.")

    # ── SB3 output parsing ───────────────────────────────────────────────────

    _PATTERNS = {
        "ep_rew_mean":     re.compile(r'\|\s+ep_rew_mean\s+\|\s+([\-\d.e+]+)\s+\|'),
        "ep_len_mean":     re.compile(r'\|\s+ep_len_mean\s+\|\s+([\-\d.e+]+)\s+\|'),
        "total_timesteps": re.compile(r'\|\s+total_timesteps\s+\|\s+(\d+)\s+\|'),
        "fps":             re.compile(r'\|\s+fps\s+\|\s+(\d+)\s+\|'),
    }

    def _parse_training_line(self, line):
        for key, pat in self._PATTERNS.items():
            m = pat.search(line)
            if not m:
                continue
            val = m.group(1)
            self._pending[key] = val
            if key == "total_timesteps":
                self.root.after(0, lambda v=val: self.lbl_timesteps.config(text=f"Timesteps:    {v}"))
            elif key == "ep_rew_mean":
                self.root.after(0, lambda v=val: self.lbl_reward.config(text=f"Mean Reward:  {v}"))
            elif key == "ep_len_mean":
                self.root.after(0, lambda v=val: self.lbl_ep_len.config(text=f"Mean Ep Len:  {v}"))
            elif key == "fps":
                self.root.after(0, lambda v=val: self.lbl_fps.config(text=f"FPS:          {v}"))

        # total_timesteps always appears after ep_rew_mean in each SB3 block.
        if "total_timesteps" in self._pending and "ep_rew_mean" in self._pending:
            t = int(self._pending["total_timesteps"])
            r = float(self._pending["ep_rew_mean"])
            self._timesteps_history.append(t)
            self._reward_history.append(r)
            self._pending.clear()
            self.root.after(0, self._refresh_chart)

    def _refresh_chart(self):
        self.reward_line.set_data(self._timesteps_history, self._reward_history)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()

    # ── Training ─────────────────────────────────────────────────────────────

    def start_training(self):
        if self.is_training:
            messagebox.showwarning("Warning", "Training is already running")
            return

        self.is_training = True
        self._timesteps_history.clear()
        self._reward_history.clear()
        self._pending.clear()
        self._refresh_chart()

        self.stop_train_btn.config(state="normal")
        self.status_label.config(text="Training...", foreground="orange")
        self.log("Starting training process...")

        threading.Thread(target=self._run_training, daemon=True).start()

    def _run_training(self):
        try:
            self.training_process = subprocess.Popen(
                [sys.executable, "train.py"],
                cwd=os.getcwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            for line in self.training_process.stdout:
                line = line.rstrip()
                self.log(line)
                self._parse_training_line(line)
                if not self.is_training:
                    break

            return_code = self.training_process.wait()
            if return_code == 0:
                self.log("Training completed successfully.")
                self.root.after(0, lambda: self.status_label.config(text="Training complete", foreground="green"))
            else:
                self.log(f"Training stopped (exit code {return_code}).")
                self.root.after(0, lambda: self.status_label.config(text="Training stopped", foreground="red"))
        except Exception as e:
            self.log(f"Training error: {e}")
            self.root.after(0, lambda: self.status_label.config(text="Training error", foreground="red"))
        finally:
            self.is_training = False
            self.root.after(0, lambda: self.stop_train_btn.config(state="disabled"))
            self.root.after(0, self._populate_tb_runs)

    def stop_training(self):
        if self.training_process and self.training_process.poll() is None:
            self.training_process.terminate()
            self.log("Training process terminated by user.")
            self.status_label.config(text="Training stopped", foreground="red")
        self.is_training = False
        self.stop_train_btn.config(state="disabled")

    # ── Demo ─────────────────────────────────────────────────────────────────

    def start_demo(self):
        if self.is_demo:
            messagebox.showwarning("Warning", "Demo is already running")
            return

        if not os.path.exists("husky_chaser_ppo_final.zip"):
            messagebox.showerror("Error", "Trained model not found. Train first!")
            return

        self.is_demo = True
        self.stop_demo_btn.config(state="normal")
        self.status_label.config(text="Demo running", foreground="blue")
        self.log("Starting visual demo...")

        threading.Thread(target=self._run_demo, daemon=True).start()

    def _run_demo(self):
        try:
            self.demo_process = subprocess.Popen(
                [sys.executable, "demo.py"],
                cwd=os.getcwd(),
            )
            self.demo_process.wait()
            self.log("Demo process exited.")
        except Exception as e:
            self.log(f"Demo error: {e}")
        finally:
            self.is_demo = False
            self.root.after(0, lambda: self.stop_demo_btn.config(state="disabled"))
            self.root.after(0, lambda: self.status_label.config(text="Ready", foreground="green"))

    def stop_demo(self):
        if self.demo_process and self.demo_process.poll() is None:
            self.demo_process.terminate()
            self.log("Demo process terminated by user.")
        self.is_demo = False
        self.stop_demo_btn.config(state="disabled")
        self.status_label.config(text="Ready", foreground="green")

    # ── TensorBoard process ───────────────────────────────────────────────────

    def open_tensorboard(self):
        try:
            subprocess.Popen([sys.executable, "-m", "tensorboard", "--logdir=./husky_ppo_logs/"])
            self.log("Opened TensorBoard on http://localhost:6006")
        except Exception as e:
            self.log(f"TensorBoard error: {e}")
            messagebox.showerror("Error", f"Could not open TensorBoard: {e}")

    # ── Sim View ──────────────────────────────────────────────────────────────

    def _poll_sim_view(self):
        path = "preview_frame.png"
        if os.path.exists(path):
            try:
                from PIL import Image, ImageTk
                img = Image.open(path).copy()
                img = img.resize((640, 400), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.sim_image_label.config(image=photo, text="", bg="#1e1e1e")
                self._sim_photo = photo
            except Exception:
                pass
        self.root.after(1000, self._poll_sim_view)

    # ── Logging ───────────────────────────────────────────────────────────────

    def log(self, message):
        def _insert():
            self.log_text.config(state=tk.NORMAL)
            timestamp = time.strftime("%H:%M:%S")
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
        self.root.after(0, _insert)


if __name__ == "__main__":
    root = tk.Tk()
    TipBotTrainingGUI(root)
    root.mainloop()
