import tkinter as tk
from tkinter import ttk, messagebox
import sys
import threading
import subprocess
import os
import time

class TipBotTrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("TipBot Training Control Panel")
        self.root.geometry("800x600")
        self.root.resizable(True, True)

        self.training_process = None
        self.demo_process = None
        self.is_training = False
        self.is_demo = False

        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        ttk.Button(control_frame, text="Start Training", command=self.start_training, width=25).pack(pady=5)
        self.stop_train_btn = ttk.Button(control_frame, text="Stop Training", command=self.stop_training, state="disabled", width=25)
        self.stop_train_btn.pack(pady=5)

        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Button(control_frame, text="Run Demo", command=self.start_demo, width=25).pack(pady=5)
        self.stop_demo_btn = ttk.Button(control_frame, text="Stop Demo", command=self.stop_demo, state="disabled", width=25)
        self.stop_demo_btn.pack(pady=5)

        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Button(control_frame, text="Open TensorBoard", command=self.open_tensorboard, width=25).pack(pady=5)

        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Label(control_frame, text="Status:", font=(None, 10, "bold")).pack(anchor=tk.W)
        self.status_label = ttk.Label(control_frame, text="Ready", foreground="green")
        self.status_label.pack(anchor=tk.W, pady=(0, 10))

        log_frame = ttk.LabelFrame(main_frame, text="Console Log", padding=10)
        log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(log_frame, wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self.log("Ready")

    def start_training(self):
        if self.is_training:
            messagebox.showwarning("Warning", "Training is already running")
            return

        self.is_training = True
        self.stop_train_btn.config(state="normal")
        self.status_label.config(text="Training...", foreground="orange")
        self.log("Starting training process...")

        thread = threading.Thread(target=self._run_training, daemon=True)
        thread.start()

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

            while True:
                line = self.training_process.stdout.readline()
                if line:
                    self.log(line.rstrip())
                if self.training_process.poll() is not None:
                    break
                if not self.is_training:
                    break
                time.sleep(0.1)

            return_code = self.training_process.poll()
            if return_code == 0:
                self.log("Training completed successfully.")
                self.status_label.config(text="Training complete", foreground="green")
            else:
                self.log(f"Training stopped with exit code {return_code}.")
                self.status_label.config(text="Training stopped", foreground="red")
        except Exception as e:
            self.log(f"Training error: {e}")
            self.status_label.config(text="Training error", foreground="red")
        finally:
            self.is_training = False
            self.stop_train_btn.config(state="disabled")

    def stop_training(self):
        if self.training_process and self.training_process.poll() is None:
            self.training_process.terminate()
            self.log("Training process terminated by user.")
            self.status_label.config(text="Training stopped", foreground="red")
        self.is_training = False
        self.stop_train_btn.config(state="disabled")

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

        thread = threading.Thread(target=self._run_demo, daemon=True)
        thread.start()

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
            self.stop_demo_btn.config(state="disabled")
            self.status_label.config(text="Ready", foreground="green")

    def stop_demo(self):
        if self.demo_process and self.demo_process.poll() is None:
            self.demo_process.terminate()
            self.log("Demo process terminated by user.")
        self.is_demo = False
        self.stop_demo_btn.config(state="disabled")
        self.status_label.config(text="Ready", foreground="green")

    def open_tensorboard(self):
        try:
            subprocess.Popen([sys.executable, "-m", "tensorboard", "--logdir=./husky_ppo_logs/"])
            self.log("Opened TensorBoard on http://localhost:6006")
        except Exception as e:
            self.log(f"TensorBoard error: {e}")
            messagebox.showerror("Error", f"Could not open TensorBoard: {e}")

    def log(self, message):
        self.log_text.config(state=tk.NORMAL)
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    TipBotTrainingGUI(root)
    root.mainloop()
