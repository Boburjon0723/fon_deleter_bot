"""
Terminalsiz ishlatish: bitta rasm yoki papkadagi barcha rasmlar.
Ishga tushirish: python gui.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from segment_products import QUALITY_PRESETS, get_inference_params, process_folder, process_image


def _open_folder(path: Path) -> None:
    path = path.resolve()
    if sys.platform == "win32":
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(path)], check=False)


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Mahsulot segmentatsiyasi — fonni olib tashlash")
        self.minsize(560, 480)
        self._build()
        self._worker: threading.Thread | None = None

    def _build(self) -> None:
        pad = {"padx": 10, "pady": 6}
        frm = ttk.Frame(self, padding=12)
        frm.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        frm.columnconfigure(1, weight=1)

        self.var_batch = tk.BooleanVar(value=False)
        self.var_input = tk.StringVar()
        self.var_output = tk.StringVar()
        self.var_mode = tk.StringVar(value="white")
        self.var_device = tk.StringVar(value="cpu")
        self.var_conf = tk.DoubleVar(value=0.25)
        self.var_min_area = tk.DoubleVar(value=0.002)
        self.var_max_obj = tk.IntVar(value=50)
        self.var_filter_nested = tk.BooleanVar(value=True)
        self.var_layout = tk.StringVar(value="centered")
        self.var_canvas_side = tk.StringVar(value="1024")
        self.var_quality = tk.StringVar(value="normal")

        r = 0
        ttk.Checkbutton(
            frm,
            text="Papka rejimi — ichidagi barcha rasmlarni qayta ishlash",
            variable=self.var_batch,
            command=self._toggle_labels,
        ).grid(row=r, column=0, columnspan=3, sticky="w", **pad)
        r += 1

        self.lbl_in = ttk.Label(frm, text="Kirish rasmi:")
        self.lbl_in.grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.var_input, width=52).grid(row=r, column=1, sticky="ew", **pad)
        self.btn_pick = ttk.Button(frm, text="Tanlash…", command=self._pick_input)
        self.btn_pick.grid(row=r, column=2, **pad)
        r += 1

        ttk.Label(frm, text="Chiqish papkasi:").grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.var_output, width=52).grid(row=r, column=1, sticky="ew", **pad)
        ttk.Button(frm, text="Tanlash…", command=self._pick_output).grid(row=r, column=2, **pad)
        r += 1

        ttk.Label(frm, text="Saqlanadi:").grid(row=r, column=0, sticky="nw", **pad)
        self.lbl_out_full = ttk.Label(
            frm,
            text="",
            wraplength=480,
            justify=tk.LEFT,
            font=("Segoe UI", 9),
            foreground="#333333",
        )
        self.lbl_out_full.grid(row=r, column=1, columnspan=2, sticky="ew", **pad)
        r += 1

        ttk.Label(frm, text="Chiqish ko‘rinishi:").grid(row=r, column=0, sticky="nw", **pad)
        lay_frm = ttk.Frame(frm)
        lay_frm.grid(row=r, column=1, columnspan=2, sticky="w", **pad)
        ttk.Radiobutton(
            lay_frm,
            text="Markazda, kattaroq (kvadrat)",
            variable=self.var_layout,
            value="centered",
            command=self._toggle_canvas_state,
        ).pack(anchor="w")
        ttk.Radiobutton(
            lay_frm,
            text="To‘liq asl kadr (ob’yekt o‘z joyida, kichik bo‘lishi mumkin)",
            variable=self.var_layout,
            value="full",
            command=self._toggle_canvas_state,
        ).pack(anchor="w")
        r += 1

        ttk.Label(frm, text="Kvadrat o‘lchami (px):").grid(row=r, column=0, sticky="w", **pad)
        cv_frm = ttk.Frame(frm)
        cv_frm.grid(row=r, column=1, sticky="w", **pad)
        self.cb_canvas = ttk.Combobox(
            cv_frm,
            textvariable=self.var_canvas_side,
            values=("512", "768", "1024", "1536", "2048"),
            width=8,
            state="readonly",
        )
        self.cb_canvas.pack(side=tk.LEFT)
        ttk.Label(cv_frm, text="  (faqat «markazda» rejimida)").pack(side=tk.LEFT)
        r += 1

        ttk.Label(frm, text="Fon rejimi:").grid(row=r, column=0, sticky="w", **pad)
        mode_frm = ttk.Frame(frm)
        mode_frm.grid(row=r, column=1, sticky="w", **pad)
        ttk.Radiobutton(
            mode_frm, text="Oq (#FFFFFF)", variable=self.var_mode, value="white"
        ).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Radiobutton(
            mode_frm, text="Shaffof (RGBA)", variable=self.var_mode, value="transparent"
        ).pack(side=tk.LEFT)
        r += 1

        ttk.Label(frm, text="Hisoblash:").grid(row=r, column=0, sticky="w", **pad)
        dev_frm = ttk.Frame(frm)
        dev_frm.grid(row=r, column=1, sticky="w", **pad)
        ttk.Radiobutton(dev_frm, text="CPU", variable=self.var_device, value="cpu").pack(
            side=tk.LEFT, padx=(0, 12)
        )
        ttk.Radiobutton(dev_frm, text="GPU (CUDA)", variable=self.var_device, value="cuda").pack(
            side=tk.LEFT
        )
        r += 1

        ttk.Label(frm, text="Segmentatsiya sifati:").grid(row=r, column=0, sticky="w", **pad)
        qf = ttk.Frame(frm)
        qf.grid(row=r, column=1, columnspan=2, sticky="w", **pad)
        self.cb_quality = ttk.Combobox(
            qf,
            textvariable=self.var_quality,
            values=("draft", "normal", "high", "max"),
            width=10,
            state="readonly",
        )
        self.cb_quality.pack(side=tk.LEFT)
        ttk.Label(
            qf,
            text="  (max = eng aniq, sekin; GPU tavsiya)",
            font=("Segoe UI", 8),
        ).pack(side=tk.LEFT)
        r += 1
        self.cb_quality.bind("<<ComboboxSelected>>", self._sync_conf_from_quality)

        ttk.Label(frm, text="Ishonchlilik (conf):").grid(row=r, column=0, sticky="w", **pad)
        ttk.Scale(
            frm, from_=0.05, to=0.9, variable=self.var_conf, orient=tk.HORIZONTAL, length=300
        ).grid(row=r, column=1, sticky="ew", **pad)
        ttk.Label(frm, textvariable=self.var_conf).grid(row=r, column=2, **pad)
        r += 1

        ttk.Label(frm, text="Minimal maydon ulushi:").grid(row=r, column=0, sticky="w", **pad)
        ttk.Scale(
            frm, from_=0.0005, to=0.05, variable=self.var_min_area, orient=tk.HORIZONTAL, length=300
        ).grid(row=r, column=1, sticky="ew", **pad)
        ttk.Label(frm, textvariable=self.var_min_area).grid(row=r, column=2, **pad)
        r += 1

        ttk.Label(frm, text="Maks. ob’yekt soni:").grid(row=r, column=0, sticky="w", **pad)
        ttk.Spinbox(frm, from_=1, to=200, textvariable=self.var_max_obj, width=8).grid(
            row=r, column=1, sticky="w", **pad
        )
        r += 1

        ttk.Checkbutton(
            frm,
            text="Ichki/kichik maskalarni filtrlash",
            variable=self.var_filter_nested,
        ).grid(row=r, column=1, sticky="w", **pad)
        r += 1

        self.btn_run = ttk.Button(frm, text="Ishlash", command=self._run)
        self.btn_run.grid(row=r, column=1, sticky="w", pady=12)
        r += 1

        self.txt_log = tk.Text(frm, height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.txt_log.grid(row=r, column=0, columnspan=3, sticky="nsew", **pad)
        frm.rowconfigure(r, weight=1)

        self.btn_open = ttk.Button(
            frm, text="Chiqish papkasini ochish", command=self._open_out, state=tk.DISABLED
        )
        self.btn_open.grid(row=r + 1, column=1, sticky="w")

        self._last_out: Path | None = None
        self._toggle_labels()
        self.var_output.trace_add("write", self._refresh_out_path)
        self._refresh_out_path()
        self._toggle_canvas_state()

    def _sync_conf_from_quality(self, *_args: object) -> None:
        q = self.var_quality.get()
        if q in QUALITY_PRESETS:
            self.var_conf.set(QUALITY_PRESETS[q]["conf"])

    def _refresh_out_path(self, *_args: object) -> None:
        t = self.var_output.get().strip()
        if not t:
            self.lbl_out_full.configure(text="(chiqish papkasi tanlanmagan — «Tanlash» yoki avtomatik)")
            return
        try:
            self.lbl_out_full.configure(text=str(Path(t).expanduser().resolve()))
        except OSError:
            self.lbl_out_full.configure(text=t)

    def _toggle_canvas_state(self) -> None:
        if self.var_layout.get() == "centered":
            self.cb_canvas.configure(state="readonly")
        else:
            self.cb_canvas.configure(state="disabled")

    def _toggle_labels(self) -> None:
        if self.var_batch.get():
            self.lbl_in.configure(text="Kirish papkasi:")
        else:
            self.lbl_in.configure(text="Kirish rasmi:")

    def _log(self, text: str) -> None:
        self.txt_log.configure(state=tk.NORMAL)
        self.txt_log.insert(tk.END, text + "\n")
        self.txt_log.see(tk.END)
        self.txt_log.configure(state=tk.DISABLED)

    def _pick_input(self) -> None:
        if self.var_batch.get():
            p = filedialog.askdirectory(title="Rasmlar joylashgan papka")
            if p:
                self.var_input.set(p)
                if not self.var_output.get().strip():
                    self.var_output.set(str(Path(p) / "output"))
        else:
            p = filedialog.askopenfilename(
                title="Rasmni tanlang",
                filetypes=[
                    ("Rasmlar", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"),
                    ("Barcha fayllar", "*.*"),
                ],
            )
            if p:
                self.var_input.set(p)
                if not self.var_output.get().strip():
                    self.var_output.set(str(Path(p).parent / "output"))

    def _pick_output(self) -> None:
        p = filedialog.askdirectory(title="Chiqish papkasi")
        if p:
            self.var_output.set(p)

    def _open_out(self) -> None:
        if self._last_out and self._last_out.is_dir():
            _open_folder(self._last_out)

    def _params(self):
        try:
            cs = int(self.var_canvas_side.get())
        except ValueError:
            cs = 1024
        inf = get_inference_params(
            self.var_quality.get(),
            conf=float(self.var_conf.get()),
        )
        return dict(
            conf=inf["conf"],
            iou=inf["iou"],
            imgsz=inf["imgsz"],
            device=self.var_device.get(),
            min_area_ratio=float(self.var_min_area.get()),
            mode=self.var_mode.get(),
            filter_subsumed=bool(self.var_filter_nested.get()),
            max_objects=int(self.var_max_obj.get()),
            layout=self.var_layout.get(),
            canvas_side=max(256, min(8192, cs)),
            pad_ratio=0.06,
            model_key=inf["model_key"],
            refine_mask=bool(inf["refine_mask"]),
        )

    def _run(self) -> None:
        inp = self.var_input.get().strip()
        out = self.var_output.get().strip()
        if not inp:
            messagebox.showwarning("Kirish yo‘q", "Rasm yoki papkani tanlang.")
            return

        p_in = Path(inp)
        if self.var_batch.get():
            if not p_in.is_dir():
                messagebox.showerror("Xato", f"Papka topilmadi:\n{p_in}")
                return
            if not out:
                out = str(p_in / "output")
        else:
            if not p_in.is_file():
                messagebox.showerror("Xato", f"Fayl topilmadi:\n{p_in}")
                return
            if not out:
                out = str(p_in.parent / "output")

        p_out = Path(out)
        try:
            out_show = str(p_out.resolve())
        except OSError:
            out_show = str(p_out)
        self.btn_run.configure(state=tk.DISABLED)
        self.btn_open.configure(state=tk.DISABLED)
        self.txt_log.configure(state=tk.NORMAL)
        self.txt_log.delete("1.0", tk.END)
        self.txt_log.configure(state=tk.DISABLED)

        batch = self.var_batch.get()
        kw = self._params()

        def work() -> None:
            try:
                self.after(
                    0,
                    lambda: self._log(
                        f"Natijalar saqlanadi:\n{out_show}\n\n"
                        "Model yuklanmoqda / segmentatsiya (birinchi marta uzoqroq)…"
                    ),
                )
                if batch:
                    result = process_folder(p_in, p_out, **kw)
                else:
                    result = process_image(p_in, p_out, **kw)
                self._last_out = p_out

                def done() -> None:
                    self.btn_run.configure(state=tk.NORMAL)
                    self._log(result.message)
                    if result.ok:
                        for f in result.saved_files[:30]:
                            self._log(f"  • {f.name}")
                        if len(result.saved_files) > 30:
                            self._log(f"  … va yana {len(result.saved_files) - 30} ta fayl")
                        self.btn_open.configure(state=tk.NORMAL)
                        short = (
                            result.message.split("\n")[0]
                            if "\n" in result.message
                            else result.message
                        )
                        messagebox.showinfo("Tayyor", short[:500])
                    else:
                        messagebox.showwarning("Natija", result.message[:800])

                self.after(0, done)
            except Exception as e:

                def err() -> None:
                    self.btn_run.configure(state=tk.NORMAL)
                    self._log(f"Xato: {e}")
                    messagebox.showerror("Xato", str(e))

                self.after(0, err)

        threading.Thread(target=work, daemon=True).start()


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
