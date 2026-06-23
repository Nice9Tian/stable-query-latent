"""PyQt6 launcher and monitor for VICReg review training.

Run from the repository root:

    C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe training.py

Extra arguments after ``--`` are passed to VICReg_review/train_vicreg_review_h5.py,
for example:

    C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe training.py -- --epochs 10 --batch-size 8
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
PYTHON_EXE = Path("C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe")
RUN_DIR = ROOT / "VICReg_review" / "heads" / "gui_run"
TRAIN_SCRIPT = ROOT / "VICReg_review" / "train_vicreg_review_h5.py"
PROBE_SCRIPT = ROOT / "VICReg_review" / "train_tag_probe.py"

TRAIN_LINE_RE = re.compile(
    r"epoch=(?P<epoch>\d+)\s+step=(?P<batch>\d+)/(?P<steps>\d+)\s+"
    r"global=(?P<global>\d+).*?\bloss=(?P<loss>[-+0-9.eE]+)"
)


def import_qt_and_matplotlib():
    try:
        from PyQt6 import QtCore, QtGui, QtWidgets
    except ImportError as exc:
        raise SystemExit(
            "PyQt6 is not installed in this environment. Install PyQt6, then run "
            "this script with the project interpreter."
        ) from exc

    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    return QtCore, QtGui, QtWidgets, FigureCanvas, Figure


QtCore, QtGui, QtWidgets, FigureCanvas, Figure = import_qt_and_matplotlib()


def qprocess_merged_channels():
    return QtCore.QProcess.ProcessChannelMode.MergedChannels


def qprocess_normal_exit():
    return QtCore.QProcess.ExitStatus.NormalExit


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def append_probe_curve(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "encoder_epoch",
        "encoder_global_step",
        "test_loss",
        "mAP",
        "micro_f1",
        "precision",
        "recall",
        "predicted_tags_per_game",
        "count_mae",
        "finished_at",
    ]
    existing_rows = []
    exists = path.exists()
    if exists:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames == fieldnames:
                existing_rows = None
            else:
                existing_rows = list(reader)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    if existing_rows is not None:
        tmp_path = path.with_name(path.name + ".tmp")
        try:
            with tmp_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
                writer.writeheader()
                for existing_row in existing_rows:
                    writer.writerow({key: existing_row.get(key, "") for key in fieldnames})
                writer.writerow({key: row.get(key, "") for key in fieldnames})
            tmp_path.replace(path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise


def unique_numbered_path(path: Path) -> Path:
    """Return path, or path with _1/_2/... before the suffix when it exists."""
    if not path.exists():
        return path
    for index in range(1, 100000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find a free numbered filename for {path}")


def sibling_history_path(checkpoint_path: Path) -> Path | None:
    candidates = [
        checkpoint_path.with_name(checkpoint_path.name.replace("latest", "history")).with_suffix(".tsv"),
        checkpoint_path.with_name("vicreg_review_h5_history.tsv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def sibling_probe_curve_path(checkpoint_path: Path) -> Path | None:
    candidate = checkpoint_path.with_name("tag_probe_curve.tsv")
    return candidate if candidate.exists() else None


class MonitorWindow(QtWidgets.QMainWindow):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.run_dir = Path(args.run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.prepare_run_paths()

        self.current_epoch = 0
        self.current_global_step = 0
        self.last_probe_epoch = 0
        self.attempted_probe_epochs = set()
        self.pending_probe_epoch = 0
        self.last_test_loss = None
        self.last_tag_f1 = None
        self.last_train_mtime = 0.0
        self.last_curve_mtime = 0.0
        self.train_buffer = ""
        self.probe_buffer = ""
        self.closing = False
        self.resume_checkpoint = None

        self.train_process = None
        self.probe_process = None

        self.setWindowTitle("VICReg Review Training")
        self.resize(1120, 760)
        self._build_ui()
        self._build_timer()
        if args.auto_start:
            self.start_training()

    def prepare_run_paths(self) -> None:
        self.checkpoint = unique_numbered_path(self.run_dir / "vicreg_review_h5_latest.pt")
        self.best_checkpoint = unique_numbered_path(self.run_dir / "vicreg_review_h5_best.pt")
        self.history_tsv = unique_numbered_path(self.run_dir / "vicreg_review_h5_history.tsv")
        self.manifest_json = unique_numbered_path(self.run_dir / "vicreg_review_h5_manifest.json")
        self.probe_curve_tsv = unique_numbered_path(self.run_dir / "tag_probe_curve.tsv")

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        grid = QtWidgets.QGridLayout()
        layout.addLayout(grid)

        self.status_label = QtWidgets.QLabel("starting")
        self.epoch_label = QtWidgets.QLabel("0")
        self.batch_label = QtWidgets.QLabel("0/0")
        self.loss_label = QtWidgets.QLabel("-")
        self.test_loss_label = QtWidgets.QLabel("-")
        self.tag_f1_label = QtWidgets.QLabel("-")

        labels = [
            ("Status", self.status_label),
            ("Epoch", self.epoch_label),
            ("Batch", self.batch_label),
            ("Loss", self.loss_label),
            ("Test loss", self.test_loss_label),
            ("Tag正确率(F1)", self.tag_f1_label),
        ]
        for col, (name, widget) in enumerate(labels):
            title = QtWidgets.QLabel(name)
            title.setStyleSheet("font-weight: 600; color: #555;")
            widget.setStyleSheet("font-size: 20px;")
            grid.addWidget(title, 0, col)
            grid.addWidget(widget, 1, col)

        self.figure = Figure(figsize=(8, 4), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.metric_ax = self.ax.twinx()
        layout.addWidget(self.canvas, stretch=1)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1200)
        self.log.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.log.setWordWrapMode(QtGui.QTextOption.WrapMode.WrapAnywhere)
        self.log.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(self.log, stretch=1)

        buttons = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("训练开始")
        self.stop_button = QtWidgets.QPushButton("训练停止")
        self.load_button = QtWidgets.QPushButton("加载")
        self.save_button = QtWidgets.QPushButton("保存checkpoint")
        self.start_button.clicked.connect(self.start_training)
        self.stop_button.clicked.connect(self.stop_processes)
        self.load_button.clicked.connect(self.load_checkpoint_dialog)
        self.save_button.clicked.connect(self.save_checkpoint_now)
        buttons.addStretch(1)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.load_button)
        buttons.addWidget(self.save_button)
        layout.addLayout(buttons)

        self.redraw_plot([], [])
        self.update_buttons()

    def _build_timer(self) -> None:
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1500)
        self.timer.timeout.connect(self.refresh_from_files)
        self.timer.start()

    def log_line(self, text: str) -> None:
        if self.closing:
            return
        self.log.appendPlainText(text.rstrip())
        cursor = self.log.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.log.setTextCursor(cursor)

    def update_buttons(self) -> None:
        if self.closing:
            return
        training = self.process_is_running(self.train_process)
        self.start_button.setEnabled(not training)
        self.stop_button.setEnabled(training or self.process_is_running(self.probe_process))
        self.load_button.setEnabled(not training)
        self.save_button.setEnabled(self.checkpoint.exists() or bool(self.resume_checkpoint and self.resume_checkpoint.exists()))

    def start_training(self) -> None:
        if self.process_is_running(self.train_process):
            self.log_line("[training] already running")
            return
        self.prepare_run_paths()
        resume_checkpoint = self.resume_checkpoint
        if resume_checkpoint:
            metadata = self.read_checkpoint_metadata(resume_checkpoint)
            self.current_epoch = int(metadata.get("epoch") or 0)
            self.current_global_step = int(metadata.get("global_step") or 0)
        else:
            self.current_epoch = 0
            self.current_global_step = 0
        self.last_probe_epoch = 0
        self.attempted_probe_epochs.clear()
        self.pending_probe_epoch = 0
        self.last_test_loss = None
        self.last_tag_f1 = None
        self.last_train_mtime = 0.0
        self.last_curve_mtime = 0.0
        self.epoch_label.setText(str(self.current_epoch))
        self.batch_label.setText("0/0")
        self.loss_label.setText("-")
        self.test_loss_label.setText("-")
        self.tag_f1_label.setText("-")
        self.redraw_plot([], [])

        command = str(Path(self.args.python_exe).resolve())
        train_args = [
            str(TRAIN_SCRIPT),
            "--checkpoint-out",
            str(self.checkpoint),
            "--best-checkpoint-out",
            str(self.best_checkpoint),
            "--history-tsv",
            str(self.history_tsv),
            "--manifest-json",
            str(self.manifest_json),
            "--amp",
        ]
        if resume_checkpoint:
            train_args.extend(["--resume-checkpoint", str(resume_checkpoint)])
        train_args.extend(self.args.train_args)

        self.train_process = QtCore.QProcess(self)
        self.train_process.setWorkingDirectory(str(ROOT))
        self.train_process.setProcessChannelMode(qprocess_merged_channels())
        self.train_process.readyReadStandardOutput.connect(self.on_train_output)
        self.train_process.finished.connect(self.on_train_finished)
        self.status_label.setText("training")
        self.log_line("$ " + " ".join([command, *train_args]))
        self.log_line(f"[checkpoint] latest={self.checkpoint}")
        self.log_line(f"[checkpoint] best={self.best_checkpoint}")
        if resume_checkpoint:
            self.log_line(f"[resume] {resume_checkpoint}")
        self.train_process.start(command, train_args)
        self.update_buttons()

    def read_checkpoint_metadata(self, checkpoint_path: Path) -> dict:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metrics = checkpoint.get("metrics") or {}
        return {
            "epoch": checkpoint.get("epoch"),
            "global_step": checkpoint.get("global_step"),
            "loss": metrics.get("loss"),
        }

    def load_checkpoint_dialog(self) -> None:
        if self.process_is_running(self.train_process):
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "加载 checkpoint",
            str(self.run_dir),
            "PyTorch checkpoints (*.pt);;All files (*)",
        )
        if not path:
            return
        self.load_checkpoint(Path(path))

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint_path = checkpoint_path.resolve()
        try:
            metadata = self.read_checkpoint_metadata(checkpoint_path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "加载失败", f"无法读取 checkpoint:\n{checkpoint_path}\n\n{exc}")
            return
        self.resume_checkpoint = checkpoint_path
        epoch = int(metadata.get("epoch") or 0)
        global_step = int(metadata.get("global_step") or 0)
        loss = metadata.get("loss")
        self.current_epoch = epoch
        self.current_global_step = global_step
        self.epoch_label.setText(str(epoch))
        self.batch_label.setText("loaded")
        self.loss_label.setText(f"{float(loss):.4f}" if loss is not None else "-")
        self.status_label.setText("loaded")
        self.log_line(f"[loaded checkpoint] {checkpoint_path}")
        self.log_line(f"[loaded checkpoint] epoch={epoch} global_step={global_step}")
        history_path = sibling_history_path(checkpoint_path)
        probe_curve_path = sibling_probe_curve_path(checkpoint_path)
        if history_path:
            self.history_tsv = history_path
        if probe_curve_path:
            self.probe_curve_tsv = probe_curve_path
        self.refresh_from_files()
        self.update_buttons()

    def on_train_output(self) -> None:
        if self.closing:
            return
        data = bytes(self.train_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self.train_buffer += data
        while "\n" in self.train_buffer:
            line, self.train_buffer = self.train_buffer.split("\n", 1)
            self.handle_train_line(line)

    def handle_train_line(self, line: str) -> None:
        self.log_line(line)
        match = TRAIN_LINE_RE.search(line)
        if not match:
            return
        epoch = int(match.group("epoch"))
        batch = int(match.group("batch"))
        steps = int(match.group("steps"))
        global_step = int(match.group("global"))
        loss = float(match.group("loss"))
        self.current_epoch = max(self.current_epoch, epoch)
        self.current_global_step = max(self.current_global_step, global_step)
        self.epoch_label.setText(str(epoch))
        self.batch_label.setText(f"{batch}/{steps}")
        self.loss_label.setText(f"{loss:.4f}")

    def on_train_finished(self, exit_code: int, exit_status) -> None:
        if self.closing:
            return
        if self.train_buffer:
            self.handle_train_line(self.train_buffer)
            self.train_buffer = ""
        ok = exit_code == 0 and exit_status == qprocess_normal_exit()
        self.status_label.setText("training done" if ok else f"training exited {exit_code}")
        self.log_line(f"[training exited: code={exit_code}]")
        self.refresh_from_files()
        self.update_buttons()
        if ok and not self.args.no_probe:
            self.maybe_start_probe(force=True)

    def refresh_from_files(self) -> None:
        if self.closing:
            return
        manifest = read_json(self.manifest_json)
        if manifest:
            epoch = int(manifest.get("epoch") or self.current_epoch or 0)
            global_step = int(manifest.get("global_step") or self.current_global_step or 0)
            self.current_epoch = max(self.current_epoch, epoch)
            self.current_global_step = max(self.current_global_step, global_step)
            self.epoch_label.setText(str(self.current_epoch))
            if manifest.get("status"):
                status = manifest["status"]
                if self.process_is_running(self.train_process):
                    status = "training"
                self.status_label.setText(status)

        train_rows = read_tsv(self.history_tsv)
        curve_rows = read_tsv(self.probe_curve_tsv)
        if train_rows:
            last = train_rows[-1]
            if last.get("loss"):
                self.loss_label.setText(f"{float(last['loss']):.4f}")
        if curve_rows:
            last = curve_rows[-1]
            if last.get("test_loss"):
                self.last_test_loss = float(last["test_loss"])
                self.test_loss_label.setText(f"{self.last_test_loss:.4f}")
            if last.get("micro_f1"):
                self.last_tag_f1 = float(last["micro_f1"])
                self.tag_f1_label.setText(f"{self.last_tag_f1:.4f}")

        train_mtime = self.history_tsv.stat().st_mtime if self.history_tsv.exists() else 0.0
        curve_mtime = self.probe_curve_tsv.stat().st_mtime if self.probe_curve_tsv.exists() else 0.0
        if train_mtime != self.last_train_mtime or curve_mtime != self.last_curve_mtime:
            self.last_train_mtime = train_mtime
            self.last_curve_mtime = curve_mtime
            self.redraw_plot(train_rows, curve_rows)

        self.maybe_start_probe()
        self.update_buttons()

    def redraw_plot(self, train_rows: list[dict[str, str]], curve_rows: list[dict[str, str]]) -> None:
        if self.closing:
            return
        self.ax.clear()
        self.metric_ax.clear()
        self.ax.set_title("VICReg training loss and tag prediction F1")
        self.ax.set_xlabel("Encoder epoch")
        self.ax.set_ylabel("Train loss")
        self.ax.grid(True, alpha=0.25)
        self.metric_ax.set_ylabel("Tag accuracy (micro-F1)")
        self.metric_ax.set_ylim(0.0, 1.0)

        if train_rows:
            x_train = [int(row["epoch"]) for row in train_rows if row.get("epoch") and row.get("loss")]
            y_train = [float(row["loss"]) for row in train_rows if row.get("epoch") and row.get("loss")]
            if x_train:
                self.ax.plot(x_train, y_train, marker="o", linewidth=1.8, label="train loss", color="#1f77b4")

        if curve_rows:
            x_test = [
                int(row["encoder_epoch"])
                for row in curve_rows
                if row.get("encoder_epoch") and row.get("micro_f1")
            ]
            y_test = [
                float(row["micro_f1"])
                for row in curve_rows
                if row.get("encoder_epoch") and row.get("micro_f1")
            ]
            if x_test:
                self.metric_ax.plot(x_test, y_test, marker="s", linewidth=1.8, label="tag accuracy (micro-F1)", color="#2ca02c")

        lines = self.ax.get_lines() + self.metric_ax.get_lines()
        if lines:
            self.ax.legend(lines, [line.get_label() for line in lines], loc="best")
        self.canvas.draw_idle()

    def process_is_running(self, process) -> bool:
        if process is None:
            return False
        return process.state() != QtCore.QProcess.ProcessState.NotRunning

    def maybe_start_probe(self, force: bool = False) -> None:
        if self.args.no_probe or self.process_is_running(self.probe_process):
            return
        if not force and not self.process_is_running(self.train_process):
            return
        if not self.checkpoint.exists() or self.current_epoch <= 0:
            return
        due = self.current_epoch >= self.last_probe_epoch + self.args.probe_every_epochs
        if not force and not due:
            return
        if self.current_epoch == self.last_probe_epoch:
            return
        if self.current_epoch in self.attempted_probe_epochs:
            return
        self.start_probe(self.current_epoch, self.current_global_step)

    def start_probe(self, encoder_epoch: int, encoder_global_step: int) -> None:
        self.pending_probe_epoch = encoder_epoch
        self.attempted_probe_epochs.add(encoder_epoch)
        command = str(Path(self.args.python_exe).resolve())
        stem = f"epoch_{encoder_epoch:04d}"
        checkpoint_snapshot = unique_numbered_path(self.run_dir / f"probe_checkpoint_{stem}.pt")
        shutil.copy2(self.checkpoint, checkpoint_snapshot)
        report_json = self.run_dir / f"tag_probe_report_{stem}.json"
        history_tsv = self.run_dir / f"tag_probe_history_{stem}.tsv"
        head_out = self.run_dir / f"tag_probe_head_{stem}.pt"
        probe_args = [
            str(PROBE_SCRIPT),
            "--checkpoint",
            str(checkpoint_snapshot),
            "--report-json",
            str(report_json),
            "--history-tsv",
            str(history_tsv),
            "--head-out",
            str(head_out),
            "--epochs",
            str(self.args.probe_epochs),
            "--patience",
            str(self.args.probe_patience),
            "--feature-views",
            str(self.args.probe_feature_views),
            "--log-every",
            str(self.args.probe_log_every),
        ]
        if self.args.probe_device:
            probe_args.extend(["--device", self.args.probe_device])
        if self.args.probe_amp:
            probe_args.append("--amp")

        self.probe_process = QtCore.QProcess(self)
        self.probe_process.setWorkingDirectory(str(ROOT))
        self.probe_process.setProcessChannelMode(qprocess_merged_channels())
        self.probe_process.readyReadStandardOutput.connect(self.on_probe_output)
        self.probe_process.finished.connect(
            lambda exit_code, exit_status, path=report_json, epoch=encoder_epoch, step=encoder_global_step: self.on_probe_finished(
                exit_code, exit_status, path, epoch, step
            )
        )
        self.status_label.setText(f"probing epoch {encoder_epoch}")
        self.log_line(f"[tag probe] checkpoint snapshot={checkpoint_snapshot}")
        self.log_line("[tag probe] $ " + " ".join([command, *probe_args]))
        self.probe_process.start(command, probe_args)
        self.update_buttons()

    def on_probe_output(self) -> None:
        if self.closing:
            return
        data = bytes(self.probe_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self.probe_buffer += data
        while "\n" in self.probe_buffer:
            line, self.probe_buffer = self.probe_buffer.split("\n", 1)
            self.log_line("[tag probe] " + line)

    def on_probe_finished(self, exit_code: int, exit_status, report_json: Path, encoder_epoch: int, encoder_global_step: int) -> None:
        if self.closing:
            return
        if self.probe_buffer:
            self.log_line("[tag probe] " + self.probe_buffer)
            self.probe_buffer = ""
        ok = exit_code == 0 and exit_status == qprocess_normal_exit()
        self.log_line(f"[tag probe exited: code={exit_code}]")
        if not ok:
            self.status_label.setText("probe failed")
            return

        report = read_json(report_json)
        best = report.get("best_metrics") or {}
        final = report.get("final_val_metrics") or {}
        metrics = final or best
        test_loss = best.get("val_loss")
        if test_loss is None:
            test_loss = final.get("val_loss")
        row = {
            "encoder_epoch": int(report.get("encoder_epoch") or encoder_epoch),
            "encoder_global_step": int(report.get("encoder_global_step") or encoder_global_step),
            "test_loss": float(test_loss) if test_loss is not None else "",
            "mAP": metrics.get("mAP", ""),
            "micro_f1": metrics.get("micro_f1", ""),
            "precision": metrics.get("precision", ""),
            "recall": metrics.get("recall", ""),
            "predicted_tags_per_game": metrics.get("predicted_tags_per_game", ""),
            "count_mae": metrics.get("count_mae", ""),
            "finished_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        }
        append_probe_curve(self.probe_curve_tsv, row)
        self.last_probe_epoch = encoder_epoch
        if row["test_loss"] != "":
            self.last_test_loss = float(row["test_loss"])
            self.test_loss_label.setText(f"{self.last_test_loss:.4f}")
        if row["micro_f1"] != "":
            self.last_tag_f1 = float(row["micro_f1"])
            self.tag_f1_label.setText(f"{self.last_tag_f1:.4f}")
        self.status_label.setText("training" if self.process_is_running(self.train_process) else "done")
        self.refresh_from_files()
        self.update_buttons()

    def stop_processes(self) -> None:
        for process in (self.probe_process, self.train_process):
            if self.process_is_running(process):
                process.terminate()
        if not self.closing:
            self.status_label.setText("stopping")
        self.update_buttons()

    def disconnect_process(self, process) -> None:
        if process is None:
            return
        for signal in (process.readyReadStandardOutput, process.finished):
            try:
                signal.disconnect()
            except TypeError:
                pass

    def shutdown_processes_for_close(self) -> None:
        self.closing = True
        if hasattr(self, "timer"):
            self.timer.stop()
        for process in (self.probe_process, self.train_process):
            self.disconnect_process(process)
            if self.process_is_running(process):
                process.terminate()

    def save_checkpoint_now(self) -> None:
        if self.closing:
            return
        source = self.checkpoint if self.checkpoint.exists() else self.resume_checkpoint
        if not source or not source.exists():
            self.log_line("[checkpoint] no checkpoint file is available yet")
            self.update_buttons()
            return
        target = unique_numbered_path(self.run_dir / "manual_checkpoint.pt")
        shutil.copy2(source, target)
        self.log_line(f"[checkpoint saved] {target}")
        self.update_buttons()

    def closeEvent(self, event) -> None:
        self.shutdown_processes_for_close()
        event.accept()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch VICReg review training with a PyQt progress window.")
    parser.add_argument("--python-exe", default=str(PYTHON_EXE))
    parser.add_argument("--run-dir", default=str(RUN_DIR))
    parser.add_argument("--auto-start", action="store_true", help="Start training immediately when the window opens.")
    parser.add_argument("--no-probe", action="store_true", help="Only show VICReg training progress.")
    parser.add_argument("--probe-every-epochs", type=int, default=5)
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--probe-patience", type=int, default=20)
    parser.add_argument("--probe-feature-views", type=int, default=8)
    parser.add_argument("--probe-log-every", type=int, default=25)
    parser.add_argument("--probe-device", default=None)
    parser.add_argument("--probe-amp", action="store_true")
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.train_args and args.train_args[0] == "--":
        args.train_args = args.train_args[1:]
    args.probe_every_epochs = max(1, args.probe_every_epochs)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    app = QtWidgets.QApplication(sys.argv)
    window = MonitorWindow(args)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
