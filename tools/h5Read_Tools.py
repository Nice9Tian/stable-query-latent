import sys
from pathlib import Path

import h5py
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTreeWidget, QTreeWidgetItem, QSplitter, 
                             QFileDialog, QPushButton, QMessageBox, QSlider, 
                             QSpinBox, QLabel, QComboBox, QStackedWidget, QTextEdit,
                             QScrollArea)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


class ShardedDataset:
    def __init__(self, name, datasets, kind="concat"):
        if not datasets:
            raise ValueError(f"{name}: no shard datasets provided")
        self.name = name
        self.datasets = datasets
        self.kind = kind
        self.dtype = datasets[0].dtype
        self.attrs = {
            "virtual": True,
            "source": "H5 shard collection",
            "shards": len(datasets),
            "layout": kind,
        }
        self._offset_cache = None

        if kind in {"review_offsets", "game_review_offsets"}:
            self.shape = (1 + sum(int(dataset.shape[0]) - 1 for dataset in datasets),)
        else:
            tail_shape = datasets[0].shape[1:]
            self.shape = (sum(int(dataset.shape[0]) for dataset in datasets),) + tail_shape
        self.ndim = len(self.shape)

    def __len__(self):
        return self.shape[0]

    def _build_offset_cache(self):
        if self._offset_cache is not None:
            return self._offset_cache

        parts = []
        cursor = 0
        for index, dataset in enumerate(self.datasets):
            values = np.asarray(dataset[:], dtype=self.dtype)
            if index > 0:
                values = values[1:]
            parts.append(values + cursor)
            cursor += int(dataset[-1])

        self._offset_cache = np.concatenate(parts, axis=0)
        return self._offset_cache

    def __getitem__(self, key):
        if self.kind in {"review_offsets", "game_review_offsets"}:
            return self._build_offset_cache()[key]

        if key is Ellipsis:
            key = slice(None)

        if isinstance(key, tuple):
            if not key:
                first_key = slice(None)
                rest_key = ()
            else:
                first_key = key[0]
                rest_key = key[1:]
        else:
            first_key = key
            rest_key = ()

        if first_key is Ellipsis:
            first_key = slice(None)

        if isinstance(first_key, (int, np.integer)):
            return self._read_index(int(first_key), rest_key)
        if isinstance(first_key, slice):
            return self._read_slice(first_key, rest_key)

        indices = np.asarray(first_key)
        return self._read_indices(indices.tolist(), rest_key)

    def _normalize_index(self, index):
        if index < 0:
            index += self.shape[0]
        if index < 0 or index >= self.shape[0]:
            raise IndexError("sharded dataset index out of range")
        return index

    def _read_index(self, index, rest_key):
        index = self._normalize_index(index)
        cursor = 0
        for dataset in self.datasets:
            shard_len = int(dataset.shape[0])
            if index < cursor + shard_len:
                return dataset[(index - cursor,) + rest_key]
            cursor += shard_len
        raise IndexError("sharded dataset index out of range")

    def _read_slice(self, row_slice, rest_key):
        start, stop, step = row_slice.indices(self.shape[0])
        if step != 1:
            return self._read_indices(range(start, stop, step), rest_key)

        pieces = []
        cursor = 0
        for dataset in self.datasets:
            shard_len = int(dataset.shape[0])
            shard_start = max(start - cursor, 0)
            shard_stop = min(stop - cursor, shard_len)
            if shard_start < shard_stop:
                pieces.append(dataset[(slice(shard_start, shard_stop),) + rest_key])
            cursor += shard_len
            if cursor >= stop:
                break

        if pieces:
            return np.concatenate(pieces, axis=0)
        return np.empty((0,) + self.shape[1:], dtype=self.dtype)

    def _read_indices(self, indices, rest_key):
        values = [self._read_index(int(index), rest_key) for index in indices]
        if not values:
            return np.empty((0,) + self.shape[1:], dtype=self.dtype)
        if np.asarray(values[0]).shape == ():
            return np.asarray(values, dtype=self.dtype)
        return np.stack(values, axis=0)


class ShardsGroup:
    def __init__(self, reader):
        self.reader = reader
        self.attrs = {
            "virtual": True,
            "source": "Opened H5 shard files",
            "shards": len(reader.files),
        }

    def keys(self):
        return list(self.reader.file_map.keys())

    def __getitem__(self, key):
        return self.reader.file_map[key]


class H5ShardCollection:
    def __init__(self, shard_paths):
        self.shard_paths = [Path(path) for path in shard_paths]
        if not self.shard_paths:
            raise ValueError("No shard H5 files selected")

        self.files = []
        try:
            self.files = [h5py.File(path, "r") for path in self.shard_paths]
            self.file_map = dict(zip(self._unique_names(self.shard_paths), self.files))
            self._datasets = self._build_virtual_datasets()
            self._shards_group = ShardsGroup(self)
            self.attrs = self._build_attrs()
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _unique_names(paths):
        counts = {}
        names = []
        for path in paths:
            name = path.name
            seen = counts.get(name, 0)
            counts[name] = seen + 1
            if seen:
                name = f"{path.stem}_{seen}{path.suffix}"
            names.append(name)
        return names

    def _build_virtual_datasets(self):
        first_file = self.files[0]
        virtual = {}
        for key in first_file.keys():
            if not all(key in file and isinstance(file[key], h5py.Dataset) for file in self.files):
                continue

            datasets = [file[key] for file in self.files]
            if not datasets[0].shape:
                continue
            if not all(dataset.shape[1:] == datasets[0].shape[1:] for dataset in datasets):
                continue
            if not all(dataset.dtype == datasets[0].dtype for dataset in datasets):
                continue

            if key == "review_offsets":
                virtual[key] = ShardedDataset(f"/{key}", datasets, kind="review_offsets")
            elif key == "game_review_offsets":
                virtual[key] = ShardedDataset(f"/{key}", datasets, kind="game_review_offsets")
            else:
                virtual[key] = ShardedDataset(f"/{key}", datasets, kind="concat")
        return virtual

    def _build_attrs(self):
        attrs = {
            "virtual": True,
            "source": "H5 shard collection",
            "shards": len(self.files),
            "paths": "\n".join(str(path) for path in self.shard_paths),
        }

        for key in ("input_dim", "dtype", "source"):
            values = [file.attrs[key] for file in self.files if key in file.attrs]
            if len(values) == len(self.files) and all(str(value) == str(values[0]) for value in values):
                attrs[key] = values[0]

        if "game_names" in self._datasets:
            attrs["games"] = self._datasets["game_names"].shape[0]
        if "review_offsets" in self._datasets:
            attrs["reviews"] = self._datasets["review_offsets"].shape[0] - 1
        if "vectors" in self._datasets:
            attrs["sentences"] = self._datasets["vectors"].shape[0]
        return attrs

    def keys(self):
        return list(self._datasets.keys()) + ["shards"]

    def __getitem__(self, path):
        path = str(path)
        if path in {"", "/"}:
            return self

        parts = [part for part in path.strip("/").split("/") if part]
        if not parts:
            return self

        if parts[0] == "shards":
            if len(parts) == 1:
                return self._shards_group
            shard_file = self.file_map[parts[1]]
            if len(parts) == 2:
                return shard_file
            return shard_file["/" + "/".join(parts[2:])]

        if len(parts) == 1 and parts[0] in self._datasets:
            return self._datasets[parts[0]]

        raise KeyError(path)

    def close(self):
        for file in self.files:
            try:
                file.close()
            except Exception:
                pass
        self.files = []


def is_group_node(node):
    return isinstance(node, h5py.Group) or isinstance(node, (H5ShardCollection, ShardsGroup))


def is_dataset_node(node):
    return isinstance(node, h5py.Dataset) or isinstance(node, ShardedDataset)


class H5Viewer(QMainWindow):
    def __init__(self, initial_file_path=None, initial_dataset_path=None):
        super().__init__()
        self.setWindowTitle("轻量化 H5 数据集查看器 (支持多模式、色彩空间与属性查看)")
        self.resize(1200, 800)
        self.current_h5_file = None
        self.current_dataset = None
        self.current_data_slice = None  
        self.current_dataset_row_mode = False
        
        self.initial_file_path = initial_file_path
        self.initial_dataset_path = initial_dataset_path
        
        self.init_ui()
        
        if self.initial_file_path:
            self.load_file(self.initial_file_path)
            if self.initial_dataset_path:
                self.display_dataset_by_path(self.initial_dataset_path)

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        open_layout = QHBoxLayout()
        self.btn_open = QPushButton("打开 .h5 文件")
        self.btn_open.clicked.connect(self.open_file)
        self.btn_open_shards = QPushButton("打开 shard 文件组")
        self.btn_open_shards.clicked.connect(self.open_shard_files)
        self.btn_open_shard_dir = QPushButton("打开 shards 目录")
        self.btn_open_shard_dir.clicked.connect(self.open_shard_dir)
        open_layout.addWidget(self.btn_open)
        open_layout.addWidget(self.btn_open_shards)
        open_layout.addWidget(self.btn_open_shard_dir)
        open_layout.addStretch()
        main_layout.addLayout(open_layout)
        
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # ================= 左侧：树状结构 =================
        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("HDF5 内部结构")
        self.tree.itemClicked.connect(self.on_item_clicked)
        main_splitter.addWidget(self.tree)
        
        # ================= 右侧：多功能显示区 =================
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        
        upper_widget = QWidget()
        upper_layout = QVBoxLayout(upper_widget)
        upper_layout.setContentsMargins(0, 0, 0, 0)
        
        # --- 1. 顶部选项栏 ---
        top_controls_layout = QHBoxLayout()
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["图像模式", "文本模式"])
        self.mode_combo.currentIndexChanged.connect(self.on_view_changed)
        
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["magma", "gray", "viridis", "plasma", "jet", "bone"])
        self.cmap_combo.currentIndexChanged.connect(self.on_view_changed)
        
        top_controls_layout.addWidget(QLabel("显示模式:"))
        top_controls_layout.addWidget(self.mode_combo)
        top_controls_layout.addWidget(QLabel(" 色彩空间/映射:"))
        top_controls_layout.addWidget(self.cmap_combo)
        top_controls_layout.addStretch() 
        
        upper_layout.addLayout(top_controls_layout)
        
        # --- 2. 中间：层叠显示区 (QStackedWidget) ---
        self.view_stack = QStackedWidget()
        
        # 图像画布滚动区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        # 【关键修改】设定滚动区域内容居中对齐，解决窄长图贴边问题
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter) 
        
        self.fig, self.ax = plt.subplots()
        # 【关键修改】消除 Matplotlib 默认的所有边距和白色背景
        self.fig.patch.set_facecolor('none')  # 画布背景设为透明
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0) # 剔除外部边距
        self.ax.margins(0) # 剔除内部边距
        
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color: transparent;") # Qt 控件层透明
        self.ax.axis('off')
        
        self.scroll_area.setWidget(self.canvas)
        self.view_stack.addWidget(self.scroll_area)
        
        # 文本框
        self.text_view = QTextEdit()
        self.text_view.setReadOnly(True)
        self.text_view.setFont(QFont("Courier New", 10)) 
        self.view_stack.addWidget(self.text_view)
        
        upper_layout.addWidget(self.view_stack, stretch=1)
        
        # --- 3. 底部导航栏 ---
        nav_layout = QHBoxLayout()
        self.lbl_index = QLabel("当前索引: 0 / 0")
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.spinbox = QSpinBox()
        self.spinbox.setEnabled(False)
        
        self.slider.valueChanged.connect(self.on_index_changed)
        self.spinbox.valueChanged.connect(self.slider.setValue)
        
        nav_layout.addWidget(self.lbl_index)
        nav_layout.addWidget(self.slider)
        nav_layout.addWidget(self.spinbox)
        
        upper_layout.addLayout(nav_layout)
        
        # --- 4. 底部属性面板 (Attributes) ---
        self.attr_view = QTextEdit()
        self.attr_view.setReadOnly(True)
        self.attr_view.setPlaceholderText("选择左侧节点以查看其 Attributes (属性信息)...")
        self.attr_view.setFont(QFont("Courier New", 10))
        
        right_splitter.addWidget(upper_widget)
        right_splitter.addWidget(self.attr_view)
        right_splitter.setSizes([600, 150]) 
        
        main_splitter.addWidget(right_splitter)
        main_splitter.setSizes([250, 950])

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择 H5 文件", "", "HDF5 Files (*.h5 *.hdf5)")
        if not file_path:
            return
        self.load_file(file_path)

    def open_shard_files(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "选择 H5 shard 文件", "", "HDF5 Files (*.h5 *.hdf5)")
        if not file_paths:
            return
        self.load_shards(file_paths)

    def open_shard_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "选择 shards 目录")
        if not directory:
            return
        shard_paths = sorted([*Path(directory).glob("*.h5"), *Path(directory).glob("*.hdf5")])
        if not shard_paths:
            QMessageBox.warning(self, "提示", f"目录中没有 .h5 shard 文件:\n{directory}")
            return
        self.load_shards(shard_paths, label=Path(directory).name)
    
    def load_file(self, file_path):
        if self.current_h5_file is not None:
            self.current_h5_file.close()
            
        try:
            self.current_h5_file = h5py.File(file_path, 'r')
            self.reset_view()
            
            root_item = QTreeWidgetItem(self.tree, [Path(file_path).name])
            root_item.setData(0, Qt.ItemDataRole.UserRole, "/") 
            self.populate_tree(self.current_h5_file, root_item, "/")
            self.tree.expandAll()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法打开文件:\n{str(e)}")

    def load_shards(self, shard_paths, label=None):
        if self.current_h5_file is not None:
            self.current_h5_file.close()

        try:
            self.current_h5_file = H5ShardCollection(shard_paths)
            self.reset_view()

            root_label = label or f"{Path(shard_paths[0]).parent.name} ({len(shard_paths)} shards)"
            root_item = QTreeWidgetItem(self.tree, [root_label])
            root_item.setData(0, Qt.ItemDataRole.UserRole, "/")
            self.populate_tree(self.current_h5_file, root_item, "/")
            self.tree.expandAll()
            self.display_attributes(self.current_h5_file)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法打开 shards:\n{str(e)}")

    def reset_view(self):
        self.tree.clear()
        self.current_dataset = None
        self.current_data_slice = None
        self.current_dataset_row_mode = False
        self.ax.clear()
        self.ax.axis('off')
        self.canvas.draw()
        self.text_view.clear()
        self.attr_view.clear()
        self.disable_navigation()

    def populate_tree(self, group, parent_item, path):
        for key in group.keys():
            current_path = f"{path}/{key}" if path != "/" else f"/{key}"
            item = QTreeWidgetItem(parent_item, [key])
            item.setData(0, Qt.ItemDataRole.UserRole, current_path)
            node = group[key]
            
            if is_group_node(node):
                self.populate_tree(node, item, current_path)
            elif is_dataset_node(node):
                shape_str = str(node.shape)
                dtype_str = str(node.dtype)
                item.setText(0, f"{key}  {shape_str} [{dtype_str}]")

    def on_item_clicked(self, item, column):
        node_path = item.data(0, Qt.ItemDataRole.UserRole)
        if not node_path or not self.current_h5_file:
            return
            
        try:
            node = self.current_h5_file[node_path]
            self.display_attributes(node)
            if is_dataset_node(node):
                self.display_dataset_by_path(node_path)
        except Exception as e:
            QMessageBox.warning(self, "错误", f"读取节点失败:\n{str(e)}")

    def display_attributes(self, node):
        attrs = node.attrs
        if not attrs:
            self.attr_view.setText("【无属性信息 (No Attributes)】")
            return
            
        attr_text = "【Attributes / 属性信息】\n" + "-"*40 + "\n"
        for key, value in attrs.items():
            attr_text += f"{key} : {value}\n"
        self.attr_view.setText(attr_text)

    def display_dataset_by_path(self, dataset_path):
        if not self.current_h5_file:
            return
            
        try:
            dataset = self.current_h5_file[dataset_path]
            self.current_dataset = dataset
            self.current_dataset_row_mode = False
            
            if len(dataset.shape) == 3:
                num_samples = dataset.shape[0]
                self.setup_navigation(max_idx=num_samples - 1)
                self.extract_and_render(0)
            elif len(dataset.shape) == 2 and self.should_use_row_navigation(dataset):
                self.current_dataset_row_mode = True
                self.setup_navigation(max_idx=dataset.shape[0] - 1)
                self.extract_and_render(0)
            elif len(dataset.shape) == 2 or len(dataset.shape) == 1:
                self.disable_navigation()
                self.extract_and_render(None)
            else:
                QMessageBox.warning(self, "提示", f"数据集维度为 {dataset.shape}，暂不支持该维度的数据可视化。")
                
            self._select_tree_item_by_path(dataset_path)
            
        except Exception as e:
            QMessageBox.warning(self, "错误", f"无法显示数据集:\n{str(e)}")

    def setup_navigation(self, max_idx):
        self.slider.blockSignals(True)
        self.spinbox.blockSignals(True)
        self.slider.setEnabled(True)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max_idx)
        self.slider.setValue(0)
        self.spinbox.setEnabled(True)
        self.spinbox.setMinimum(0)
        self.spinbox.setMaximum(max_idx)
        self.spinbox.setValue(0)
        self.lbl_index.setText(f"当前索引: 0 / {max_idx}")
        self.slider.blockSignals(False)
        self.spinbox.blockSignals(False)

    def disable_navigation(self):
        self.slider.setEnabled(False)
        self.spinbox.setEnabled(False)
        self.lbl_index.setText("当前索引: - / -")

    def should_use_row_navigation(self, dataset):
        if isinstance(dataset, ShardedDataset):
            return dataset.shape[0] > 1
        return int(np.prod(dataset.shape)) > 1_000_000 and dataset.shape[0] > 1

    def on_index_changed(self, value):
        if self.current_dataset is not None:
            max_idx = self.slider.maximum()
            self.lbl_index.setText(f"当前索引: {value} / {max_idx}")
            if self.spinbox.value() != value:
                self.spinbox.blockSignals(True)
                self.spinbox.setValue(value)
                self.spinbox.blockSignals(False)
            self.extract_and_render(value)

    def extract_and_render(self, index):
        if self.current_dataset is None:
            return
        if index is not None and len(self.current_dataset.shape) == 3:
            self.current_data_slice = self.current_dataset[index, :, :]
        elif index is not None and self.current_dataset_row_mode and len(self.current_dataset.shape) == 2:
            self.current_data_slice = self.current_dataset[index, :]
        else:
            self.current_data_slice = self.current_dataset[:]
        self.on_view_changed()

    def center_scrollbars(self):
        """将滚动条强制设定为正中间"""
        h_bar = self.scroll_area.horizontalScrollBar()
        v_bar = self.scroll_area.verticalScrollBar()
        h_bar.setValue((h_bar.minimum() + h_bar.maximum()) // 2)
        v_bar.setValue((v_bar.minimum() + v_bar.maximum()) // 2)

    def on_view_changed(self):
        if self.current_data_slice is None:
            return
            
        is_text_mode = (self.mode_combo.currentText() == "文本模式")
        
        # 渲染前先重置 Canvas 的尺寸限制
        self.canvas.setMinimumSize(0, 0)
        self.canvas.setMaximumSize(16777215, 16777215) # PyQt中的最大尺寸常量
        
        if is_text_mode:
            self.view_stack.setCurrentIndex(1)
            self.cmap_combo.setEnabled(False) 
            
            array_str = np.array2string(self.current_data_slice, threshold=1000, edgeitems=10, precision=4)
            info_header = f"Data Shape: {self.current_data_slice.shape}\nData Type: {self.current_data_slice.dtype}\n\n"
            self.text_view.setText(info_header + array_str)
            
        else:
            self.view_stack.setCurrentIndex(0)
            self.cmap_combo.setEnabled(True)
            
            if len(self.current_data_slice.shape) == 2:
                selected_cmap = self.cmap_combo.currentText()
                self.ax.clear()
                
                h, w = self.current_data_slice.shape
                ratio = w / h if h != 0 else 1
                
                is_extreme_ratio = (ratio > 10 or ratio < 0.1)
                
                if is_extreme_ratio:
                    viewport_w = self.scroll_area.viewport().width()
                    viewport_h = self.scroll_area.viewport().height()
                    
                    if viewport_w < 100: viewport_w = 600
                    if viewport_h < 100: viewport_h = 400
                    
                    if ratio > 1: # 极端宽图
                        target_h = viewport_h - 20 
                        target_w = int(target_h * ratio)
                    else: # 极端高图
                        target_w = viewport_w - 20
                        target_h = int(target_w / ratio)
                        
                    # 【关键修改】同时锁定最小值和最大值，彻底阻止 Matplotlib 自动填充白边
                    self.canvas.setMinimumSize(target_w, target_h)
                    self.canvas.setMaximumSize(target_w, target_h)
                
                self.ax.imshow(self.current_data_slice, cmap=selected_cmap, aspect='equal', interpolation='nearest')
                self.ax.axis('off')
                self.canvas.draw()
                
                # 如果是极端长图，在 UI 刷新后延迟 50 毫秒将滚动条拉到中间
                if is_extreme_ratio:
                    QTimer.singleShot(50, self.center_scrollbars)
            else:
                self.ax.clear()
                self.ax.text(0.5, 0.5, "1D 数据无法作为图像渲染\n请切换至'文本模式'", 
                             horizontalalignment='center', verticalalignment='center', wrap=True, color='white')
                self.ax.axis('off')
                self.canvas.draw()

    def _select_tree_item_by_path(self, dataset_path):
        def find_item(parent, path_parts):
            if not path_parts:
                return None
            for i in range(parent.childCount()):
                child = parent.child(i)
                child_path = child.data(0, Qt.ItemDataRole.UserRole)
                if child_path == dataset_path:
                    return child
                if child.childCount() > 0:
                    result = find_item(child, path_parts)
                    if result:
                        return result
            return None
        
        root = self.tree.invisibleRootItem()
        target_item = find_item(root, dataset_path.split('/'))
        if target_item:
            self.tree.setCurrentItem(target_item)
            self.tree.scrollToItem(target_item)

    def closeEvent(self, event):
        if self.current_h5_file is not None:
            self.current_h5_file.close()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = H5Viewer(initial_dataset_path='/sample')
    viewer.show()
    sys.exit(app.exec())
