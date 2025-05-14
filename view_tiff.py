#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import rasterio
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class TiffViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("MODIS LST TIF Viewer - Vietnam")
        self.root.geometry("1200x800")
        
        # Định nghĩa các đường dẫn thư mục
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.processed_dir = os.path.join(self.base_dir, "MODIS_LST_VN_PROCESSED")
        
        # Biến cho file 1
        self.tif_path1 = None
        self.current_data1 = None
        self.current_transform1 = None
        self.original_data1 = None
        
        # Biến cho file 2
        self.tif_path2 = None
        self.current_data2 = None
        self.current_transform2 = None
        self.original_data2 = None
        
        # Tạo frame chính
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Layout chính
        self.left_frame = ttk.Frame(self.main_frame, width=200)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Frame điều khiển (bên trái)
        self.create_left_panel()
        
        # Frame hiển thị kết quả (bên phải)
        self.display_frame = ttk.Frame(self.right_frame)
        self.display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Tạo layout cho 2 hình ảnh
        self.display_frame1 = ttk.Frame(self.display_frame)
        self.display_frame1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.display_frame2 = ttk.Frame(self.display_frame)
        self.display_frame2.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Figure cho matplotlib - file 1
        self.fig1 = plt.Figure(figsize=(5, 5), dpi=100)
        self.ax1 = self.fig1.add_subplot(111)
        
        # Canvas FigureCanvasTkAgg - file 1
        self.canvas1 = FigureCanvasTkAgg(self.fig1, self.display_frame1)
        self.canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Figure cho matplotlib - file 2
        self.fig2 = plt.Figure(figsize=(5, 5), dpi=100)
        self.ax2 = self.fig2.add_subplot(111)
        
        # Canvas FigureCanvasTkAgg - file 2
        self.canvas2 = FigureCanvasTkAgg(self.fig2, self.display_frame2)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_left_panel(self):
        # Frame cho File 1
        file1_frame = ttk.LabelFrame(self.left_frame, text="File 1", padding=(5, 5))
        file1_frame.pack(fill=tk.X, expand=False, pady=(0, 10))
        
        # Nút chọn file 1
        ttk.Button(file1_frame, text="Chọn file TIF", 
                  command=lambda: self.select_tif_file(1)).pack(fill=tk.X, pady=2)
        
        # Nút chọn từ thư mục đã xử lý cho file 1
        ttk.Button(file1_frame, text="Chọn từ thư mục đã xử lý", 
                  command=lambda: self.select_from_processed(1)).pack(fill=tk.X, pady=2)
        
        # Label để hiển thị đường dẫn file 1
        self.file_label1 = ttk.Label(file1_frame, text="Chưa chọn file", wraplength=180)
        self.file_label1.pack(pady=2)
        
        # Thông tin file 1
        self.info_label1 = ttk.Label(file1_frame, text="", wraplength=180)
        self.info_label1.pack(fill=tk.X, pady=2)
        
        # Frame cho File 2
        file2_frame = ttk.LabelFrame(self.left_frame, text="File 2", padding=(5, 5))
        file2_frame.pack(fill=tk.X, expand=False, pady=(0, 10))
        
        # Nút chọn file 2
        ttk.Button(file2_frame, text="Chọn file TIF", 
                  command=lambda: self.select_tif_file(2)).pack(fill=tk.X, pady=2)
        
        # Nút chọn từ thư mục đã xử lý cho file 2
        ttk.Button(file2_frame, text="Chọn từ thư mục đã xử lý", 
                  command=lambda: self.select_from_processed(2)).pack(fill=tk.X, pady=2)
        
        # Label để hiển thị đường dẫn file 2
        self.file_label2 = ttk.Label(file2_frame, text="Chưa chọn file", wraplength=180)
        self.file_label2.pack(pady=2)
        
        # Thông tin file 2
        self.info_label2 = ttk.Label(file2_frame, text="", wraplength=180)
        self.info_label2.pack(fill=tk.X, pady=2)
        
        # Common controls frame
        common_frame = ttk.LabelFrame(self.left_frame, text="Cài đặt chung", padding=(5, 5))
        common_frame.pack(fill=tk.X, expand=False, pady=(0, 10))
        
        # Min/Max value sliders
        ttk.Label(common_frame, text="Phạm vi nhiệt độ:").pack(anchor=tk.W, pady=(5, 2))
        
        # Min temperature controls (slider and entry)
        min_frame = ttk.Frame(common_frame)
        min_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(min_frame, text="Min:").grid(row=0, column=0, sticky=tk.W)
        
        # Min slider
        self.min_var = tk.DoubleVar(value=-5)
        self.min_scale = ttk.Scale(min_frame, from_=-50, to=50, 
                                  variable=self.min_var, orient=tk.HORIZONTAL, 
                                  command=self.on_min_scale_change)
        self.min_scale.grid(row=0, column=1, sticky="ew", padx=5)
        
        # Min entry
        validate_cmd = self.root.register(self.validate_float)
        self.min_entry = ttk.Entry(min_frame, width=8, validate="key", 
                               validatecommand=(validate_cmd, '%P'))
        self.min_entry.grid(row=0, column=2, padx=(0, 5))
        self.min_entry.insert(0, "-5.0")
        self.min_entry.bind("<Return>", self.on_min_entry_change)
        self.min_entry.bind("<FocusOut>", self.on_min_entry_change)
        
        # Max temperature controls (slider and entry)
        max_frame = ttk.Frame(common_frame)
        max_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(max_frame, text="Max:").grid(row=0, column=0, sticky=tk.W)
        
        # Max slider
        self.max_var = tk.DoubleVar(value=35)
        self.max_scale = ttk.Scale(max_frame, from_=-50, to=50, 
                                  variable=self.max_var, orient=tk.HORIZONTAL,
                                  command=self.on_max_scale_change)
        self.max_scale.grid(row=0, column=1, sticky="ew", padx=5)
        
        # Max entry
        self.max_entry = ttk.Entry(max_frame, width=8, validate="key", 
                               validatecommand=(validate_cmd, '%P'))
        self.max_entry.grid(row=0, column=2, padx=(0, 5))
        self.max_entry.insert(0, "35.0")
        self.max_entry.bind("<Return>", self.on_max_entry_change)
        self.max_entry.bind("<FocusOut>", self.on_max_entry_change)
        
        min_frame.columnconfigure(1, weight=1)
        max_frame.columnconfigure(1, weight=1)
        
        # Nút để đồng bộ hiển thị hai file
        ttk.Button(common_frame, text="Đồng bộ hiển thị", 
                  command=self.sync_displays).pack(fill=tk.X, pady=5)
    
    def validate_float(self, value):
        """Kiểm tra xem giá trị nhập vào có phải số thực không"""
        if value == "":
            return True
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def on_min_scale_change(self, value):
        """Xử lý khi thanh trượt min thay đổi"""
        min_val = self.min_var.get()
        self.min_entry.delete(0, tk.END)
        self.min_entry.insert(0, f"{min_val:.1f}")
        self.update_display_after_range_change()
    
    def on_max_scale_change(self, value):
        """Xử lý khi thanh trượt max thay đổi"""
        max_val = self.max_var.get()
        self.max_entry.delete(0, tk.END)
        self.max_entry.insert(0, f"{max_val:.1f}")
        self.update_display_after_range_change()
    
    def on_min_entry_change(self, event):
        """Xử lý khi giá trị nhập vào ô min thay đổi"""
        try:
            min_val = float(self.min_entry.get())
            self.min_var.set(min_val)
            self.update_display_after_range_change()
        except ValueError:
            pass  # Bỏ qua nếu không phải số hợp lệ
    
    def on_max_entry_change(self, event):
        """Xử lý khi giá trị nhập vào ô max thay đổi"""
        try:
            max_val = float(self.max_entry.get())
            self.max_var.set(max_val)
            self.update_display_after_range_change()
        except ValueError:
            pass  # Bỏ qua nếu không phải số hợp lệ
    
    def update_display_after_range_change(self):
        """Cập nhật hiển thị sau khi thay đổi phạm vi nhiệt độ"""
        # Đảm bảo min < max
        min_val = self.min_var.get()
        max_val = self.max_var.get()
        
        if min_val >= max_val:
            # Điều chỉnh để đảm bảo min < max
            if min_val >= 49.5:  # Nếu min gần giới hạn trên
                min_val = max_val - 1
                self.min_var.set(min_val)
                self.min_entry.delete(0, tk.END)
                self.min_entry.insert(0, f"{min_val:.1f}")
            else:
                max_val = min_val + 1
                self.max_var.set(max_val)
                self.max_entry.delete(0, tk.END)
                self.max_entry.insert(0, f"{max_val:.1f}")
        
        # Cập nhật hiển thị
        if self.current_data1 is not None:
            self.update_display(1)
            
        if self.current_data2 is not None:
            self.update_display(2)
    
    def select_tif_file(self, file_num):
        filepath = filedialog.askopenfilename(
            title=f"Chọn file TIF {file_num}",
            filetypes=[("TIF files", "*.tif"), ("All files", "*.*")]
        )
        if filepath:
            self.load_tif(filepath, file_num)
    
    def select_from_processed(self, file_num):
        if os.path.exists(self.processed_dir):
            # Lấy danh sách tệp TIF
            tif_files = [f for f in os.listdir(self.processed_dir) 
                        if f.lower().endswith('.tif')]
            
            if tif_files:
                # Tạo cửa sổ lựa chọn
                selection_window = tk.Toplevel(self.root)
                selection_window.title(f"Chọn file TIF {file_num} đã xử lý")
                selection_window.geometry("400x300")
                
                # Tạo listbox cho danh sách tệp
                listbox_frame = ttk.Frame(selection_window)
                listbox_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                
                scrollbar = ttk.Scrollbar(listbox_frame)
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                
                listbox = tk.Listbox(listbox_frame, yscrollcommand=scrollbar.set)
                listbox.pack(fill=tk.BOTH, expand=True)
                
                scrollbar.config(command=listbox.yview)
                
                # Thêm các tệp vào listbox
                for file in sorted(tif_files):
                    listbox.insert(tk.END, file)
                
                # Nút chọn
                def select_item():
                    selection = listbox.curselection()
                    if selection:
                        selected_file = listbox.get(selection[0])
                        filepath = os.path.join(self.processed_dir, selected_file)
                        self.load_tif(filepath, file_num)
                        selection_window.destroy()
                
                ttk.Button(selection_window, text="Chọn", 
                          command=select_item).pack(pady=10)
            else:
                info_label = self.info_label1 if file_num == 1 else self.info_label2
                info_label.config(text="Không tìm thấy tệp TIF trong thư mục đã xử lý")
        else:
            info_label = self.info_label1 if file_num == 1 else self.info_label2
            info_label.config(text="Không tìm thấy thư mục đã xử lý")
    
    def load_tif(self, filepath, file_num):
        try:
            with rasterio.open(filepath) as src:
                if file_num == 1:
                    self.original_data1 = src.read(1)
                    self.current_data1 = self.original_data1.copy()
                    self.current_transform1 = src.transform
                    self.tif_path1 = filepath
                    
                    # Hiển thị đường dẫn file
                    self.file_label1.config(text=os.path.basename(filepath))
                    
                    # Cập nhật thông tin
                    self.update_info_label(1)
                    
                    # Hiển thị dữ liệu
                    self.update_display(1)
                else:
                    self.original_data2 = src.read(1)
                    self.current_data2 = self.original_data2.copy()
                    self.current_transform2 = src.transform
                    self.tif_path2 = filepath
                    
                    # Hiển thị đường dẫn file
                    self.file_label2.config(text=os.path.basename(filepath))
                    
                    # Cập nhật thông tin
                    self.update_info_label(2)
                    
                    # Hiển thị dữ liệu
                    self.update_display(2)
        except Exception as e:
            info_label = self.info_label1 if file_num == 1 else self.info_label2
            info_label.config(text=f"Lỗi khi mở file: {str(e)}")
    
    def update_info_label(self, file_num):
        data = self.current_data1 if file_num == 1 else self.current_data2
        info_label = self.info_label1 if file_num == 1 else self.info_label2
        
        if data is not None:
            info_text = f"Kích thước: {data.shape}\n"
            info_text += f"Min: {np.nanmin(data):.2f}\n"
            info_text += f"Max: {np.nanmax(data):.2f}\n"
            info_text += f"Mean: {np.nanmean(data):.2f}"
            info_label.config(text=info_text)
    
    def update_range(self):
        min_val = self.min_var.get()
        max_val = self.max_var.get()
        
        # Cập nhật ô nhập liệu
        self.min_entry.delete(0, tk.END)
        self.min_entry.insert(0, f"{min_val:.1f}")
        
        self.max_entry.delete(0, tk.END)
        self.max_entry.insert(0, f"{max_val:.1f}")
        
        # Cập nhật hiển thị
        if self.current_data1 is not None:
            self.update_display(1)
            
        if self.current_data2 is not None:
            self.update_display(2)
    
    def update_display(self, file_num):
        if file_num == 1:
            if self.current_data1 is None:
                return
            
            # Xóa figure hiện tại và tạo lại axes
            self.fig1.clear()
            self.ax1 = self.fig1.add_subplot(111)
            
            # Lấy phạm vi giá trị
            vmin = self.min_var.get()
            vmax = self.max_var.get()
            
            # Clip dữ liệu
            data_display = np.copy(self.current_data1)
            data_display[data_display < vmin] = vmin
            data_display[data_display > vmax] = vmax
            
            # Tạo colormap giống GEE
            colors = ['blue', 'cyan', 'green', 'yellow', 'orange', 'red']
            cmap = LinearSegmentedColormap.from_list('gee_lst', colors, N=256)
            
            # Tạo hình ảnh
            img = self.ax1.imshow(data_display, cmap=cmap, vmin=vmin, vmax=vmax)
            
            # Thêm colorbar
            cbar = self.fig1.colorbar(img, ax=self.ax1, label='Nhiệt độ (°C)')
            
            # Thêm tiêu đề
            title = f"{os.path.basename(self.tif_path1) if self.tif_path1 else ''}"
            self.ax1.set_title(title)
            
            # Ẩn các trục
            self.ax1.axis('off')
            
            # Cập nhật canvas
            self.fig1.tight_layout()
            self.canvas1.draw()
        else:
            if self.current_data2 is None:
                return
            
            # Xóa figure hiện tại và tạo lại axes
            self.fig2.clear()
            self.ax2 = self.fig2.add_subplot(111)
            
            # Lấy phạm vi giá trị
            vmin = self.min_var.get()
            vmax = self.max_var.get()
            
            # Clip dữ liệu
            data_display = np.copy(self.current_data2)
            data_display[data_display < vmin] = vmin
            data_display[data_display > vmax] = vmax
            
            # Tạo colormap giống GEE
            colors = ['blue', 'cyan', 'green', 'yellow', 'orange', 'red']
            cmap = LinearSegmentedColormap.from_list('gee_lst', colors, N=256)
            
            # Tạo hình ảnh
            img = self.ax2.imshow(data_display, cmap=cmap, vmin=vmin, vmax=vmax)
            
            # Thêm colorbar
            cbar = self.fig2.colorbar(img, ax=self.ax2, label='Nhiệt độ (°C)')
            
            # Thêm tiêu đề
            title = f"{os.path.basename(self.tif_path2) if self.tif_path2 else ''}"
            self.ax2.set_title(title)
            
            # Ẩn các trục
            self.ax2.axis('off')
            
            # Cập nhật canvas
            self.fig2.tight_layout()
            self.canvas2.draw()
            
    def sync_displays(self):
        """Đồng bộ hiển thị của hai file tiff với cùng thang màu"""
        if self.current_data1 is None or self.current_data2 is None:
            messagebox.showwarning("Cảnh báo", "Cần chọn cả hai file để đồng bộ hiển thị")
            return
            
        # Lấy giá trị min-max hiện tại từ các ô nhập
        try:
            min_val = float(self.min_entry.get())
            max_val = float(self.max_entry.get())
        except ValueError:
            messagebox.showerror("Lỗi", "Giá trị nhiệt độ không hợp lệ")
            return
            
        if min_val >= max_val:
            messagebox.showerror("Lỗi", "Giá trị min phải nhỏ hơn giá trị max")
            return
        
        # Cập nhật sliders để đồng bộ với giá trị nhập
        self.min_var.set(min_val)
        self.max_var.set(max_val)
        
        # Làm mới hoàn toàn cả hai hiển thị để tránh chồng chéo
        self.fig1.clear()
        self.fig2.clear()
        self.ax1 = self.fig1.add_subplot(111)
        self.ax2 = self.fig2.add_subplot(111)
        
        # Cập nhật cả hai hiển thị
        self.update_display(1)
        self.update_display(2)
        
        messagebox.showinfo("Thông báo", f"Đã đồng bộ hiển thị với phạm vi: {min_val:.1f}°C - {max_val:.1f}°C")

if __name__ == "__main__":
    root = tk.Tk()
    app = TiffViewer(root)
    root.mainloop() 