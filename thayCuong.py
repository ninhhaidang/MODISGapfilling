import os
import sys
import logging
import warnings

# Cấu hình logging chi tiết
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='modis_processing.log',  # Log vào file thay vì console
    filemode='w'
)

# Thiết lập trực tiếp để tắt mọi thông báo về CuPy, Numba và MKL
os.environ['CUPY_LOG_LEVEL'] = '30'  # WARNING level
os.environ['NUMBA_WARNINGS'] = '0'
os.environ['NUMBA_DEBUG'] = '0'
os.environ['MKL_VERBOSE'] = '0'
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Tắt TensorFlow warnings nếu có

# Tắt tất cả warnings
warnings.filterwarnings('ignore')

# Lưu trữ stdout gốc
original_stdout = sys.stdout

# Tạo class để lọc output
class OutputFilter:
    def __init__(self):
        self.blocked_messages = [
            "GPU support enabled with CuPy",
            "MKL multi-threading enabled with",
            "Numba optimization available"
        ]
        self.shown_messages = set()
    
    def write(self, text):
        # Chỉ cho phép hiển thị thông báo một lần
        should_print = True
        for msg in self.blocked_messages:
            if msg in text and msg in self.shown_messages:
                should_print = False
                break
            elif msg in text:
                self.shown_messages.add(msg)
        
        if should_print:
            original_stdout.write(text)
    
    def flush(self):
        original_stdout.flush()

# Áp dụng bộ lọc
sys.stdout = OutputFilter()

# Sau đó tiếp tục với các import thông thường
import datetime
import rasterio
import numpy as np
from glob import glob
from datetime import datetime, timedelta
import pandas as pd
from scipy import ndimage
from scipy.interpolate import griddata
import multiprocessing # Thêm thư viện multiprocessing
from functools import partial # Hữu ích cho việc truyền đối số vào pool.map
import gc # Để quản lý bộ nhớ (tùy chọn)
import psutil # Thêm thư viện để theo dõi và quản lý tài nguyên
import math

# Biến toàn cục để tránh hiển thị trùng lặp thông báo
_VERBOSE_MODE = True # Đặt False để giảm bớt thông báo

# Hàm để hiển thị thông báo, chỉ hiển thị khi cần thiết
def log_message(message, force=False, level="INFO"):
    if not _VERBOSE_MODE and not force:
        return
    if level == "INFO":
        print(message)
    elif level == "WARNING":
        print(f"WARNING: {message}")
    elif level == "ERROR":
        print(f"ERROR: {message}")

# Hàm khởi tạo cho tiến trình con
def _init_worker():
    """Khởi tạo môi trường im lặng cho worker process"""
    import os
    import sys
    import warnings
    
    # Tắt tất cả logging và warning trong worker
    import logging
    logging.basicConfig(level=logging.CRITICAL)
    
    # Thiết lập các biến môi trường
    os.environ['CUPY_LOG_LEVEL'] = '30'
    os.environ['NUMBA_WARNINGS'] = '0'
    os.environ['NUMBA_DEBUG'] = '0'
    os.environ['MKL_VERBOSE'] = '0'
    os.environ['PYTHONWARNINGS'] = 'ignore'
    
    # Tắt warnings
    warnings.filterwarnings('ignore')
    
    # Đổi hướng stdout để ẩn hết thông báo
    class NullWriter:
        def write(self, text): pass
        def flush(self): pass
    
    # Chỉ đổi hướng stdout khi không phải main process
    if multiprocessing.current_process().name != 'MainProcess':
        sys.stdout = NullWriter()
        sys.stderr = NullWriter()

# Thêm Numba để tối ưu hóa các vòng lặp
try:
    import numba
    from numba import jit, prange
    numba.config.NUMBA_DISABLE_JIT = 0
    HAS_NUMBA = True
    log_message("Numba optimization available!", force=True)
except ImportError:
    HAS_NUMBA = False
    log_message("Numba not available. Some functions will run slower.", force=True)

# Thêm hỗ trợ GPU nếu có thể
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cu_ndimage
    HAS_GPU = True
    log_message("GPU support enabled with CuPy!", force=True)
except ImportError:
    HAS_GPU = False
    log_message("GPU support not available. Using CPU only.", force=True)

# Cấu hình để sử dụng đa luồng với NumPy (nếu có Intel MKL)
try:
    import mkl
    threads_to_use = 24  # Đặt theo số threads của CPU E5-2678 v3
    mkl.set_num_threads(threads_to_use)
    log_message(f"MKL multi-threading enabled with {mkl.get_max_threads()} threads", force=True)
except ImportError:
    # Nếu không có MKL, thử với OpenBLAS
    try:
        threads_to_use = 24  # Đặt theo số threads của E5-2678 v3
        os.environ["OMP_NUM_THREADS"] = str(threads_to_use)
        os.environ["OPENBLAS_NUM_THREADS"] = str(threads_to_use)
        os.environ["MKL_NUM_THREADS"] = str(threads_to_use)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads_to_use)
        os.environ["NUMEXPR_NUM_THREADS"] = str(threads_to_use)
        log_message(f"NumPy multi-threading configured with {threads_to_use} threads", force=True)
    except:
        log_message("Could not configure NumPy multi-threading", level="WARNING")

# --- LỚP CONFIG GIỮ NGUYÊN VỚI CÁC THAY ĐỔI ---
class Config:
    # Thiết lập khoảng thời gian xử lý dữ liệu mặc định
    START_DATE, END_DATE = '2020-01-01', '2020-02-01'
    # Đường dẫn thư mục gốc của dự án
    BASE_DIR = 'D:\HaiDang\MODISGapfilling' # Sử dụng dấu / hoặc \\ cho đường dẫn Windows
    # Đường dẫn thư mục chứa dữ liệu MODIS LST thô (chưa xử lý)
    RAW_DATA_DIR = f'{BASE_DIR}/MODIS_LST_VN_RAW'
    # Đường dẫn thư mục chính để lưu dữ liệu đã xử lý
    OUTPUT_DIR = f'{BASE_DIR}/MODIS_LST_VN_PROCESSED'
    # Đường dẫn thư mục để lưu các hình ảnh trực quan hóa
    VISUALIZATION_DIR = f'{BASE_DIR}/MODIS_LST_VN_VISUALIZATIONS'

    # Tên thư mục con cho dữ liệu LST cuối cùng đã được lấp đầy và xử lý
    FINAL_SUBDIR_NAME = 'Final_LST_Data'
    # Tên thư mục con cho dữ liệu LST gốc (đã kết hợp Terra/Aqua nhưng chưa qua lấp đầy chính)
    ORIGINAL_SUBDIR_NAME = 'Original_LST_Data'

    # Đường dẫn đầy đủ đến thư mục con chứa dữ liệu LST cuối cùng
    FINAL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, FINAL_SUBDIR_NAME)
    # Đường dẫn đầy đủ đến thư mục con chứa dữ liệu LST gốc
    ORIGINAL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, ORIGINAL_SUBDIR_NAME)
    
    # Tham số cho việc lấp đầy không gian: số lần lặp cho phép giãn nở và sigma cho làm mịn Gaussian
    SPATIAL_DILATION_ITERATIONS, SPATIAL_SMOOTHING_SIGMA = 5, 1.5
    # Tham số cho việc lấp đầy thời gian: số ngày lân cận, số lượng lân cận tối thiểu, kích thước cửa sổ thời gian
    TEMPORAL_NEIGHBOR_DAYS, TEMPORAL_MIN_NEIGHBORS, TEMPORAL_WINDOW_SIZE = 7, 2, 15
    # Ngưỡng nhiệt độ tối thiểu và tối đa cho việc trực quan hóa
    VISUALIZATION_MIN_TEMP, VISUALIZATION_MAX_TEMP = -5, 35
    # Danh sách màu sắc và số lượng bin cho bản đồ màu trực quan hóa
    COLOR_MAP_COLORS = ['blue', 'cyan', 'green', 'yellow', 'orange', 'red']
    COLOR_MAP_BINS = 100
    # Tham số để xác định giá trị ngoại lai: hệ số IQR, phân vị thấp và cao
    OUTLIER_IQR_FACTOR, OUTLIER_PERCENTILE_LOW, OUTLIER_PERCENTILE_HIGH = 2.0, 0.5, 99.5
    
    # Cờ điều khiển: bật/tắt làm mịn cạnh và áp dụng mặt nạ nước (hiện tại tắt)
    ENABLE_EDGE_SMOOTHING, APPLY_WATER_MASK = True, False # Đã sửa APPLY_WATER_MASK từ True thành False như trong mã gốc
    
    # Kích thước cửa sổ cho bộ lọc uniform khi làm mịn phần dư không gian (mô phỏng focal_mean radius=3 của GEE ~ 7x7)
    RESIDUAL_UNIFORM_FILTER_SIZE = 7
    
    # Cấu hình tối ưu cho xử lý đa luồng
    NUM_CPU_WORKERS = None  # Sẽ được tính dựa trên CPU
    MEMORY_LIMIT_PERCENT = 85  # Sử dụng tối đa 85% RAM có sẵn
    CHUNK_SIZE = 10  # Số lượng mẫu xử lý trong mỗi batch cho việc song song hóa
    USE_GPU = True  # Mặc định bật GPU nếu có
    
    # Cấu hình buffer I/O
    RASTERIO_BUFFER_SIZE = 2**26  # ~64MB cho buffer đọc/ghi
    
    @classmethod
    def initialize_dirs(cls):
        """Tạo các thư mục đầu ra nếu chúng chưa tồn tại."""
        for dir_path in [cls.OUTPUT_DIR, cls.VISUALIZATION_DIR, cls.FINAL_OUTPUT_DIR, cls.ORIGINAL_OUTPUT_DIR]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
    
    @classmethod
    def optimize_for_system(cls):
        """Tối ưu hóa cấu hình dựa trên tài nguyên hệ thống."""
        # Xác định số lượng core CPU tối ưu
        try:
            logical_cores = psutil.cpu_count(logical=True)
            physical_cores = psutil.cpu_count(logical=False)
            
            # E5-2678 v3 có 12 cores/24 threads
            if logical_cores and physical_cores:
                # Sử dụng tất cả các logical cores ngoại trừ 1-2 cho hệ thống
                cls.NUM_CPU_WORKERS = max(1, logical_cores - 2)
                log_message(f"System has {physical_cores} physical cores, {logical_cores} logical cores", force=True)
                log_message(f"Using {cls.NUM_CPU_WORKERS} worker processes", force=True)
            else:
                cls.NUM_CPU_WORKERS = 22  # Giá trị mặc định cho E5-2678 v3 (24 threads - 2)
        except:
            cls.NUM_CPU_WORKERS = 22
        
        # Xác định giới hạn RAM
        try:
            mem = psutil.virtual_memory()
            total_ram_gb = mem.total / (1024 ** 3)
            log_message(f"System has {total_ram_gb:.1f} GB RAM", force=True)
            
            # Tính toán kích thước chunk dựa trên RAM
            if total_ram_gb > 16:
                cls.CHUNK_SIZE = 20  # Tăng kích thước xử lý đồng thời cho RAM cao
            
            # Đặt giới hạn sử dụng RAM
            usable_ram = (mem.total * cls.MEMORY_LIMIT_PERCENT) // 100
            log_message(f"Will use up to {cls.MEMORY_LIMIT_PERCENT}% RAM ({usable_ram/(1024**3):.1f} GB)", force=True)
        except:
            pass
        
        # Kiểm tra GPU
        cls.USE_GPU = HAS_GPU and cls.USE_GPU
        
        # Tối ưu hóa các thông số rasterio
        rasterio_env = {
            'GDAL_CACHEMAX': '1024',  # 1024MB cache cho GDAL
            'GDAL_DISABLE_READDIR_ON_OPEN': 'TRUE',  # Giảm I/O
            'GDAL_MAX_DATASET_POOL_SIZE': '1024'  # Tăng kích thước pool
        }
        
        # Áp dụng các cài đặt rasterio
        for key, value in rasterio_env.items():
            os.environ[key] = value
        
        return cls

# --- CÁC HÀM WORKER CHO MULTIPROCESSING (ĐẶT Ở TOP-LEVEL) ---
def _worker_load_and_preprocess_tif(file_path_and_config_tuple):
    """Hàm worker để tải và tiền xử lý TIF, nhận tuple (file_path, config_obj)"""
    file_path, config_obj = file_path_and_config_tuple
    # Tạo một instance tạm thời của Processor hoặc gọi một hàm static nếu có thể
    # Để đơn giản, giả sử hàm load_and_preprocess_tif có thể được gọi với config
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1).astype(np.float32)
            profile = src.profile
            data = data * 0.02 - 273.15 # Kelvin sang Celsius
            valid_data_mask = (data > -50) & (data < 50) & (np.abs(data - (-273.15)) > 1e-5)
            processed_data = np.full_like(data, np.nan, dtype=np.float32)
            processed_data[valid_data_mask] = data[valid_data_mask]
            return processed_data, profile, os.path.basename(file_path) # Trả về thêm tên file để lấy ngày
    except Exception as e:
        print(f"Worker Error loading or preprocessing TIF {file_path}: {e}")
        return None, None, os.path.basename(file_path)

def _worker_spatial_gapfill(data_slice_and_config_tuple):
    """Hàm worker cho spatial_gapfill."""
    data_slice, config = data_slice_and_config_tuple # config là instance của Config
    if data_slice is None or np.all(np.isnan(data_slice)): return data_slice
    valid_mask = ~np.isnan(data_slice)
    if np.sum(valid_mask) == 0 or np.sum(valid_mask) == data_slice.size: return data_slice
    
    rows, cols = data_slice.shape
    y, x = np.mgrid[0:rows, 0:cols]
    points = np.column_stack((y[valid_mask].ravel(), x[valid_mask].ravel()))
    values = data_slice[valid_mask].ravel()

    if points.shape[0] < 2: return data_slice 

    dilated_mask = ndimage.binary_dilation(valid_mask, iterations=config.SPATIAL_DILATION_ITERATIONS)
    interpolation_mask = dilated_mask & ~valid_mask
    filled_data = np.copy(data_slice)
    interp_points = np.column_stack((y[interpolation_mask].ravel(), x[interpolation_mask].ravel()))

    if len(interp_points) > 0:
        interp_values_linear = griddata(points, values, interp_points, method='linear')
        linear_fill_idx = ~np.isnan(interp_values_linear)
        if np.any(linear_fill_idx): # Chỉ gán nếu có giá trị không NaN
             filled_data[interpolation_mask][linear_fill_idx] = interp_values_linear[linear_fill_idx]
        
        remaining_nan_mask_after_linear = interpolation_mask & np.isnan(filled_data)
        if np.any(remaining_nan_mask_after_linear):
            points_for_nearest = np.column_stack((y[remaining_nan_mask_after_linear].ravel(), x[remaining_nan_mask_after_linear].ravel()))
            if len(points_for_nearest) > 0:
                interp_values_nearest = griddata(points, values, points_for_nearest, method='nearest')
                filled_data[remaining_nan_mask_after_linear] = interp_values_nearest
                
        actually_filled_mask = interpolation_mask & ~np.isnan(filled_data)
        if np.any(actually_filled_mask):
            temp_smoothed = ndimage.gaussian_filter(filled_data, sigma=config.SPATIAL_SMOOTHING_SIGMA, mode='reflect')
            filled_data[actually_filled_mask] = temp_smoothed[actually_filled_mask]
            
    return filled_data

def _worker_nan_robust_uniform_filter(data_slice_and_size_tuple):
    """Hàm worker cho _nan_robust_uniform_filter."""
    data, size = data_slice_and_size_tuple
    if data is None or data.size == 0: return data
    data_copy = np.copy(data)
    nan_mask = np.isnan(data_copy)
    temp_data_for_sum = np.copy(data_copy)
    temp_data_for_sum[nan_mask] = 0
    weights = np.ones_like(data_copy, dtype=float)
    weights[nan_mask] = 0
    sum_filter = ndimage.uniform_filter(temp_data_for_sum, size=size, mode='constant', cval=0)
    count_filter = ndimage.uniform_filter(weights, size=size, mode='constant', cval=0)
    result = np.full_like(data, np.nan, dtype=np.float64)
    valid_counts_mask = count_filter > 1e-9
    result[valid_counts_mask] = sum_filter[valid_counts_mask] / count_filter[valid_counts_mask]
    return result

def _worker_filter_outliers(data_slice_and_config_tuple):
    """Hàm worker cho filter_outliers (cho một slice 2D)."""
    current_slice, config = data_slice_and_config_tuple
    if np.all(np.isnan(current_slice)): return current_slice
    
    filtered_slice = current_slice.copy()
    valid_mask_slice = ~np.isnan(filtered_slice)
    if not np.any(valid_mask_slice): return filtered_slice
    
    valid_data = filtered_slice[valid_mask_slice]
    if valid_data.size == 0: return filtered_slice
    
    q_low = np.percentile(valid_data, config.OUTLIER_PERCENTILE_LOW)
    q_high = np.percentile(valid_data, config.OUTLIER_PERCENTILE_HIGH)
    iqr = q_high - q_low
    
    if iqr < 1e-6: 
        lower_bound, upper_bound = q_low - 1, q_high + 1
    else: 
        lower_bound, upper_bound = q_low - config.OUTLIER_IQR_FACTOR * iqr, q_high + config.OUTLIER_IQR_FACTOR * iqr
        
    outlier_mask_slice = (filtered_slice < lower_bound) | (filtered_slice > upper_bound)
    filtered_slice[outlier_mask_slice & valid_mask_slice] = np.nan
    return filtered_slice

# Thêm hàm Numba-optimized cho temporal gap filling
if HAS_NUMBA:
    @jit(nopython=True, parallel=False)
    def _numba_temporal_gapfill_pixel(time_series, temporal_neighbor_days, temporal_min_neighbors):
        """Tối ưu hóa với Numba: Nội suy thiếu dữ liệu trong một chuỗi thời gian pixel"""
        if np.all(~np.isnan(time_series)):  # Nếu không có giá trị NaN
            return time_series

        result = np.copy(time_series)
        valid_idx = np.where(~np.isnan(time_series))[0]
        missing_idx = np.where(np.isnan(time_series))[0]

        if len(valid_idx) < 2 or len(missing_idx) == 0:
            return result

        min_valid_time = np.min(valid_idx)
        max_valid_time = np.max(valid_idx)

        for idx_miss in missing_idx:
            if min_valid_time < idx_miss < max_valid_time:
                relevant_neighbors = []
                relevant_values = []
                
                # Tìm các lân cận có giá trị
                for idx in valid_idx:
                    if abs(idx - idx_miss) <= temporal_neighbor_days:
                        relevant_neighbors.append(idx)
                        relevant_values.append(time_series[idx])
                
                if len(relevant_neighbors) >= temporal_min_neighbors:
                    # Sắp xếp lân cận để nội suy chính xác
                    sorted_indices = np.argsort(relevant_neighbors)
                    sorted_neighbors = np.array([relevant_neighbors[i] for i in sorted_indices])
                    sorted_values = np.array([relevant_values[i] for i in sorted_indices])
                    
                    # Kiểm tra xem idx_miss có nằm trong phạm vi lân cận không
                    if sorted_neighbors[0] <= idx_miss <= sorted_neighbors[-1]:
                        # Nội suy tuyến tính
                        result[idx_miss] = np.interp(idx_miss, sorted_neighbors, sorted_values)
        
        return result

    @jit(nopython=True, parallel=True)
    def _numba_temporal_gapfill_block(data_block, temporal_neighbor_days, temporal_min_neighbors):
        """Tối ưu hóa với Numba: Xử lý song song nhiều pixel trong một khối"""
        rows, cols = data_block.shape[1:3]
        result = np.copy(data_block)
        
        for i in prange(rows):
            for j in prange(cols):
                result[:, i, j] = _numba_temporal_gapfill_pixel(
                    data_block[:, i, j], 
                    temporal_neighbor_days, 
                    temporal_min_neighbors
                )
        
        return result

# Thêm hàm xử lý block cho temporal gap filling
def _process_temporal_block(args):
    """Xử lý một khối dữ liệu trong temporal_gapfill"""
    block, config = args
    if HAS_NUMBA:
        return _numba_temporal_gapfill_block(
            block, 
            config.TEMPORAL_NEIGHBOR_DAYS, 
            config.TEMPORAL_MIN_NEIGHBORS
        )
    else:
        # Fallback không dùng Numba - xử lý tuần tự
        result = np.copy(block)
        for i in range(block.shape[1]):
            for j in range(block.shape[2]):
                ts = block[:, i, j]
                if not np.any(np.isnan(ts)):
                    continue

                valid_idx = np.where(~np.isnan(ts))[0]
                missing_idx = np.where(np.isnan(ts))[0]

                if len(valid_idx) < 2 or len(missing_idx) == 0:
                    continue
                
                min_valid_time = np.min(valid_idx)
                max_valid_time = np.max(valid_idx)

                for idx_miss in missing_idx:
                    if min_valid_time < idx_miss < max_valid_time:
                        relevant_neighbors_indices = valid_idx[np.abs(valid_idx - idx_miss) <= config.TEMPORAL_NEIGHBOR_DAYS]
                        if len(relevant_neighbors_indices) >= config.TEMPORAL_MIN_NEIGHBORS:
                            sorted_relevant_neighbors = np.sort(relevant_neighbors_indices)
                            if sorted_relevant_neighbors.size > 0 and \
                               np.min(sorted_relevant_neighbors) <= idx_miss <= np.max(sorted_relevant_neighbors):
                                try:
                                    unique_neighbors, unique_indices = np.unique(sorted_relevant_neighbors, return_index=True)
                                    if len(unique_neighbors) >=2:
                                        result[idx_miss, i, j] = np.interp(idx_miss, unique_neighbors, ts[unique_neighbors])
                                except Exception as e:
                                    # Silent exception, leave as NaN
                                    pass
        return result

# --- LỚP MODISLSTProcessor ĐÃ ĐƯỢC SỬA ĐỔI ---
class MODISLSTProcessor:
    def __init__(self, config=None, num_workers=None):
        self.config = config or Config.optimize_for_system()
        self.config.initialize_dirs()
        self.water_mask = None
        
        # Sử dụng num_workers từ tham số hoặc từ cấu hình
        self.num_workers = num_workers if num_workers is not None else self.config.NUM_CPU_WORKERS
        self.use_gpu = self.config.USE_GPU and HAS_GPU
        
        log_message(f"Initializing MODISLSTProcessor with {self.num_workers} workers and GPU={self.use_gpu}")
        
        # Cấu hình rasterio - Sửa: chuyển đổi RASTERIO_BUFFER_SIZE thành integer
        rasterio_options = {
            'GDAL_CACHEMAX': 1024,  # Số nguyên thay vì chuỗi
            'RASTERIO_BUFFER_SIZE': int(self.config.RASTERIO_BUFFER_SIZE)  # Chuyển đổi thành integer
        }
        rasterio.Env(**rasterio_options).__enter__()
        
        # Cấu hình xử lý đa luồng
        if hasattr(np, 'core'):
            try:
                np.core.arrayprint._line_width = 160  # Tối ưu output cho terminal rộng
            except:
                pass

    # _nan_robust_uniform_filter có thể giữ nguyên hoặc gọi worker nếu cần
    def _nan_robust_uniform_filter(self, data, size):
        if self.use_gpu:
            try:
                # Không hiển thị thông báo ở đây để tránh lặp lại
                # Chuyển dữ liệu lên GPU
                gpu_data = cp.asarray(data)
                nan_mask = cp.isnan(gpu_data)
                temp_data_for_sum = cp.copy(gpu_data)
                temp_data_for_sum[nan_mask] = 0
                weights = cp.ones_like(gpu_data, dtype=float)
                weights[nan_mask] = 0
                
                # Áp dụng bộ lọc
                sum_filter = cu_ndimage.uniform_filter(temp_data_for_sum, size=size, mode='constant', cval=0)
                count_filter = cu_ndimage.uniform_filter(weights, size=size, mode='constant', cval=0)
                
                # Tính kết quả
                result = cp.full_like(gpu_data, cp.nan, dtype=cp.float64)
                valid_counts_mask = count_filter > 1e-9
                result[valid_counts_mask] = sum_filter[valid_counts_mask] / count_filter[valid_counts_mask]
                
                # Chuyển kết quả trở lại CPU
                return cp.asnumpy(result)
            except Exception as e:
                # Chỉ hiển thị cảnh báo một lần
                warnings.warn(f"GPU processing failed with error: {e}. Falling back to CPU.", stacklevel=2)
                return _worker_nan_robust_uniform_filter((data, size))
        else:
            return _worker_nan_robust_uniform_filter((data, size))

    def load_and_preprocess_tif(self, file_path):
        # Hàm này sẽ được gọi bởi worker, nên logic chính nằm trong _worker_load_and_preprocess_tif
        # Tuy nhiên, vẫn giữ lại để có thể gọi tuần tự nếu cần test
        try:
            with rasterio.open(file_path) as src:
                data = src.read(1).astype(np.float32)
                profile = src.profile
                data = data * 0.02 - 273.15
                valid_data_mask = (data > -50) & (data < 50) & (np.abs(data - (-273.15)) > 1e-5)
                processed_data = np.full_like(data, np.nan, dtype=np.float32)
                processed_data[valid_data_mask] = data[valid_data_mask]
                return processed_data, profile
        except Exception as e:
            print(f"Error loading or preprocessing TIF {file_path}: {e}")
            return None, None

    def get_date_from_filename(self, filename):
        try:
            return datetime.strptime(filename.split('_')[-1].split('.')[0], '%Y-%m-%d')
        except ValueError: # Xử lý trường hợp tên file không đúng định dạng mong đợi
            print(f"Warning: Could not parse date from filename: {filename}")
            # Trả về một ngày mặc định hoặc raise lỗi tùy theo logic mong muốn
            return datetime.min # Hoặc một giá trị không bao giờ nằm trong khoảng start/end date

    def clip_values(self, data, min_val=None, max_val=None):
        if data is None: return None
        min_val = min_val if min_val is not None else self.config.VISUALIZATION_MIN_TEMP
        max_val = max_val if max_val is not None else self.config.VISUALIZATION_MAX_TEMP
        clipped_data = np.copy(data)
        # Xử lý mask cẩn thận hơn khi so sánh với NaN
        non_nan_mask = ~np.isnan(clipped_data)
        clipped_data[non_nan_mask & (clipped_data[non_nan_mask] < min_val)] = min_val
        clipped_data[non_nan_mask & (clipped_data[non_nan_mask] > max_val)] = max_val
        return clipped_data

    def spatial_gapfill(self, data_slice):
        # Hàm này sẽ được gọi bởi worker _worker_spatial_gapfill
        return _worker_spatial_gapfill((data_slice, self.config))

    def temporal_gapfill(self, data_cube):
        """Lấp đầy khoảng trống (NaN) trong chuỗi thời gian bằng nội suy tuyến tính.
        Phiên bản cải tiến: Chia dữ liệu thành các khối và xử lý song song.
        """
        if data_cube is None or data_cube.shape[0] == 0:
            log_message("Warning: Empty data cube in temporal_gapfill", level="WARNING")
            return data_cube

        log_message(f"Starting temporal gap filling on data cube shape {data_cube.shape}...", force=True)
        start_time = datetime.now()
        
        # Xác định số lượng khối có thể xử lý dựa trên bộ nhớ có sẵn
        num_time_steps, rows, cols = data_cube.shape
        total_pixels = rows * cols
        
        # Tính số lượng khối để cân bằng giữa hiệu suất và bộ nhớ
        # Mặc định chia thành 8-12 khối tùy thuộc vào RAM
        try:
            mem = psutil.virtual_memory()
            free_mem_gb = (mem.available / (1024**3))
            # Mỗi phần tử float32 chiếm 4 bytes
            data_size_gb = (data_cube.size * 4) / (1024**3)
            
            # Tối ưu hóa số khối để tận dụng CPU tốt hơn
            # Dùng nhiều worker hơn khi dữ liệu lớn
            optimal_blocks = min(self.num_workers, max(8, int(math.sqrt(total_pixels) / 200)))
            num_blocks = max(optimal_blocks, min(24, int(data_size_gb / (free_mem_gb * 0.1))))
            log_message(f"Data size: {data_size_gb:.2f} GB, Free memory: {free_mem_gb:.2f} GB")
            log_message(f"Dividing into {num_blocks} blocks for temporal gap filling")
        except:
            # Fallback nếu không thể đo bộ nhớ
            num_blocks = max(12, self.num_workers // 2)
            log_message(f"Using default {num_blocks} blocks for temporal gap filling")
        
        # Sử dụng multithreading nếu có Numba, multiprocessing nếu không
        if HAS_NUMBA:
            log_message("Using Numba-optimized temporal gap filling (parallel)", force=True)
            # Kích hoạt parallel execution trong Numba
            if hasattr(numba, 'set_num_threads'):
                # Sử dụng tất cả các core có sẵn
                numba.set_num_threads(self.num_workers)
                log_message(f"Numba parallelism enabled with {self.num_workers} threads")
            
            block_rows = max(1, math.ceil(rows / num_blocks))
            result_cube = np.copy(data_cube)
            
            for i in range(0, rows, block_rows):
                end_i = min(i + block_rows, rows)
                # Không hiển thị thông báo lặp lại cho mỗi block
                if i % (3 * block_rows) == 0:  # Hiển thị thông báo mỗi 3 block
                    log_message(f"Processing block rows {i} to {end_i}...")
                
                block = data_cube[:, i:end_i, :]
                # Khối được xử lý với tối ưu Numba
                result_block = _numba_temporal_gapfill_block(
                    block, 
                    self.config.TEMPORAL_NEIGHBOR_DAYS, 
                    self.config.TEMPORAL_MIN_NEIGHBORS
                )
                
                result_cube[:, i:end_i, :] = result_block
                
                # Báo cáo tiến độ, nhưng không quá thường xuyên
                progress = (end_i / rows) * 100
                if progress % 20 <= (block_rows / rows * 100):  # Báo cáo khoảng mỗi 20%
                    log_message(f"Temporal gap filling progress: {progress:.1f}%", force=True)
                
                # Giải phóng bộ nhớ
                del block, result_block
                gc.collect()
        else:
            log_message("Using multiprocessing for temporal gap filling", force=True)
            # Chia thành các khối hàng và cột để tận dụng tốt hơn CPU
            # Mỗi worker xử lý một phần nhỏ dữ liệu thay vì những dải lớn
            block_size = max(50, min(200, int(math.sqrt(rows*cols / num_blocks))))
            blocks = []
            
            # Tạo lưới các khối nhỏ hơn để xử lý song song
            for i in range(0, rows, block_size):
                end_i = min(i + block_size, rows)
                for j in range(0, cols, block_size):
                    end_j = min(j + block_size, cols)
                    block = data_cube[:, i:end_i, j:end_j]
                    if not np.all(np.isnan(block)):  # Chỉ xử lý khối có dữ liệu
                        blocks.append((block, self.config))
            
            log_message(f"Created {len(blocks)} blocks of approx. {block_size}x{block_size} pixels each")
            
            # Xử lý song song các khối với multiprocessing
            with multiprocessing.Pool(processes=min(len(blocks), self.num_workers), initializer=_init_worker) as pool:
                processed_blocks = pool.map(_process_temporal_block, blocks)
            
            # Kết hợp kết quả
            result_cube = np.copy(data_cube)
            block_idx = 0
            
            for i in range(0, rows, block_size):
                end_i = min(i + block_size, rows)
                for j in range(0, cols, block_size):
                    end_j = min(j + block_size, cols)
                    if block_idx < len(processed_blocks) and not np.all(np.isnan(data_cube[:, i:end_i, j:end_j])):
                        result_cube[:, i:end_i, j:end_j] = processed_blocks[block_idx]
                        block_idx += 1
                    
                # Báo cáo tiến độ theo dòng
                if (i + block_size) % (max(rows // 5, block_size)) < block_size:
                    progress = (i + block_size) / rows * 100
                    log_message(f"Temporal gap filling progress: {progress:.1f}%", force=True)
            
            # Giải phóng bộ nhớ
            del blocks, processed_blocks
            gc.collect()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        log_message(f"Temporal gap filling completed in {duration:.2f} seconds", force=True)
        
        return result_cube
    
    def filter_outliers(self, data_array):
        """Lọc giá trị ngoại lai, giữ NaN. Có thể song song hóa theo từng ảnh."""
        if data_array is None: return None
        
        is_3d = data_array.ndim == 3
        if not is_3d: # Xử lý trường hợp mảng 2D
            return _worker_filter_outliers((data_array, self.config))

        num_images = data_array.shape[0]
        slices_to_process = [(data_array[i], self.config) for i in range(num_images)]
        
        log_message(f"Filtering outliers for {num_images} images using {self.num_workers} workers...")
        # Thêm initializer
        with multiprocessing.Pool(processes=self.num_workers, initializer=_init_worker) as pool:
            filtered_slices = pool.map(_worker_filter_outliers, slices_to_process)
        
        return np.stack(filtered_slices) if filtered_slices else data_array


    def apply_temporal_smoothing(self, data):
        if data is None or data.shape[0] < 1: return data
        if data.shape[0] == 1: return data 
        window = min(self.config.TEMPORAL_WINDOW_SIZE, data.shape[0])
        
        # Hàm nan_rolling_mean_pd nên được định nghĩa bên ngoài nếu dùng apply_along_axis với pool,
        # hoặc giữ nguyên nếu apply_along_axis đủ nhanh.
        # Pandas rolling có thể đã được tối ưu hóa.
        def nan_rolling_mean_pd(arr_1d):
            series = pd.Series(arr_1d)
            return series.rolling(window=window, center=True, min_periods=1).mean().to_numpy()
        try:
            # np.apply_along_axis có thể chậm với hàm Python thuần túy.
            # Nếu đây là điểm nghẽn, xem xét việc viết lại bằng Numba hoặc tối ưu hóa Pandas.
            return np.apply_along_axis(nan_rolling_mean_pd, axis=0, arr=data)
        except Exception as e:
            print(f"Error during temporal smoothing: {e}. Returning original data.")
            return data

    def combine_terra_aqua_data(self, final_lst_input, original_data_input, indices, file_sources):
        # Hàm này logic phức tạp, giữ nguyên xử lý tuần tự cho mỗi ngày.
        # Việc song song hóa ở mức này có thể không mang lại nhiều lợi ích so với độ phức tạp.
        if not indices or final_lst_input is None or original_data_input is None:
            # print("Warning: Empty inputs to combine_terra_aqua_data.")
            ref_shape_fallback = (1,1) 
            if final_lst_input is not None and final_lst_input.ndim > 2 and final_lst_input.shape[0] > 0:
                ref_shape_fallback = final_lst_input.shape[1:]
            elif original_data_input is not None and original_data_input.ndim > 2 and original_data_input.shape[0] > 0:
                ref_shape_fallback = original_data_input.shape[1:]
            return (np.full(ref_shape_fallback, np.nan, dtype=np.float32), 
                    np.full(ref_shape_fallback, np.nan, dtype=np.float32), [])

        # final_lst và original_data phải là mảng 3D (time, lat, lon) chứa các slice cho ngày đó.
        # `indices` là danh sách các chỉ số của các file thuộc về cùng một ngày.
        final_lst_slices_for_day = final_lst_input[indices]
        original_data_slices_for_day = original_data_input[indices]
        current_file_sources_for_day = [file_sources[i] for i in indices]

        if len(indices) == 1:
            return (final_lst_slices_for_day[0], original_data_slices_for_day[0], [current_file_sources_for_day[0]])

        ref_shape = final_lst_slices_for_day[0].shape
        combined_data = np.full(ref_shape, np.nan, dtype=np.float32)
        combined_original = np.full(ref_shape, np.nan, dtype=np.float32)
        
        unique_sources_in_day = sorted(list(set(s.split('_')[0] for s in current_file_sources_for_day))) # Lấy Terra/Aqua từ tên

        terra_final_slices = [final_lst_slices_for_day[i] for i, src_full_name in enumerate(current_file_sources_for_day) if src_full_name.startswith('Terra') and not np.all(np.isnan(final_lst_slices_for_day[i]))]
        aqua_final_slices = [final_lst_slices_for_day[i] for i, src_full_name in enumerate(current_file_sources_for_day) if src_full_name.startswith('Aqua') and not np.all(np.isnan(final_lst_slices_for_day[i]))]

        if terra_final_slices and aqua_final_slices:
            terra_mean_final = np.nanmean(np.stack(terra_final_slices, axis=0), axis=0)
            aqua_mean_final = np.nanmean(np.stack(aqua_final_slices, axis=0), axis=0)
            combined_data = np.nanmean(np.stack([terra_mean_final, aqua_mean_final], axis=0), axis=0)
        elif aqua_final_slices:
            combined_data = np.nanmean(np.stack(aqua_final_slices, axis=0), axis=0)
        elif terra_final_slices:
            combined_data = np.nanmean(np.stack(terra_final_slices, axis=0), axis=0)

        terra_orig_slices = [original_data_slices_for_day[i] for i, src_full_name in enumerate(current_file_sources_for_day) if src_full_name.startswith('Terra') and not np.all(np.isnan(original_data_slices_for_day[i]))]
        aqua_orig_slices = [original_data_slices_for_day[i] for i, src_full_name in enumerate(current_file_sources_for_day) if src_full_name.startswith('Aqua') and not np.all(np.isnan(original_data_slices_for_day[i]))]
        
        if terra_orig_slices and aqua_orig_slices:
            terra_mean_orig = np.nanmean(np.stack(terra_orig_slices, axis=0), axis=0)
            aqua_mean_orig = np.nanmean(np.stack(aqua_orig_slices, axis=0), axis=0)
            combined_original = np.nanmean(np.stack([terra_mean_orig, aqua_mean_orig], axis=0), axis=0)
        elif aqua_orig_slices:
            combined_original = np.nanmean(np.stack(aqua_orig_slices, axis=0), axis=0)
        elif terra_orig_slices:
            combined_original = np.nanmean(np.stack(terra_orig_slices, axis=0), axis=0)

        if self.config.ENABLE_EDGE_SMOOTHING and np.any(~np.isnan(combined_data)):
            valid_data_for_smoothing_mask = ~np.isnan(combined_data)
            if np.sum(valid_data_for_smoothing_mask) > 0:
                smoothed_edge_data = ndimage.gaussian_filter(combined_data, sigma=0.75, mode='reflect') #Sử dụng combined_data thay vì combined_data[valid_data_for_smoothing_mask]
                combined_data[valid_data_for_smoothing_mask] = smoothed_edge_data[valid_data_for_smoothing_mask]
        
        return combined_data, combined_original, unique_sources_in_day
    
    def _combine_original_data(self, original_data_input, indices, file_sources, combined_original_output, ref_shape):
        pass # Không còn dùng

    def load_modis_data(self, start_date, end_date):
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        file_patterns = {
            'terra_day': 'Terra_Day_*.tif', 'terra_night': 'Terra_Night_*.tif',
            'aqua_day': 'Aqua_Day_*.tif', 'aqua_night': 'Aqua_Night_*.tif'
        }
        all_files_in_range = []
        # Thu thập tất cả các file trước, sau đó lọc theo ngày
        all_raw_files = glob(os.path.join(self.config.RAW_DATA_DIR, '*.tif'))

        files_dict = {key: [] for key in file_patterns}
        
        for f_path in all_raw_files:
            basename = os.path.basename(f_path)
            try:
                file_date = self.get_date_from_filename(basename)
                if start_dt <= file_date <= end_dt:
                    for key, pattern_start in [('terra_day', 'Terra_Day_'), ('terra_night', 'Terra_Night_'),
                                               ('aqua_day', 'Aqua_Day_'), ('aqua_night', 'Aqua_Night_')]:
                        if basename.startswith(pattern_start):
                            files_dict[key].append(f_path)
                            break 
            except Exception as e: # Bắt lỗi nếu get_date_from_filename thất bại
                print(f"Skipping file due to date parsing error: {basename} - {e}")

        for key in files_dict: # Sắp xếp lại
            files_dict[key].sort()

        print(f"Loaded: {len(files_dict['terra_day'])} TerraD, {len(files_dict['aqua_day'])} AquaD, {len(files_dict['terra_night'])} TerraN, {len(files_dict['aqua_night'])} AquaN files.")
        return files_dict['terra_day'] + files_dict['aqua_day'], files_dict['terra_night'] + files_dict['aqua_night']

    def process_files(self, file_list, time_of_day):
        if not file_list:
            log_message(f"No files to process for {time_of_day}.")
            return None

        log_message(f"Processing {len(file_list)} files for {time_of_day} using {self.num_workers} workers...", force=True)
        
        # Chuẩn bị đối số cho worker: list các tuple (file_path, self.config)
        args_for_pool = [(file_path, self.config) for file_path in file_list]

        # Sử dụng initializer để tắt thông báo trong worker processes
        with multiprocessing.Pool(processes=self.num_workers, initializer=_init_worker) as pool:
            # results_from_workers là list các tuple (processed_data, profile, basename)
            results_from_workers = pool.map(_worker_load_and_preprocess_tif, args_for_pool)
        
        del args_for_pool # Giải phóng bộ nhớ
        gc.collect()

        data_series, dates, file_sources_names, profiles = [], [], [], []
        common_profile = None

        for data, profile, basename in results_from_workers:
            if data is not None and profile is not None:
                if common_profile is None:
                    common_profile = profile
                # Kiểm tra profile (width, height) có nhất quán không
                elif common_profile['width'] != profile['width'] or common_profile['height'] != profile['height']:
                    print(f"Warning: Mismatched profile for {basename}. Expected {common_profile['width']}x{common_profile['height']}, got {profile['width']}x{profile['height']}. Skipping file.")
                    continue
                
                data_series.append(data)
                try:
                    dates.append(self.get_date_from_filename(basename))
                    # Xác định nguồn từ basename (ví dụ 'Terra_Day_...' -> 'Terra')
                    if basename.startswith('Terra'):
                        file_sources_names.append('Terra_' + time_of_day) # Thêm time_of_day để phân biệt
                    elif basename.startswith('Aqua'):
                        file_sources_names.append('Aqua_' + time_of_day)
                    else:
                        file_sources_names.append('Unknown_' + time_of_day)
                except Exception as e:
                    print(f"Error processing file metadata for {basename}: {e}. Skipping.")
                    if data_series: data_series.pop() # Xóa data đã thêm nếu không lấy được metadata
                    continue # Bỏ qua file này
            else:
                print(f"Skipping file {basename} due to load/preprocess error from worker.")
        
        del results_from_workers # Giải phóng bộ nhớ
        gc.collect()

        if not data_series:
            print(f"No valid data loaded for {time_of_day}.")
            return None
        
        # Đảm bảo tất cả các mảng trong data_series có cùng shape trước khi stack
        first_shape = data_series[0].shape
        
        consistent_indices = [idx for idx, arr in enumerate(data_series) if arr.shape == first_shape]
        
        if len(consistent_indices) != len(data_series):
            print(f"Warning: {len(data_series) - len(consistent_indices)} arrays had mismatched shapes and were excluded for {time_of_day}.")
        
        if not consistent_indices:
            print(f"No consistently shaped data arrays for {time_of_day}.")
            return None

        consistent_data_series = [data_series[i] for i in consistent_indices]
        consistent_dates = [dates[i] for i in consistent_indices]
        consistent_file_sources = [file_sources_names[i] for i in consistent_indices]
        
        del data_series, dates, file_sources_names # Giải phóng
        gc.collect()

        if not consistent_data_series: # Kiểm tra lại sau khi lọc
            print(f"No consistently shaped data arrays left for {time_of_day}.")
            return None

        data_array = np.stack(consistent_data_series)
        original_data_array = data_array.copy() 

        del consistent_data_series # Giải phóng
        gc.collect()

        processed_data_array = self.process_data_array(data_array)
        del data_array # Giải phóng
        gc.collect()

        if processed_data_array is None:
            print(f"Processing data array failed for {time_of_day}.")
            return None
        if common_profile is None: # Nên lấy common_profile từ consistent_profiles nếu có
            print(f"Error: No common raster profile found for {time_of_day}.")
            return None
            
        return self.organize_results_by_date(processed_data_array, original_data_array, consistent_dates, consistent_file_sources, time_of_day, common_profile)

    def process_data_array(self, data_array_input):
        if data_array_input is None or data_array_input.size == 0:
            log_message("Error: Input data_array is None/empty.", level="ERROR")
            return None
        
        data_array = data_array_input.copy()
        del data_array_input # Giải phóng
        gc.collect()

        num_images = data_array.shape[0]

        log_message(f"Applying spatial gap filling for {num_images} images using {self.num_workers} workers...", force=True)
        args_for_spatial_pool = [(data_array[i], self.config) for i in range(num_images)]
        with multiprocessing.Pool(processes=self.num_workers, initializer=_init_worker) as pool:
            filled_spatial_slices = pool.map(_worker_spatial_gapfill, args_for_spatial_pool)
        del args_for_spatial_pool
        gc.collect()
        
        if not filled_spatial_slices or len(filled_spatial_slices) != num_images:
            log_message("Error during spatial gap filling: not all slices were processed.", level="ERROR")
            return None # Hoặc xử lý lỗi khác
        data_array = np.stack(filled_spatial_slices)
        del filled_spatial_slices
        gc.collect()

        log_message("Applying temporal gap filling (parallel)...", force=True)
        data_array = self.temporal_gapfill(data_array) # Đã tối ưu hóa song song
        if data_array is None: log_message("Error: data_array became None after temporal_gapfill.", level="ERROR"); return None
        if np.all(np.isnan(data_array)): log_message("Warning: Data array is all NaN before long-term mean.", level="WARNING"); return data_array
        
        log_message("Calculating long-term mean...", force=True)
        mean_lst = np.nanmean(data_array, axis=0)
        if np.all(np.isnan(mean_lst)): log_message("Warning: Long-term mean is all NaN.", level="WARNING"); return data_array
        
        log_message(f"Computing and smoothing residuals for {num_images} images using {self.num_workers} workers...", force=True)
        residuals = data_array - mean_lst[np.newaxis, :, :]
        del data_array # Giải phóng
        gc.collect()

        args_for_residual_smoothing_pool = [(residuals[i], self.config.RESIDUAL_UNIFORM_FILTER_SIZE) for i in range(num_images)]
        
        smoothed_residuals_slices = []
        with multiprocessing.Pool(processes=self.num_workers, initializer=_init_worker) as pool:
            # Áp dụng _nan_robust_uniform_filter 2 lần
            log_message("First pass of residual smoothing...", force=True)
            temp_smoothed_once = pool.map(_worker_nan_robust_uniform_filter, args_for_residual_smoothing_pool)
            
            # Cập nhật args cho lần chạy thứ 2
            args_for_second_pass = [(s, self.config.RESIDUAL_UNIFORM_FILTER_SIZE) for s in temp_smoothed_once if s is not None]
            # Chỉ xử lý những slice không None
            if args_for_second_pass:
                log_message("Second pass of residual smoothing...", force=True)
                smoothed_residuals_slices = pool.map(_worker_nan_robust_uniform_filter, args_for_second_pass)
            else: # Nếu tất cả là None sau lần đầu
                 smoothed_residuals_slices = temp_smoothed_once # Giữ nguyên kết quả None
        
        del args_for_residual_smoothing_pool, temp_smoothed_once, args_for_second_pass
        gc.collect()

        if not smoothed_residuals_slices or len(smoothed_residuals_slices) != num_images:
            # Xử lý trường hợp một số slice là None hoặc số lượng không khớp
            log_message("Warning: Some residual slices might be None or count mismatch after smoothing.", level="WARNING")
            # Tạo lại mảng smoothed_residuals với kích thước đúng, điền NaN nếu cần
            valid_smoothed_residuals = [s for s in smoothed_residuals_slices if s is not None]
            if not valid_smoothed_residuals: # Nếu tất cả đều None
                log_message("All residual slices are None after smoothing. Returning original residuals or handling error.", level="WARNING")
                # Quyết định trả về residuals gốc hoặc một mảng NaN
                # Ở đây, giả sử lỗi và trả về None để dừng xử lý
                return None
            
            # Cố gắng stack những gì có, nếu không thì xử lý lỗi
            try:
                if len(valid_smoothed_residuals) == num_images:
                     smoothed_residuals = np.stack(valid_smoothed_residuals)
                else: # Nếu số lượng không khớp, đây là vấn đề
                     log_message(f"Error: Mismatch in smoothed residual slices count. Expected {num_images}, got {len(valid_smoothed_residuals)}", level="ERROR")
                     # Cần chiến lược điền bù hoặc dừng lại.
                     # Ví dụ: Điền bằng residuals gốc cho những slice bị lỗi
                     reconstructed_residuals = []
                     original_idx = 0
                     processed_idx = 0
                     while len(reconstructed_residuals) < num_images:
                         if processed_idx < len(smoothed_residuals_slices) and smoothed_residuals_slices[processed_idx] is not None:
                             reconstructed_residuals.append(smoothed_residuals_slices[processed_idx])
                             processed_idx +=1
                         else: # Thêm residual gốc hoặc NaN
                             reconstructed_residuals.append(residuals[original_idx]) # Hoặc np.full_like(residuals[0], np.nan)
                             if processed_idx < len(smoothed_residuals_slices): processed_idx +=1 # Bỏ qua slice None
                         original_idx +=1
                     smoothed_residuals = np.stack(reconstructed_residuals)

            except ValueError as ve:
                log_message(f"ValueError when stacking smoothed residuals: {ve}. Dimensions might be inconsistent.", level="ERROR")
                # Xử lý lỗi ở đây, ví dụ trả về residuals chưa làm mịn hoặc None
                return None # Hoặc residuals
        else:
             smoothed_residuals = np.stack(smoothed_residuals_slices)

        del smoothed_residuals_slices, residuals # Giải phóng
        gc.collect()

        log_message("Applying temporal smoothing to residuals...", force=True)
        smoothed_temporal_residuals = self.apply_temporal_smoothing(smoothed_residuals) # Tuần tự
        
        if smoothed_temporal_residuals is None: # Nếu apply_temporal_smoothing trả về None (do lỗi)
            log_message("Warning: Temporal smoothing of residuals failed. Using unsmoothed residuals.", level="WARNING")
            final_lst = smoothed_residuals + mean_lst[np.newaxis, :, :]
        else:
            final_lst = smoothed_temporal_residuals + mean_lst[np.newaxis, :, :]
        
        del smoothed_residuals, smoothed_temporal_residuals, mean_lst
        gc.collect()

        log_message("Filtering outliers from final LST (parallel)...", force=True)
        return self.filter_outliers(final_lst) # Song song

    def organize_results_by_date(self, final_lst_input, original_data_input, dates_input, file_sources_input, time_of_day, profile):
        if final_lst_input is None or not dates_input:
            print(f"Error: Invalid inputs to organize_results for {time_of_day}.")
            return []
        
        # Sắp xếp dữ liệu theo ngày tháng trước khi nhóm
        # Điều này quan trọng để đảm bảo các chỉ số (indices) là chính xác khi nhóm
        sorted_indices = np.argsort(dates_input)
        
        final_lst_all = final_lst_input[sorted_indices]
        original_data_all = original_data_input[sorted_indices] if original_data_input is not None else np.full_like(final_lst_all, np.nan)
        dates_all = [dates_input[i] for i in sorted_indices]
        file_sources_all = [file_sources_input[i] for i in sorted_indices] # file_sources_input là tên nguồn (Terra/Aqua_Day/Night)

        unique_dates_dict = {}
        for i, date_obj in enumerate(dates_all):
            date_str = date_obj.strftime('%Y-%m-%d')
            if date_str not in unique_dates_dict:
                unique_dates_dict[date_str] = {'date_obj': date_obj, 'indices': []}
            unique_dates_dict[date_str]['indices'].append(i) # i bây giờ là chỉ số trong mảng đã sắp xếp

        print(f"Found {len(unique_dates_dict)} unique dates for {time_of_day}")
        organized_results = []
        for date_str, info in unique_dates_dict.items():
            unique_date = info['date_obj']
            current_indices_in_sorted_array = info['indices']
            
            # Gọi combine_terra_aqua_data với các slice đã sắp xếp và chỉ số tương ứng
            combined_data, combined_original, sources = self.combine_terra_aqua_data(
                final_lst_all,  # Mảng 3D đầy đủ (đã sắp xếp)
                original_data_all, # Mảng 3D đầy đủ (đã sắp xếp)
                current_indices_in_sorted_array, # List các chỉ số cho ngày này trong mảng đã sắp xếp
                file_sources_all # List đầy đủ các nguồn (đã sắp xếp)
            )
            
            if combined_data is not None and not np.all(np.isnan(combined_data)):
                self.save_and_visualize_result(unique_date, combined_data, combined_original, sources, time_of_day, profile)
                organized_results.append((unique_date, combined_data, combined_original, sources))
            else:
                print(f"Skipping save for {date_str} ({time_of_day}) as combined data is None or all NaN.")
        return organized_results

    def save_and_visualize_result(self, date_obj, combined_data, combined_original, sources, time_of_day, profile):
        # Giữ nguyên, việc ghi file thường bị giới hạn bởi I/O, song song hóa ít lợi hơn.
        if combined_data is None: print(f"Skipping save for {date_obj} {time_of_day}, data is None."); return
        date_file_str = date_obj.strftime('%Y_%m_%d')
        processed_filename = f'Final_LST_{time_of_day}_{date_file_str}.tif'
        original_filename = f'Original_LST_{time_of_day}_{date_file_str}.tif'
        processed_output_path = os.path.join(self.config.FINAL_OUTPUT_DIR, processed_filename)
        original_output_path = os.path.join(self.config.ORIGINAL_OUTPUT_DIR, original_filename)

        def write_tif(output_path, data_to_save, base_profile, data_type_tag, source_tags):
            if data_to_save is None or (isinstance(data_to_save, np.ndarray) and data_to_save.size == 0):
                print(f"Skipping TIF save for {output_path}, data is None or empty.")
                return
            
            is_all_nan = np.all(np.isnan(data_to_save))
            if is_all_nan: print(f"Warning: Data for {output_path} is all NaN. Saving all-NaN TIF.")
            
            min_val = np.nanmin(data_to_save) if not is_all_nan else np.nan
            max_val = np.nanmax(data_to_save) if not is_all_nan else np.nan
            mean_val = np.nanmean(data_to_save) if not is_all_nan else np.nan
            
            print(f"Stats for {os.path.basename(output_path)}: Min={min_val:.2f}, Max={max_val:.2f}, Mean={mean_val:.2f}")
            
            save_prof = base_profile.copy()
            save_prof.update(dtype=rasterio.float32, count=1, nodata=np.nan, 
                             width=data_to_save.shape[1], height=data_to_save.shape[0]) # Đảm bảo width, height đúng
            try:
                with rasterio.open(output_path, 'w', **save_prof) as dst:
                    dst.write(data_to_save.astype(rasterio.float32), 1)
                    tags = {'TEMPERATURE_UNIT': 'Celsius', 'MIN_TEMP': str(min_val), 'MAX_TEMP': str(max_val), 
                            'MEAN_TEMP': str(mean_val), 'DATA_TYPE': data_type_tag, 
                            'SOURCE': '+'.join(sorted(list(set(source_tags)))) if source_tags else 'Unknown'}
                    filtered_tags = {k: v for k, v in tags.items() if str(v).lower() != 'nan'}
                    dst.update_tags(**filtered_tags)
                print(f"Saved: {output_path}")
            except Exception as e: 
                print(f"Error saving TIF {output_path}: {e}")

        write_tif(processed_output_path, combined_data, profile, 'LST_gapfilled_combined', sources)
        if combined_original is not None:
            write_tif(original_output_path, combined_original, profile, 'LST_original_combined', sources)
        else:
            print(f"Skipped saving original combined data as it was None for {date_obj} {time_of_day}.")

    def process_time_series(self, start_date=None, end_date=None):
        start_date = start_date or self.config.START_DATE
        end_date = end_date or self.config.END_DATE
        
        # Giải phóng bộ nhớ trước khi bắt đầu xử lý
        gc.collect()
        
        log_message(f"\nLOADING DATA: {start_date} to {end_date}", force=True)
        day_files, night_files = self.load_modis_data(start_date, end_date)
        
        log_message("\nSYSTEM RESOURCE USAGE:", force=True)
        self._print_resource_usage()
        
        day_results, night_results = None, None
        
        if day_files:
            log_message("\n--- Processing DAY data ---", force=True)
            day_results = self.process_files(day_files, 'Day')
            # Đảm bảo giải phóng bộ nhớ sau khi xử lý xong dữ liệu ban ngày
            gc.collect()
            if self.use_gpu:
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                    log_message("GPU memory cleared after day processing")
                except:
                    pass
        else:
            log_message("No day files to process.", force=True)
            
        if night_files:
            log_message("\n--- Processing NIGHT data ---", force=True)
            night_results = self.process_files(night_files, 'Night')
            gc.collect()
            if self.use_gpu:
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                    log_message("GPU memory cleared after night processing")
                except:
                    pass
        else:
            log_message("No night files to process.", force=True)
            
        log_message("\nFINAL RESOURCE USAGE:", force=True)
        self._print_resource_usage()
            
        log_message(f"\nProcessing complete. Results (if any) saved in {self.config.OUTPUT_DIR}", force=True)
        return day_results, night_results
    
    def _print_resource_usage(self):
        """In thông tin về tài nguyên đang được sử dụng."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.5)
            
            # Memory usage
            mem = psutil.virtual_memory()
            mem_used_gb = mem.used / (1024 ** 3)
            mem_total_gb = mem.total / (1024 ** 3)
            mem_percent = mem.percent
            
            log_message(f"CPU Usage: {cpu_percent}%", force=True)
            log_message(f"RAM Usage: {mem_used_gb:.2f}GB / {mem_total_gb:.2f}GB ({mem_percent}%)", force=True)
            
            # GPU memory if available
            if self.use_gpu:
                try:
                    gpu_mem_used = cp.cuda.Device().mem_info[1]
                    gpu_mem_total = cp.cuda.Device().mem_info[0]
                    gpu_percent = (gpu_mem_used / gpu_mem_total) * 100
                    log_message(f"GPU Memory: {gpu_mem_used/(1024**3):.2f}GB / {gpu_mem_total/(1024**3):.2f}GB ({gpu_percent:.1f}%)", force=True)
                except:
                    log_message("Could not get GPU memory info")
        except:
            log_message("Could not get resource usage information")

# Hàm chính để chạy quy trình xử lý
def main():
    # Quan trọng cho multiprocessing trên Windows
    multiprocessing.freeze_support()
    
    try:
        # Đặt phương thức khởi tạo process phù hợp với hệ thống
        # 'spawn' là cách an toàn nhất cho Windows, sẽ tránh clone toàn bộ process
        multiprocessing.set_start_method('spawn', force=True)
        log_message("Using 'spawn' multiprocessing start method", force=True)
    except RuntimeError:
        # Nếu start method đã được đặt từ trước
        log_message(f"Using existing multiprocessing start method: {multiprocessing.get_start_method()}", force=True)
    
    # Thiết lập để sử dụng tối đa CPU cores mà không gây quá tải
    os.environ['OPENBLAS_NUM_THREADS'] = str(max(1, os.cpu_count() // 2))
    os.environ['MKL_NUM_THREADS'] = str(max(1, os.cpu_count() // 2))
    
    # Cài đặt numba nếu có
    if HAS_NUMBA:
        try:
            # Tắt JIT warning và thông tin debug
            os.environ['NUMBA_WARNINGS'] = '0'
            os.environ['NUMBA_DEBUG'] = '0'
            # Tắt việc quảng bá tính năng
            if hasattr(numba, 'config'):
                numba.config.ENABLE_CUDASIM = 0
        except:
            pass
    
    # Tối ưu hóa cấu hình theo tài nguyên hệ thống
    config = Config.optimize_for_system()
    config.initialize_dirs()
    
    # Kiểm tra và in thông tin hệ thống
    log_message("\n=== SYSTEM INFORMATION ===", force=True)
    log_message(f"CPU: {psutil.cpu_count(logical=True)} logical cores, {psutil.cpu_count(logical=False)} physical cores", force=True)
    mem = psutil.virtual_memory()
    log_message(f"RAM: {mem.total / (1024**3):.2f} GB total", force=True)
    log_message(f"GPU: {'Available' if HAS_GPU else 'Not available'}", force=True)
    
    # Hiển thị thêm thông tin về tối ưu hóa, nhưng chỉ một lần
    log_message(f"Numba optimization: {'Available' if HAS_NUMBA else 'Not available'}", force=True)
    
    # Khởi tạo processor với cấu hình đã tối ưu
    processor = MODISLSTProcessor(config=config)
    
    log_message(f"\nStarting LST gap filling: {config.START_DATE} to {config.END_DATE}", force=True)
    # Không lặp lại thông báo về số workers và GPU
    
    # Chạy quá trình xử lý
    day_results, night_results = processor.process_time_series()
    
    if not day_results and not night_results:
        log_message("No results generated.", force=True)
    else:
        log_message("Gap filling completed.", force=True)
    log_message(f"For visualization, consider MODISGapfilling/visualizer.py.", force=True)

if __name__ == "__main__":
    main()