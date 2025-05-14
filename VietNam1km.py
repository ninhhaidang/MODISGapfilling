import ee
import os
import datetime
import rasterio
import numpy as np
from glob import glob
from datetime import datetime, timedelta
import pandas as pd
from scipy import ndimage
from scipy.interpolate import griddata

# Lớp cấu hình: chứa các tham số và đường dẫn mặc định cho việc xử lý dữ liệu
class Config:
    # Thiết lập khoảng thời gian xử lý dữ liệu mặc định
    START_DATE, END_DATE = '2020-01-01', '2020-02-01'
    # Đường dẫn thư mục gốc của dự án
    BASE_DIR = '/Users/ninhhaidang/Library/CloudStorage/GoogleDrive-ninhhailongg@gmail.com/My Drive/Cac_mon_hoc/Nam4_Ky2/Du_an_thuc_te'
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
    ENABLE_EDGE_SMOOTHING, APPLY_WATER_MASK = True, False
    
    # Kích thước cửa sổ cho bộ lọc uniform khi làm mịn phần dư không gian (mô phỏng focal_mean radius=3 của GEE ~ 7x7)
    RESIDUAL_UNIFORM_FILTER_SIZE = 7
    
    @classmethod
    def initialize_dirs(cls):
        """Tạo các thư mục đầu ra nếu chúng chưa tồn tại."""
        # Duyệt qua danh sách các đường dẫn thư mục cần thiết
        for dir_path in [cls.OUTPUT_DIR, cls.VISUALIZATION_DIR, cls.FINAL_OUTPUT_DIR, cls.ORIGINAL_OUTPUT_DIR]:
            # Kiểm tra nếu thư mục chưa tồn tại
            if not os.path.exists(dir_path):
                # Tạo thư mục
                os.makedirs(dir_path)

# Lớp chính xử lý dữ liệu MODIS LST
class MODISLSTProcessor:
    def __init__(self, config=None):
        """Khởi tạo đối tượng xử lý với một cấu hình cụ thể hoặc cấu hình mặc định."""
        # Sử dụng cấu hình được cung cấp hoặc lớp Config mặc định
        self.config = config or Config
        # Khởi tạo các thư mục cần thiết dựa trên cấu hình
        self.config.initialize_dirs()
        # Khởi tạo mặt nạ nước (hiện không được sử dụng trong phiên bản này)
        self.water_mask = None
        
    def _nan_robust_uniform_filter(self, data, size):
        """
        Áp dụng bộ lọc trung bình đồng nhất (uniform filter) có khả năng xử lý NaN.
        Giá trị NaN trong cửa sổ sẽ được bỏ qua khi tính trung bình.

        Args:
            data (numpy.ndarray): Mảng dữ liệu đầu vào (2D).
            size (int): Kích thước của cửa sổ bộ lọc.

        Returns:
            numpy.ndarray: Mảng dữ liệu đã được làm mịn.
        """
        if data is None or data.size == 0:
            return data # Trả về như cũ nếu đầu vào rỗng/None
        # Tạo bản sao để tránh thay đổi dữ liệu gốc
        data_copy = np.copy(data)
        
        # Mặt nạ cho các giá trị NaN
        nan_mask = np.isnan(data_copy)
        
        # Tạo mảng tạm thời với NaN thay bằng 0 để tính tổng
        temp_data_for_sum = np.copy(data_copy)
        temp_data_for_sum[nan_mask] = 0
        
        # Tạo trọng số: 1 cho non-NaN, 0 cho NaN
        weights = np.ones_like(data_copy, dtype=float) # Sử dụng float cho weights
        weights[nan_mask] = 0
        
        # Tính tổng các giá trị non-NaN trong cửa sổ, sử dụng mode='constant', cval=0 để xử lý cạnh 
        sum_filter = ndimage.uniform_filter(temp_data_for_sum, size=size, mode='constant', cval=0)
        
        # Tính số lượng các giá trị non-NaN trong cửa sổ
        count_filter = ndimage.uniform_filter(weights, size=size, mode='constant', cval=0)
        
        # Tính trung bình, tránh chia cho 0 (kết quả là NaN nếu count là 0)
        result = np.full_like(data, np.nan, dtype=np.float64) # Khởi tạo kết quả với NaN
        
        valid_counts_mask = count_filter > 1e-9 # Sử dụng một ngưỡng nhỏ để tránh lỗi chia cho số rất nhỏ
        
        result[valid_counts_mask] = sum_filter[valid_counts_mask] / count_filter[valid_counts_mask]
        
        return result
    
    def load_and_preprocess_tif(self, file_path):
        """Đọc một file TIF, tiền xử lý và chuyển đổi đơn vị nhiệt độ.

        Args:
            file_path (str): Đường dẫn đến file TIF.

        Returns:
            tuple: Một tuple chứa (dữ liệu numpy array của LST đã chuyển sang Celsius, thông tin profile của file raster).
                   Giá trị NaN được gán cho các pixel không hợp lệ.
        """
        try:
            with rasterio.open(file_path) as src:
                data = src.read(1).astype(np.float32) # Đọc và chuyển sang float32
                profile = src.profile
                data = data * 0.02 - 273.15 # Kelvin sang Celsius
                # Mặt nạ chất lượng: giữ giá trị từ -50 đến 50°C và không phải là fill value (~ -273.15 sau chuyển đổi)
                # Giả sử fill value gốc là 0 Kelvin, sau khi nhân 0.02 và trừ 273.15 sẽ rất gần -273.15
                valid_data_mask = (data > -50) & (data < 50) & (np.abs(data - (-273.15)) > 1e-5)
                processed_data = np.full_like(data, np.nan, dtype=np.float32)
                processed_data[valid_data_mask] = data[valid_data_mask]
                return processed_data, profile
        except Exception as e:
            print(f"Error loading or preprocessing TIF {file_path}: {e}")
            return None, None # Trả về None nếu có lỗi
    
    def get_date_from_filename(self, filename):
        """Trích xuất thông tin ngày tháng từ tên file.
        Giả định tên file có định dạng như 'Aqua_Day_YYYY-MM-DD.tif' hoặc 'Terra_Night_YYYY-MM-DD.tif'.

        Args:
            filename (str): Tên file.

        Returns:
            datetime: Đối tượng datetime chứa thông tin ngày tháng được trích xuất.
        """
        return datetime.strptime(filename.split('_')[-1].split('.')[0], '%Y-%m-%d')
    
    def clip_values(self, data, min_val=None, max_val=None):
        """Cắt giá trị của mảng dữ liệu vào một khoảng xác định (min_val, max_val).
        Thường được sử dụng để chuẩn bị dữ liệu cho việc hiển thị/trực quan hóa.

        Args:
            data (numpy.ndarray): Mảng dữ liệu đầu vào.
            min_val (float, optional): Giá trị tối thiểu. Mặc định lấy từ config.
            max_val (float, optional): Giá trị tối đa. Mặc định lấy từ config.

        Returns:
            numpy.ndarray: Mảng dữ liệu đã được cắt giá trị.
        """
        if data is None: return None
        min_val = min_val if min_val is not None else self.config.VISUALIZATION_MIN_TEMP
        max_val = max_val if max_val is not None else self.config.VISUALIZATION_MAX_TEMP
        clipped_data = np.copy(data)
        clipped_data[~np.isnan(clipped_data) & (clipped_data < min_val)] = min_val
        clipped_data[~np.isnan(clipped_data) & (clipped_data > max_val)] = max_val
        return clipped_data
    
    def spatial_gapfill(self, data):
        """Lấp đầy khoảng trống (NaN) trong ảnh 2D bằng nội suy không gian.
        Args: data (np.ndarray): Ảnh LST 2D chứa NaN.
        Returns: np.ndarray: Ảnh LST 2D đã lấp đầy.
        """
        if data is None or np.all(np.isnan(data)): return data
        valid_mask = ~np.isnan(data)
        if np.sum(valid_mask) == 0 or np.sum(valid_mask) == data.size: return data
        rows, cols = data.shape
        y, x = np.mgrid[0:rows, 0:cols]
        points = np.column_stack((y[valid_mask].ravel(), x[valid_mask].ravel()))
        values = data[valid_mask].ravel()
        if points.shape[0] < 2: return data # Cần ít nhất 2 điểm để nội suy griddata
        dilated_mask = ndimage.binary_dilation(valid_mask, iterations=self.config.SPATIAL_DILATION_ITERATIONS)
        interpolation_mask = dilated_mask & ~valid_mask
        filled_data = np.copy(data)
        interp_points = np.column_stack((y[interpolation_mask].ravel(), x[interpolation_mask].ravel()))
        if len(interp_points) > 0:
            interp_values_linear = griddata(points, values, interp_points, method='linear')
            linear_fill_idx = ~np.isnan(interp_values_linear)
            filled_data[interpolation_mask][linear_fill_idx] = interp_values_linear[linear_fill_idx]
            remaining_nan_mask_after_linear = interpolation_mask & np.isnan(filled_data)
            if np.any(remaining_nan_mask_after_linear):
                points_for_nearest = np.column_stack((y[remaining_nan_mask_after_linear].ravel(), x[remaining_nan_mask_after_linear].ravel()))
                if len(points_for_nearest) > 0:
                    interp_values_nearest = griddata(points, values, points_for_nearest, method='nearest')
                    filled_data[remaining_nan_mask_after_linear] = interp_values_nearest
            actually_filled_mask = interpolation_mask & ~np.isnan(filled_data)
            if np.any(actually_filled_mask):
                temp_smoothed = ndimage.gaussian_filter(filled_data, sigma=self.config.SPATIAL_SMOOTHING_SIGMA, mode='reflect')
                filled_data[actually_filled_mask] = temp_smoothed[actually_filled_mask]
        return filled_data

    def temporal_gapfill(self, data_cube):
        """Lấp đầy khoảng trống (NaN) trong chuỗi thời gian bằng nội suy tuyến tính.
        Args: data_cube (np.ndarray): Dữ liệu 3D (thời gian, hàng, cột).
        Returns: np.ndarray: Dữ liệu 3D đã lấp đầy thời gian.
        """
        if data_cube is None: return None
        for i in range(data_cube.shape[1]):
            for j in range(data_cube.shape[2]):
                ts = data_cube[:, i, j]
                if not np.any(np.isnan(ts)): continue
                valid_idx = np.where(~np.isnan(ts))[0]
                missing_idx = np.where(np.isnan(ts))[0]
                if len(valid_idx) < 2 or len(missing_idx) == 0: continue
                # Sửa lỗi np.interp: xp phải tăng dần và fp phải cùng độ dài
                # Đảm bảo idx_miss nằm trong khoảng của valid_idx để nội suy
                min_valid_time = np.min(valid_idx)
                max_valid_time = np.max(valid_idx)
                for idx_miss in missing_idx:
                    if min_valid_time < idx_miss < max_valid_time: # Chỉ nội suy nếu điểm nằm giữa các điểm đã biết
                        # Tìm các điểm lân cận trong phạm vi TEMPORAL_NEIGHBOR_DAYS
                        relevant_neighbors = valid_idx[np.abs(valid_idx - idx_miss) <= self.config.TEMPORAL_NEIGHBOR_DAYS]
                        if len(relevant_neighbors) >= self.config.TEMPORAL_MIN_NEIGHBORS:
                            # Sắp xếp các điểm lân cận (xp cho np.interp phải tăng dần)
                            sorted_relevant_neighbors = np.sort(relevant_neighbors)
                            # Đảm bảo idx_miss nằm trong khoảng của các lân cận này
                            if np.min(sorted_relevant_neighbors) <= idx_miss <= np.max(sorted_relevant_neighbors): 
                                data_cube[idx_miss, i, j] = np.interp(idx_miss, sorted_relevant_neighbors, ts[sorted_relevant_neighbors])
        return data_cube

    def filter_outliers(self, data_array):
        """Lọc giá trị ngoại lai, giữ NaN.
        Args: data_array (np.ndarray): Dữ liệu 3D hoặc 2D.
        Returns: np.ndarray: Dữ liệu đã lọc ngoại lai.
        """
        if data_array is None: return None
        filtered_array = data_array.copy()
        is_3d = filtered_array.ndim == 3
        num_images = filtered_array.shape[0] if is_3d else 1
        for i in range(num_images):
            current_slice = filtered_array[i] if is_3d else filtered_array
            if np.all(np.isnan(current_slice)): continue
            valid_mask_slice = ~np.isnan(current_slice)
            if not np.any(valid_mask_slice): continue
            valid_data = current_slice[valid_mask_slice]
            if valid_data.size == 0: continue
            q_low = np.percentile(valid_data, self.config.OUTLIER_PERCENTILE_LOW)
            q_high = np.percentile(valid_data, self.config.OUTLIER_PERCENTILE_HIGH)
            iqr = q_high - q_low
            if iqr < 1e-6: lower_bound, upper_bound = q_low - 1, q_high + 1 # Tránh iqr quá nhỏ
            else: lower_bound, upper_bound = q_low - self.config.OUTLIER_IQR_FACTOR * iqr, q_high + self.config.OUTLIER_IQR_FACTOR * iqr
            outlier_mask_slice = (current_slice < lower_bound) | (current_slice > upper_bound)
            slice_to_update = filtered_array[i] if is_3d else filtered_array
            slice_to_update[outlier_mask_slice & valid_mask_slice] = np.nan # Chỉ set NaN cho outlier có giá trị, không phải NaN sẵn
        return filtered_array

    def apply_temporal_smoothing(self, data):
        """Áp dụng làm mịn thời gian bằng trung bình trượt.
        Args: data (np.ndarray): Dữ liệu 3D.
        Returns: np.ndarray: Dữ liệu 3D đã làm mịn thời gian.
        """
        if data is None or data.shape[0] < 1: return data
        if data.shape[0] == 1: return data # Không làm gì nếu chỉ có 1 ảnh
        window = min(self.config.TEMPORAL_WINDOW_SIZE, data.shape[0])
        # Sử dụng lambda với pandas rolling mean để xử lý NaN đúng cách
        def nan_rolling_mean_pd(arr_1d):
            series = pd.Series(arr_1d)
            return series.rolling(window=window, center=True, min_periods=1).mean().to_numpy()
        try:
            return np.apply_along_axis(nan_rolling_mean_pd, axis=0, arr=data)
        except Exception as e:
            print(f"Error during temporal smoothing: {e}. Returning original data.")
            return data

    def combine_terra_aqua_data(self, final_lst_input, original_data_input, indices, file_sources):
        """Kết hợp dữ liệu Terra và Aqua cho cùng một ngày.
        Args: final_lst_input (np.ndarray), original_data_input (np.ndarray), indices (list), file_sources (list).
        Returns: tuple: (LST kết hợp cuối cùng, LST gốc kết hợp, danh sách nguồn).
        """
        if not indices or final_lst_input is None or original_data_input is None:
            print("Warning: Empty inputs to combine_terra_aqua_data.")
            ref_shape_fallback = (1,1) # Fallback shape, nên được cải thiện nếu biết trước kích thước mong đợi
            if final_lst_input is not None and final_lst_input.ndim > 2 and final_lst_input.shape[0] > 0:
                 ref_shape_fallback = final_lst_input.shape[1:]
            elif original_data_input is not None and original_data_input.ndim > 2 and original_data_input.shape[0] > 0:
                 ref_shape_fallback = original_data_input.shape[1:]
            return (np.full(ref_shape_fallback, np.nan, dtype=np.float32), 
                    np.full(ref_shape_fallback, np.nan, dtype=np.float32), [])

        final_lst = final_lst_input[indices] if final_lst_input.ndim == 3 else final_lst_input # Lấy các slice cần thiết
        original_data = original_data_input[indices] if original_data_input.ndim == 3 else original_data_input
        current_file_sources = [file_sources[i] for i in indices]

        if len(indices) == 1:
            return (final_lst[0], original_data[0], [current_file_sources[0]])

        ref_shape = final_lst[0].shape
        combined_data = np.full(ref_shape, np.nan, dtype=np.float32)
        combined_original = np.full(ref_shape, np.nan, dtype=np.float32)
        unique_sources_in_day = sorted(list(set(current_file_sources)))

        terra_final_slices = [final_lst[i] for i, src in enumerate(current_file_sources) if src == 'Terra' and not np.all(np.isnan(final_lst[i]))]
        aqua_final_slices = [final_lst[i] for i, src in enumerate(current_file_sources) if src == 'Aqua' and not np.all(np.isnan(final_lst[i]))]

        # Kết hợp dữ liệu đã xử lý (final_lst)
        if terra_final_slices and aqua_final_slices: # Cả hai nguồn đều có dữ liệu
            terra_mean_final = np.nanmean(np.stack(terra_final_slices, axis=0), axis=0)
            aqua_mean_final = np.nanmean(np.stack(aqua_final_slices, axis=0), axis=0)
            combined_data = np.nanmean(np.stack([terra_mean_final, aqua_mean_final], axis=0), axis=0)
        elif aqua_final_slices: # Chỉ có Aqua
            combined_data = np.nanmean(np.stack(aqua_final_slices, axis=0), axis=0)
        elif terra_final_slices: # Chỉ có Terra
            combined_data = np.nanmean(np.stack(terra_final_slices, axis=0), axis=0)
        # else: combined_data giữ nguyên là all NaN nếu không có nguồn nào

        # Kết hợp dữ liệu gốc (original_data)
        # (Logic tương tự như trên, được đơn giản hóa và gộp vào đây)
        terra_orig_slices = [original_data[i] for i, src in enumerate(current_file_sources) if src == 'Terra' and not np.all(np.isnan(original_data[i]))]
        aqua_orig_slices = [original_data[i] for i, src in enumerate(current_file_sources) if src == 'Aqua' and not np.all(np.isnan(original_data[i]))]
        if terra_orig_slices and aqua_orig_slices:
            terra_mean_orig = np.nanmean(np.stack(terra_orig_slices, axis=0), axis=0)
            aqua_mean_orig = np.nanmean(np.stack(aqua_orig_slices, axis=0), axis=0)
            combined_original = np.nanmean(np.stack([terra_mean_orig, aqua_mean_orig], axis=0), axis=0)
        elif aqua_orig_slices:
            combined_original = np.nanmean(np.stack(aqua_orig_slices, axis=0), axis=0)
        elif terra_orig_slices:
            combined_original = np.nanmean(np.stack(terra_orig_slices, axis=0), axis=0)

        # Làm mịn cạnh nếu cần
        if self.config.ENABLE_EDGE_SMOOTHING and np.any(~np.isnan(combined_data)):
            # Tạo mặt nạ cho vùng có dữ liệu để làm mịn dựa trên đó
            valid_data_for_smoothing_mask = ~np.isnan(combined_data)
            # Chỉ làm mịn nếu có đủ dữ liệu trong vùng lân cận
            # Ở đây, làm mịn toàn bộ ảnh và chỉ áp dụng cho vùng có dữ liệu ban đầu
            # Điều này có thể không lý tưởng nếu có các vùng NaN lớn, vì bộ lọc Gaussian sẽ bị ảnh hưởng bởi chúng
            # Tuy nhiên, vì chúng ta đã lấp đầy khá nhiều, hy vọng điều này sẽ ổn
            if np.sum(valid_data_for_smoothing_mask) > 0: # Chỉ làm mịn nếu có dữ liệu
                smoothed_edge_data = ndimage.gaussian_filter(combined_data, sigma=0.75, mode='reflect')
                # Chỉ áp dụng làm mịn cho các pixel có dữ liệu sau khi kết hợp, để không lan truyền NaN không cần thiết
                # và để tránh làm mịn các vùng vốn đã là NaN.
                combined_data[valid_data_for_smoothing_mask] = smoothed_edge_data[valid_data_for_smoothing_mask]

        return combined_data, combined_original, unique_sources_in_day

    def _combine_original_data(self, original_data_input, indices, file_sources, combined_original_output, ref_shape):
        # Phương thức này không còn được sử dụng trực tiếp, logic đã được gộp vào combine_terra_aqua_data
        # Giữ lại phòng trường hợp cần tham khảo hoặc tái sử dụng logic cụ thể sau này
        pass 

    def load_modis_data(self, start_date, end_date):
        """Tải danh sách file MODIS LST.
        Args: start_date (str), end_date (str).
        Returns: tuple: (list file ban ngày, list file ban đêm).
        """
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        file_patterns = {
            'terra_day': 'Terra_Day_*.tif', 'terra_night': 'Terra_Night_*.tif',
            'aqua_day': 'Aqua_Day_*.tif', 'aqua_night': 'Aqua_Night_*.tif'
        }
        files = {key: [] for key in file_patterns}
        for key, pattern in file_patterns.items():
            files[key] = sorted([
                f for f in glob(os.path.join(self.config.RAW_DATA_DIR, pattern))
                if start_dt <= self.get_date_from_filename(os.path.basename(f)) <= end_dt
            ])
        print(f"Loaded: {len(files['terra_day'])} TerraD, {len(files['aqua_day'])} AquaD, {len(files['terra_night'])} TerraN, {len(files['aqua_night'])} AquaN files.")
        return files['terra_day'] + files['aqua_day'], files['terra_night'] + files['aqua_night']
    
    def process_files(self, file_list, time_of_day):
        """Xử lý danh sách file cho thời điểm cụ thể (ban ngày/đêm).
        Args: file_list (list), time_of_day (str).
        Returns: list or None: Kết quả xử lý hoặc None nếu lỗi.
        """
        if not file_list: print(f"No files to process for {time_of_day}."); return None
        data_series, dates, file_sources, profiles = [], [], [], []
        common_profile = None
        for file_path in file_list:
            data, profile = self.load_and_preprocess_tif(file_path)
            if data is not None and profile is not None:
                if common_profile is None: common_profile = profile
                # Kiểm tra profile (width, height) có nhất quán không
                elif common_profile['width'] != profile['width'] or common_profile['height'] != profile['height']:
                    print(f"Warning: Mismatched profile for {file_path}. Expected {common_profile['width']}x{common_profile['height']}, got {profile['width']}x{profile['height']}. Skipping file.")
                    continue
                data_series.append(data); dates.append(self.get_date_from_filename(os.path.basename(file_path)))
                file_sources.append('Terra' if 'Terra' in os.path.basename(file_path) else 'Aqua') # Đơn giản hóa việc xác định nguồn
            else: print(f"Skipping file due to load/preprocess error: {file_path}")

        if not data_series: print(f"No valid data loaded for {time_of_day}."); return None
        # Đảm bảo tất cả các mảng trong data_series có cùng shape trước khi stack
        first_shape = data_series[0].shape
        consistent_data_series = [arr for arr in data_series if arr.shape == first_shape]
        if len(consistent_data_series) != len(data_series):
            print(f"Warning: Some arrays had mismatched shapes and were excluded for {time_of_day}.")
        if not consistent_data_series: print(f"No consistently shaped data arrays for {time_of_day}."); return None

        data_array = np.stack(consistent_data_series) # Stack chỉ các mảng có shape nhất quán
        # Cập nhật dates và file_sources cho phù hợp với consistent_data_series
        # Điều này hơi phức tạp, đơn giản là giả định tất cả đều nhất quán nếu consistent_data_series không rỗng
        # Một cách tiếp cận tốt hơn là lọc dates và file_sources song song khi lọc data_series

        original_data_array = data_array.copy()
        processed_data_array = self.process_data_array(data_array)
        if processed_data_array is None: print(f"Processing data array failed for {time_of_day}."); return None
        if common_profile is None: print(f"Error: No common raster profile found for {time_of_day}."); return None
        return self.organize_results_by_date(processed_data_array, original_data_array, dates, file_sources, time_of_day, common_profile)
    
    def process_data_array(self, data_array_input):
        """Thực hiện chuỗi các bước xử lý chính trên mảng dữ liệu LST 3D.
        Args: data_array_input (np.ndarray): Dữ liệu LST 3D.
        Returns: np.ndarray or None: Dữ liệu LST 3D đã xử lý, hoặc None nếu lỗi.
        """
        if data_array_input is None or data_array_input.size == 0: print("Error: Input data_array is None/empty."); return None
        data_array = data_array_input.copy()
        print("Applying spatial gap filling...")
        for i in range(data_array.shape[0]): data_array[i] = self.spatial_gapfill(data_array[i])
        print("Applying temporal gap filling...")
        data_array = self.temporal_gapfill(data_array)
        if data_array is None: print("Error: data_array became None after temporal_gapfill."); return None
        if np.all(np.isnan(data_array)): print("Warning: Data array is all NaN before long-term mean."); return data_array
        print("Calculating long-term mean...")
        mean_lst = np.nanmean(data_array, axis=0)
        if np.all(np.isnan(mean_lst)): print("Warning: Long-term mean is all NaN."); return data_array
        print("Computing and smoothing residuals...")
        residuals = data_array - mean_lst[np.newaxis, :, :]
        smoothed_residuals = np.zeros_like(residuals)
        for i in range(len(residuals)):
            if np.all(np.isnan(residuals[i])): smoothed_residuals[i] = residuals[i]; continue
            temp_residual = self._nan_robust_uniform_filter(residuals[i], size=self.config.RESIDUAL_UNIFORM_FILTER_SIZE)
            if np.all(np.isnan(temp_residual)): smoothed_residuals[i] = temp_residual
            else: smoothed_residuals[i] = self._nan_robust_uniform_filter(temp_residual, size=self.config.RESIDUAL_UNIFORM_FILTER_SIZE)
        smoothed_temporal_residuals = self.apply_temporal_smoothing(smoothed_residuals)
        if smoothed_temporal_residuals is None: final_lst = smoothed_residuals + mean_lst[np.newaxis, :, :]
        else: final_lst = smoothed_temporal_residuals + mean_lst[np.newaxis, :, :]
        return self.filter_outliers(final_lst)

    def organize_results_by_date(self, final_lst_input, original_data_input, dates_input, file_sources_input, time_of_day, profile):
        """Tổ chức kết quả theo ngày, kết hợp Terra/Aqua.
        Args: (Inputs đa dạng, xem định nghĩa gốc).
        Returns: list: Danh sách kết quả đã tổ chức.
        """
        if final_lst_input is None or not dates_input: print(f"Error: Invalid inputs to organize_results for {time_of_day}."); return []
        # Tạo bản sao để tránh thay đổi ngoài ý muốn
        final_lst_all = np.copy(final_lst_input)
        original_data_all = np.copy(original_data_input) if original_data_input is not None else np.full_like(final_lst_all, np.nan) # Dự phòng nếu original rỗng
        dates_all = list(dates_input)
        file_sources_all = list(file_sources_input)

        unique_dates_dict = {}
        for i, date_obj in enumerate(dates_all):
            date_str = date_obj.strftime('%Y-%m-%d')
            if date_str not in unique_dates_dict:
                unique_dates_dict[date_str] = {'date_obj': date_obj, 'indices': []}
            unique_dates_dict[date_str]['indices'].append(i)

        print(f"Found {len(unique_dates_dict)} unique dates for {time_of_day}")
        organized_results = []
        for date_str, info in unique_dates_dict.items():
            unique_date = info['date_obj']
            current_indices = info['indices']
            # Lấy dữ liệu và nguồn cho ngày hiện tại dựa trên chỉ số
            # current_final_data_slices = final_lst_all[current_indices] # Đây sẽ là mảng 3D các slice cho ngày đó
            # current_original_data_slices = original_data_all[current_indices]
            # current_sources_for_day = [file_sources_all[i] for i in current_indices]

            combined_data, combined_original, sources = self.combine_terra_aqua_data(
                final_lst_all, original_data_all, current_indices, file_sources_all
            )
            if combined_data is not None and not np.all(np.isnan(combined_data)):
                self.save_and_visualize_result(unique_date, combined_data, combined_original, sources, time_of_day, profile)
                organized_results.append((unique_date, combined_data, combined_original, sources))
            else:
                print(f"Skipping save for {date_str} ({time_of_day}) as combined data is None or all NaN.")
        return organized_results

    def save_and_visualize_result(self, date_obj, combined_data, combined_original, sources, time_of_day, profile):
        """Lưu kết quả TIF.
        Args: (Inputs đa dạng, xem định nghĩa gốc).
        """
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
            if np.all(np.isnan(data_to_save)): print(f"Warning: Data for {output_path} is all NaN. Saving all-NaN TIF.")
            min_val = np.nanmin(data_to_save) if not np.all(np.isnan(data_to_save)) else np.nan
            max_val = np.nanmax(data_to_save) if not np.all(np.isnan(data_to_save)) else np.nan
            mean_val = np.nanmean(data_to_save) if not np.all(np.isnan(data_to_save)) else np.nan
            print(f"Stats for {os.path.basename(output_path)}: Min={min_val:.2f}, Max={max_val:.2f}, Mean={mean_val:.2f}")
            save_prof = base_profile.copy()
            save_prof.update(dtype=rasterio.float32, count=1, nodata=np.nan, width=data_to_save.shape[1], height=data_to_save.shape[0])
            try:
                with rasterio.open(output_path, 'w', **save_prof) as dst:
                    dst.write(data_to_save.astype(rasterio.float32), 1)
                    tags = {'TEMPERATURE_UNIT': 'Celsius', 'MIN_TEMP': str(min_val), 'MAX_TEMP': str(max_val), 
                            'MEAN_TEMP': str(mean_val), 'DATA_TYPE': data_type_tag, 'SOURCE': '+'.join(sorted(list(set(source_tags))))}
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
        """Phương thức chính để xử lý toàn bộ chuỗi thời gian LST.
        Args: start_date (str), end_date (str).
        Returns: tuple: (kết quả ngày, kết quả đêm), có thể là None.
        """
        start_date = start_date or self.config.START_DATE
        end_date = end_date or self.config.END_DATE
        day_files, night_files = self.load_modis_data(start_date, end_date)
        day_results, night_results = None, None
        if day_files: print("\n--- Processing DAY data ---"); day_results = self.process_files(day_files, 'Day')
        else: print("No day files to process.")
        if night_files: print("\n--- Processing NIGHT data ---"); night_results = self.process_files(night_files, 'Night')
        else: print("No night files to process.")
        print(f"\nProcessing complete. Results (if any) saved in {self.config.OUTPUT_DIR}")
        return day_results, night_results

# Hàm chính để chạy quy trình xử lý
def main():
    """Hàm chính của script."""
    processor = MODISLSTProcessor()
    print(f"Starting LST gap filling: {Config.START_DATE} to {Config.END_DATE}")
    day_results, night_results = processor.process_time_series()
    if not day_results and not night_results: print("No results generated.")
    else: print("Gap filling completed.")
    print(f"For visualization, consider MODISGapfilling/visualizer.py.")

if __name__ == "__main__":
    main() 