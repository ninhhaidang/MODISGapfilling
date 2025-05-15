import ee
import os
import datetime
import rasterio
import numpy as np
from glob import glob
from datetime import datetime, timedelta
import pandas as pd
from scipy import ndimage
from scipy.interpolate import griddata, UnivariateSpline
from scipy.stats import pearsonr, linregress
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
# Remove ee_monitor import as it might not be installed
# import ee_monitor

# Comment out Earth Engine initialization for now
# ee.Initialize(project='ee-bonglantrungmuoi')

# Cấu hình mặc định
class Config:
    # Thiết lập thời gian
    START_DATE = '2020-01-01'
    END_DATE = '2020-07-01' # Adjust for longer periods for better stats
    
    # Đường dẫn thư mục
    RAW_DATA_DIR = '/Users/ninhhaidang/Library/CloudStorage/GoogleDrive-ninhhailongg@gmail.com/My Drive/Cac_mon_hoc/Nam4_Ky2/Du_an_thuc_te/MODIS_LST_VN_RAW'
    # TODO: Add path for QA data if separate files
    # RAW_QA_DATA_DIR = '/path/to/modis/qa/data' 
    OUTPUT_DIR = '/Users/ninhhaidang/Library/CloudStorage/GoogleDrive-ninhhailongg@gmail.com/My Drive/Cac_mon_hoc/Nam4_Ky2/Du_an_thuc_te/MODIS_LST_VN_PROCESSED_ADVANCED'
    VISUALIZATION_DIR = '/Users/ninhhaidang/Library/CloudStorage/GoogleDrive-ninhhailongg@gmail.com/My Drive/Cac_mon_hoc/Nam4_Ky2/Du_an_thuc_te/MODIS_LST_VN_VISUALIZATIONS_ADVANCED'
    
    # Tham số tiền xử lý
    QA_ERROR_THRESHOLD_KELVIN = 3.0
    PVD_THRESHOLD = 0.05 # 5%

    # Tham số nội suy không gian (cho ICW - simplified)
    SPATIAL_NEIGHBOR_WINDOW_SIZE = 3 # e.g., 3x3 window for neighbors in simplified ICW
    
    # Tham số nội suy thời gian (spline)
    SPLINE_SMOOTHING_FACTOR = None # Let UnivariateSpline estimate, or set a value e.g., len(data) * 0.1

    # Tham số hiển thị
    VISUALIZATION_MIN_TEMP = -5
    VISUALIZATION_MAX_TEMP = 35
    COLOR_MAP_COLORS = ['blue', 'cyan', 'green', 'yellow', 'orange', 'red']
    COLOR_MAP_BINS = 100
    
    # Cờ điều khiển
    APPLY_WATER_MASK = True
    
    @classmethod
    def initialize_dirs(cls):
        """Tạo các thư mục đầu ra nếu chưa tồn tại"""
        if not os.path.exists(cls.OUTPUT_DIR):
            os.makedirs(cls.OUTPUT_DIR)
        if not os.path.exists(cls.VISUALIZATION_DIR):
            os.makedirs(cls.VISUALIZATION_DIR)

# Lớp xử lý dữ liệu MODIS LST nâng cao
class MODISLSTProcessorAdvanced:
    def __init__(self, config=None):
        self.config = config or Config
        self.config.initialize_dirs()

    def get_date_from_filename(self, filename):
        """Trích xuất ngày từ tên file format Platform_Time_YYYY-MM-DD.tif"""
        # Assuming filename format like 'Terra_Day_2020-01-01.tif' or 'Aqua_Night_2020-01-01.tif'
        parts = os.path.basename(filename).split('_')
        return datetime.strptime(parts[-1].split('.')[0], '%Y-%m-%d')

    def get_platform_time_from_filename(self, filename):
        """Trích xuất platform và time_of_day từ tên file."""
        # e.g., Terra_Day_2020-01-01.tif -> ("Terra", "Day")
        basename = os.path.basename(filename)
        parts = basename.split('_')
        if len(parts) >= 2:
            return parts[0], parts[1] # Platform, Time (Day/Night)
        return "Unknown", "Unknown"

    def load_and_preprocess_tif_with_qa(self, lst_file_path, qa_file_path=None):
        """Đọc file LST TIF, chuyển đổi sang Celsius, và áp dụng lọc QA."""
        with rasterio.open(lst_file_path) as src:
            data = src.read(1).astype(np.float32)
            profile = src.profile
            
            # Chuyển từ Kelvin sang Celsius (MODIS LST scale factor 0.02)
            data = data * 0.02 - 273.15
            
            # Placeholder cho việc đọc và áp dụng QA
            # Giả sử QA data được cung cấp và giá trị QA nhỏ hơn ngưỡng là tốt
            # Ví dụ: qa_value <= Config.QA_ERROR_THRESHOLD_KELVIN (cần logic cụ thể dựa trên format QA)
            # Đây là một bước quan trọng và cần file QA tương ứng.
            # For now, we'll use a simple validity check like in the original script.
            # In a real implementation, QA flags (e.g., from MOD11_L2 or MYD11_L2 QA bands)
            # would be used to identify LST error estimations.
            # Errors > 3K are filtered.
            # This requires interpreting QA bits.
            
            # Simple validity mask (replace with actual QA processing)
            # For example, if QA data indicates error in Kelvin:
            # error_k = ... (derived from QA band) ...
            # quality_mask = error_k <= self.config.QA_ERROR_THRESHOLD_KELVIN
            
            # Fallback to simple range check if QA not implemented
            quality_mask = (data > -60) & (data < 70) # Generous range, QA is better
            data[~quality_mask] = np.nan
            
            return data, profile

    def _calculate_pvd(self, data_series, expected_days_in_year=365):
        """
        Tính toán Tỷ lệ dữ liệu hợp lệ (PVD) cho một chuỗi thời gian.
        data_series: list of 2D numpy arrays (hoặc 3D stack)
        expected_days_in_year: tổng số ngày kỳ vọng trong giai đoạn (để tính tỷ lệ)
        Returns: PVD (float), or 2D array of PVDs if data_series is many pixels over time.
        NOTE: This is a simplified PVD for the loaded series. True PVD needs a full year.
        """
        if not data_series:
            return 0.0
        
        data_stack = np.stack(data_series) if isinstance(data_series, list) else data_series
        valid_pixels_per_image = np.sum(~np.isnan(data_stack), axis=(1,2)) # Sum over spatial dims
        total_pixels_per_image = data_stack.shape[1] * data_stack.shape[2]
        
        # PVD for the entire image series (average PVD)
        # For pixel-wise PVD, this logic would change
        if total_pixels_per_image == 0: return 0.0
        pvd = np.sum(valid_pixels_per_image) / (len(data_stack) * total_pixels_per_image)
        
        # If we need pixel-wise PVD:
        # valid_counts_pixelwise = np.sum(~np.isnan(data_stack), axis=0)
        # pvd_pixelwise = valid_counts_pixelwise / data_stack.shape[0]
        # return pvd_pixelwise
        
        return pvd # Returns average PVD for the series for now

    def _linear_regression_fill(self, target_data_to_fill, source_data_for_model, dates):
        """
        Lấp đầy target_data bằng hồi quy tuyến tính từ source_data.
        Assumes target_data_to_fill is a 2D array (single image for the day).
        source_data_for_model is a 3D stack (time, H, W) used to build pixel-wise models.
        This is highly conceptual. Pixel-wise regression over a year is needed.
        """
        filled_data = np.copy(target_data_to_fill)
        if source_data_for_model is None or source_data_for_model.shape[0] < 2:
            print("Not enough source data for linear regression.")
            return filled_data # Cannot build model

        # Placeholder: Perform pixel-wise linear regression.
        # For each pixel (y,x):
        #   y_series = source_data_for_model[:, y, x] (independent var, e.g. T1 LSTs)
        #   x_series_target = corresponding historical T2 LSTs for that pixel (dependent var)
        #   Fit model: T2_hist = a * T1_hist + b
        #   Predict: filled_data[y,x] = a * source_data_for_model[-1, y, x] + b (using current day's T1)
        
        # Simplified: if source has data on the current day, use it as a proxy (very naive)
        # A real implementation would fit models using a longer time series.
        # For now, we'll just demonstrate the idea if the source data for *this specific day* is valid
        # This is NOT the described regression.
        
        # Example: For pixels where target is NaN and source (for this day) is not NaN
        source_current_day = source_data_for_model[-1] # Assuming last entry is current day
        nan_mask_target = np.isnan(filled_data)
        valid_mask_source = ~np.isnan(source_current_day)
        
        fillable_mask = nan_mask_target & valid_mask_source
        # This would be: filled_data[fillable_mask] = slope * source_current_day[fillable_mask] + intercept
        # For now, direct copy if source is valid (very simplified)
        filled_data[fillable_mask] = source_current_day[fillable_mask] 
        print("Linear regression fill (simplified) applied.")
        return filled_data


    def _shift_method_fill(self, target_data_to_fill, source_data_for_model, dates_for_model):
        """
        Lấp đầy target_data bằng phương pháp dịch chuyển từ source_data.
        Calculates monthly mean LST difference: diff = mean(T2_hist) - mean(T3_hist or T4_hist)
        Predicts: T2_filled = T3_current_day (or T4_current_day) + diff_for_current_month
        This is also conceptual and needs historical data.
        """
        filled_data = np.copy(target_data_to_fill)
        if source_data_for_model is None or source_data_for_model.shape[0] == 0:
            print("Not enough source data for shift method.")
            return filled_data

        # Placeholder: Calculate monthly average differences pixel-wise.
        # Example: current_month = dates_for_model[-1].month
        # monthly_diff_map = precalculated_monthly_diffs[current_month] (a 2D map of diffs)
        # filled_data[nan_mask] = source_data_for_model[-1][nan_mask] + monthly_diff_map[nan_mask]

        # Simplified: if source has data on current day, use it as a proxy (very naive)
        source_current_day = source_data_for_model[-1] # Assuming last entry is current day
        nan_mask_target = np.isnan(filled_data)
        valid_mask_source = ~np.isnan(source_current_day)
        fillable_mask = nan_mask_target & valid_mask_source
        
        # This would be: filled_data[fillable_mask] = source_current_day[fillable_mask] + monthly_avg_diff
        # For now, direct copy (very simplified)
        filled_data[fillable_mask] = source_current_day[fillable_mask]
        print("Shift method fill (simplified) applied.")
        return filled_data

    def merge_daily_observations(self, daily_data, daily_dates, all_sensor_data_history):
        """
        Ghép dữ liệu hàng ngày từ Terra/Aqua (T1, T2, T3, T4).
        Focuses on filling T2 (Aqua Day) and T4 (Aqua Night).
        daily_data: dict { "T1": data_T1, "T2": data_T2, ...} for the current day. Can be None if no file.
        daily_dates: dict { "T1": date_T1, ...}
        all_sensor_data_history: dict {"T1": [historical_data_T1_stack], ...} for PVD, regression, shift.
        Returns: filled_T2, filled_T4 for the current day.
        """
        
        # Initialize with current day's data or NaNs if missing
        t2_shape = daily_data.get("T2", daily_data.get("T1", daily_data.get("T4", daily_data.get("T3")))).shape
        
        filled_T2 = daily_data.get("T2", np.full(t2_shape, np.nan))
        filled_T4 = daily_data.get("T4", np.full(t2_shape, np.nan))

        # --- Process T2 (Aqua Day ~13:30) ---
        # Calculate PVDs (needs yearly data, simplified here with available history)
        # PVD calculation needs to be pixel-wise for the "PVD of T2 < 5%" condition.
        # For simplicity, using an overall PVD for the entire historical series for now.
        pvd_T1 = self._calculate_pvd(all_sensor_data_history.get("T1")[0] if all_sensor_data_history.get("T1") and all_sensor_data_history.get("T1")[0] is not None else None)
        pvd_T2 = self._calculate_pvd(all_sensor_data_history.get("T2")[0] if all_sensor_data_history.get("T2") and all_sensor_data_history.get("T2")[0] is not None else None)
        pvd_T3 = self._calculate_pvd(all_sensor_data_history.get("T3")[0] if all_sensor_data_history.get("T3") and all_sensor_data_history.get("T3")[0] is not None else None)
        pvd_T4 = self._calculate_pvd(all_sensor_data_history.get("T4")[0] if all_sensor_data_history.get("T4") and all_sensor_data_history.get("T4")[0] is not None else None)


        # If T2 is largely missing (based on its PVD, or if current T2 is all NaN)
        # The rule "PVD of T2 < 5%" ideally applies to each pixel over a year.
        # Simplified: if current T2 is mostly NaN or its historical PVD is low.
        if np.all(np.isnan(filled_T2)) or pvd_T2 < self.config.PVD_THRESHOLD:
            print(f"Attempting to fill T2 for date {daily_dates.get('T2', daily_dates.get('T1'))}: PVD_T2 ({pvd_T2:.2f}) may be low.")
            
            # Try T1 (Terra Day ~10:30) with Linear Regression
            if pvd_T1 > self.config.PVD_THRESHOLD and daily_data.get("T1") is not None:
                print("Using T1 (Terra Day) with Linear Regression for T2.")
                filled_T2 = self._linear_regression_fill(filled_T2, 
                                                         all_sensor_data_history.get("T1")[0] if all_sensor_data_history.get("T1") else None, 
                                                         daily_dates.get("T1"))
            
            # If T2 still missing, try T4 (Aqua Night ~01:30) with Shift Method
            if np.all(np.isnan(filled_T2)) and pvd_T4 > self.config.PVD_THRESHOLD and daily_data.get("T4") is not None:
                print("Using T4 (Aqua Night) with Shift Method for T2.")
                filled_T2 = self._shift_method_fill(filled_T2, 
                                                    all_sensor_data_history.get("T4")[0] if all_sensor_data_history.get("T4") else None, 
                                                    daily_dates.get("T4"))

            # If T2 still missing, try T3 (Terra Night ~22:30) with Shift Method
            if np.all(np.isnan(filled_T2)) and pvd_T3 > self.config.PVD_THRESHOLD and daily_data.get("T3") is not None:
                print("Using T3 (Terra Night) with Shift Method for T2.")
                filled_T2 = self._shift_method_fill(filled_T2, 
                                                    all_sensor_data_history.get("T3")[0] if all_sensor_data_history.get("T3") else None,
                                                    daily_dates.get("T3"))
        
        # --- Process T4 (Aqua Night ~01:30) --- (Similar logic could be applied if T4 is primary target)
        # For this example, we primarily focus on filling T2 and assume T4 is either used as is or filled by other means.
        # The paper mentions T2 and T4 are processed independently.
        # So, T4 would have its own PVD checks and filling rules, potentially using T3, T1, T2.
        # For now, T4 is returned as is or from its direct observation.
        if np.all(np.isnan(filled_T4)):
             print(f"T4 for date {daily_dates.get('T4', daily_dates.get('T1'))} is initially all NaN.")
        # Add filling logic for T4 if needed, similar to T2

        return filled_T2, filled_T4


    def _fit_temporal_trend_spline(self, pixel_timeseries, doy_series, smoothing_factor=None):
        """Fits a smoothing spline to a single pixel's time series using Day of Year."""
        valid_mask = ~np.isnan(pixel_timeseries)
        if np.sum(valid_mask) < 4: # Need enough points for spline
            return np.full_like(pixel_timeseries, np.nan)
        
        # Sort by DOY if not already sorted, and handle duplicates for spline
        unique_doy, unique_indices = np.unique(doy_series[valid_mask], return_index=True)
        unique_values = pixel_timeseries[valid_mask][unique_indices]

        if len(unique_doy) < 4: # Still not enough unique points
             return np.full_like(pixel_timeseries, np.nan)

        s_val = smoothing_factor
        if s_val is None:
            # Default s in UnivariateSpline is len(weights) which is len(unique_doy) here.
            # If "s too small" warning occurs, this default is too aggressive.
            # Try a larger s for more smoothing. Aim for s > m, where m is number of points.
            s_val = len(unique_doy) * 2.0 # Heuristic: double the default smoothing factor
            # An alternative could be: s_val = len(unique_doy) + np.sqrt(2 * len(unique_doy))
        
        # Ensure DOY is scaled appropriately if needed, though UnivariateSpline handles it.
        try:
            spl = UnivariateSpline(unique_doy, unique_values, s=s_val, k=3) # k=3 for cubic spline
            # Predict trend for all DOYs in the original series length
            # If doy_series contains all DOYs from 1 to 365/366 for interpolation:
            # trend = spl(np.arange(1, len(pixel_timeseries) + 1)) # Or full DOY range
            # For now, predict for the given doy_series
            trend = spl(doy_series)
        except Exception as e:
            # print(f"Warning: Spline fitting failed for pixel. s_val={s_val}, N_points={len(unique_doy)}. Error: {e}. Returning NaNs for trend.")
            return np.full_like(pixel_timeseries, np.nan)
        return trend

    def calculate_temporal_trend_and_residuals(self, data_cube, dates):
        """
        data_cube: (time, height, width)
        dates: list of datetime objects corresponding to the time axis
        Returns: trend_cube, residuals_cube
        """
        num_times, height, width = data_cube.shape
        doy_series = np.array([date.timetuple().tm_yday for date in dates])
        
        trend_cube = np.full_like(data_cube, np.nan)
        residuals_cube = np.full_like(data_cube, np.nan)
        
        print("Calculating temporal trend (spline) and residuals...")
        for r in range(height):
            for c in range(width):
                pixel_ts = data_cube[:, r, c]
                if np.all(np.isnan(pixel_ts)):
                    continue
                
                # Apply water mask: skip spline fitting for permanent water bodies if desired
                

                trend_line = self._fit_temporal_trend_spline(pixel_ts, doy_series, 
                                                             self.config.SPLINE_SMOOTHING_FACTOR)
                trend_cube[:, r, c] = trend_line
                residuals_cube[:, r, c] = pixel_ts - trend_line
            progress_divisor_r = max(1, height // 10)
            if r % progress_divisor_r == 0: # Progress update
                 print(f"  Spline fitting: {int((r/height)*100)}% done")
        
        return trend_cube, residuals_cube

    def _get_neighbors(self, r, c, shape, window_size=3):
        """Helper to get neighbor coordinates for a pixel (r,c)"""
        neighbors = []
        half_win = window_size // 2
        for dr in range(-half_win, half_win + 1):
            for dc in range(-half_win, half_win + 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < shape[0] and 0 <= nc < shape[1]:
                    neighbors.append((nr, nc))
        return neighbors

    def interpolate_residuals_correlation_based(self, residuals_cube, doy_series):
        """
        Interpolates missing residuals using a simplified correlation-based method.
        residuals_cube: (time, height, width) with NaNs for missing residuals.
        doy_series: array of day-of-year for the time axis.
        Returns: filled_residuals_cube
        
        This is a simplified version of ICW. Full ICW involves:
        - 8 specific neighbors (not just a window)
        - 1% representative pixels (10km spacing)
        - OLS regression with most correlated neighbor (Formula 3 & 4)
        - Block processing (10x10) with IDW for block centers.
        - Pearson correlation over the year's daily values.
        """
        num_times, height, width = residuals_cube.shape
        filled_residuals = np.copy(residuals_cube)
        
        print("Interpolating residuals (simplified correlation-based)...")

        # Pre-calculate correlations (conceptually, over a year)
        # For simplicity, use correlations over the available `residuals_cube`
        # This should ideally use a full year of residuals for robust correlations.

        for t in range(num_times):
            for r in range(height):
                for c in range(width):
                    if np.isnan(filled_residuals[t, r, c]):
                        # This pixel's residual is missing for this day (t)
                        target_pixel_historical_residuals = residuals_cube[:, r, c]
                        
                        # Skip if historical data is all NaN (no basis for correlation)
                        if np.all(np.isnan(target_pixel_historical_residuals)):
                            continue

                        # Apply water mask: do not interpolate water pixels
                        

                        neighbors = self._get_neighbors(r, c, (height, width), 
                                                        self.config.SPATIAL_NEIGHBOR_WINDOW_SIZE)
                        
                        weighted_sum_residuals = 0
                        sum_weights = 0
                        
                        best_corr = -1
                        best_neighbor_residual_today = np.nan

                        for nr, nc in neighbors:
                            neighbor_residual_today = residuals_cube[t, nr, nc]
                            if np.isnan(neighbor_residual_today): # Neighbor also missing today
                                continue

                            neighbor_pixel_historical_residuals = residuals_cube[:, nr, nc]
                            if np.all(np.isnan(neighbor_pixel_historical_residuals)):
                                continue

                            # Calculate Pearson correlation between target's history and neighbor's history
                            # Mask NaNs for correlation
                            valid_mask_corr = ~np.isnan(target_pixel_historical_residuals) & \
                                              ~np.isnan(neighbor_pixel_historical_residuals)
                            
                            if np.sum(valid_mask_corr) < 10: # Need enough points for meaningful correlation
                                continue
                                
                            corr, _ = pearsonr(target_pixel_historical_residuals[valid_mask_corr], 
                                               neighbor_pixel_historical_residuals[valid_mask_corr])
                            
                            if np.isnan(corr): continue

                            # Simplified ICW: use neighbor with best positive correlation
                            # A more complete ICW would use OLS regression (Formula 3)
                            # R_target = a * R_neighbor + b
                            if corr > 0 and corr > best_corr: # Taking positive correlations
                                best_corr = corr
                                # For OLS: fit on historical target_pixel_historical_residuals and neighbor_pixel_historical_residuals
                                # Then predict: best_neighbor_residual_today = slope * residuals_cube[t,nr,nc] + intercept
                                # Simplified: just use the neighbor's value, weighted by corr
                                # Or, for Formula (4) like simplification: use the value from the most correlated neighbor directly
                                # For now, collect for weighted average based on positive correlation
                                weighted_sum_residuals += corr * neighbor_residual_today
                                sum_weights += corr
                        
                        if sum_weights > 0:
                            filled_residuals[t, r, c] = weighted_sum_residuals / sum_weights
                        # Else: still NaN if no good neighbors found

            progress_divisor_t = max(1, num_times // 10)
            if t % progress_divisor_t == 0: # Progress update
                 print(f"  Residual interpolation: {int((t/num_times)*100)}% done for daily slices")

        return filled_residuals

    def load_all_sensor_data_for_period(self, start_dt, end_dt):
        """
        Loads all LST data for T1, T2, T3, T4 within the period.
        Returns a dictionary: {"T1": (data_stack, dates_list), "T2": ..., }
        This data is used for PVD, regression model fitting, shift calculations.
        """
        sensor_data_history = {"T1": [], "T2": [], "T3": [], "T4": []} # T1: TerraDay, T2: AquaDay, etc.
        sensor_dates_history = {"T1": [], "T2": [], "T3": [], "T4": []}
        
        all_files = sorted(glob(os.path.join(self.config.RAW_DATA_DIR, '*.tif')))
        
        profile_shape = None # To ensure consistent shapes

        for f_path in all_files:
            try:
                f_date = self.get_date_from_filename(f_path)
                if not (start_dt <= f_date <= end_dt):
                    continue

                platform, time_of_day = self.get_platform_time_from_filename(f_path)
                sensor_key = None
                if platform == "Terra" and time_of_day == "Day": sensor_key = "T1"
                elif platform == "Aqua" and time_of_day == "Day": sensor_key = "T2"
                elif platform == "Terra" and time_of_day == "Night": sensor_key = "T3"
                elif platform == "Aqua" and time_of_day == "Night": sensor_key = "T4"
                
                if sensor_key:
                    # TODO: Add QA file path resolution if needed
                    # qa_f_path = resolve_qa_file(f_path, self.config.RAW_QA_DATA_DIR)
                    data, profile = self.load_and_preprocess_tif_with_qa(f_path) # qa_f_path
                    
                    if profile_shape is None:
                        profile_shape = data.shape
                    elif data.shape != profile_shape:
                        print(f"Skipping {f_path} due to shape mismatch: {data.shape} vs {profile_shape}")
                        continue

                    sensor_data_history[sensor_key].append(data)
                    sensor_dates_history[sensor_key].append(f_date)
                    
            except Exception as e:
                print(f"Error loading historical file {os.path.basename(f_path)}: {e}")
        
        # Stack data for easier use later
        for key in sensor_data_history:
            if sensor_data_history[key]:
                sensor_data_history[key] = (np.stack(sensor_data_history[key]), sensor_dates_history[key])
            else: # Ensure it's (None, []) if no data
                sensor_data_history[key] = (None, [])
                
        return sensor_data_history


    def process_time_series_advanced(self, start_date_str=None, end_date_str=None):
        """Main advanced processing workflow."""
        start_date_str = start_date_str or self.config.START_DATE
        end_date_str = end_date_str or self.config.END_DATE
        start_dt = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date_str, '%Y-%m-%d')

        print(f"Loading all sensor data for period {start_date_str} to {end_date_str} for historical context...")
        # This loads ALL data in the window to serve as 'history' for PVD, regression, etc.
        # For true yearly PVD/regression, this period should be at least a year.
        all_sensor_data_history_map = self.load_all_sensor_data_for_period(start_dt, end_dt)
        
        # Get global profile from any loaded data
        global_profile = None
        # Try to get profile from a file that exists within the date range and type
        for key_to_try in ["T2", "T1", "T4", "T3"]:
            data_stack_check, dates_check = all_sensor_data_history_map.get(key_to_try, (None, []))
            if data_stack_check is not None and len(dates_check) > 0:
                # Find the original file path for the first image of this type
                # This is a bit convoluted; ideally profile is stored with loaded data more directly
                platform_search = "Aqua" if key_to_try in ["T2", "T4"] else "Terra"
                time_search = "Day" if key_to_try in ["T1", "T2"] else "Night"
                first_date_str_search = dates_check[0].strftime("%Y-%m-%d")
                
                # Construct expected filename pattern part
                # Example: Terra_Day_2020-01-01.tif
                fn_pattern = f"{platform_search}_{time_search}_{first_date_str_search}.tif"
                potential_files = glob(os.path.join(self.config.RAW_DATA_DIR, fn_pattern))
                
                if potential_files:
                    try:
                        _, global_profile = self.load_and_preprocess_tif_with_qa(potential_files[0])
                        if global_profile: break
                    except Exception as e:
                        print(f"Could not load profile from {potential_files[0]}: {e}")      

        if global_profile is None:
            print("Could not determine global raster profile from available data. Exiting.")
            return

        # --- Iterate day by day for processing ---
        # The core spatiotemporal fitting (spline + residual) is done on the *entire series*.
        # Daily saving/visualization can then pick from the processed series.

        # Process T2 (Aqua Day) Series
        target_data_series_T2, dates_T2 = all_sensor_data_history_map.get("T2", (None, []))
        final_T2_filled_series = None
        if target_data_series_T2 is None or target_data_series_T2.shape[0] == 0:
            print("No T2 (Aqua Day) data available for spatiotemporal fitting. Skipping T2 processing.")
        else:
            print("\n--- Processing Full T2 (Aqua Day) Series ---")
            print("Step A: Calculating temporal trend and residuals for T2 series...")
            t2_trend_cube, t2_residuals_cube = self.calculate_temporal_trend_and_residuals(
                target_data_series_T2, dates_T2
            )
            print("Step B: Interpolating residuals for T2 series...")
            t2_filled_residuals = self.interpolate_residuals_correlation_based(
                t2_residuals_cube, 
                np.array([d.timetuple().tm_yday for d in dates_T2])
            )
            print("Step C: Reconstructing final T2 LST series...")
            # Step C: Reconstruct final T2 LST series by adding interpolated residuals back to trend
            if t2_trend_cube is not None and t2_filled_residuals is not None:
                final_T2_filled_series = t2_trend_cube + t2_filled_residuals
                if final_T2_filled_series is not None:
                    print(f"Clipping final T2 series. Original min/max: {np.nanmin(final_T2_filled_series):.2f}/{np.nanmax(final_T2_filled_series):.2f}")
                    final_T2_filled_series = np.clip(
                        final_T2_filled_series, 
                        self.config.VISUALIZATION_MIN_TEMP, 
                        self.config.VISUALIZATION_MAX_TEMP
                    )
                    print(f"Clipped T2 series min/max: {np.nanmin(final_T2_filled_series):.2f}/{np.nanmax(final_T2_filled_series):.2f}")
            else:
                final_T2_filled_series = None

        # Process T4 (Aqua Night) Series
        target_data_series_T4, dates_T4 = all_sensor_data_history_map.get("T4", (None, []))
        final_T4_filled_series = None
        if target_data_series_T4 is None or target_data_series_T4.shape[0] == 0:
            print("No T4 (Aqua Night) data available for spatiotemporal fitting. Skipping T4 processing.")
        else:
            print("\n--- Processing Full T4 (Aqua Night) Series ---")
            print("Step A: Calculating temporal trend and residuals for T4 series...")
            t4_trend_cube, t4_residuals_cube = self.calculate_temporal_trend_and_residuals(
                target_data_series_T4, dates_T4
            )
            print("Step B: Interpolating residuals for T4 series...")
            t4_filled_residuals = self.interpolate_residuals_correlation_based(
                t4_residuals_cube,
                np.array([d.timetuple().tm_yday for d in dates_T4])
            )
            print("Step C: Reconstructing final T4 LST series...")
            # Step C: Reconstruct final T4 LST series
            if t4_trend_cube is not None and t4_filled_residuals is not None:
                final_T4_filled_series = t4_trend_cube + t4_filled_residuals
                if final_T4_filled_series is not None:
                    print(f"Clipping final T4 series. Original min/max: {np.nanmin(final_T4_filled_series):.2f}/{np.nanmax(final_T4_filled_series):.2f}")
                    final_T4_filled_series = np.clip(
                        final_T4_filled_series, 
                        self.config.VISUALIZATION_MIN_TEMP, 
                        self.config.VISUALIZATION_MAX_TEMP
                    )
                    print(f"Clipped T4 series min/max: {np.nanmin(final_T4_filled_series):.2f}/{np.nanmax(final_T4_filled_series):.2f}")
            else:
                final_T4_filled_series = None

        # Daily saving and visualization loop
        current_date_iter = start_dt
        while current_date_iter <= end_dt:
            print(f"\nSaving/Visualizing for date: {current_date_iter.strftime('%Y-%m-%d')}")
            
            # Handle T2 saving
            if final_T2_filled_series is not None and dates_T2:
                try:
                    current_date_idx_t2 = dates_T2.index(current_date_iter)
                    self.save_and_visualize_single_day_result(
                        current_date_iter,
                        final_T2_filled_series[current_date_idx_t2],
                        target_data_series_T2[current_date_idx_t2],
                        "T2_Aqua_Day", global_profile, ["AquaDay_AdvProcessed"]
                    )
                except ValueError:
                    print(f"Date {current_date_iter} not found in T2 processed series for saving.")
            else:
                print(f"Skipping T2 save for {current_date_iter} as no processed T2 series available.")

            # Handle T4 saving
            if final_T4_filled_series is not None and dates_T4:
                try:
                    current_date_idx_t4 = dates_T4.index(current_date_iter)
                    self.save_and_visualize_single_day_result(
                        current_date_iter,
                        final_T4_filled_series[current_date_idx_t4],
                        target_data_series_T4[current_date_idx_t4],
                        "T4_Aqua_Night", global_profile, ["AquaNight_AdvProcessed"]
                    )
                except ValueError:
                    print(f"Date {current_date_iter} not found in T4 processed series for saving.")
            else:
                print(f"Skipping T4 save for {current_date_iter} as no processed T4 series available.")
            
            current_date_iter += timedelta(days=1)
        
        print("\nAdvanced processing and daily saving complete.")
        return final_T2_filled_series, final_T4_filled_series 


    def save_and_visualize_single_day_result(self, date_obj, processed_data, original_data, 
                                             time_of_day_label, profile, sources_label_list):
        """Saves and visualizes the result for a single day."""
        date_file_str = date_obj.strftime('%Y_%m_%d')
        output_filename = f'Final_LST_{time_of_day_label}_{date_file_str}.tif'
        output_path = os.path.join(self.config.OUTPUT_DIR, output_filename)

        min_val = np.nanmin(processed_data)
        max_val = np.nanmax(processed_data)
        print(f"Output for {output_filename}: Min={min_val:.2f}, Max={max_val:.2f}")

        save_profile = profile.copy()
        save_profile.update(dtype=rasterio.float32, count=1, nodata=float('nan'))
        
        with rasterio.open(output_path, 'w', **save_profile) as dst:
            dst.write(processed_data.astype(rasterio.float32), 1)
            # Add relevant tags
            dst.update_tags(TEMPERATURE_UNIT='Celsius', SOURCE='+'.join(sources_label_list))
        print(f"Saved: {output_path}")

        # Visualization
        vis_filename_base = f'LST_{time_of_day_label}_{date_file_str}'
        self.create_comparison_figure(
            original_data, processed_data,
            [f'Original MODIS {time_of_day_label} ({date_obj.strftime("%Y-%m-%d")})',
             f'Gap-filled {time_of_day_label} ({date_obj.strftime("%Y-%m-%d")})'],
            os.path.join(self.config.VISUALIZATION_DIR, f'{vis_filename_base}_comparison.png')
        )
        print(f"Saved visualization: {vis_filename_base}_comparison.png")


    # --- Helper and Visualization methods from original script (can be reused/adapted) ---
    def clip_values(self, data, min_val=None, max_val=None):
        """Cắt giá trị trong khoảng xác định cho hiển thị"""
        if min_val is None: min_val = self.config.VISUALIZATION_MIN_TEMP
        if max_val is None: max_val = self.config.VISUALIZATION_MAX_TEMP
        # Use a copy to avoid modifying the original array directly if it's part of a larger structure
        data_copy = np.copy(data) 
        # Apply clipping
        clipped_data = np.clip(data_copy, min_val, max_val)
        # Preserve NaNs from the original data_copy (before clipping)
        clipped_data[np.isnan(data_copy)] = np.nan 
        return clipped_data

    def create_comparison_figure(self, original_data, processed_data, titles, output_path, vmin=None, vmax=None):
        """Tạo hình ảnh so sánh trước và sau khi xử lý"""
        if vmin is None: vmin = self.config.VISUALIZATION_MIN_TEMP
        if vmax is None: vmax = self.config.VISUALIZATION_MAX_TEMP
            
        original_clipped = self.clip_values(original_data, vmin, vmax)
        processed_clipped = self.clip_values(processed_data, vmin, vmax)
        
        colors = self.config.COLOR_MAP_COLORS
        cmap = LinearSegmentedColormap.from_list('gee_lst', colors, N=self.config.COLOR_MAP_BINS)
        cmap.set_bad('white', 1.0) # Color for NaN
        
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        
        masked_original = np.ma.masked_invalid(original_clipped)
        masked_processed = np.ma.masked_invalid(processed_clipped)
        
        # Common plot function
        def plot_data_ax(ax, data_to_plot, title_str):
            # Ensure data_to_plot is a masked array for consistent .filled behavior
            if not isinstance(data_to_plot, np.ma.MaskedArray):
                data_to_plot = np.ma.masked_invalid(data_to_plot)

            im = ax.imshow(data_to_plot, cmap=cmap, vmin=vmin, vmax=vmax)
            
            ax.set_title(title_str, fontsize=14)
            ax.axis('off')

        im1 = plot_data_ax(axs[0], masked_original, titles[0])
        im2 = plot_data_ax(axs[1], masked_processed, titles[1])
        
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap=cmap), cax=cbar_ax, extend='both')
        cbar.set_label('Temperature (°C)')
        
        fig.suptitle("MODIS LST Comparison (Advanced)", fontsize=16, y=0.95)
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

def main_advanced():
    print("Starting Advanced MODIS LST Gap Filling Process...")
    processor = MODISLSTProcessorAdvanced()
    
    processor.process_time_series_advanced()
    
    print("Advanced gap filling completed successfully!")
    print(f"Outputs saved in: {Config.OUTPUT_DIR}")
    print(f"Visualizations saved in: {Config.VISUALIZATION_DIR}")

if __name__ == "__main__":
    main_advanced() 