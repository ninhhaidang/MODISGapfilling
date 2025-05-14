import ee
import datetime
import time
import sys
import argparse
import os
from ee_monitor import initialize_ee, start_export_task, monitor_tasks

# ===== THIẾT LẬP THÔNG SỐ MẶC ĐỊNH =====
# Thay đổi các giá trị này nếu bạn muốn thay đổi mặc định
start_date = '2020-01-01'  # Định dạng: YYYY-MM-DD
end_date = '2020-02-01'    # Định dạng: YYYY-MM-DD
export_folder = 'LST_Vietnam'
# ==========================================

# ===== THIẾT LẬP PHẠM VI KHÔNG GIAN =====
# Tọa độ bounding box Việt Nam
WEST = 102.138448   # Kinh độ Tây
EAST = 117.840999   # Kinh độ Đông
SOUTH = 7.177539    # Vĩ độ Nam
NORTH = 23.401113   # Vĩ độ Bắc
# ==========================================

def authenticate_ee():
    """Authenticate Earth Engine using the available methods"""
    try:
        # Try to initialize EE without authentication
        ee.Initialize()
        print("Earth Engine already authenticated")
        return True
    except Exception as e:
        print(f"Initial authentication attempt failed: {e}")
        
        try:
            # Try to authenticate
            ee.Authenticate()
            ee.Initialize()
            print("Earth Engine authenticated successfully")
            return True
        except Exception as e:
            print(f"Authentication failed: {e}")
            print("Please authenticate manually by running: earthengine authenticate")
            return False

def main():
    # Parse command line arguments
    global start_date, end_date, export_folder
    
    parser = argparse.ArgumentParser(description='MODIS LST Gap-filling for Vietnam')
    parser.add_argument('--start_date', default=start_date, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', default=end_date, help='End date (YYYY-MM-DD)')
    parser.add_argument('--project_id', default=None, help='GEE project ID')
    parser.add_argument('--export_folder', default=export_folder, help='Google Drive folder to export to')
    parser.add_argument('--monitor_interval', type=int, default=30, help='Monitoring interval in seconds')
    parser.add_argument('--skip_exports', action='store_true', help='Skip the export tasks and only compute the results')
    parser.add_argument('--auth', action='store_true', help='Authenticate Earth Engine before processing')
    args = parser.parse_args()

    # Authenticate if requested or try using initialize_ee
    if args.auth:
        if not authenticate_ee():
            sys.exit(1)
    elif not initialize_ee(args.project_id):
        print("Attempting alternative authentication method...")
        if not authenticate_ee():
            sys.exit(1)
    
    # Set up dates
    start_date = args.start_date
    end_date = args.end_date
    export_folder = args.export_folder
    
    print(f"Processing MODIS LST data from {start_date} to {end_date}")
    
    # ===== 1. Định nghĩa khu vực quan tâm bằng bounding box =====
    # Tạo bounding box từ tọa độ
    vietnam_bbox = ee.Geometry.Rectangle([WEST, SOUTH, EAST, NORTH])
    print(f"Using bounding box: West={WEST}, South={SOUTH}, East={EAST}, North={NORTH}")
    
    # ===== 2. Tải dữ liệu MODIS LST =====
    print("Loading MODIS Terra and Aqua data...")
    modisTerra = ee.ImageCollection('MODIS/061/MOD11A1') \
        .filterDate(start_date, end_date) \
        .filterBounds(vietnam_bbox) \
        .select(['LST_Day_1km', 'LST_Night_1km', 'QC_Day', 'QC_Night']) \
        .map(lambda image: image.reproject('EPSG:4326', None, 1000))
    
    modisAqua = ee.ImageCollection('MODIS/061/MYD11A1') \
        .filterDate(start_date, end_date) \
        .filterBounds(vietnam_bbox) \
        .select(['LST_Day_1km', 'LST_Night_1km', 'QC_Day', 'QC_Night']) \
        .map(lambda image: image.reproject('EPSG:4326', None, 1000))
    
    # ===== 3. Hàm lọc chất lượng =====
    def filter_quality(image):
        qc_day = image.select('QC_Day')
        qc_night = image.select('QC_Night')
        quality_mask_day = qc_day.bitwiseAnd(3).neq(3)
        quality_mask_night = qc_night.bitwiseAnd(3).neq(3)
        return image.addBands(image.select('LST_Day_1km').updateMask(quality_mask_day).rename('filtered_LST_Day_1km')) \
            .addBands(image.select('LST_Night_1km').updateMask(quality_mask_night).rename('filtered_LST_Night_1km'))
    
    print("Applying quality filtering...")
    modisTerraFiltered = modisTerra.map(filter_quality)
    modisAquaFiltered = modisAqua.map(filter_quality)
    
    # ===== 4. Kết hợp dữ liệu =====
    combined = modisTerraFiltered.merge(modisAquaFiltered)
    
    # ===== 5. Xử lý riêng ban ngày và ban đêm (đổi từ K sang °C) =====
    print("Converting LST from Kelvin to Celsius...")
    dayData = combined.select('filtered_LST_Day_1km').map(lambda image: 
        image.multiply(0.02).subtract(273.15)
            .rename('LST_Day_1km_C')
            .set('system:time_start', image.get('system:time_start'))
    )
    
    nightData = combined.select('filtered_LST_Night_1km').map(lambda image: 
        image.multiply(0.02).subtract(273.15)
            .rename('LST_Night_1km_C')
            .set('system:time_start', image.get('system:time_start'))
    )
    
    # ===== 6. Trung bình dài hạn =====
    print("Calculating long-term means...")
    meanDay = dayData.mean().reproject('EPSG:4326', None, 1000)
    meanNight = nightData.mean().reproject('EPSG:4326', None, 1000)
    
    # ===== 7. Tính phần dư =====
    print("Computing residuals...")
    residualsDay = dayData.map(lambda image: 
        image.subtract(meanDay)
            .rename('residual_day')
            .set('system:time_start', image.get('system:time_start'))
            .reproject('EPSG:4326', None, 1000)
    )
    
    residualsNight = nightData.map(lambda image: 
        image.subtract(meanNight)
            .rename('residual_night')
            .set('system:time_start', image.get('system:time_start'))
            .reproject('EPSG:4326', None, 1000)
    )
    
    # ===== 8. Làm mượt không gian =====
    print("Applying spatial smoothing...")
    smoothedResidualsDay = residualsDay.map(lambda image: 
        image.focal_mean(radius=3, kernelType='square', units='pixels')
            .focal_mean(radius=3, kernelType='square', units='pixels')
            .set('system:time_start', image.get('system:time_start'))
            .reproject('EPSG:4326', None, 1000)
    )
    
    smoothedResidualsNight = residualsNight.map(lambda image: 
        image.focal_mean(radius=3, kernelType='square', units='pixels')
            .focal_mean(radius=3, kernelType='square', units='pixels')
            .set('system:time_start', image.get('system:time_start'))
            .reproject('EPSG:4326', None, 1000)
    )
    
    # ===== 9. Làm mượt thời gian =====
    print("Applying temporal smoothing...")
    
    def temporal_smoothing(collection):
        return collection.map(lambda image:
            ee.Algorithms.If(
                ee.Algorithms.IsEqual(image.get('system:time_start'), None),
                image,
                (lambda: 
                    (lambda date: 
                        (lambda before, after: 
                            image.unmask(before.merge(after).mean())
                                .set('system:time_start', image.get('system:time_start'))
                                .reproject('EPSG:4326', None, 1000)
                        )(
                            collection.filterDate(date.advance(-7, 'day'), date),
                            collection.filterDate(date, date.advance(7, 'day'))
                        )
                    )(ee.Date(image.get('system:time_start')))
                )()
            )
        )
    
    smoothedResidualsDayTemporal = temporal_smoothing(smoothedResidualsDay)
    smoothedResidualsNightTemporal = temporal_smoothing(smoothedResidualsNight)
    
    # ===== 10. Tạo dữ liệu cuối cùng (clip theo bounding box Việt Nam) =====
    print("Creating final gap-filled LST data...")
    finalDay = smoothedResidualsDayTemporal.map(lambda image: 
        image.add(meanDay)
            .rename('final_LST_Day_1km_C')
            .clip(vietnam_bbox)
            .reproject('EPSG:4326', None, 1000)
            .set('system:time_start', image.get('system:time_start'))
    )
    
    finalNight = smoothedResidualsNightTemporal.map(lambda image: 
        image.add(meanNight)
            .rename('final_LST_Night_1km_C')
            .clip(vietnam_bbox)
            .reproject('EPSG:4326', None, 1000)
            .set('system:time_start', image.get('system:time_start'))
    )
    
    # Format date for export file naming
    date_str = datetime.datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y_%m')
    
    if args.skip_exports:
        print("Export tasks skipped as requested.")
        return
    
    # ===== Export tasks =====
    print("Preparing export tasks...")
    task_ids = []
    export_folder = args.export_folder
    
    # Helper function to create export task
    def create_export_task(image, description):
        # Clip ảnh theo bounding box
        image_clipped = image.clip(vietnam_bbox)
        
        task = ee.batch.Export.image.toDrive(
            image=image_clipped.reproject('EPSG:4326', None, 1000),
            description=description,
            folder=export_folder,
            scale=1000,
            maxPixels=1e13,
            crs='EPSG:4326'
        )
        return task
    
    # Process display images for the requested time frame
    displayStart = ee.Date(start_date)
    displayEnd = ee.Date(end_date)
    
    # Chỉ xuất ảnh Final LST (ảnh cuối cùng) cho từng ngày
    print("Setting up final LST export tasks for each day in the date range...")
    
    # Chuyển đổi khoảng thời gian thành danh sách các ngày
    start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    
    # Tạo danh sách ngày trong khoảng thời gian
    delta = end_dt - start_dt
    date_list = [start_dt + datetime.timedelta(days=i) for i in range(delta.days + 1)]
    
    print(f"Exporting data for {len(date_list)} days from {start_date} to {end_date}")
    
    # Với mỗi ngày, xuất ảnh LST ban ngày và ban đêm
    for current_date in date_list:
        # Định dạng ngày cho filter và tên file
        date_str = current_date.strftime('%Y-%m-%d')
        date_file_str = current_date.strftime('%Y_%m_%d')
        
        print(f"Processing date: {date_str}")
        
        # Tạo filter date cho 1 ngày
        current_start = ee.Date(date_str)
        current_end = current_start.advance(1, 'day')
        
        # Lọc ảnh final cho ngày hiện tại
        finalDayForDate = finalDay.filterDate(current_start, current_end)
        finalNightForDate = finalNight.filterDate(current_start, current_end)
        
        # Nếu có ảnh cho ngày này, xuất ảnh
        if finalDayForDate.size().getInfo() > 0:
            task_ids.append(start_export_task(create_export_task(
                finalDayForDate.first(),  # Lấy ảnh đầu tiên nếu có nhiều ảnh
                f'Final_LST_Day_{date_file_str}'
            )))
        else:
            print(f"No day data available for {date_str}")
        
        if finalNightForDate.size().getInfo() > 0:
            task_ids.append(start_export_task(create_export_task(
                finalNightForDate.first(),  # Lấy ảnh đầu tiên nếu có nhiều ảnh
                f'Final_LST_Night_{date_file_str}'
            )))
        else:
            print(f"No night data available for {date_str}")
    
    # Monitor export tasks
    if task_ids:
        print(f"\nStarted {len(task_ids)} export tasks. Starting monitoring...")
        monitor_tasks(task_ids, args.monitor_interval)
    else:
        print("No tasks to monitor. No data available for the specified date range.")

if __name__ == "__main__":
    main() 