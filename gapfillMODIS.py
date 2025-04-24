import ee
import os
import datetime
import ee_monitor  # Module để theo dõi tiến trình các task export EarthEngine

# Authenticate Earth Engine với project
# Thay 'your-project-id' bằng ID Google Cloud Project của bạn
ee.Initialize(project='ee-bonglantrungmuoi')

# ===== 0. Thiết lập thời gian =====
startDate = '2003-01-01'
endDate = '2025-01-01'

# Sử dụng toàn bộ khoảng thời gian cho việc hiển thị/xuất dữ liệu
displayStart = ee.Date(startDate)
displayEnd = ee.Date(endDate)

# ===== 1. Định nghĩa khu vực quan tâm từ shapefile Việt Nam =====
viet_nam = ee.FeatureCollection("projects/ee-bonglantrungmuoi/assets/viet_nam")
# Lấy geometry từ FeatureCollection để tránh lỗi
viet_nam_geometry = viet_nam.geometry()

# ===== 2. Tải dữ liệu MODIS LST =====
modisTerra = ee.ImageCollection('MODIS/061/MOD11A1') \
    .filterDate(startDate, endDate) \
    .filterBounds(viet_nam) \
    .select(['LST_Day_1km', 'LST_Night_1km', 'QC_Day', 'QC_Night'])

modisAqua = ee.ImageCollection('MODIS/061/MYD11A1') \
    .filterDate(startDate, endDate) \
    .filterBounds(viet_nam) \
    .select(['LST_Day_1km', 'LST_Night_1km', 'QC_Day', 'QC_Night'])

# ===== 3. Hàm lọc chất lượng =====
def filterQuality(image):
    qcDay = image.select('QC_Day')
    qcNight = image.select('QC_Night')
    qualityMaskDay = qcDay.bitwiseAnd(3).neq(3)
    qualityMaskNight = qcNight.bitwiseAnd(3).neq(3)
    return image.addBands(image.select('LST_Day_1km').updateMask(qualityMaskDay).rename('filtered_LST_Day_1km')) \
        .addBands(image.select('LST_Night_1km').updateMask(qualityMaskNight).rename('filtered_LST_Night_1km'))

modisTerraFiltered = modisTerra.map(filterQuality)
modisAquaFiltered = modisAqua.map(filterQuality)

# ===== 4. Kết hợp dữ liệu =====
combined = modisTerraFiltered.merge(modisAquaFiltered)

# ===== 5. Xử lý riêng ban ngày và ban đêm (đổi từ K sang °C) =====
def convert_to_celsius(image):
    return image.multiply(0.02).subtract(273.15) \
        .set('system:time_start', image.get('system:time_start'))

dayData = combined.select('filtered_LST_Day_1km').map(lambda image: 
    convert_to_celsius(image).rename('LST_Day_1km_C'))

nightData = combined.select('filtered_LST_Night_1km').map(lambda image: 
    convert_to_celsius(image).rename('LST_Night_1km_C'))

# ===== 6. Trung bình dài hạn =====
meanDay = dayData.mean()
meanNight = nightData.mean()

# ===== 7. Tính phần dư =====
residualsDay = dayData.map(lambda image: 
    image.subtract(meanDay)
    .rename('residual_day')
    .set('system:time_start', image.get('system:time_start')))

residualsNight = nightData.map(lambda image: 
    image.subtract(meanNight)
    .rename('residual_night')
    .set('system:time_start', image.get('system:time_start')))

# ===== 8. Làm mượt không gian =====
smoothedResidualsDay = residualsDay.map(lambda image: 
    image.focal_mean(radius=3, kernelType='square', units='pixels')
    .focal_mean(radius=3, kernelType='square', units='pixels')
    .set('system:time_start', image.get('system:time_start')))

smoothedResidualsNight = residualsNight.map(lambda image: 
    image.focal_mean(radius=3, kernelType='square', units='pixels')
    .focal_mean(radius=3, kernelType='square', units='pixels')
    .set('system:time_start', image.get('system:time_start')))

# ===== 9. Làm mượt thời gian =====
def temporalSmoothing(collection):
    def apply_smoothing(image):
        time = image.get('system:time_start')
        
        def process_image():
            date = ee.Date(time)
            before = collection.filterDate(date.advance(-7, 'day'), date)
            after = collection.filterDate(date, date.advance(7, 'day'))
            window = before.merge(after).mean()
            return image.unmask(window).set('system:time_start', time)
        
        return ee.Algorithms.If(
            ee.Algorithms.IsEqual(time, None),
            image,
            process_image()
        )
    
    return collection.map(apply_smoothing)

smoothedResidualsDayTemporal = temporalSmoothing(smoothedResidualsDay)
smoothedResidualsNightTemporal = temporalSmoothing(smoothedResidualsNight)

# ===== 10. Tạo dữ liệu cuối cùng (clip theo shapefile Việt Nam) =====
finalDay = smoothedResidualsDayTemporal.map(lambda image: 
    image.add(meanDay)
    .rename('final_LST_Day_1km_C')
    .clip(viet_nam)
    .set('system:time_start', image.get('system:time_start')))

finalNight = smoothedResidualsNightTemporal.map(lambda image: 
    image.add(meanNight)
    .rename('final_LST_Night_1km_C')
    .clip(viet_nam)
    .set('system:time_start', image.get('system:time_start')))

# ===== 11. Export dữ liệu =====
# Tạo thư mục để lưu dữ liệu
output_folder = 'LST_Vietnam'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 11.1 Xuất dữ liệu cuối cùng
finalDayDisplay = finalDay.filterDate(displayStart, displayEnd)
finalNightDisplay = finalNight.filterDate(displayStart, displayEnd)

# Format date for filename
start_date_str = displayStart.format('yyyy-MM-dd').getInfo()
end_date_str = displayEnd.format('yyyy-MM-dd').getInfo()
date_filename = f"{start_date_str}_to_{end_date_str}"

# Tạo một hàm để khởi tạo và bắt đầu export task
def export_image(image, description):
    task = ee.batch.Export.image.toDrive(
        image=image.mean(),
        description=description,
        folder='LST_Vietnam',
        region=viet_nam_geometry,
        scale=1000,
        maxPixels=1e13,
        crs='EPSG:4326'
    )
    task.start()
    return task.id

# Danh sách các cặp (image, description) để xuất
export_list = [
    (finalDayDisplay, f'Final_LST_Day_{date_filename}'),
    (finalNightDisplay, f'Final_LST_Night_{date_filename}')
]

# Tạo và bắt đầu tất cả các task, đồng thời lưu IDs
task_ids = [export_image(img, desc) for img, desc in export_list]

print("Các tác vụ đã được bắt đầu. Vui lòng kiểm tra Google Drive của bạn sau khi hoàn thành.")
print("Dữ liệu sẽ được lưu trong thư mục 'LST_Vietnam' trên Google Drive của bạn.") 

# Theo dõi tất cả các tasks
print("\nĐang bắt đầu theo dõi tiến trình...")
ee_monitor.monitor_tasks(task_ids) 