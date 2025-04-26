import ee
import geemap
import os
import datetime

# Authenticate với Earth Engine
ee.Initialize(project='ee-bonglantrungmuoi')

# Thiết lập thời gian
startDate = '2003-01-01'
endDate = '2025-01-01'

# Định nghĩa khu vực Việt Nam từ shapefile
viet_nam = ee.FeatureCollection("projects/ee-bonglantrungmuoi/assets/viet_nam")
viet_nam_geometry = viet_nam.geometry()

# Thiết lập thư mục đầu ra theo yêu cầu
output_dir = r"D:\HaiDang\MODISGapfilling\MODIS_LST_RAW"
os.makedirs(output_dir, exist_ok=True)

# Tạo thư mục con cho ảnh ngày và đêm
day_dir = os.path.join(output_dir, "Day")
night_dir = os.path.join(output_dir, "Night")
os.makedirs(day_dir, exist_ok=True)
os.makedirs(night_dir, exist_ok=True)

# Tải dữ liệu MODIS LST Terra và Aqua (raw)
modisTerra = ee.ImageCollection('MODIS/061/MOD11A1') \
    .filterDate(startDate, endDate) \
    .filterBounds(viet_nam) \
    .select(['LST_Day_1km', 'LST_Night_1km', 'QC_Day', 'QC_Night'])

modisAqua = ee.ImageCollection('MODIS/061/MYD11A1') \
    .filterDate(startDate, endDate) \
    .filterBounds(viet_nam) \
    .select(['LST_Day_1km', 'LST_Night_1km', 'QC_Day', 'QC_Night'])

# Chuyển các bộ dữ liệu thành danh sách ảnh
terra_list = modisTerra.toList(modisTerra.size())
aqua_list = modisAqua.toList(modisAqua.size())

print(f"Tổng số ảnh Terra: {modisTerra.size().getInfo()}")
print(f"Tổng số ảnh Aqua: {modisAqua.size().getInfo()}")

# Hàm tải xuống ảnh
def download_collection(collection_list, collection_name, total_images):
    for i in range(total_images):
        try:
            image = ee.Image(collection_list.get(i))
            image_date = ee.Date(image.get('system:time_start')).format('yyyy-MM-dd').getInfo()
            
            # Tạo ảnh riêng cho ban ngày và ban đêm
            day_image = image.select(['LST_Day_1km', 'QC_Day'])
            night_image = image.select(['LST_Night_1km', 'QC_Night'])
            
            # Tạo tên file xuất cho ảnh ngày
            day_filename = os.path.join(day_dir, f"{collection_name}_Day_{image_date}.tif")
            
            # Xuất ảnh ngày
            geemap.ee_export_image(
                ee_object=day_image,
                filename=day_filename,
                scale=1000,
                region=viet_nam_geometry,
                crs="EPSG:4326",
                file_per_band=False
            )
            
            # Tạo tên file xuất cho ảnh đêm
            night_filename = os.path.join(night_dir, f"{collection_name}_Night_{image_date}.tif")
            
            # Xuất ảnh đêm
            geemap.ee_export_image(
                ee_object=night_image,
                filename=night_filename,
                scale=1000,
                region=viet_nam_geometry,
                crs="EPSG:4326",
                file_per_band=False
            )
            
            print(f"Đã tải xuống {day_filename} và {night_filename}")
        except Exception as e:
            print(f"Lỗi khi tải ảnh {i} của {collection_name}: {e}")

# Tải xuống dữ liệu Terra và Aqua
print("Bắt đầu tải xuống dữ liệu Terra...")
download_collection(terra_list, "Terra", modisTerra.size().getInfo())

print("Bắt đầu tải xuống dữ liệu Aqua...")
download_collection(aqua_list, "Aqua", modisAqua.size().getInfo())

print(f"Tất cả ảnh đã được tải về thư mục: {output_dir}")
print(f"- Ảnh ngày: {day_dir}")
print(f"- Ảnh đêm: {night_dir}") 