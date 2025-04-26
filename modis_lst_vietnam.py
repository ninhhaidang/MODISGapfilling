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
output_dir = r"D:\HaiDang\MODISGapfilling\MODIS_LST_VN_RAW"
os.makedirs(output_dir, exist_ok=True)

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
            
            # Tạo tên file xuất
            filename = os.path.join(output_dir, f"{collection_name}_{image_date}.tif")
            
            # Xuất ảnh
            geemap.ee_export_image(
                ee_object=image,
                filename=filename,
                scale=1000,
                region=viet_nam_geometry,
                crs="EPSG:4326",
                file_per_band=False
            )
            
            print(f"Đã tải xuống {filename}")
        except Exception as e:
            print(f"Lỗi khi tải ảnh {i} của {collection_name}: {e}")

# Tải xuống dữ liệu Terra và Aqua
print("Bắt đầu tải xuống dữ liệu Terra...")
download_collection(terra_list, "Terra", modisTerra.size().getInfo())

print("Bắt đầu tải xuống dữ liệu Aqua...")
download_collection(aqua_list, "Aqua", modisAqua.size().getInfo())

print(f"Tất cả ảnh đã được tải về thư mục: {output_dir}") 