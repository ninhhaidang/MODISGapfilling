import ee
import geemap
import os
import datetime
import concurrent.futures
import time

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

# Hàm tải xuống một ảnh
def download_single_image(args):
    collection_list, collection_name, index, total_images = args
    try:
        image = ee.Image(collection_list.get(index))
        image_date = ee.Date(image.get('system:time_start')).format('yyyy-MM-dd').getInfo()
        
        # Tạo ảnh riêng cho ban ngày và ban đêm
        day_image = image.select(['LST_Day_1km', 'QC_Day'])
        night_image = image.select(['LST_Night_1km', 'QC_Night'])
        
        # Tạo tên file xuất cho ảnh ngày
        day_filename = os.path.join(output_dir, f"{collection_name}_Day_{image_date}.tif")
        
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
        night_filename = os.path.join(output_dir, f"{collection_name}_Night_{image_date}.tif")
        
        # Xuất ảnh đêm
        geemap.ee_export_image(
            ee_object=night_image,
            filename=night_filename,
            scale=1000,
            region=viet_nam_geometry,
            crs="EPSG:4326",
            file_per_band=False
        )
        
        return f"Đã tải xuống {day_filename} và {night_filename}"
    except Exception as e:
        return f"Lỗi khi tải ảnh {index} của {collection_name}: {e}"

# Số luồng xử lý song song (điều chỉnh theo cấu hình máy)
max_workers = 8  # Tăng lên vì CPU có 12 cores, 24 threads

# Tải xuống dữ liệu Terra và Aqua song song
print("Bắt đầu tải xuống song song...")
start_time = time.time()

# Tải Terra và Aqua đồng thời để tận dụng tối đa băng thông
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers*2) as executor:
    # Chuẩn bị args cho tất cả các ảnh Terra và Aqua
    terra_args = [(terra_list, "Terra", i, modisTerra.size().getInfo()) for i in range(modisTerra.size().getInfo())]
    aqua_args = [(aqua_list, "Aqua", i, modisAqua.size().getInfo()) for i in range(modisAqua.size().getInfo())]
    
    # Chạy cả hai bộ dữ liệu song song
    all_futures = []
    
    for args in terra_args:
        future = executor.submit(download_single_image, args)
        all_futures.append(future)
    
    for args in aqua_args:
        future = executor.submit(download_single_image, args)
        all_futures.append(future)
    
    # Xử lý kết quả khi hoàn thành
    completed = 0
    total = len(all_futures)
    
    for future in concurrent.futures.as_completed(all_futures):
        completed += 1
        result = future.result()
        print(f"[{completed}/{total}] {result}")

end_time = time.time()
print(f"Tổng thời gian tải: {end_time - start_time:.2f} giây")
print(f"Tất cả ảnh đã được tải về thư mục: {output_dir}") 