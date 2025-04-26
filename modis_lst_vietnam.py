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
output_dir = r"D:\HaiDang\MODISGapfilling\RAW"
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

terra_size = modisTerra.size().getInfo()
aqua_size = modisAqua.size().getInfo()

print(f"Tổng số ảnh Terra: {terra_size}")
print(f"Tổng số ảnh Aqua: {aqua_size}")

# Hàm tải xuống một ảnh
def download_single_image(args):
    collection_list, collection_name, index, total_images = args
    try:
        print(f"Bắt đầu tải {collection_name} ảnh {index+1}/{total_images}...")
        image = ee.Image(collection_list.get(index))
        image_date = ee.Date(image.get('system:time_start')).format('yyyy-MM-dd').getInfo()
        
        # Tạo ảnh riêng cho ban ngày và ban đêm
        day_image = image.select(['LST_Day_1km', 'QC_Day'])
        night_image = image.select(['LST_Night_1km', 'QC_Night'])
        
        # Tạo tên file xuất cho ảnh ngày
        day_filename = os.path.join(output_dir, f"{collection_name}_Day_{image_date}.tif")
        
        # Xuất ảnh ngày
        print(f"  - Tải ảnh ngày {collection_name} {image_date}...")
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
        print(f"  - Tải ảnh đêm {collection_name} {image_date}...")
        geemap.ee_export_image(
            ee_object=night_image,
            filename=night_filename,
            scale=1000,
            region=viet_nam_geometry,
            crs="EPSG:4326",
            file_per_band=False
        )
        
        return f"✓ Hoàn thành: {collection_name} {image_date} (ngày + đêm)"
    except Exception as e:
        return f"✗ Lỗi: {collection_name} ảnh {index+1}/{total_images}: {str(e)}"

# Số luồng xử lý song song (điều chỉnh theo cấu hình máy)
max_workers = 12  # Tăng lên để tận dụng băng thông mạng 86Mbps

# Kích thước lô - Số ảnh xử lý mỗi lần
batch_size = 100  # Tăng kích thước lô để tận dụng băng thông tốt hơn

# Tải Terra và Aqua đồng thời để tận dụng tối đa băng thông
print("\n=== BẮT ĐẦU TẢI DỮ LIỆU ===")
print(f"Thư mục lưu: {output_dir}")
print(f"Số luồng xử lý: {max_workers}")
print(f"Tổng số ảnh cần tải: Terra ({terra_size}) + Aqua ({aqua_size})")
print(f"Kích thước lô: {batch_size} ảnh mỗi lần")
print("===========================\n")

start_time = time.time()
total_completed = 0

# Tạo danh sách chỉ số cần tải
all_indices = []
for i in range(terra_size):
    all_indices.append(("Terra", i))
for i in range(aqua_size):
    all_indices.append(("Aqua", i))

total_images = len(all_indices)
print(f"Tổng cộng {total_images} ảnh cần tải")

# Xử lý theo lô
for batch_start in range(0, total_images, batch_size):
    batch_end = min(batch_start + batch_size, total_images)
    current_batch = all_indices[batch_start:batch_end]
    batch_size_actual = len(current_batch)
    
    print(f"\nBắt đầu tải lô {batch_start//batch_size + 1}/{(total_images + batch_size - 1)//batch_size}, "
          f"ảnh {batch_start+1}-{batch_end} / {total_images}")
    
    # Chuẩn bị danh sách công việc cho lô hiện tại
    batch_args = []
    for sat_name, idx in current_batch:
        if sat_name == "Terra":
            batch_args.append((terra_list, sat_name, idx, terra_size))
        else:
            batch_args.append((aqua_list, sat_name, idx, aqua_size))
    
    # Khởi chạy tải song song cho lô hiện tại
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, batch_size_actual)) as executor:
        futures = [executor.submit(download_single_image, args) for args in batch_args]
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            total_completed += 1
            print(f"[{total_completed}/{total_images}] {result}")
            # Đảm bảo output được hiển thị ngay lập tức
            import sys
            sys.stdout.flush()
    
    current_time = time.time()
    elapsed = current_time - start_time
    percent_complete = total_completed / total_images * 100
    
    print(f"\nĐã hoàn thành: {total_completed}/{total_images} ảnh ({percent_complete:.1f}%)")
    print(f"Thời gian đã trôi qua: {elapsed//3600:.0f}h {(elapsed%3600)//60:.0f}m {elapsed%60:.0f}s")
    
    if total_completed > 0:
        avg_time_per_image = elapsed / total_completed
        remaining_images = total_images - total_completed
        est_remaining_time = remaining_images * avg_time_per_image
        
        print(f"Ước tính thời gian còn lại: {est_remaining_time//3600:.0f}h {(est_remaining_time%3600)//60:.0f}m {est_remaining_time%60:.0f}s")
        print(f"Thời gian trung bình mỗi ảnh: {avg_time_per_image:.1f}s")

end_time = time.time()
total_time = end_time - start_time

print("\n=== TẢI XONG ===")
print(f"Tổng thời gian: {total_time//3600:.0f}h {(total_time%3600)//60:.0f}m {total_time%60:.0f}s")
print(f"Tất cả ảnh đã được tải về thư mục: {output_dir}") 