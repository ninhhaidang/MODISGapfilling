import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
from glob import glob
from datetime import datetime

# Cấu hình đơn giản cho mục đích trực quan hóa
class VisConfig:
    VISUALIZATION_DIR = '/Users/ninhhaidang/Library/CloudStorage/GoogleDrive-ninhhailongg@gmail.com/My Drive/Cac_mon_hoc/Nam4_Ky2/Du_an_thuc_te/MODIS_LST_VN_VISUALIZATIONS'
    # Thư mục đầu ra cơ sở, giống như trong Config của VietNam1km.py
    BASE_OUTPUT_DIR = '/Users/ninhhaidang/Library/CloudStorage/GoogleDrive-ninhhailongg@gmail.com/My Drive/Cac_mon_hoc/Nam4_Ky2/Du_an_thuc_te/MODIS_LST_VN_PROCESSED'
    # RAW_DATA_DIR = '/Users/ninhhaidang/Library/CloudStorage/GoogleDrive-ninhhailongg@gmail.com/My Drive/Cac_mon_hoc/Nam4_Ky2/Du_an_thuc_te/MODIS_LST_VN_RAW' 

    # Đường dẫn đến dữ liệu GEE LST
    GEE_LST_DIR = '/Users/ninhhaidang/Library/CloudStorage/GoogleDrive-ninhhailongg@gmail.com/My Drive/Cac_mon_hoc/Nam4_Ky2/Du_an_thuc_te/GEE_LST_Vietnam'

    # Tên thư mục con, cần khớp với Config của VietNam1km.py
    FINAL_SUBDIR_NAME = 'Final_LST_Data'
    ORIGINAL_SUBDIR_NAME = 'Original_LST_Data'

    # Đường dẫn đầy đủ đến các thư mục con chứa file TIF
    FINAL_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, FINAL_SUBDIR_NAME)
    ORIGINAL_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, ORIGINAL_SUBDIR_NAME)

    VISUALIZATION_MIN_TEMP, VISUALIZATION_MAX_TEMP = -5, 35
    COLOR_MAP_COLORS = ['blue', 'cyan', 'green', 'yellow', 'orange', 'red']
    COLOR_MAP_BINS = 100

    @classmethod
    def initialize_dirs(cls):
        """Tạo các thư mục đầu ra nếu chưa tồn tại"""
        # Chỉ cần thư mục VISUALIZATION_DIR ở đây cho visualizer
        for dir_path in [cls.VISUALIZATION_DIR]: 
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

def clip_values(data, min_val, max_val):
    """Cắt giá trị trong khoảng xác định cho hiển thị"""
    clipped_data = np.copy(data)
    clipped_data[clipped_data < min_val] = min_val
    clipped_data[clipped_data > max_val] = max_val
    return clipped_data

def visualize_single_map(data, title, output_path, vmin=None, vmax=None, config=None):
    """Tạo biểu đồ cho dữ liệu LST (chỉnh sửa từ visualize_results)"""
    cfg = config or VisConfig
    cfg.initialize_dirs()

    vmin = vmin or cfg.VISUALIZATION_MIN_TEMP
    vmax = vmax or cfg.VISUALIZATION_MAX_TEMP
    
    data_clipped = clip_values(data, vmin, vmax) if vmin is not None and vmax is not None else data.copy()
    
    cmap = LinearSegmentedColormap.from_list(
        'gee_lst', cfg.COLOR_MAP_COLORS, N=cfg.COLOR_MAP_BINS
    )
    cmap.set_bad('white', 1.0)
    
    plt.figure(figsize=(10, 9))
    img = plt.imshow(np.ma.masked_invalid(data_clipped), cmap=cmap, vmin=vmin, vmax=vmax)
    
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap=cmap), 
                      extend='both', orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Temperature (°C)', fontsize=12)
    
    plt.title(title, fontsize=16)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu hình trực quan: {output_path}")

def create_comparison_figure(original_data, processed_data, gee_data, titles, output_path, vmin=None, vmax=None, config=None):
    """Tạo hình ảnh so sánh dữ liệu MODIS gốc, MODIS đã xử lý, và dữ liệu GEE"""
    cfg = config or VisConfig
    cfg.initialize_dirs()

    vmin = vmin or cfg.VISUALIZATION_MIN_TEMP
    vmax = vmax or cfg.VISUALIZATION_MAX_TEMP
        
    if vmin is not None and vmax is not None:
        original_clipped = clip_values(original_data, vmin, vmax) if original_data is not None else None
        processed_clipped = clip_values(processed_data, vmin, vmax) if processed_data is not None else None
        gee_clipped = clip_values(gee_data, vmin, vmax) if gee_data is not None else None
    else:
        original_clipped = original_data.copy() if original_data is not None else None
        processed_clipped = processed_data.copy() if processed_data is not None else None
        gee_clipped = gee_data.copy() if gee_data is not None else None
    
    cmap = LinearSegmentedColormap.from_list(
        'gee_lst', cfg.COLOR_MAP_COLORS, N=cfg.COLOR_MAP_BINS
    )
    cmap.set_bad('white', 1.0)
    
    fig, axs = plt.subplots(1, 3, figsize=(24, 8)) # Đã thay đổi thành 1 hàng, 3 cột
    
    # Vẽ dữ liệu MODIS gốc
    if original_clipped is not None:
        masked_original = np.ma.masked_invalid(original_clipped)
        im1 = axs[0].imshow(masked_original, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[0].set_title(titles[0], fontsize=14)
    else:
        axs[0].text(0.5, 0.5, 'Dữ liệu MODIS gốc\nKhông có sẵn', horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes)
    axs[0].axis('off')
    
    # Vẽ dữ liệu MODIS đã xử lý
    if processed_clipped is not None:
        masked_processed = np.ma.masked_invalid(processed_clipped)
        im2 = axs[1].imshow(masked_processed, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[1].set_title(titles[1], fontsize=14)
    else:
        axs[1].text(0.5, 0.5, 'Dữ liệu MODIS đã xử lý\nKhông có sẵn', horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
    axs[1].axis('off')

    # Vẽ dữ liệu GEE LST
    if gee_clipped is not None:
        masked_gee = np.ma.masked_invalid(gee_clipped)
        im3 = axs[2].imshow(masked_gee, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[2].set_title(titles[2], fontsize=14)
    else:
        axs[2].text(0.5, 0.5, 'Dữ liệu GEE LST\nKhông có sẵn', horizontalalignment='center', verticalalignment='center', transform=axs[2].transAxes)
    axs[2].axis('off')
    
    # Điều chỉnh layout để có không gian cho thanh màu
    fig.subplots_adjust(right=0.90) 
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # Điều chỉnh vị trí cho 3 biểu đồ
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap=cmap), cax=cbar_ax)
    cbar.set_label('Temperature (°C)')
    
    # plt.tight_layout(rect=[0, 0, 0.9, 1]) # tight_layout có thể xung đột với add_axes, bỏ qua hoặc điều chỉnh
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu hình so sánh: {output_path}")

def load_tif_data(file_path):
    """Đọc dữ liệu từ file TIF"""
    if not os.path.exists(file_path):
        print(f"Lỗi: Không tìm thấy file tại {file_path}")
        return None, None
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1)
            # Giả sử dữ liệu đã ở độ C và NaN thể hiện không có dữ liệu.
            # Nếu file TIF gốc là Kelvin: data = data * 0.02 - 273.15
            # Đối với các file TIF đã xử lý, chúng đã ở độ C.
            # Đối với các file TIF thô, chúng có thể cần chuyển đổi nếu visualizer tải trực tiếp.
            # Hiện tại, giả sử visualizer tải các file TIF *đã xử lý* hoặc *gốc đã kết hợp*
            # (combined_original) vốn đã ở độ C.
            profile = src.profile
            return data, profile
    except Exception as e:
        print(f"Lỗi khi tải file TIF {file_path}: {e}")
        return None, None

def find_matching_original_file(processed_filename, raw_data_dir, date_str, time_of_day, sources_str):
    """
    Tìm (các) file dữ liệu gốc tương ứng với một file đã xử lý.
    Đây là một giải pháp tạm thời và cần được cải thiện để mạnh mẽ hơn.
    Để đơn giản, chúng ta có thể cần lưu dữ liệu gốc đã kết hợp dưới dạng TIF,
    hoặc cải tiến hàm này để tái tạo nó từ các file Terra/Aqua thô.

    Một cách tiếp cận đơn giản hơn hiện tại: giả sử VietNam1km.py lưu một file TIF 'combined_original'.
    Nếu `Final_LST_Day_YYYY_MM_DD.tif` được xử lý, chúng ta tìm `Original_LST_Day_YYYY_MM_DD.tif`.
    """
    base_name = os.path.basename(processed_filename) # ví dụ: Final_LST_Day_2020_01_01.tif
    if base_name.startswith("Final_LST_"):
        original_base_name = base_name.replace("Final_LST_", "Original_LST_")
        # Tìm file gốc trong ORIGINAL_OUTPUT_DIR
        original_file_path = os.path.join(VisConfig.ORIGINAL_OUTPUT_DIR, original_base_name)
        if os.path.exists(original_file_path):
            return original_file_path
    print(f"Cảnh báo: Không thể tìm thấy file gốc khớp với {processed_filename} trong {VisConfig.ORIGINAL_OUTPUT_DIR}")
    return None

def parse_date_input(date_str_input, date_format='%Y-%m-%d'):
    """Chuyển đổi chuỗi ngày nhập vào thành đối tượng datetime và xử lý lỗi."""
    try:
        return datetime.strptime(date_str_input, date_format)
    except ValueError:
        print(f"Định dạng ngày không hợp lệ: '{date_str_input}'. Vui lòng sử dụng định dạng {date_format}.")
        return None

def main_visualize():
    """Hàm chính để chạy trực quan hóa với lựa chọn tương tác từ người dùng."""
    cfg = VisConfig
    cfg.initialize_dirs()

    print("MODIS LST Visualizer - Chào mừng bạn!")
    print("=======================================")

    print("Vui lòng chọn loại trực quan hóa bạn muốn thực hiện cho mỗi bộ dữ liệu tìm thấy:")
    print("1. Hiển thị ảnh MODIS Final (đã xử lý)")
    print("2. Hiển thị ảnh MODIS Original (gốc)")
    print("3. Hiển thị ảnh GEE LST")
    print("4. So sánh MODIS Final và MODIS Original")
    print("5. So sánh MODIS Final và GEE LST")
    print("6. So sánh MODIS Original và GEE LST")
    print("7. So sánh cả 3 ảnh (MODIS Original, MODIS Final, GEE LST)")
    print("8. Chạy tất cả các hình ảnh riêng lẻ và so sánh 3 chiều (nếu có đủ dữ liệu)")
    print("9. Thoát")
    choice = input("Nhập lựa chọn của bạn (1-9): ")

    # Chuyển đổi lựa chọn thành các cờ boolean (tương tự args trước đây)
    show_final_single = False
    show_original_single = False
    show_gee_single = False
    compare_final_original = False
    compare_final_gee = False
    compare_original_gee = False
    compare_3_way = False

    if choice == '1':
        show_final_single = True
    elif choice == '2':
        show_original_single = True
    elif choice == '3':
        show_gee_single = True
    elif choice == '4':
        compare_final_original = True
    elif choice == '5':
        compare_final_gee = True
    elif choice == '6':
        compare_original_gee = True
    elif choice == '7':
        compare_3_way = True
    elif choice == '8':
        show_final_single = True
        show_original_single = True
        show_gee_single = True
        compare_3_way = True
    elif choice == '9':
        print("Đang thoát chương trình.")
        return
    else:
        print("Lựa chọn không hợp lệ. Đang thoát.")
        return

    # Hỏi về việc lọc ngày
    filter_by_date = False
    single_date_filter = None
    date_range_filter_start = None
    date_range_filter_end = None

    filter_choice_date = input("\nBạn có muốn lọc kết quả theo ngày không? (c/k): ").lower()
    if filter_choice_date == 'c':
        filter_by_date = True
        date_filter_option = input("Lọc theo (1) Một ngày cụ thể hay (2) Một khoảng ngày? Nhập 1 hoặc 2: ")
        if date_filter_option == '1':
            date_str = input("Nhập ngày cụ thể (YYYY-MM-DD): ")
            single_date_filter = parse_date_input(date_str)
            if single_date_filter is None:
                return # Thoát nếu ngày không hợp lệ
        elif date_filter_option == '2':
            start_date_str = input("Nhập ngày bắt đầu (YYYY-MM-DD): ")
            date_range_filter_start = parse_date_input(start_date_str)
            if date_range_filter_start is None: return

            end_date_str = input("Nhập ngày kết thúc (YYYY-MM-DD): ")
            date_range_filter_end = parse_date_input(end_date_str)
            if date_range_filter_end is None: return

            if date_range_filter_start > date_range_filter_end:
                print("Ngày bắt đầu không thể lớn hơn ngày kết thúc. Đang thoát.")
                return
        else:
            print("Lựa chọn lọc ngày không hợp lệ. Sẽ xử lý tất cả các ngày.")
            filter_by_date = False # Reset lại nếu lựa chọn không đúng

    all_processed_files_pattern = os.path.join(cfg.FINAL_OUTPUT_DIR, 'Final_LST_*.tif')
    all_processed_files = glob(all_processed_files_pattern)

    if not all_processed_files:
        print(f"Không tìm thấy file đã xử lý nào khớp với mẫu: {all_processed_files_pattern}")
        return
    
    # Lọc danh sách file nếu có yêu cầu
    files_to_process = []
    if filter_by_date:
        for file_path in all_processed_files:
            try:
                base = os.path.basename(file_path)
                fn_parts = base.replace('.tif','').split('_')
                # Final_LST_Day_YYYY_MM_DD or Final_LST_Night_YYYY_MM_DD
                if len(fn_parts) >= 6 and fn_parts[0] == "Final" and fn_parts[1] == "LST":
                    file_date_str = f"{fn_parts[3]}_{fn_parts[4]}_{fn_parts[5]}" 
                    file_date_obj = datetime.strptime(file_date_str, '%Y_%m_%d')
                    
                    if single_date_filter and file_date_obj.date() == single_date_filter.date():
                        files_to_process.append(file_path)
                    elif date_range_filter_start and date_range_filter_end and \
                         date_range_filter_start.date() <= file_date_obj.date() <= date_range_filter_end.date():
                        files_to_process.append(file_path)
            except Exception as e:
                print(f"Lỗi khi trích xuất ngày từ file {file_path}: {e}. Bỏ qua file này.")
                continue
        if not files_to_process:
            print("Không tìm thấy file nào khớp với tiêu chí lọc ngày của bạn.")
            return
    else:
        files_to_process = all_processed_files

    found_any_visuals_to_generate = False

    for processed_file_path in files_to_process: # Duyệt qua danh sách đã lọc hoặc toàn bộ
        print(f"\nĐang xử lý trực quan hóa cho: {processed_file_path}")
        
        processed_data, _ = load_tif_data(processed_file_path)
        if processed_data is None:
            print(f"Không thể tải dữ liệu từ: {processed_file_path}")
            continue

        filename_parts = os.path.basename(processed_file_path).replace('.tif','').split('_')
        if len(filename_parts) < 5 or filename_parts[0] != "Final" or filename_parts[1] != "LST":
            print(f"Bỏ qua định dạng file không nhận dạng được: {processed_file_path}")
            continue
            
        time_of_day = filename_parts[2] 
        date_str_fn = f"{filename_parts[3]}_{filename_parts[4]}_{filename_parts[5]}"
        date_obj = datetime.strptime(date_str_fn, '%Y_%m_%d')
        
        title_processed = f'Final MODIS LST {time_of_day} (Đã lấp đầy, °C) - {date_obj.strftime("%Y-%m-%d")}'
        title_original = f'Original MODIS LST {time_of_day} (Thô, °C) - {date_obj.strftime("%Y-%m-%d")}'
        title_gee = f'GEE LST {time_of_day} (°C) - {date_obj.strftime("%Y-%m-%d")}'
        standard_titles_for_comparison = [title_original, title_processed, title_gee]

        original_data = None
        original_file_path = os.path.join(cfg.ORIGINAL_OUTPUT_DIR, f'Original_LST_{time_of_day}_{date_str_fn}.tif')
        if os.path.exists(original_file_path):
            original_data, _ = load_tif_data(original_file_path)
            if original_data is None:
                 print(f"Lỗi tải file MODIS gốc: {original_file_path}")
        
        gee_data = None
        gee_file_path = os.path.join(cfg.GEE_LST_DIR, f'GEE_LST_{time_of_day}_{date_str_fn}.tif')
        if os.path.exists(gee_file_path):
            gee_data, _ = load_tif_data(gee_file_path)
            if gee_data is None:
                print(f"Lỗi tải file GEE LST: {gee_file_path}")

        # === Xử lý các tùy chọn hiển thị riêng lẻ ===
        if show_final_single:
            if processed_data is not None:
                vis_output_path_processed = os.path.join(cfg.VISUALIZATION_DIR, f'Final_MODIS_{time_of_day}_{date_str_fn}.png')
                visualize_single_map(processed_data, title_processed, vis_output_path_processed, config=cfg)
                found_any_visuals_to_generate = True
            else:
                print(f"Không có dữ liệu Final MODIS để hiển thị cho {date_str_fn} {time_of_day}")

        if show_original_single:
            if original_data is not None:
                vis_output_path_original = os.path.join(cfg.VISUALIZATION_DIR, f'Original_MODIS_{time_of_day}_{date_str_fn}.png')
                visualize_single_map(original_data, title_original, vis_output_path_original, config=cfg)
                found_any_visuals_to_generate = True
            else:
                print(f"Không tìm thấy/lỗi tải file dữ liệu MODIS gốc cho {date_str_fn} {time_of_day}, bỏ qua hiển thị riêng lẻ.")

        if show_gee_single:
            if gee_data is not None:
                vis_output_path_gee = os.path.join(cfg.VISUALIZATION_DIR, f'GEE_LST_{time_of_day}_{date_str_fn}.png')
                visualize_single_map(gee_data, title_gee, vis_output_path_gee, config=cfg)
                found_any_visuals_to_generate = True
            else:
                print(f"Không tìm thấy/lỗi tải file dữ liệu GEE LST cho {date_str_fn} {time_of_day}, bỏ qua hiển thị riêng lẻ.")

        # === Xử lý các tùy chọn so sánh ===
        if compare_final_original:
            comp_fo_output_path = os.path.join(cfg.VISUALIZATION_DIR, f'Comparison_Final_Original_{time_of_day}_{date_str_fn}.png')
            print(f"Đang tạo so sánh Final MODIS & Original MODIS: {comp_fo_output_path}")
            create_comparison_figure(original_data, processed_data, None, standard_titles_for_comparison, comp_fo_output_path, config=cfg)
            found_any_visuals_to_generate = True

        if compare_final_gee:
            comp_fg_output_path = os.path.join(cfg.VISUALIZATION_DIR, f'Comparison_Final_GEE_{time_of_day}_{date_str_fn}.png')
            print(f"Đang tạo so sánh Final MODIS & GEE LST: {comp_fg_output_path}")
            create_comparison_figure(None, processed_data, gee_data, standard_titles_for_comparison, comp_fg_output_path, config=cfg)
            found_any_visuals_to_generate = True
        
        if compare_original_gee:
            comp_og_output_path = os.path.join(cfg.VISUALIZATION_DIR, f'Comparison_Original_GEE_{time_of_day}_{date_str_fn}.png')
            print(f"Đang tạo so sánh Original MODIS & GEE LST: {comp_og_output_path}")
            create_comparison_figure(original_data, None, gee_data, standard_titles_for_comparison, comp_og_output_path, config=cfg)
            found_any_visuals_to_generate = True

        if compare_3_way:
            comp_3way_output_path = os.path.join(cfg.VISUALIZATION_DIR, f'Comparison_3Way_MODIS_GEE_{time_of_day}_{date_str_fn}.png')
            print(f"Đang tạo so sánh 3 chiều: {comp_3way_output_path}")
            create_comparison_figure(original_data, processed_data, gee_data, standard_titles_for_comparison, comp_3way_output_path, config=cfg)
            found_any_visuals_to_generate = True
    
    if not found_any_visuals_to_generate and choice not in ['9']:
        print("\nKhông có hình ảnh nào được tạo dựa trên lựa chọn và dữ liệu có sẵn.")

if __name__ == "__main__":
    main_visualize() 