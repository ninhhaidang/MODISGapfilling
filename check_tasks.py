import ee
import sys
import os

# Thêm thư mục gốc vào path để import module ee_monitor
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import ee_monitor

def main():
    # Khởi tạo Earth Engine
    print("Đang khởi tạo Earth Engine...")
    project_id = input("Nhập ID project (nhấn Enter để sử dụng 'ee-bonglantrungmuoi'): ").strip()
    if not project_id:
        project_id = 'ee-bonglantrungmuoi'
    
    try:
        ee.Initialize(project=project_id)
        print(f"Đã khởi tạo Earth Engine với project '{project_id}'")
    except Exception as e:
        print(f"Lỗi khi khởi tạo Earth Engine: {e}")
        return
    
    # Lấy danh sách các task gần đây
    try:
        print("\nĐang lấy danh sách các task gần đây...")
        limit = int(input("Số lượng task muốn hiển thị (mặc định: 10): ") or "10")
        recent_tasks = ee_monitor.get_recent_tasks(limit=limit)
        
        if not recent_tasks:
            print("Không tìm thấy task nào.")
            return
            
        print("\n===== DANH SÁCH TASKS =====")
        ee_monitor.print_task_list(recent_tasks)
        
        # Yêu cầu người dùng chọn các task để theo dõi
        print("\nNhập STT các task muốn theo dõi (phân cách bằng dấu phẩy, ví dụ: 1,2,5)")
        print("Hoặc nhập 'all' để theo dõi tất cả các task hiển thị")
        choices = input("Lựa chọn của bạn: ").strip()
        
        task_ids = []
        if choices.lower() == 'all':
            task_ids = [task['id'] for task in recent_tasks]
        else:
            try:
                # Chuyển đổi từ STT sang indices (STT bắt đầu từ 1, indices từ 0)
                indices = [int(idx.strip()) - 1 for idx in choices.split(',') if idx.strip()]
                for idx in indices:
                    if 0 <= idx < len(recent_tasks):
                        task_ids.append(recent_tasks[idx]['id'])
                    else:
                        print(f"Cảnh báo: STT {idx+1} không hợp lệ, bỏ qua.")
            except ValueError:
                print("Lỗi: Vui lòng nhập các số nguyên phân cách bằng dấu phẩy.")
                return
        
        # Theo dõi các task đã chọn
        if task_ids:
            print(f"\nĐang theo dõi {len(task_ids)} task...")
            interval = int(input("Khoảng thời gian cập nhật (giây, mặc định: 10): ") or "10")
            ee_monitor.monitor_tasks(task_ids, interval=interval)
        else:
            print("Không có task nào được chọn.")
            
    except Exception as e:
        print(f"Lỗi không mong muốn: {e}")

if __name__ == "__main__":
    print("==== CÔNG CỤ THEO DÕI TASK EARTH ENGINE ====")
    main()
    print("\nĐã kết thúc chương trình.") 