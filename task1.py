import cv2 as cv
import numpy as np
import os
import math

# --- 1. Thiết lập các biến ---
input_video = 'task1.mp4'
# --- THAY ĐỔI: Đổi lại output sang .mp4 ---
output_video = 'output_task1.mp4' # Cập nhật tên file

if not os.path.exists(input_video):
    print(f"Lỗi: Không tìm thấy file '{input_video}'.")
    exit()

# --- 2. Khởi tạo đối tượng đọc/ghi video ---
cap = cv.VideoCapture(input_video)

if not cap.isOpened():
    print("Lỗi: Không thể mở video file")
    exit()

frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CAP_PROP_FPS))
size = (frame_width, frame_height) # Giữ kích thước output là ảnh gốc

# --- THAY ĐỔI: Đổi lại codec sang mp4v ---
fourcc = cv.VideoWriter_fourcc(*'mp4v')
result = cv.VideoWriter(output_video, fourcc, fps, size)

print(f"Đang xử lý video... (Chiến lược: Pass 1 - Separate Contours)") # Cập nhật print
print(f"File output: {output_video}")

# --- THIẾT LẬP VÙNG ROI TĨNH (STATIC ROI) ---
roi_y_start = 0
roi_y_end = int(frame_height * 55) # Chỉ lấy 40% trên
print(f"Kích thước gốc: {frame_width}x{frame_height}. Xử lý ROI: {frame_width}x{roi_y_end}")


# --- THÔNG SỐ TINH CHỈNH TỪ BẠN BẠN ---
# (Giữ nguyên các thông số màu sắc, blur, clahe, ksize, iter)
# --- Thông số cho MÀU XANH DƯƠNG (Blue) ---
blue_lower_h = 102
blue_upper_h = 144
blue_lower_s = 150
blue_upper_s = 255
blue_lower_v = 81
blue_upper_v = 227
blue_ksize = 7
blue_open_iter = 1
blue_close_iter = 5
blue_clahe_clip_limit = 30
blue_blur_ksize = 7

# --- Thông số cho MÀU ĐỎ (Red) ---
red_lower_h1 = 0
red_upper_h1 = 10
red_lower_h2 = 117
red_upper_h2 = 179
red_lower_s = 40
red_upper_s = 255
red_lower_v = 0
red_upper_v = 255
red_ksize = 3
red_open_iter = 2
red_close_iter = 5
red_clahe_clip_limit = 30
red_blur_ksize = 5

# --- Thông số cho MÀU VÀNG (Yellow) ---
yellow_lower_h = 8
yellow_upper_h = 18
yellow_lower_s = 111
yellow_upper_s = 255
yellow_lower_v = 100
yellow_upper_v = 255
yellow_ksize = 3
yellow_open_iter = 1
yellow_close_iter = 5
yellow_clahe_clip_limit = 30
yellow_blur_ksize = 7

# --- Ngưỡng lọc Contours (Giữ nguyên ngưỡng khắt khe của bạn) ---
min_contour_area = 1000
min_hull_circularity = 0.88
min_aspect_ratio = 0.7
max_aspect_ratio = 1.3

# --- Hàm xử lý cho từng kênh màu ---
# (Hàm process_color_channel giữ nguyên)
def process_color_channel(frame, lower_h, upper_h, lower_s, upper_s, lower_v, upper_v,
                          blur_ksize, clahe_clip_limit, ksize, open_iter, close_iter,
                          is_red=False, lower_h2=None, upper_h2=None):
    """Xử lý một kênh màu riêng biệt."""
    try:
        # Tiền xử lý riêng
        blur_ksize_odd = max(1, blur_ksize if blur_ksize % 2 != 0 else blur_ksize + 1)
        blurred = cv.medianBlur(frame, blur_ksize_odd)
        hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)

        # CLAHE
        clahe = cv.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(1, 1))
        v_clahe = clahe.apply(v)
        hsv_enhanced = cv.merge([h, s, v_clahe])

        # Lọc màu
        lower_bound = np.array([lower_h, lower_s, lower_v])
        upper_bound = np.array([upper_h, upper_s, upper_v])
        mask = cv.inRange(hsv_enhanced, lower_bound, upper_bound)

        # Xử lý riêng cho màu đỏ (2 dải H)
        if is_red and lower_h2 is not None and upper_h2 is not None:
            lower_bound2 = np.array([lower_h2, lower_s, lower_v])
            upper_bound2 = np.array([upper_h2, upper_s, upper_v])
            mask2 = cv.inRange(hsv_enhanced, lower_bound2, upper_bound2)
            mask = cv.add(mask, mask2)

        # Lọc hình thái riêng
        ksize_odd = max(1, ksize if ksize % 2 != 0 else ksize + 1)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (ksize_odd, ksize_odd))
        mask_opened = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=open_iter)
        mask_closed = cv.morphologyEx(mask_opened, cv.MORPH_CLOSE, kernel, iterations=close_iter)
        return mask_closed
    except Exception as e:
        print(f"Lỗi khi xử lý màu H={lower_h}-{upper_h}: {e}")
        return np.zeros_like(frame[:,:,0], dtype=np.uint8)


# --- THÊM MỚI: HÀM TÌM VÀ LỌC CONTOURS ---
def find_and_filter_contours(mask_processed, output_frame_to_draw, y_offset, debug_color_bgr):
    """
    Tìm contours trên một mask, lọc chúng, và vẽ text debug.
    Trả về một danh sách các bounding box (x, y, w, h) đã được lọc.
    """
    contours, _ = cv.findContours(mask_processed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    detected_rects = []

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < min_contour_area: # Lọc bỏ nhiễu siêu nhỏ trước
            continue

        try:
            hull = cv.convexHull(cnt)
        except cv.error as e:
            continue

        hull_area = cv.contourArea(hull)
        hull_perimeter = cv.arcLength(hull, True)
        
        x, y, w, h = cv.boundingRect(cnt)
        if w == 0 or h == 0: continue

        # Tính toán các giá trị
        circularity = 0
        if hull_perimeter > 0:
            circularity = (4 * np.pi * hull_area) / (hull_perimeter ** 2)
        
        aspect_ratio = float(w) / h
        
        # --- Vẽ thông số Area và Circularity (Debug) ---
        text_area = f"A: {area}"
        text_circ = f"C: {circularity:.2f}"
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        passes_circularity = circularity >= min_hull_circularity # Ngưỡng 0.88
        passes_aspect = min_aspect_ratio <= aspect_ratio <= max_aspect_ratio # Ngưỡng 0.8-1.2
        
        is_passed = passes_circularity and passes_aspect
        
        # Quyết định màu sắc: Xanh mạ (Cyan) nếu ĐẠT, nếu KHÔNG ĐẠT thì dùng màu debug của kênh
        text_color = (255, 255, 0) if is_passed else debug_color_bgr

        # Lấy tọa độ Y tuyệt đối (cộng offset ROI)
        y_on_frame = y + y_offset
        
        # Vẽ 2 dòng text
        cv.putText(output_frame_to_draw, text_area, (x, y_on_frame - 25), font, font_scale, text_color, thickness)
        cv.putText(output_frame_to_draw, text_circ, (x, y_on_frame - 5), font, font_scale, text_color, thickness)
        
        # Áp dụng các bộ lọc
        if not passes_circularity:
            continue
            
        if not passes_aspect:
            continue

        # Nếu vượt qua, thêm (x, y, w, h) của ROI (chưa có offset)
        detected_rects.append((x, y, w, h))
        
    return detected_rects
# --- KẾT THÚC HÀM MỚI ---


# --- 3. Vòng lặp xử lý video ---
id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    id += 1
    if id % 100 == 0:
        print(f"Đã xử lý frame {id}...")

    # === PASS 1: DETECTION (MULTI-CHANNEL + SEPARATE CONTOURS) ===
    
    # --- BƯỚC 0: CẮT LẤY VÙNG ROI ---
    roi_frame = frame[roi_y_start:roi_y_end, :] 
    output_frame = frame.copy()

    # --- BƯỚC 1: XỬ LÝ TỪNG KÊNH MÀU (TRÊN ROI) ---
    mask_processed_red = process_color_channel(
        roi_frame,
        red_lower_h1, red_upper_h1, red_lower_s, red_upper_s, red_lower_v, red_upper_v,
        red_blur_ksize, red_clahe_clip_limit, red_ksize, red_open_iter, red_close_iter,
        is_red=True, lower_h2=red_lower_h2, upper_h2=red_upper_h2
    )

    mask_processed_blue = process_color_channel(
        roi_frame,
        blue_lower_h, blue_upper_h, blue_lower_s, blue_upper_s, blue_lower_v, blue_upper_v,
        blue_blur_ksize, blue_clahe_clip_limit, blue_ksize, blue_open_iter, blue_close_iter
    )

    mask_processed_yellow = process_color_channel(
        roi_frame,
        yellow_lower_h, yellow_upper_h, yellow_lower_s, yellow_upper_s, yellow_lower_v, yellow_upper_v,
        yellow_blur_ksize, yellow_clahe_clip_limit, yellow_ksize, yellow_open_iter, yellow_close_iter
    )

    # --- BƯỚC 2: TÌM VÀ LỌC CONTOURS RIÊNG BIỆT ---
    # Bỏ qua bước gộp mask
    
    # Xử lý contours từ mask ĐỎ (debug text màu Đỏ)
    red_rects = find_and_filter_contours(
        mask_processed_red, output_frame, roi_y_start, (0, 0, 255) 
    )
    
    # Xử lý contours từ mask XANH (debug text màu Xanh)
    blue_rects = find_and_filter_contours(
        mask_processed_blue, output_frame, roi_y_start, (255, 0, 0)
    )
    
    # Xử lý contours từ mask VÀNG (debug text màu Vàng)
    yellow_rects = find_and_filter_contours(
        mask_processed_yellow, output_frame, roi_y_start, (0, 255, 255)
    )

    # Gộp tất cả các rects đã được lọc
    all_detected_rects = red_rects + blue_rects + yellow_rects

    # --- BƯỚC 3: VẼ KẾT QUẢ (Bounding Box màu xanh lá) ---
    for (x, y, w, h) in all_detected_rects:
        # QUAN TRỌNG: Phải cộng y với roi_y_start
        y_on_frame = y + roi_y_start
        
        cv.rectangle(output_frame, (x, y_on_frame), (x + w, y_on_frame + h), (0, 255, 0), 2)


    # ============================

    # --- THAY ĐỔI: Hiển thị các mask riêng lẻ ---
    cv.imshow("Mask Red", mask_processed_red)
    cv.imshow("Mask Blue", mask_processed_blue)
    cv.imshow("Mask Yellow", mask_processed_yellow)
    cv.imshow("Output", output_frame) # Giữ lại imshow Output
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    # Ghi frame (Vẫn comment lại theo code của bạn)
    result.write(output_frame)

# --- 4. Dọn dẹp ---
print(f"Xử lý hoàn tất! Video đã được lưu tại: {output_video}") # Cập nhật print
cap.release()
result.release() # Bỏ comment nếu muốn lưu video
cv.destroyAllWindows()

