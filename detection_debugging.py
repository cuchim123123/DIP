import cv2 as cv
import numpy as np
import os
import math
import collections

# --- 1. Thiết lập các biến ---
INPUT_VIDEO = '../task1.mp4'
# --- THAY ĐỔI: Đổi lại output sang .mp4 ---
OUTPUT_VIDEO = 'debug.mp4' # Output debug video

if not os.path.exists(INPUT_VIDEO):
    print(f"Lỗi: Không tìm thấy file '{INPUT_VIDEO}'.")
    exit()

# --- 2. Khởi tạo đối tượng đọc/ghi video ---
cap = cv.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    print("Lỗi: Không thể mở video file")
    exit()

FRAME_WIDTH = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
FRAME_HEIGHT = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
FPS = int(cap.get(cv.CAP_PROP_FPS))
SIZE = (FRAME_WIDTH, FRAME_HEIGHT) # Giữ kích thước output là ảnh gốc

# --- THAY ĐỔI: Đổi lại codec sang mp4v ---
fourcc = cv.VideoWriter_fourcc(*'mp4v')
result = cv.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, SIZE)

# --- THIẾT LẬP VÙNG ROI TĨNH (STATIC ROI) ---
ROI_Y_START = 0
ROI_Y_END = int(FRAME_HEIGHT * 0.44) 
print(f"Kích thước gốc: {FRAME_WIDTH}x{FRAME_HEIGHT}. Xử lý ROI: {FRAME_WIDTH}x{ROI_Y_END}")


# --- THÔNG SỐ TINH CHỈNH TỪ BẠN BẠN ---
# (Giữ nguyên các thông số màu sắc, blur, clahe, ksize, iter)
# --- Thông số cho MÀU XANH DƯƠNG (Blue) ---
BLUE_LOWER_H = 102
BLUE_UPPER_H = 117
BLUE_LOWER_S = 110
BLUE_UPPER_S = 255
BLUE_LOWER_V = 111
BLUE_UPPER_V = 255
BLUE_KSIZE = 7
BLUE_OPEN_ITER = 1
BLUE_CLOSE_ITER = 5 
BLUE_CLAHE_CLIP_LIMIT = 30
BLUE_BLUR_KSIZE = 7

# --- Thông số cho MÀU ĐỎ (Red) ---
# Only using high H values (dark red) - removed 0-10 range (bright red)
RED_LOWER_H = 135
RED_UPPER_H = 179
RED_LOWER_S = 31
RED_UPPER_S = 245
RED_LOWER_V = 7
RED_UPPER_V = 246
RED_KSIZE = 3
RED_OPEN_ITER = 2
RED_CLOSE_ITER = 5
RED_CLAHE_CLIP_LIMIT = 30
RED_BLUR_KSIZE = 5

# --- Thông số cho MÀU VÀNG (Yellow) ---
YELLOW_LOWER_H = 7
YELLOW_UPPER_H = 20
YELLOW_LOWER_S = 150
YELLOW_UPPER_S = 235
YELLOW_LOWER_V = 140
YELLOW_UPPER_V = 230
YELLOW_KSIZE = 3
YELLOW_OPEN_ITER = 1
YELLOW_CLOSE_ITER = 5
YELLOW_CLAHE_CLIP_LIMIT = 30
YELLOW_BLUR_KSIZE = 5

# --- Ngưỡng lọc Contours (Giữ nguyên ngưỡng khắt khe của bạn) ---
MIN_CONTOUR_AREA = 1100
MIN_HULL_CIRCULARITY = 0.88
MIN_ASPECT_RATIO = 0.7
MAX_ASPECT_RATIO = 1.3

# --- Ngưỡng lọc Triangle (for Yellow signs) ---
TRIANGLE_VERTICES_MIN = 3
TRIANGLE_VERTICES_MAX = 4
TRIANGLE_AREA_RATIO_MIN = 0.45  # Triangles fill less of bbox than circles
TRIANGLE_MIN_ASPECT = 0.7      # Minimum height/width ratio
TRIANGLE_MAX_ASPECT = 1.3      # Maximum height/width ratio

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


# --- SHAPE DETECTION FUNCTIONS ---
def is_triangle(contour, hull):
    """
    Check if contour is triangle-shaped.
    Returns (is_triangle, num_vertices)
    """
    hull_perimeter = cv.arcLength(hull, True)
    if hull_perimeter == 0:
        return False, 0
    
    # Get bounding box for aspect ratio check
    x, y, w, h = cv.boundingRect(hull)
    if w == 0 or h == 0:
        return False, 0
    
    aspect_ratio = float(h) / w  # Height / Width
    
    # Approximate polygon
    epsilon = 0.04 * hull_perimeter  # 4% tolerance
    approx = cv.approxPolyDP(hull, epsilon, True)
    
    num_vertices = len(approx)
    
    # Triangle should have 3-4 vertices (allowing for rounded corners)
    if TRIANGLE_VERTICES_MIN <= num_vertices <= TRIANGLE_VERTICES_MAX:
        # Additional check: triangle area ratio
        hull_area = cv.contourArea(hull)
        bbox_area = w * h
        if bbox_area == 0:
            return False, num_vertices
        area_ratio = hull_area / bbox_area
        
        # Check aspect ratio (reject very wide or very tall shapes)
        passes_aspect = TRIANGLE_MIN_ASPECT <= aspect_ratio <= TRIANGLE_MAX_ASPECT
        
        # Triangles fill less of their bbox than circles
        if area_ratio >= TRIANGLE_AREA_RATIO_MIN and passes_aspect:
            return True, num_vertices
    
    return False, num_vertices


def is_circle(contour, hull):
    """
    Enhanced circle detection with tighter convex hull approximation.
    Returns (is_circle, circularity)
    """
    # Step 1: Approximate contour to remove edge irregularities
    perimeter = cv.arcLength(contour, True)
    if perimeter == 0:
        return False, 0
    
    # Smooth contour before creating hull (removes edge gaps/noise)
    epsilon = 0.02 * perimeter  # 2% tolerance for smoothing
    approx_contour = cv.approxPolyDP(contour, epsilon, True)
    
    # Step 2: Create better convex hull from approximated contour
    try:
        better_hull = cv.convexHull(approx_contour)
    except:
        return False, 0
    
    hull_area = cv.contourArea(better_hull)
    if hull_area == 0:
        return False, 0
    
    hull_perimeter = cv.arcLength(better_hull, True)
    if hull_perimeter == 0:
        return False, 0
    
    # Circularity check on enhanced hull
    circularity = (4 * np.pi * hull_area) / (hull_perimeter ** 2)
    
    # Aspect ratio check
    x, y, w, h = cv.boundingRect(better_hull)
    if w == 0 or h == 0:
        return False, circularity
    aspect_ratio = float(w) / h
    
    # Circle criteria
    passes_circularity = circularity >= MIN_HULL_CIRCULARITY
    passes_aspect = MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO
    
    return (passes_circularity and passes_aspect), circularity


# --- THÊM MỚI: HÀM TÌM VÀ LỌC CONTOURS ---
def find_and_filter_contours(mask_processed, output_frame_to_draw, y_offset, debug_color_bgr, color_name):
    """
    Tìm contours trên một mask, lọc chúng theo shape (circle/triangle), và vẽ text debug.
    color_name: 'red', 'blue', or 'yellow'
    Trả về một danh sách các bounding box (x, y, w, h) đã được lọc.
    """
    contours, _ = cv.findContours(mask_processed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    detected_rects = []

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < MIN_CONTOUR_AREA: # Lọc bỏ nhiễu siêu nhỏ trước
            continue

        try:
            hull = cv.convexHull(cnt)
        except cv.error as e:
            continue

        hull_area = cv.contourArea(hull)
        hull_perimeter = cv.arcLength(hull, True)
        
        x, y, w, h = cv.boundingRect(cnt)
        if w == 0 or h == 0: continue

        # Shape detection based on color
        shape_metric = 0  # Will store circularity OR vertices count
        is_passed = False
        
        if color_name == 'yellow':
            # Yellow signs should be triangles
            is_triangle_shape, num_vertices = is_triangle(cnt, hull)
            shape_metric = num_vertices
            is_passed = is_triangle_shape
        else:
            # Red and blue signs should be circles
            is_circle_shape, circularity = is_circle(cnt, hull)
            shape_metric = circularity
            is_passed = is_circle_shape
        
        # Áp dụng shape filter
        if not is_passed:
            continue

        # Nếu vượt qua, thêm (x, y, w, h) của ROI (chưa có offset)
        detected_rects.append((x, y, w, h))
        
    return detected_rects
# --- KẾT THÚC HÀM MỚI ---


def remove_overlapping_detections(all_rects, iou_threshold=0.6):
    """
    Remove smaller rectangles that are contained within larger ones.
    Keeps the outermost (largest) detection when there's overlap.
    """
    if len(all_rects) <= 1:
        return all_rects
    
    # Sort by area (largest first)
    rects_with_area = [(x, y, w, h, w * h) for x, y, w, h in all_rects]
    rects_with_area.sort(key=lambda r: r[4], reverse=True)
    
    keep = []
    
    for i, (x1, y1, w1, h1, area1) in enumerate(rects_with_area):
        is_overlapped = False
        
        # Check against already kept (larger) rectangles
        for (x2, y2, w2, h2) in keep:
            area2 = w2 * h2
            
            # Calculate intersection
            ix1 = max(x1, x2)
            iy1 = max(y1, y2)
            ix2 = min(x1 + w1, x2 + w2)
            iy2 = min(y1 + h1, y2 + h2)
            
            if ix1 < ix2 and iy1 < iy2:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                
                # Calculate overlap ratio for both rectangles
                overlap_ratio_1 = intersection / area1  # How much of rect1 is covered
                overlap_ratio_2 = intersection / area2  # How much of rect2 is covered
                
                # Remove if significantly overlapped by a larger rectangle
                if overlap_ratio_1 > iou_threshold or overlap_ratio_2 > iou_threshold:
                    is_overlapped = True
                    break
        
        if not is_overlapped:
            keep.append((x1, y1, w1, h1))
    
    return keep


# --- Simple consecutive frame tracking ---
prev_frame_detections = []

def is_detection_confirmed(x, y, w, h, prev_detections, tolerance=30):
    """
    Check if detection appeared in previous frame (within tolerance pixels).
    """
    for (px, py, pw, ph) in prev_detections:
        if abs(x - px) < tolerance and abs(y - py) < tolerance:
            return True
    return False


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
    roi_frame = frame[ROI_Y_START:ROI_Y_END, :] 
    output_frame = frame.copy()

    # --- BƯỚC 1: XỬ LÝ TỪNG KÊNH MÀU (TRÊN ROI) ---
    mask_processed_red = process_color_channel(
        roi_frame,
        RED_LOWER_H, RED_UPPER_H, RED_LOWER_S, RED_UPPER_S, RED_LOWER_V, RED_UPPER_V,
        RED_BLUR_KSIZE, RED_CLAHE_CLIP_LIMIT, RED_KSIZE, RED_OPEN_ITER, RED_CLOSE_ITER
    )

    mask_processed_blue = process_color_channel(
        roi_frame,
        BLUE_LOWER_H, BLUE_UPPER_H, BLUE_LOWER_S, BLUE_UPPER_S, BLUE_LOWER_V, BLUE_UPPER_V,
        BLUE_BLUR_KSIZE, BLUE_CLAHE_CLIP_LIMIT, BLUE_KSIZE, BLUE_OPEN_ITER, BLUE_CLOSE_ITER
    )

    mask_processed_yellow = process_color_channel(
        roi_frame,
        YELLOW_LOWER_H, YELLOW_UPPER_H, YELLOW_LOWER_S, YELLOW_UPPER_S, YELLOW_LOWER_V, YELLOW_UPPER_V,
        YELLOW_BLUR_KSIZE, YELLOW_CLAHE_CLIP_LIMIT, YELLOW_KSIZE, YELLOW_OPEN_ITER, YELLOW_CLOSE_ITER
    )

    # --- BƯỚC 2: TÌM VÀ LỌC CONTOURS RIÊNG BIỆT ---
    # Bỏ qua bước gộp mask
    
    # Xử lý contours từ mask ĐỎ (debug text màu Đỏ, circles only)
    red_rects = find_and_filter_contours(
        mask_processed_red, output_frame, ROI_Y_START, (0, 0, 255), 'red'
    )
    
    # Xử lý contours từ mask XANH (debug text màu Xanh, circles only)
    blue_rects = find_and_filter_contours(
        mask_processed_blue, output_frame, ROI_Y_START, (255, 0, 0), 'blue'
    )
    
    # Xử lý contours từ mask VÀNG (debug text màu Vàng, triangles only)
    yellow_rects = find_and_filter_contours(
        mask_processed_yellow, output_frame, ROI_Y_START, (0, 255, 255), 'yellow'
    )

    # Gộp tất cả các rects đã được lọc
    all_detected_rects = red_rects + blue_rects + yellow_rects
    
    # Remove overlapping detections (keep larger/outer ones)
    all_detected_rects = remove_overlapping_detections(all_detected_rects, iou_threshold=0.6)

    # Filter: only draw detections that appeared in previous frame
    confirmed_rects = []
    for (x, y, w, h) in all_detected_rects:
        if is_detection_confirmed(x, y, w, h, prev_frame_detections):
            confirmed_rects.append((x, y, w, h))
    
    # Update previous frame detections for next iteration
    prev_frame_detections = all_detected_rects.copy()

    # --- BƯỚC 3: VẼ KẾT QUẢ (Bounding Box màu xanh lá) ---
    for (x, y, w, h) in confirmed_rects:
        # QUAN TRỌNG: Phải cộng y với roi_y_start
        y_on_frame = y + ROI_Y_START
        
        cv.rectangle(output_frame, (x, y_on_frame), (x + w, y_on_frame + h), (0, 255, 0), 2)


    # ============================

    # --- Display only final output ---
    cv.imshow("Output", output_frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    # Ghi frame (Vẫn comment lại theo code của bạn)
    result.write(output_frame)

# --- 4. Dọn dẹp ---
print(f"Xử lý hoàn tất! Video đã được lưu tại: {OUTPUT_VIDEO}") # Cập nhật print
cap.release()
result.release() # Bỏ comment nếu muốn lưu video
cv.destroyAllWindows()

