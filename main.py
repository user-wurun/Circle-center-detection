import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = r"C:\Windows\Fonts\NotoSansSC-VF.ttf"

def put_chinese_text(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int],
    size: int,
) -> None:
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(rgb_image)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT_PATH, size)
    draw.text(position, text, font=font, fill=color[::-1])
    image[:] = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)

def high_pass_preserve(
    frame: np.ndarray, radius: float, strength: float, contrast: float = 1.8
) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), radius)
    result = cv2.addWeighted(gray, 1.0 + strength, blurred, -strength, 95)
    neutral_gray = np.full_like(result, 95)
    return cv2.addWeighted(result, contrast, neutral_gray, 1.0 - contrast, 0)

def detect_black_curves(
    image: np.ndarray,
    black_threshold: int,
    min_curve_length: float,
    blackhat_size: int = 15,
    min_curve_span: float = 50.0,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    blackhat_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (blackhat_size, blackhat_size)
    )
    dark_strokes = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, blackhat_kernel)
    dark_strokes = cv2.GaussianBlur(dark_strokes, (3, 3), 0)
    _, binary = cv2.threshold(
        dark_strokes, black_threshold, 255, cv2.THRESH_BINARY
    )
    connected = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8)
    )
    contours, _ = cv2.findContours(
        connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    curve_points = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter < min_curve_length:
            continue
        area = cv2.contourArea(contour)
        stroke_width = area / max(perimeter, 1.0)
        points = contour.reshape(-1, 2)
        fit_line = cv2.fitLine(
            points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01
        )
        direction = np.asarray(fit_line[:2], dtype=np.float32).reshape(2)
        projections = points.astype(np.float32) @ direction
        first = tuple(int(value) for value in points[np.argmin(projections)])
        second = tuple(int(value) for value in points[np.argmax(projections)])
        span = float(np.linalg.norm(np.subtract(first, second)))
        if span < min_curve_span or (area > 0 and stroke_width > 4.0):
            continue
        curve_points.append((first, second))

    return sorted(
        curve_points,
        key=lambda points: np.linalg.norm(np.subtract(points[0], points[1])),
        reverse=True,
    )

def line_intersection(
    first_point: np.ndarray,
    first_direction: np.ndarray,
    second_point: np.ndarray,
    second_direction: np.ndarray,
) -> np.ndarray | None:
    denominator = (
        first_direction[0] * second_direction[1]
        - first_direction[1] * second_direction[0]
    )
    if abs(float(denominator)) < 1e-6:
        return None
    offset = second_point - first_point
    scale = (
        offset[0] * second_direction[1]
        - offset[1] * second_direction[0]
    ) / denominator
    return first_point + scale * first_direction

def perpendicular_bisector(
    first: tuple[int, int], second: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray] | None:
    first_point = np.array(first, dtype=np.float32)
    second_point = np.array(second, dtype=np.float32)
    direction = second_point - first_point
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return None
    midpoint = (first_point + second_point) / 2.0
    perpendicular = np.array([-direction[1], direction[0]], dtype=np.float32)
    perpendicular /= length
    return midpoint, perpendicular

def find_candidate_centers(
    curves: list[tuple[tuple[int, int], tuple[int, int]]],
    image_shape: tuple[int, ...],
) -> list[tuple[int, int]]:
    bisectors = [
        bisector
        for curve in curves
        if (bisector := perpendicular_bisector(*curve)) is not None
    ]
    height, width = image_shape[:2]
    centers = []
    for index, (first_midpoint, first_direction) in enumerate(bisectors):
        for second_midpoint, second_direction in bisectors[index + 1 :]:
            intersection = line_intersection(
                first_midpoint,
                first_direction,
                second_midpoint,
                second_direction,
            )
            if intersection is None:
                continue
            x, y = (int(round(value)) for value in intersection)
            if 0 <= x < width and 0 <= y < height:
                centers.append((x, y))
    return centers

def draw_curve_points(
    image: np.ndarray,
    curves: list[tuple[tuple[int, int], tuple[int, int]]],
) -> np.ndarray:
    annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for index, (first, second) in enumerate(curves, start=1):
        cv2.circle(annotated, first, 6, (0, 0, 255), -1)
        cv2.circle(annotated, second, 6, (0, 255, 0), -1)
        cv2.line(annotated, first, second, (255, 180, 0), 2)
        cv2.putText(
            annotated,
            f"{index}: {first}, {second}",
            (10, 30 + (index - 1) * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    bisectors = [
        bisector
        for curve in curves
        if (bisector := perpendicular_bisector(*curve)) is not None
    ]

    height, width = image.shape[:2]
    line_scale = float(np.hypot(width, height))
    colors = [
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 0),
        (255, 128, 0),
        (0, 255, 128),
        (128, 0, 255),
    ]
    for index, (midpoint, direction) in enumerate(bisectors):
        start = midpoint - direction * line_scale
        end = midpoint + direction * line_scale
        cv2.line(
            annotated,
            tuple(int(value) for value in start),
            tuple(int(value) for value in end),
            colors[index % len(colors)],
            1,
        )

    candidate_centers = find_candidate_centers(curves, image.shape)
    for center_count, (x, y) in enumerate(candidate_centers, start=1):
        cv2.circle(annotated, (x, y), 8, (0, 0, 255), 2)
        cv2.circle(annotated, (x, y), 3, (255, 255, 255), -1)
        put_chinese_text(
            annotated,
            f"交点{center_count}",
            (x + 10, y - 10),
            (0, 0, 255),
            18,
        )

    if candidate_centers:
        average_center = np.mean(np.asarray(candidate_centers, dtype=np.float32), axis=0)
        avg_x, avg_y = (int(round(value)) for value in average_center)
        avg_x = max(0, min(width - 1, avg_x))
        avg_y = max(0, min(height - 1, avg_y))
        cv2.circle(annotated, (avg_x, avg_y), 5, (50, 50, 50), -1)
        put_chinese_text(
            annotated,
            f"平均圆心: ({avg_x}, {avg_y})",
            (avg_x + 12, avg_y - 12),
            (50, 50, 50),
            24,
        )

    put_chinese_text(
        annotated,
        f"曲线: {len(curves)} | 交点: {len(candidate_centers)}",
        (10, annotated.shape[0] - 34),
        (255, 255, 0),
        26,
    )
    return annotated

def main() -> None:
    camera_index = 0
    radius = 3.0
    strength = 1.0
    contrast = 1.0
    black_threshold = 18
    min_curve_length = 30.0
    blackhat_size = 15
    min_curve_span = 50.0

    if radius <= 0 or strength < 0 or contrast <= 0:
        raise ValueError("半径必须大于 0，强度不能小于 0，对比度必须大于 0")
    if (
        not 0 <= black_threshold <= 255
        or min_curve_length <= 0
        or min_curve_span <= 0
    ):
        raise ValueError(
            "黑色阈值应在 0 到 255 之间，曲线长度和跨度必须大于 0"
        )
    if blackhat_size < 3 or blackhat_size % 2 == 0:
        raise ValueError("黑帽窗口大小必须是不小于 3 的奇数")

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"无法打开摄像头 {camera_index}")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("读取摄像头画面失败")
                break

            result = high_pass_preserve(
                frame, radius, strength, contrast
            )
            curves = detect_black_curves(
                result,
                black_threshold,
                min_curve_length,
                blackhat_size,
                min_curve_span,
            )
            annotated = frame.copy()
            height, width = frame.shape[:2]
            candidate_centers = find_candidate_centers(curves, frame.shape)
            for x, y in candidate_centers:
                cv2.circle(annotated, (x, y), 3, (0, 0, 255), -1)
            if candidate_centers:
                average_center = np.mean(
                    np.asarray(candidate_centers, dtype=np.float32), axis=0
                )
                avg_x, avg_y = (int(round(value)) for value in average_center)
                avg_x = max(0, min(width - 1, avg_x))
                avg_y = max(0, min(height - 1, avg_y))
                window_center = (width // 2, height // 2)
                cv2.line(
                    annotated,
                    window_center,
                    (avg_x, avg_y),
                    (255, 0, 255),
                    2,
                )
                cv2.circle(annotated, (avg_x, avg_y), 5, (50, 50, 50), -1)
                put_chinese_text(
                    annotated,
                    f"平均圆心: ({avg_x}, {avg_y})",
                    (avg_x + 12, avg_y - 12),
                    (50, 50, 50),
                    24,
                )
            cv2.imshow("Camera Circle Search", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()