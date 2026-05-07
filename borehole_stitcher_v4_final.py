import cv2
import numpy as np
import os
import sys


class BoreholeStitcher:
    """
    钻孔全景拼接器 v6

    核心：一次 cv2.remap 完成极坐标展开 + 鱼眼径向畸变补偿。

    畸变模型
    ────────
    摄像头为鱼眼（等距投影）：r_pixel = f * θ
    对应物理半径：r_phys = H * tan(θ) = H * tan(r_pixel / f)

    图像中 r_pixel 均匀采样时，外圈单位 Δr_pixel 对应更大 Δr_phys：
        dr_phys/dr_pixel = (H/f) * sec²(r_pixel/f)  → 随 r 增大而增大
    → 展开图中外圈格子更高、内圈格子更矮（正是观察到的现象）。

    补偿
    ────
    令输出行 row 对应等物理间距的 r_phys（线性），则采样坐标：
        t = row / (output_height - 1)
        r_phys(t) = r_phys_inner + t * (r_phys_outer - r_phys_inner)   线性
        r_px_sample = f * arctan(r_phys(t) / H)
                    = f * arctan(tan(inner_r/f) + t*(tan(outer_r/f)-tan(inner_r/f)))

    其中 H 约掉，只需一个参数 f（等效焦距，像素单位）。
    f 越小 → 鱼眼畸变越强 → 补偿幅度越大。
    f → ∞ → arctan(x)≈x → 退化为线性采样（无畸变）。

    纵向整体比例
    ────────────
    补偿后仍可通过 aspect_ratio 对 output_height 整体缩放，
    使格子达到方形（弥补 circumference 与 annular_height 比例不匹配）。
    output_height = round(annular_height * aspect_ratio)
    """

    def __init__(
        self,
        inner_radius=190,
        outer_radius=360,
        focal_px=400.0,       # 鱼眼等效焦距（像素）。越小补偿越强。
                              # f→∞ 退化为线性（无畸变补偿）。
                              # 用 calibrate_focal() 从实测格子高度比标定。
        output_height=None,   # 直接指定输出行数（优先级最高）
        aspect_ratio=1.0,     # 整体纵向比例。output_height = annular_h * aspect_ratio
    ):
        assert inner_radius > 0
        assert inner_radius < outer_radius

        self.inner_r = inner_radius
        self.outer_r = outer_radius
        self.circumference = int(2 * np.pi * self.outer_r)
        self.annular_height = self.outer_r - self.inner_r
        self.focal_px = float(focal_px)

        if output_height is not None:
            self.output_height = output_height
        else:
            self.output_height = round(self.annular_height * aspect_ratio)

        # 预计算完整的 2D remap（只算一次）
        self._map_x, self._map_y = None, None
        self._last_center = None

        self.panorama = None
        self.last_unwrapped = None
        self.prev_center = None
        self.template_h = 60
        self.match_score_threshold = 0.65
        self.max_dy = self.output_height // 2

        os.makedirs("temp", exist_ok=True)

    # ------------------------------------------------------------------
    # 核心：生成完整 2D remap（展开 + 鱼眼补偿一体化）
    # ------------------------------------------------------------------

    def _build_remap(self, center):
        """
        生成 shape=(output_height, circumference) 的 map_x / map_y。

        r 方向：鱼眼补偿采样
            t = row / (H_out - 1)  ∈ [0, 1]
            r_c(t) = f * arctan(tan(inner_r/f) + t*(tan(outer_r/f)-tan(inner_r/f)))

        f→∞ 时 r_c(t) → inner_r + t*annular_height（线性，无补偿）
        f 较小时外圈行被压缩，内圈行被拉伸，使格子等高。

        φ 方向：均匀角度采样（无畸变）
        """
        cx, cy = float(center[0]), float(center[1])
        H_out = self.output_height
        W = self.circumference
        f = self.focal_px

        t = np.arange(H_out, dtype=np.float64) / max(H_out - 1, 1)

        tan_i = np.tan(self.inner_r / f)
        tan_o = np.tan(self.outer_r / f)
        r_c = f * np.arctan(tan_i + t * (tan_o - tan_i))   # (H_out,)

        cols = np.arange(W, dtype=np.float64)
        phi = cols / W * 2.0 * np.pi                        # (W,)

        r2d     = r_c[:, None]
        cos_phi = np.cos(phi)[None, :]
        sin_phi = np.sin(phi)[None, :]

        map_x = (cx + r2d * cos_phi).astype(np.float32)
        map_y = (cy + r2d * sin_phi).astype(np.float32)
        return map_x, map_y

    def unwrap_and_correct(self, frame, center):
        """
        一步完成极坐标展开 + 深度补偿。
        当圆心变化超过阈值时重新生成 remap 表。
        """
        cx, cy = center
        # 圆心偏移超过 1px 时重建（低通滤波后基本稳定）
        if (self._map_x is None or self._last_center is None
                or abs(cx - self._last_center[0]) > 1.0
                or abs(cy - self._last_center[1]) > 1.0):
            self._map_x, self._map_y = self._build_remap(center)
            self._last_center = (cx, cy)

        result = cv2.remap(
            frame,
            self._map_x, self._map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        return result   # shape=(output_height, circumference, 3)

    # ------------------------------------------------------------------
    # 圆心检测
    # ------------------------------------------------------------------

    def detect_center(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT,
            dp=1.5, minDist=self.outer_r,
            param1=80, param2=40,
            minRadius=self.inner_r, maxRadius=self.outer_r
        )

        if circles is not None:
            cx, cy = float(circles[0, 0, 0]), float(circles[0, 0, 1])
        else:
            k = int(frame.shape[1] / 6)
            if k % 2 == 0: k += 1
            bb = cv2.GaussianBlur(gray, (k, k), 0)
            _, _, _, ml = cv2.minMaxLoc(bb)
            cx, cy = float(ml[0]), float(ml[1])

        if self.prev_center is None:
            self.prev_center = (cx, cy)
        else:
            a = 0.15
            cx = a * cx + (1 - a) * self.prev_center[0]
            cy = a * cy + (1 - a) * self.prev_center[1]
            self.prev_center = (cx, cy)
        return (cx, cy)

    # ------------------------------------------------------------------
    # 位移估计
    # ------------------------------------------------------------------

    def find_vertical_shift(self, prev_img, curr_img):
        template = prev_img[-self.template_h:, :]
        res = cv2.matchTemplate(curr_img, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        dy = curr_img.shape[0] - self.template_h - max_loc[1]
        dy = int(max(0, min(dy, self.max_dy)))
        dx = self._circular_horizontal_shift(
            prev_img[-self.template_h:, :],
            curr_img[-self.template_h:, :]
        )
        return dx, dy, max_val

    def _circular_horizontal_shift(self, img_a, img_b):
        def to_row(img):
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            return np.mean(g, axis=0).astype(np.float32)
        row_a, row_b = to_row(img_a), to_row(img_b)
        W = len(row_a)
        fa, fb = np.fft.rfft(row_a, n=W), np.fft.rfft(row_b, n=W)
        cross = fa * np.conj(fb)
        norm = np.abs(cross); norm[norm == 0] = 1
        corr = np.fft.irfft(cross / norm, n=W)
        peak = int(np.argmax(corr))
        if peak > W // 2: peak -= W
        return peak

    # ------------------------------------------------------------------
    # 主处理流程
    # ------------------------------------------------------------------

    def process_video(self, video_path, step=2):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频：{video_path}"); return

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0

        print(f"[输出尺寸] 宽={self.circumference}  高={self.output_height}")
        print(f"[原始环带] annular_height={self.annular_height}")
        print(f"[纵向比例] aspect_ratio={self.output_height/self.annular_height:.3f}\n")

        while True:
            ret, frame = cap.read()
            if not ret: break

            if frame_count % step == 0:
                sys.stdout.write(f"\r进度: {frame_count}/{total}...")
                sys.stdout.flush()

                center = self.detect_center(frame)

                # 一步完成展开 + 补偿（左右对称）
                curr_unwrapped = self.unwrap_and_correct(frame, center)

                move_y = 0
                match_score = 0

                if self.panorama is None:
                    self.panorama = curr_unwrapped
                else:
                    dx, dy, match_score = self.find_vertical_shift(
                        self.last_unwrapped, curr_unwrapped
                    )
                    if dy >= 1 and match_score > self.match_score_threshold:
                        move_y = dy
                        if abs(dx) > 0:
                            curr_unwrapped = np.roll(curr_unwrapped, -dx, axis=1)
                        self.panorama = np.vstack((self.panorama, curr_unwrapped[-move_y:, :]))

                # 调试图
                viz = frame.copy()
                c_int = (int(center[0]), int(center[1]))
                cv2.circle(viz, c_int, 5, (0, 0, 255), -1)
                cv2.circle(viz, c_int, self.inner_r, (255, 0, 0), 2)
                cv2.circle(viz, c_int, self.outer_r, (0, 255, 0), 2)

                th = viz.shape[0]
                def fit(img):
                    h, w = img.shape[:2]
                    return cv2.resize(img, (int(w * th / h), th))

                debug = np.hstack([viz, fit(curr_unwrapped)])
                cv2.putText(debug,
                    f"F:{frame_count} Dy:{move_y} Score:{match_score:.2f}",
                    (20, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.imwrite(f"temp/step_{frame_count:05d}.jpg", debug)
                self.last_unwrapped = curr_unwrapped

            frame_count += 1

        cap.release()
        if self.panorama is not None:
            out = f"panorama_ar{self.output_height/self.annular_height:.2f}.jpg"
            cv2.imwrite(out, self.panorama)
            print(f"\n完成！输出：{out}")
            print(f"全景尺寸：宽={self.circumference}  高={self.panorama.shape[0]}")


# ------------------------------------------------------------------
# 标定工具1：测量展开图格子的实际宽高比（用于确定 aspect_ratio 参数）
# ------------------------------------------------------------------

def calibrate_aspect(
    raw_frame_path,
    center_xy,
    inner_radius,
    outer_radius,
    output_dir="aspect_calibration"
):
    """
    对一帧原始图，生成无补偿的展开图，自动测量格子宽高比，
    并生成不同 aspect_ratio 的对比图辅助确认。

    返回值：测量到的 aspect_ratio（格子宽/格子高，原始展开图像素）。

    使用步骤：
      1. 将标定视频帧（含已知正方形格子或管壁等间距纹理）保存为 raw_frame_path
      2. 调用本函数，观察 output_dir/comparison.jpg
      3. 将返回的 aspect_ratio 填入 BoreholeStitcher(aspect_ratio=...)
    """
    import os
    from scipy.signal import find_peaks

    os.makedirs(output_dir, exist_ok=True)
    frame = cv2.imread(raw_frame_path)
    if frame is None:
        print(f"无法读取：{raw_frame_path}"); return 1.0

    # 生成无补偿展开图（output_height=annular_height，不做拉伸）
    s0 = BoreholeStitcher(
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        output_height=outer_radius - inner_radius,
    )
    unw = s0.unwrap_and_correct(frame, center_xy)
    cv2.imwrite(os.path.join(output_dir, "unwrapped_raw.jpg"), unw)

    gray = cv2.cvtColor(unw, cv2.COLOR_BGR2GRAY).astype(float)
    H, W = gray.shape

    # 水平周期（列方向梯度）
    dx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    col_e = dx.mean(axis=0)
    v_peaks, _ = find_peaks(col_e, height=np.percentile(col_e, 75), distance=10)
    pw = float(np.diff(v_peaks).mean()) if len(v_peaks) > 1 else float(W) / 10

    # 垂直周期（行方向梯度）
    dy_img = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
    row_e = dy_img.mean(axis=1)
    h_peaks, _ = find_peaks(row_e, height=np.percentile(row_e, 75), distance=5)
    ph = float(np.diff(h_peaks).mean()) if len(h_peaks) > 1 else float(H)

    ar = pw / ph
    print(f"格子水平周期（原始展开图）: {pw:.1f} px")
    print(f"格子垂直周期（原始展开图）: {ph:.1f} px")
    print(f"测量宽高比 aspect_ratio = {ar:.3f}")

    # 生成对比图：用不同 aspect_ratio 展开，检查是否方形
    test_ratios = sorted(set([1.0, round(ar*0.5,2), round(ar*0.75,2),
                               round(ar,2), round(ar*1.25,2), round(ar*1.5,2)]))
    target_h = 400
    strips = []
    for r in test_ratios:
        s = BoreholeStitcher(
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            aspect_ratio=r,
        )
        img_r = s.unwrap_and_correct(frame, center_xy)
        rh, rw = img_r.shape[:2]
        scaled = cv2.resize(img_r, (min(int(rw * target_h / rh), 600), target_h))
        cv2.putText(scaled, f"ar={r:.2f}", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        strips.append(scaled)

    # 高度统一
    max_h = max(s.shape[0] for s in strips)
    padded = []
    for s in strips:
        ph2 = max_h - s.shape[0]
        padded.append(np.pad(s, ((0,ph2),(0,0),(0,0))))
    comparison = np.hstack(padded)
    out_path = os.path.join(output_dir, "aspect_comparison.jpg")
    cv2.imwrite(out_path, comparison)
    print(f"对比图：{out_path}  （选格子最接近正方形的 aspect_ratio）")
    return ar


# ------------------------------------------------------------------
# 入口
# ------------------------------------------------------------------

def calibrate_focal(
    raw_frame_path,
    center_xy,
    inner_radius,
    outer_radius,
    focal_values=None,
    output_dir="focal_calibration"
):
    """
    标定鱼眼等效焦距 focal_px。

    原理：
      鱼眼投影下，展开图中内圈格子矮、外圈格子高。
      focal_px 越小 → 补偿越强（外圈压缩幅度越大）。
      正确的 focal_px 使各行格子等高。

    使用步骤：
      1. 提供含均匀网格（管壁等间距纹理或标定格）的原始帧
      2. 调用本函数，查看 focal_calibration/comparison.jpg
      3. 选择格子行高最均匀的 focal_px 填入 BoreholeStitcher

    focal_values 默认覆盖从强补偿到无补偿（f=∞=线性）的范围。
    """
    os.makedirs(output_dir, exist_ok=True)
    frame = cv2.imread(raw_frame_path)
    if frame is None:
        print(f"无法读取：{raw_frame_path}"); return

    annular_h = outer_radius - inner_radius
    if focal_values is None:
        # 从强补偿到弱补偿，包含"无穷大"（线性）
        focal_values = [200, 300, 400, 500, 700, 1000, 9999]

    target_h = 400
    strips = []
    for f in focal_values:
        s = BoreholeStitcher(
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            focal_px=f,
            output_height=annular_h * 2,  # 固定高度方便对比
        )
        img_f = s.unwrap_and_correct(frame, center_xy)
        rh, rw = img_f.shape[:2]
        scaled = cv2.resize(img_f, (min(int(rw * target_h / rh), 500), target_h))
        label = f"f={f}" if f < 9000 else "f=inf(linear)"
        cv2.putText(scaled, label, (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        strips.append(scaled)

    comparison = np.hstack(strips)
    cv2.imwrite(os.path.join(output_dir, "comparison.jpg"), comparison)
    print(f"对比图：{output_dir}/comparison.jpg")
    print("选各行格子高度最均匀的 focal_px。")


# ------------------------------------------------------------------
# 入口
# ------------------------------------------------------------------

if __name__ == "__main__":
    # ── 第一步：标定 focal_px（消除内外圈格子高度不一致）──────────
    # calibrate_focal(
    #     "frame_000.jpg",
    #     center_xy=(960, 540),
    #     inner_radius=190,
    #     outer_radius=360,
    # )

    # ── 第二步：标定 aspect_ratio（使格子整体变方形）──────────────
    # calibrate_aspect(
    #     "frame_000.jpg",
    #     center_xy=(960, 540),
    #     inner_radius=190,
    #     outer_radius=360,
    # )

    # ── 第三步：正式处理 ─────────────────────────────────────────────
    # focal_px=397.6 由 step_00016.jpg 实测格子高度比(1.54x)反算得到
    # aspect_ratio 由 calibrate_aspect() 精确测量后填入
    stitcher = BoreholeStitcher(
        inner_radius=190,
        outer_radius=360,
        focal_px=397.6,    # 鱼眼焦距，由 calibrate_focal() 确定
        aspect_ratio=2.1,  # 整体纵向比例，由 calibrate_aspect() 确定
    )
    stitcher.process_video("borehole_video.mp4", step=2)
