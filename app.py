import numpy as np
from PIL import Image
import sys
import time
import json
import os
import warnings
import multiprocessing as mp
import soundfile as sf
from mutagen.flac import FLAC

# ====== 默认参数配置（编码阶段使用） ======
F_MIN = 500                         # 音频频率最小值（对应图像黑色）
F_MAX = 3000                        # 音频频率最大值（对应图像白色）
SAMPLE_RATE = 44100                 # 音频采样率
SAMPLES_PER_PIXEL = 48             # 每个像素点对应的音频采样点数（决定音频时长和分辨率）

# ====== 解码与显示参数配置 ======
SCREEN_W = 1280                     # 显示窗口宽度（像素）
SCREEN_H = 720                      # 显示窗口高度（像素）
N_FFT = 512                         # 快速傅里叶变换的点数（影响频率解析精度）
VOLUME_PERCENT = 5                  # 播放音频时的音量（百分比，避免音量过大）
ENCODE_COLUMN_BLOCK = 8             # 编码时每个任务处理的列数（越小越省内存，推荐值为CPU核心数（并非线程数）的1/2）
DECODE_PIXEL_BLOCK = 8192           # 解码时每批处理的像素数（越小越省内存）
# =====================


def _resolve_workers(workers):
    cpu_total = os.cpu_count() or 1
    if workers is None:
        return cpu_total
    return max(1, min(int(workers), cpu_total))


def _import_pygame():
    # 仅在需要显示/播放时导入pygame，避免编码多进程重复打印初始化信息
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="pkg_resources is deprecated as an API.*",
            category=UserWarning,
            module="pygame.pkgdata"
        )
        import pygame as pg
    return pg


def _encode_chunk(args):
    chunk, f_min, f_max, samples_per_pixel, sample_rate = args
    if chunk.size == 0:
        return np.empty(0, dtype=np.float32)

    t_base = np.arange(samples_per_pixel, dtype=np.float32) / sample_rate
    freqs = f_min + chunk.reshape(-1) * (f_max - f_min)
    waves = np.sin(2 * np.pi * freqs[:, None] * t_base[None, :])
    return waves.reshape(-1).astype(np.float32)


def _decode_chunk(args):
    frame_chunk, n_fft, f_min_idx, f_max_idx, freqs, f_min, f_max = args
    if frame_chunk.size == 0:
        return np.empty(0, dtype=np.float32)

    window = np.hanning(frame_chunk.shape[1]).astype(np.float32)
    frame_win = frame_chunk * window

    fft_result = np.fft.rfft(frame_win, n=n_fft, axis=1)
    sub_mag = np.abs(fft_result[:, f_min_idx:f_max_idx + 1])
    peak_rel_idx = np.argmax(sub_mag, axis=1)
    estimated_freq = freqs[f_min_idx + peak_rel_idx]

    gray = (estimated_freq - f_min) / (f_max - f_min)
    return np.clip(gray, 0.0, 1.0).astype(np.float32)


def _handle_quit_events(pg):
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            sys.exit()


def _refresh_preview(pg, screen, image_surface, width, height):
    scale_ratio = min(SCREEN_W / width, SCREEN_H / height)
    new_w = int(width * scale_ratio)
    new_h = int(height * scale_ratio)

    scaled_surface = pg.transform.smoothscale(image_surface, (new_w, new_h))
    offset_x = (SCREEN_W - new_w) // 2
    offset_y = (SCREEN_H - new_h) // 2

    screen.fill((0, 0, 0))
    screen.blit(scaled_surface, (offset_x, offset_y))
    pg.display.flip()


def _draw_gray_block(pg, screen, image_surface, img_uint8, gray_block, start_idx, height):
    for offset, gray in enumerate(gray_block):
        _handle_quit_events(pg)

        i = start_idx + offset
        x = i // height
        y = i % height
        val = int(gray * 255)

        image_surface.set_at((x, y), (val, val, val))
        img_uint8[y, x] = val

        if i % 1024 == 0:
            _refresh_preview(pg, screen, image_surface, img_uint8.shape[1], img_uint8.shape[0])


def image_to_audio(image_path, output_flac, workers=None):

    print(f"🖼️ 加载图像: {image_path}")
    total_start = time.perf_counter()

    # 打开图像并转换为灰度模式（L模式：0=黑，255=白）
    prep_start = time.perf_counter()
    img = Image.open(image_path).convert('L')
    width, height = img.size         # 获取图像的宽和高
    # 将像素值转换为0-1之间的浮点数
    pixels = np.asarray(img, dtype=np.float32) / 255.0
    data = pixels.T
    workers = _resolve_workers(workers)
    total_samples = width * height * SAMPLES_PER_PIXEL
    column_block = max(1, min(ENCODE_COLUMN_BLOCK, width))
    prep_time = time.perf_counter() - prep_start

    def _encode_tasks():
        for start_col in range(0, data.shape[0], column_block):
            chunk = data[start_col:start_col + column_block]
            yield (chunk, F_MIN, F_MAX, SAMPLES_PER_PIXEL, SAMPLE_RATE)

    # 分块写入FLAC，避免完整音频常驻内存
    encode_start = time.perf_counter()
    with sf.SoundFile(output_flac, mode="w", samplerate=SAMPLE_RATE, channels=1, format="FLAC") as out_file:
        if workers > 1:
            with mp.Pool(processes=workers) as pool:
                for audio_part in pool.imap(_encode_chunk, _encode_tasks(), chunksize=1):
                    out_file.write(np.clip(audio_part, -1.0, 1.0))
        else:
            for task in _encode_tasks():
                audio_part = _encode_chunk(task)
                out_file.write(np.clip(audio_part, -1.0, 1.0))
    encode_time = time.perf_counter() - encode_start

    print(f"⚙️ 编码并行进程数: {workers}")

    # 构建元数据字典：保存解码所需的关键参数
    metadata = {
        "width": width,                  # 原始图像宽度
        "height": height,                # 原始图像高度
        "F_MIN": F_MIN,                  # 编码时的最小频率
        "F_MAX": F_MAX,                  # 编码时的最大频率
        "SAMPLES_PER_PIXEL": SAMPLES_PER_PIXEL,  # 每个像素的采样点数
        "SAMPLE_RATE": SAMPLE_RATE,      # 音频采样率
        "N_FFT": N_FFT                   # 解码时用的FFT点数
    }

    # 将元数据写入FLAC文件的标签中（方便解码时读取）
    meta_start = time.perf_counter()
    flac_file = FLAC(output_flac)
    flac_file["IMAGE_METADATA"] = json.dumps(metadata)  # 序列化为JSON字符串
    flac_file.save()
    meta_time = time.perf_counter() - meta_start

    # 计算并输出音频时长
    duration = total_samples / SAMPLE_RATE
    print("✅ 编码完成")
    print(f"   分辨率: {width}x{height}")
    print(f"   音频时长: {duration:.2f} 秒")
    print("⏱️ 编码计时:")
    print(f"   图像预处理: {prep_time:.3f} 秒")
    print(f"   音频生成与写盘: {encode_time:.3f} 秒")
    print(f"   元数据写入: {meta_time:.3f} 秒")
    print(f"   编码总耗时: {time.perf_counter() - total_start:.3f} 秒")


def decode_play_draw(input_flac, output_image_path, workers=None):

    print(f"🔊 加载并播放: {input_flac}")
    total_start = time.perf_counter()

    # 读取FLAC文件的元数据
    metadata_start = time.perf_counter()
    flac_file = FLAC(input_flac)
    # 检查是否存在编码时写入的图像元数据
    if "IMAGE_METADATA" not in flac_file:
        print("❌ 未找到 IMAGE_METADATA")
        return

    # 解析元数据（从JSON字符串还原为字典）
    metadata = json.loads(flac_file["IMAGE_METADATA"][0])

    # 从元数据中读取解码所需的参数
    width = metadata["width"]
    height = metadata["height"]
    F_MIN = metadata["F_MIN"]
    F_MAX = metadata["F_MAX"]
    SAMPLES_PER_PIXEL = metadata["SAMPLES_PER_PIXEL"]
    SAMPLE_RATE = metadata["SAMPLE_RATE"]
    N_FFT = metadata["N_FFT"]
    workers = _resolve_workers(workers)
    metadata_time = time.perf_counter() - metadata_start

    # 计算总像素数和预期的音频采样点数
    num_pixels = width * height
    expected_samples = num_pixels * SAMPLES_PER_PIXEL

    # 计算FFT对应的频率轴（获取每个FFT点对应的实际频率）
    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / SAMPLE_RATE)
    # 筛选出在目标频率范围内的索引（±100Hz容错，避免频率偏移）
    valid_idx = np.where((freqs >= F_MIN - 100) & (freqs <= F_MAX + 100))[0]

    # 检查是否有有效的频率索引
    if len(valid_idx) == 0:
        print("❌ 频率范围错误")
        return

    # 获取有效频率范围的起止索引
    f_min_idx, f_max_idx = valid_idx[0], valid_idx[-1]

    pg = _import_pygame()
    pg.init()
    pg.mixer.init(SAMPLE_RATE, -16, 1, 1024)
    screen = pg.display.set_mode((SCREEN_W, SCREEN_H))
    
    window_title = f"🌌 藏在声音里的图片 | 图片分辨率：{width}x{height}"
    pg.display.set_caption(window_title)

    # 使用music接口流式播放，避免整段音频常驻内存
    pg.mixer.music.load(input_flac)
    pg.mixer.music.set_volume(VOLUME_PERCENT / 100.0)
    pg.mixer.music.play()

    print(f"⚙️ 解码并行进程数: {workers}")

    # 创建一个和原始图像大小一致的pygame表面（用于绘制还原的像素）
    image_surface = pg.Surface((width, height))
    img_uint8 = np.zeros((height, width), dtype=np.uint8)
    pixel_block = max(1, min(DECODE_PIXEL_BLOCK, num_pixels))
    io_time = 0.0
    fft_time = 0.0
    draw_time = 0.0
    decode_start = time.perf_counter()

    with sf.SoundFile(input_flac, mode="r") as audio_file:
        if audio_file.frames < expected_samples:
            print("⚠️ 音频过短，无法完整解码")
            return

        processed_pixels = 0

        if workers > 1:
            with mp.Pool(processes=workers) as pool:
                while processed_pixels < num_pixels:
                    current_pixels = min(pixel_block, num_pixels - processed_pixels)
                    block_frames = current_pixels * SAMPLES_PER_PIXEL

                    io_start = time.perf_counter()
                    audio_block = audio_file.read(block_frames, dtype="float32", always_2d=True)
                    io_time += time.perf_counter() - io_start
                    if audio_block.shape[0] < block_frames:
                        print("⚠️ 音频过短，无法完整解码")
                        return

                    mono = audio_block[:, 0]
                    frames = mono.reshape(current_pixels, SAMPLES_PER_PIXEL)

                    frame_chunks = [
                        chunk for chunk in np.array_split(frames, workers, axis=0)
                        if chunk.size > 0
                    ]
                    task_args = [
                        (chunk, N_FFT, f_min_idx, f_max_idx, freqs, F_MIN, F_MAX)
                        for chunk in frame_chunks
                    ]

                    fft_start = time.perf_counter()
                    gray_chunks = pool.map(_decode_chunk, task_args)
                    gray_block = np.concatenate(gray_chunks)
                    fft_time += time.perf_counter() - fft_start

                    draw_start = time.perf_counter()
                    _draw_gray_block(
                        pg, screen, image_surface, img_uint8, gray_block,
                        processed_pixels, height
                    )
                    draw_time += time.perf_counter() - draw_start
                    processed_pixels += current_pixels
        else:
            while processed_pixels < num_pixels:
                current_pixels = min(pixel_block, num_pixels - processed_pixels)
                block_frames = current_pixels * SAMPLES_PER_PIXEL

                io_start = time.perf_counter()
                audio_block = audio_file.read(block_frames, dtype="float32", always_2d=True)
                io_time += time.perf_counter() - io_start
                if audio_block.shape[0] < block_frames:
                    print("⚠️ 音频过短，无法完整解码")
                    return

                mono = audio_block[:, 0]
                frames = mono.reshape(current_pixels, SAMPLES_PER_PIXEL)
                fft_start = time.perf_counter()
                gray_block = _decode_chunk(
                    (frames, N_FFT, f_min_idx, f_max_idx, freqs, F_MIN, F_MAX)
                )
                fft_time += time.perf_counter() - fft_start

                draw_start = time.perf_counter()
                _draw_gray_block(
                    pg, screen, image_surface, img_uint8, gray_block,
                    processed_pixels, height
                )
                draw_time += time.perf_counter() - draw_start
                processed_pixels += current_pixels

    decode_time = time.perf_counter() - decode_start

    _refresh_preview(pg, screen, image_surface, width, height)

    save_start = time.perf_counter()
    Image.fromarray(img_uint8, mode='L').save(output_image_path)
    save_time = time.perf_counter() - save_start

    print(f"✅ 解码完成并保存: {output_image_path}")
    print("⏱️ 解码计时:")
    print(f"   元数据读取: {metadata_time:.3f} 秒")
    print(f"   音频分块读取: {io_time:.3f} 秒")
    print(f"   FFT反推: {fft_time:.3f} 秒")
    print(f"   画面绘制: {draw_time:.3f} 秒")
    print(f"   解码主流程(含读取/FFT/绘制): {decode_time:.3f} 秒")
    print(f"   图像保存: {save_time:.3f} 秒")
    print(f"   解码总耗时(不含等待关窗): {time.perf_counter() - total_start:.3f} 秒")

    # 保持窗口显示，直到用户关闭
    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
        time.sleep(0.01)

if __name__ == "__main__":
    mp.freeze_support()

    if len(sys.argv) < 2:
        print("用法:")
        print("编码: python app.py encode input.jpg signal.flac [workers]")
        print("解码: python app.py draw signal.flac result.png [workers]")
        sys.exit(1)

    # 获取操作模式（encode/draw）
    mode = sys.argv[1]
    if mode == "encode":
        # 编码模式：参数2=输入图像，参数3=输出FLAC，参数4=可选并行进程数
        if len(sys.argv) < 4:
            print("❌ 参数不足")
            print("编码: python app.py encode input.jpg signal.flac [workers]")
            sys.exit(1)
        workers = int(sys.argv[4]) if len(sys.argv) >= 5 else None
        image_to_audio(sys.argv[2], sys.argv[3], workers)
    elif mode == "draw":
        # 解码模式：参数2=输入FLAC，参数3=输出图像，参数4=可选并行进程数
        if len(sys.argv) < 4:
            print("❌ 参数不足")
            print("解码: python app.py draw signal.flac result.png [workers]")
            sys.exit(1)
        workers = int(sys.argv[4]) if len(sys.argv) >= 5 else None
        decode_play_draw(sys.argv[2], sys.argv[3], workers)
    else:
        print("❌ 未知模式")
