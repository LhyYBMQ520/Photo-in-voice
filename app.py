import numpy as np
from PIL import Image
import pygame
import sys
import time
import json
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
# =====================


def image_to_audio(image_path, output_flac):

    print(f"🖼️ 加载图像: {image_path}")

    # 打开图像并转换为灰度模式（L模式：0=黑，255=白）
    img = Image.open(image_path).convert('L')
    width, height = img.size         # 获取图像的宽和高
    # 将像素值转换为0-1之间的浮点数
    pixels = np.array(img) / 255.0
    data = pixels.T

    # 计算音频总采样点数：总像素数 × 每个像素对应的采样点数
    total_samples = data.size * SAMPLES_PER_PIXEL
    # 创建空的音频数组（float32格式，符合音频标准）
    audio = np.zeros(total_samples, dtype=np.float32)
    # 生成时间基轴：对应每个像素的采样点的时间坐标（单位：秒）
    t_base = np.arange(SAMPLES_PER_PIXEL) / SAMPLE_RATE

    idx = 0  # 音频数组的写入位置指针
    # 遍历每一列像素
    for column in data:
        # 遍历列中的每个像素
        for pixel in column:
            # 将像素灰度值（0-1）映射到F_MIN到F_MAX的频率
            freq = F_MIN + pixel * (F_MAX - F_MIN)
            # 生成对应频率的正弦波（音频信号）
            wave = np.sin(2 * np.pi * freq * t_base)
            # 将生成的正弦波写入音频数组的对应位置
            audio[idx:idx + SAMPLES_PER_PIXEL] = wave
            # 移动指针到下一个像素的位置
            idx += SAMPLES_PER_PIXEL

    # 音频信号限幅：确保所有值在[-1, 1]范围内（避免音频失真）
    audio = np.clip(audio, -1, 1)

    # 将音频数据写入FLAC文件（无损压缩格式，保留完整音频信息）
    sf.write(output_flac, audio, SAMPLE_RATE, format="FLAC")

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
    flac_file = FLAC(output_flac)
    flac_file["IMAGE_METADATA"] = json.dumps(metadata)  # 序列化为JSON字符串
    flac_file.save()

    # 计算并输出音频时长
    duration = len(audio) / SAMPLE_RATE
    print("✅ 编码完成")
    print(f"   分辨率: {width}x{height}")
    print(f"   音频时长: {duration:.2f} 秒")


def decode_play_draw(input_flac, output_image_path):

    print(f"🔊 加载并播放: {input_flac}")

    # 读取FLAC文件的元数据
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

    # 读取音频数据（忽略返回的采样率，使用元数据中的采样率保证一致性）
    audio, _ = sf.read(input_flac)

    # 如果是立体声（二维数组），只取左声道（第一列）转为单声道
    if audio.ndim > 1:
        audio = audio[:, 0]

    # 计算总像素数和预期的音频采样点数
    num_pixels = width * height
    expected_samples = num_pixels * SAMPLES_PER_PIXEL

    if len(audio) < expected_samples:
        print("⚠️ 音频过短，无法完整解码")
        return

    # 截取刚好能还原完整图像的音频段
    audio = audio[:expected_samples]

    pygame.init()
    pygame.mixer.init(SAMPLE_RATE, -16, 1, 1024)
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    
    window_title = f"🌌 藏在声音里的图片 | 图片分辨率：{width}x{height}"
    pygame.display.set_caption(window_title)

    # 加载音频文件并准备播放
    sound = pygame.mixer.Sound(input_flac)
    # 设置播放音量（转换为0-1的浮点数）
    sound.set_volume(VOLUME_PERCENT / 100.0)
    sound.play()

    # 生成像素坐标列表：按列优先的顺序（和编码时的处理顺序一致）
    coords = [(x, y) for x in range(width) for y in range(height)]

    # 创建空数组存储还原后的像素值（0-1）
    recovered_pixels = np.zeros(num_pixels, dtype=np.float32)

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

    # 创建一个和原始图像大小一致的pygame表面（用于绘制还原的像素）
    image_surface = pygame.Surface((width, height))

    # 逐个像素还原图像
    for i in range(num_pixels):
        # 处理pygame窗口事件（如关闭窗口）
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # 计算当前像素对应的音频段的起止位置
        start = i * SAMPLES_PER_PIXEL
        end = start + SAMPLES_PER_PIXEL
        frame = audio[start:end]  # 提取当前像素的音频段

        # 应用汉宁窗（减少FFT的频谱泄漏，提高频率检测精度）
        window = np.hanning(len(frame))
        frame_win = frame * window

        # 补零到N_FFT长度（提高FFT的频率分辨率）
        padded = np.zeros(N_FFT)
        padded[:len(frame_win)] = frame_win

        # 对加窗后的音频段进行快速傅里叶变换（时域转频域）
        fft_result = np.fft.rfft(padded)
        # 计算频域信号的幅值（反映各频率的能量）
        magnitude = np.abs(fft_result)

        # 提取目标频率范围内的幅值
        sub_mag = magnitude[f_min_idx:f_max_idx + 1]
        # 找到幅值最大的频率索引（能量最强的频率，即编码时的原始频率）
        peak_idx = f_min_idx + np.argmax(sub_mag)
        # 获取该索引对应的实际频率
        estimated_freq = freqs[peak_idx]

        # 将检测到的频率映射回0-1的灰度值
        gray = (estimated_freq - F_MIN) / (F_MAX - F_MIN)
        # 限幅确保灰度值在0-1范围内
        gray = np.clip(gray, 0.0, 1.0)

        # 保存还原后的像素值
        recovered_pixels[i] = gray

        # 获取当前像素的坐标并在pygame表面上绘制
        x, y = coords[i]
        val = int(gray * 255)  # 转换为0-255的整数灰度值
        image_surface.set_at((x, y), (val, val, val))  # 设置像素颜色（灰度）

        # 每处理1024个像素刷新一次显示（平衡性能和流畅度）
        if i % 1024 == 0:
            # 计算缩放比例（保证图像完整显示在窗口中）
            scale_ratio = min(SCREEN_W / width, SCREEN_H / height)
            new_w = int(width * scale_ratio)
            new_h = int(height * scale_ratio)

            # 平滑缩放图像到窗口大小
            scaled_surface = pygame.transform.smoothscale(
                image_surface, (new_w, new_h)
            )

            # 计算图像在窗口中的居中偏移量
            offset_x = (SCREEN_W - new_w) // 2
            offset_y = (SCREEN_H - new_h) // 2

            # 清屏（黑色背景）
            screen.fill((0, 0, 0))
            # 在窗口中绘制缩放后的图像
            screen.blit(scaled_surface, (offset_x, offset_y))
            # 更新显示
            pygame.display.flip()

    # 最终刷新显示（完整还原后的图像）
    scale_ratio = min(SCREEN_W / width, SCREEN_H / height)
    new_w = int(width * scale_ratio)
    new_h = int(height * scale_ratio)

    scaled_surface = pygame.transform.smoothscale(
        image_surface, (new_w, new_h)
    )

    offset_x = (SCREEN_W - new_w) // 2
    offset_y = (SCREEN_H - new_h) // 2

    screen.fill((0, 0, 0))
    screen.blit(scaled_surface, (offset_x, offset_y))
    pygame.display.flip()

    # 按列优先模式重建图像数组（转置还原为原始的行优先格式）
    img_array = recovered_pixels.reshape((width, height)).T

    # 转换为0-255的uint8格式并保存图像
    img_uint8 = (img_array * 255).astype(np.uint8)
    Image.fromarray(img_uint8, mode='L').save(output_image_path)

    print(f"✅ 解码完成并保存: {output_image_path}")

    # 保持窗口显示，直到用户关闭
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        time.sleep(0.01)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法:")
        print("编码: python app.py encode input.jpg signal.flac")
        print("解码: python app.py draw signal.flac result.png")
        sys.exit(1)

    # 获取操作模式（encode/draw）
    mode = sys.argv[1]
    if mode == "encode":
        # 编码模式：参数2=输入图像，参数3=输出FLAC
        image_to_audio(sys.argv[2], sys.argv[3])
    elif mode == "draw":
        # 解码模式：参数2=输入FLAC，参数3=输出图像
        decode_play_draw(sys.argv[2], sys.argv[3])
    else:
        print("❌ 未知模式")
