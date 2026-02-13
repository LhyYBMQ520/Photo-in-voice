import numpy as np
from scipy.io.wavfile import write, read
from PIL import Image
import pygame
import sys
import time

# ====== 配置参数（必须与编码一致）======
F_MIN = 500                   # 黑色对应频率 (Hz)
F_MAX = 3000                  # 白色对应频率 (Hz)
IMAGE_SIZE = (512, 512)       # 图像尺寸
ENCODE_DIRECTION = "col"      # "row" 或 "col"
SAMPLE_RATE = 44100           # 音频采样率
SAMPLES_PER_PIXEL = 48       # 每个像素对应的音频采样点数
# ======================================

def image_to_audio(image_path, output_wav):
    print(f"🖼️  加载图像: {image_path}")
    img = Image.open(image_path).convert('L')
    img = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    pixels = np.array(img) / 255.0

    if ENCODE_DIRECTION == "row":
        data = pixels
    else:
        data = pixels.T

    total_samples = data.size * SAMPLES_PER_PIXEL
    audio = np.zeros(total_samples, dtype=np.float32)
    t_base = np.arange(SAMPLES_PER_PIXEL) / SAMPLE_RATE

    idx = 0
    for unit in data:
        for pixel in unit:
            freq = F_MIN + pixel * (F_MAX - F_MIN)
            wave = np.sin(2 * np.pi * freq * t_base)
            audio[idx:idx + SAMPLES_PER_PIXEL] = wave
            idx += SAMPLES_PER_PIXEL

    audio = np.clip(audio, -1, 1)
    write(output_wav, SAMPLE_RATE, (audio * 32767).astype(np.int16))
    
    actual_duration = len(audio) / SAMPLE_RATE
    print(f"✅ 编码完成！图像尺寸: {IMAGE_SIZE}")
    print(f"   音频实际时长: {actual_duration:.2f} 秒")


def decode_play_draw(input_wav, output_image_path, volume=0.8):
    """
    边播放音频、边实时绘制图像、最后保存结果
    """
    print(f"🔊 加载并播放: {input_wav}")
    rate, audio_data = read(input_wav)

    if audio_data.dtype == np.int16:
        audio = audio_data.astype(np.float32) / 32767.0
    else:
        audio = audio_data.astype(np.float32)

    num_pixels = IMAGE_SIZE[0] * IMAGE_SIZE[1]
    expected_samples = num_pixels * SAMPLES_PER_PIXEL

    if len(audio) >= expected_samples:
        audio = audio[:expected_samples]
    else:
        print("⚠️ 音频过短，无法完整解码")
        return

    # 初始化 Pygame：音频 + 画布
    pygame.init()
    pygame.mixer.init(SAMPLE_RATE, -16, 1, 1024)
    screen = pygame.display.set_mode(IMAGE_SIZE)
    pygame.display.set_caption("正在播放并绘制图像...")
    screen.fill((0, 0, 0))
    pygame.display.flip()

    sound = pygame.mixer.Sound(input_wav)
    sound.set_volume(volume)
    sound.play()

    # 生成像素绘制顺序（匹配编码方向）
    w, h = IMAGE_SIZE
    coords = [(x, y) for y in range(h) for x in range(w)] if ENCODE_DIRECTION == "row" \
             else [(x, y) for x in range(w) for y in range(h)]

    recovered_pixels = np.zeros(num_pixels, dtype=np.float32)

    # FFT 设置（与编码完全对应）
    n_fft = 1024
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / SAMPLE_RATE)
    valid_idx = np.where((freqs >= F_MIN - 100) & (freqs <= F_MAX + 100))[0]
    f_min_idx, f_max_idx = valid_idx[0], valid_idx[-1]

    # 实时解码 + 绘图
    for i in range(num_pixels):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        start = i * SAMPLES_PER_PIXEL
        end = start + SAMPLES_PER_PIXEL
        frame = audio[start:end]

        window = np.hanning(len(frame))
        frame_win = frame * window
        padded = np.zeros(n_fft)
        padded[:len(frame_win)] = frame_win

        fft_result = np.fft.rfft(padded)
        magnitude = np.abs(fft_result)
        sub_mag = magnitude[f_min_idx:f_max_idx+1]
        peak_idx = f_min_idx + np.argmax(sub_mag)
        estimated_freq = freqs[peak_idx]

        gray = (estimated_freq - F_MIN) / (F_MAX - F_MIN)
        gray = np.clip(gray, 0.0, 1.0)
        recovered_pixels[i] = gray

        x, y = coords[i]
        val = int(gray * 255)
        screen.set_at((x, y), (val, val, val))

        if i % 256 == 0:
            pygame.display.flip()

    pygame.display.flip()

    # 保存图像
    if ENCODE_DIRECTION == "row":
        img_array = recovered_pixels.reshape(IMAGE_SIZE)
    else:
        img_array = recovered_pixels.reshape((h, w)).T

    img_uint8 = (img_array * 255).astype(np.uint8)
    Image.fromarray(img_uint8, mode='L').save(output_image_path)
    print(f"✅ 播放+绘制+保存完成：{output_image_path}")

    # 保持窗口直到关闭
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        time.sleep(0.01)


# ====== 主程序入口 ======
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法:")
        print("  编码：     python app.py encode input.jpg signal.wav")
        print("  边播边画： python app.py draw signal.wav result.png")
        print("\n💡 音量可在 decode_play_draw() 调用处修改（默认 0.1）")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "encode":
        image_to_audio(sys.argv[2], sys.argv[3])
    elif mode == "draw":
        decode_play_draw(sys.argv[2], sys.argv[3], volume=0.1)  # 可调音量
    else:
        print("❌ 未知模式，请使用 'encode' 或 'draw'")
        sys.exit(1)
