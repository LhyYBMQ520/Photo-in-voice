# 藏在声音里的图片（Photo in Voice）

## 对旅行者一号金唱片的一次现代致敬

> 把人类的图像，刻进声波之中。

本项目灵感来自旅行者 1 号（Voyager 1）携带的镀金唱片（Voyager Golden Record）。
1977 年，人类把声音、音乐与图像编码后送入星际，期待某一天被另一个文明解读。

这个项目尝试用 Python 复现这一思路：

- 把灰度图片编码为 FLAC 无损音频
- 把解码所需参数写入 FLAC 元数据
- 解码时一边播放音频，一边实时重建图像
- 最终输出原始分辨率的还原图

它不是隐写术，而是一种跨模态转译：视觉与听觉之间的来回映射。

## 功能概览

- 图片编码：`encode` 模式将灰度图转为单声道 FLAC。
- 音频解码：`draw` 模式从 FLAC 频谱反推灰度，并实时绘制。
- 元数据自描述：宽高、频率范围、采样率等参数写入 `IMAGE_METADATA`。
- 大图支持：可通过开关允许超大分辨率图片。
- 并行优化：支持 `sync/thread/process`，并在 `auto` 模式下自适应选择后端。

## 核心原理

### 1) 像素到频率

每个灰度像素（0~255）映射为一个频率（默认 500~3000 Hz）：

```text
freq = F_MIN + (pixel / 255) * (F_MAX - F_MIN)
```

像素值越亮，频率越高。

### 2) 编码流程（image -> flac）

1. 读取图片并转灰度。
2. 使用列优先顺序展开像素（先整列，再下一列）。
3. 每个像素生成固定长度正弦波（`SAMPLES_PER_PIXEL`）。
4. 分块并行生成音频，流式写入 FLAC，避免整段常驻内存。
5. 将参数写入 FLAC 的 `IMAGE_METADATA` 字段。

### 3) 解码流程（flac -> image）

1. 读取 FLAC 及 `IMAGE_METADATA`。
2. 按像素块读取音频，重组为帧。
3. 对每帧加汉宁窗，做 FFT，取目标频段峰值。
4. 把频率反映射为灰度值并逐像素绘制。
5. 播放音频同时显示还原过程，最终保存为图片。

## 安装

建议 Python 3.12（其他 3.x 通常也可运行）：

```bash
pip install -r requirements.txt
```

依赖如下：

- `numpy`
- `Pillow`
- `pygame`
- `soundfile`
- `mutagen`

## 使用方式

### 编码

```bash
python app.py encode input.jpg signal.flac [workers]
```

- `workers` 可选，默认自动使用 CPU 线程数。
- 输出文件为 `signal.flac`。

### 解码与播放

```bash
python app.py draw signal.flac result.png [workers]
```

- 程序会打开 `1280x720` 窗口实时绘制。
- 音频会同步播放（默认音量 5%）。
- 最终保存 `result.png`（原始分辨率灰度图）。

## 可调参数（app.py 顶部）

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `F_MIN` | 500 | 黑色对应最小频率 |
| `F_MAX` | 3000 | 白色对应最大频率 |
| `SAMPLE_RATE` | 44100 | 采样率 |
| `SAMPLES_PER_PIXEL` | 48 | 每像素采样点数 |
| `SCREEN_W` | 1280 | 预览窗口宽 |
| `SCREEN_H` | 720 | 预览窗口高 |
| `N_FFT` | 512 | FFT 点数 |
| `VOLUME_PERCENT` | 5 | 播放音量百分比 |
| `ENCODE_COLUMN_BLOCK` | 8 | 编码列分块大小 |
| `DECODE_PIXEL_BLOCK` | 8192 | 解码像素分块大小 |
| `ALLOW_LARGE_IMAGES` | `True` | 是否允许超大图 |
| `ENCODE_PARALLEL_MODE` | `auto` | 编码并行模式 |
| `DECODE_PARALLEL_MODE` | `auto` | 解码并行模式 |
| `MAX_ENCODE_TASK_BYTES` | 32MB | 编码任务目标字节上限 |
| `MAX_DECODE_TASK_BYTES` | 32MB | 解码任务目标字节上限 |

## 并行策略说明

- `workers <= 1` 时走同步模式。
- `auto` 模式下：
  - Windows 默认优先线程并行（降低进程通信开销与资源错误风险）。
  - 任务块过大时也会倾向线程并行。
- 若进程并行失败，程序会自动降级到线程并行重试。

## FLAC 元数据结构

写入字段：`IMAGE_METADATA`

```json
{
  "width": 1920,
  "height": 1080,
  "F_MIN": 500,
  "F_MAX": 3000,
  "SAMPLES_PER_PIXEL": 48,
  "SAMPLE_RATE": 44100,
  "N_FFT": 512
}
```

因此解码时不需要额外配置文件。

## 注意事项

1. 音频主要集中在 500~3000 Hz，请先降低耳机或音箱音量。
2. 高对比度图片（文字、剪影）更容易获得清晰还原。
3. 程序按列优先扫描，重建画面会从左到右展开。
4. 解码后窗口会保持，需手动关闭退出。
5. 若 FLAC 缺失 `IMAGE_METADATA`，解码会直接报错退出。

## 常见问题

### Q1: 为什么还原图有轻微误差？

这是频谱估计与离散采样导致的正常现象，可尝试：

- 提高 `SAMPLES_PER_PIXEL`
- 增大 `N_FFT`
- 使用更高对比度输入图像

### Q2: 处理大图时内存压力大怎么办？

- 调小 `ENCODE_COLUMN_BLOCK`
- 调小 `DECODE_PIXEL_BLOCK`
- 保持 `auto` 并行模式，让程序自动选择更稳妥后端

## 致敬星际讯息

> "The spacecraft will be encountered and the record played only if there are advanced spacefaring civilizations in interstellar space."  
> Carl Sagan

也许我们收不到回信。  
但至少能让一张图片，  
以声音的形式，在寂静中再次诞生。