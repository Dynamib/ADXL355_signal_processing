# ADXL355 Moxibustion Vibration Signal Processing

基于 ADXL355 低噪声三轴加速度计的艾灸「得气」振动信号处理分析。

## 背景

中医艾灸临床实践中，施灸者有时感觉到艾条出现微细振动（「得气」现象）。本项目通过传感器信号处理技术，对 ADXL355 采集的加速度计数据进行降噪、时频分析和统计验证，探索「得气」振动的物理特征。

传感器: ADXL355 (25 µg/√Hz 超低噪声, ±2g 量程)

## 处理流程

```
原始 CSV 数据 (不规则采样)
    │
    ├── [Step 0] 预处理
    │   线性插值 → 均匀 1000Hz → 去均值(去重力)
    │
    ├── [Step 1] 降噪与特征提取
    │   巴特沃斯带通 1-200Hz → MED 盲反卷积 → 希尔伯特包络 → 对数变换
    │
    ├── [Step 2] 高分辨率时频分析
    │   STFT (Hann 512pt, 75%重叠) + SST 同步压缩变换 (Morlet 小波)
    │
    └── [Step 3] 统计验证
        Welch PSD → 滑动窗口频谱特征 → Z-score 异常检测
```

## 快速开始

```bash
pip install -r requirements.txt
python run_pipeline.py
```

输出图片和报告保存在 `output/figures/` 目录。

## 依赖

| 库 | 用途 |
|----|------|
| numpy, scipy | 基础信号处理、滤波 |
| pandas | 数据加载 |
| ssqueezepy | SST 同步压缩变换 |
| matplotlib, seaborn | 可视化 |

## 参数配置

所有可调参数在 `config.py` 中集中管理：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| BP_LOWCUT / BP_HIGHCUT | 1.0 / 200.0 Hz | 带通滤波截止频率 |
| BP_ORDER | 4 | 巴特沃斯阶数 |
| MED_FILTER_LENGTH | 30 | MED 逆滤波器长度 |
| MED_MAX_ITER | 50 | MED 最大迭代次数 |
| STFT_NPERSEG | 512 | STFT 窗口长度 |
| WINDOW_DURATION_SEC | 2.0 | 滑动窗口时长 |
| Z_THRESHOLD | 3.0 | 异常检测阈值 |

## 输出文件

| 文件 | 内容 |
|------|------|
| `01_raw_time_series.png` | 三轴原始时域波形 |
| `02_psd_comparison.png` | 三轴 Welch PSD 对比 |
| `03_bandpass_filtered_X/Y/Z.png` | 每轴滤波前后对比 |
| `04_envelope_kurtosis_X/Y/Z.png` | MED 峭度变化 + 包络分析 |
| `05_stft_X/Y/Z.png` | STFT 时频图 |
| `06_sst_X/Y/Z.png` | SST 高分辨率时频图 |
| `07_sliding_window_features_X/Y/Z.png` | 滑动窗口频谱特征时间序列 |
| `08_anomaly_heatmap_X/Y/Z.png` | 异常检测热力图 |
| `anomaly_windows_X/Y/Z.csv` | 异常窗口数据 |

## 项目结构

```
├── config.py           参数集中管理
├── run_pipeline.py     一键运行全流程
├── requirements.txt    依赖声明
├── src/
│   ├── io_utils.py         CSV加载 + 重采样
│   ├── preprocessing.py    去重力 + 带通滤波
│   ├── denoising.py        MED + 包络 + 对数变换
│   ├── time_frequency.py   STFT + SST
│   ├── statistical.py      PSD + 频谱特征 + 异常检测
│   └── visualization.py   全部绑图函数
└── output/figures/     输出图片和报告
```
