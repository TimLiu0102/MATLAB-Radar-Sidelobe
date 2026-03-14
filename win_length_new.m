clc;clear;close all;

clc;clear;close all;

%% 参数设置
% 雷达参数
B = 20e6;               % 带宽: 20 MHz
T_pulse = 50e-6;        % 脉宽: 50 μs
fs = 60e6;              % 采样率: 60 MHz

% 处理参数
N_pulse = round(T_pulse * fs);      % 脉冲内采样点数
N_fft = 2^nextpow2(N_pulse * 8);    % FFT点数（8倍过采样）
freq = (-N_fft/2:N_fft/2-1) * (fs/N_fft);  % 频率轴

% 补偿参数
alpha_reg = 1e-2;       % 正则化因子系数（增大了）
A_max = 4.0;            % 最大增益限制（适当增大）

% 显示参数
f_center_idx = find(abs(freq) < B/2);

%% 步骤0.1: 生成增强的系统频率响应 H[k]
% 增强幅度响应：更大的波动和不平坦度
H_mag = zeros(N_fft, 1);
H_mag(f_center_idx) = 0.7 + 0.4*cos(2*pi*freq(f_center_idx)/B*6) .* ...
                      exp(-(freq(f_center_idx)/(0.5*B)).^2) + ...
                      0.1*sin(2*pi*freq(f_center_idx)/B*3);

% 增强相位响应：更大的非线性相位失真
H_phase = zeros(N_fft, 1);
H_phase(f_center_idx) = 0.4*pi*(freq(f_center_idx)/B).^3 + ...
                        0.3*pi*sin(2*pi*freq(f_center_idx)/B*5) + ...
                        0.05*pi*(freq(f_center_idx)/B).^2;

% 组合成复数频率响应
H_k = H_mag .* exp(1j*H_phase);
H_k = fftshift(H_k);  % 转换为MATLAB的FFT顺序

%% 步骤1: 生成理想LFM信号及其频谱
% 生成时域LFM信号
t = (-N_pulse/2:N_pulse/2-1)' / fs;
k_chirp = B / T_pulse;  % 调频率
s_lfm = exp(1j * pi * k_chirp * t.^2);  % 基带LFM

% 过采样并补零到FFT长度
s_lfm_padded = zeros(N_fft, 1);
s_lfm_padded(1:N_pulse) = s_lfm;

% 计算LFM频谱
S_LFM_k = fft(s_lfm_padded, N_fft);

% 设置目标参考频谱 R[k]（使用未加权的理想LFM频谱）
R_k = S_LFM_k;

%% 步骤2: 计算发射端预补偿谱（幅度+相位全补偿）
% 计算正则化项：使用均值而不是最大值，更稳定
epsilon = alpha_reg * mean(abs(H_k .* S_LFM_k));

% 核心补偿公式：G_tx[k] = R[k] / (H[k] * S_LFM[k] + ε)
G_tx_k = R_k ./ (H_k .* S_LFM_k + epsilon);

% 幅度限制：防止过高的增益导致PA饱和
G_tx_mag = abs(G_tx_k);
max_G = max(G_tx_mag);
if max_G > A_max
    G_tx_k = G_tx_k ./ max(1, G_tx_mag/A_max);
end

%% 步骤2: 频域窗/带宽约束 - 使用海明窗

f_idx1 = find(abs(freq) <= B);
%创建海明窗
hamming_window1 = zeros(N_fft, 1);
% 生成对应带宽长度的海明窗（仅在LFM信号带宽内施加海明窗，带宽外为0）
hamming_win_local = hamming(length(f_idx1));  % 生成局部海明窗
hamming_window1(f_idx1) = hamming_win_local;   % 填充到频域对应位置
W_k1 = fftshift(hamming_window1);      % 转换为FFT顺序
% 应用海明窗
G_tx_k_windowed = G_tx_k .* W_k1;

f_idx2 = find(abs(freq) <= B/2);
%创建海明窗
hamming_window2 = zeros(N_fft, 1);
% 生成对应带宽长度的海明窗（仅在LFM信号带宽内施加海明窗，带宽外为0）
hamming_win_local2 = hamming(length(f_idx2));  % 生成局部海明窗
hamming_window2(f_idx2) = hamming_win_local2;   % 填充到频域对应位置
W_k2 = fftshift(hamming_window2);      % 转换为FFT顺序

%% 步骤4: 形成发射频谱并IFFT回时域
% 应用预补偿：S_tx[k] = S_LFM[k] * G_tx[k] * W[k]
S_tx_k = S_LFM_k .* G_tx_k_windowed;
S_tx_hamming_k = S_LFM_k .* W_k2;

% IFFT得到时域信号
s_tx_time = ifft(S_tx_k, N_fft);
s_tx_hamming_time = ifft(S_tx_hamming_k, N_fft);

% 截取有效部分（保留原始脉冲长度）
s_tx_pulse = s_tx_time(1:N_pulse);

% 能量归一化（而不是幅度归一化）
s_tx_pulse = s_tx_pulse / sqrt(sum(abs(s_tx_pulse).^2));

% 情况3: 预补偿LFM经过系统H（有预补偿）
S_tx_full = fft(s_tx_time, N_fft);  % 预补偿信号频谱
S_tx_with_H = S_tx_full .* H_k;  % 经过系统响应
s_tx_with_H_time = ifft(S_tx_with_H, N_fft);
s_tx_with_H = s_tx_with_H_time(1:N_pulse);
s_tx_with_H = s_tx_with_H / sqrt(sum(abs(s_tx_with_H).^2));

% 情况3: 加海明窗
s_hamming_pulse = s_tx_hamming_time(1:N_pulse);  % 截取原始脉冲长度
s_hamming_pulse = s_hamming_pulse / sqrt(sum(abs(s_hamming_pulse).^2));  % 能量归一化

%% 步骤5: 自相关分析
% 计算自相关函数
t_corr = (-N_pulse+1:N_pulse-1) / fs * 1e6;  % 微秒

% 计算三种情况的自相关
auto_corr_with_comp = xcorr(s_tx_with_H, s_tx_with_H);
auto_corr_hamming = xcorr(s_hamming_pulse, s_hamming_pulse);

% 归一化自相关（能量归一化）
auto_corr_with_comp = auto_corr_with_comp / max(abs(auto_corr_with_comp));
auto_corr_hamming = auto_corr_hamming / max(abs(auto_corr_hamming)); 

% 转换为dB显示
auto_corr_with_comp_db = 20*log10(abs(auto_corr_with_comp) + 1e-10);
auto_corr_hamming_db = 20*log10(abs(auto_corr_hamming) + 1e-10);

center_idx = ceil(length(t_corr)/2);
range_idx = center_idx-50:center_idx+50;

%% 步骤6: 自相关函数绘图
% 自相关对数
figure(1);
plot(t_corr, auto_corr_hamming_db, 'b-', 'LineWidth', 1.5);hold on;
plot(t_corr, auto_corr_with_comp_db, 'r--', 'LineWidth', 1.5);
xlabel('Time Delay (μs)'); ylabel('Amplitude(dB)');
legend('Window Width:B','Window Width:2B', 'Location', 'best');
grid on;
ylim([-120, 0]);

% 自相关主瓣区域（线性）
figure(2);
center_idx = ceil(length(t_corr)/2);
range_idx = center_idx-50:center_idx+50;
plot(t_corr(range_idx), abs(auto_corr_hamming(range_idx)), 'b-', 'LineWidth', 1.5); hold on;
plot(t_corr(range_idx), abs(auto_corr_with_comp(range_idx)), 'r--', 'LineWidth', 1.5);
xlabel('Time Delay (μs)'); ylabel('Normalized Amplitude');
legend('Window Width:B', 'Window Width:2B', 'Location', 'best');
grid on;
grid on;

figure(3);
s_ideal_padded = zeros(N_fft, 1);
s_ideal_padded(1:N_pulse) = s_tx_with_H;
S_ideal_mag = fftshift(abs(fft(s_ideal_padded, N_fft)));

s_with_H_padded = zeros(N_fft, 1);
s_with_H_padded(1:N_pulse) = s_hamming_pulse;
S_no_comp_mag = fftshift(abs(fft(s_with_H_padded, N_fft)));

plot(freq/1e6, 20*log10(S_ideal_mag/max(S_ideal_mag)), 'k-', 'LineWidth', 1.5); hold on;
plot(freq/1e6, 20*log10(S_no_comp_mag/max(S_no_comp_mag)), 'r--', 'LineWidth', 1.5);
xlabel('Frequency (MHz)'); ylabel('Amplitude (dB)');
legend('Window Width:2B', 'Window Width:B', 'Location', 'best');
grid on;

figure(4);
plot(freq/1e6, unwrap(angle(fftshift(fft(s_ideal_padded, N_fft)))), 'k-', 'LineWidth', 1.5); hold on;
plot(freq/1e6, unwrap(angle(fftshift(fft(s_with_H_padded, N_fft)))), 'r--', 'LineWidth', 1.5);
xlabel('Frequency (MHz)'); ylabel('Amplitude (dB)');
legend('Window Width:2B', 'Window Width:B', 'Location', 'best');
grid on;
%% 步骤8: 计算和显示性能指标
% 峰值旁瓣比（PSLR）
center_idx = ceil(length(auto_corr_hamming)/2);
pslr_no_comp = compute_pslr_corrected(auto_corr_hamming, center_idx, fs, B);
pslr_with_comp = compute_pslr_corrected(auto_corr_with_comp, center_idx, fs, B);
fprintf('B: %.3f(dB)\n',pslr_no_comp);
fprintf('2B: %.3f(dB)\n',pslr_with_comp);
% 主瓣宽度（3dB）
bw3db_no_comp = compute_3db_width_corrected(auto_corr_hamming, center_idx, fs, B);
bw3db_with_comp = compute_3db_width_corrected(auto_corr_with_comp, center_idx, fs, B);
fprintf('B: %.3f μs\n', bw3db_no_comp);
fprintf('2B: %.3f μs\n', bw3db_with_comp);

% 计算PAPR（峰均功率比） - 使用能量归一化后的信号
PAPR_ideal = 10*log10(max(abs(s_tx_with_H).^2) / mean(abs(s_tx_with_H).^2));
PAPR_no_comp = 10*log10(max(abs(s_hamming_pulse).^2) / mean(abs(s_hamming_pulse).^2));
fprintf('B: %.3f (dB)\n', PAPR_no_comp);
fprintf('2B: %.3f (dB)\n', PAPR_ideal);

%% 修正后的辅助函数
function pslr = compute_pslr_corrected(corr_signal, peak_idx, fs, B)
    % 修正的峰值旁瓣比计算函数
    % 基于主瓣理论宽度定义区域
    % 主瓣宽度 ≈ 1/B，转换为采样点数
    mainlobe_width_samples = ceil(2 * fs / B);  % 取2倍作为安全余量
    
    % 确保边界有效
    mainlobe_start = max(1, peak_idx - mainlobe_width_samples);
    mainlobe_end = min(length(corr_signal), peak_idx + mainlobe_width_samples);
    
    % 创建掩码排除主瓣
    mask = true(length(corr_signal), 1);
    mask(mainlobe_start:mainlobe_end) = false;
    
    % 如果没有旁瓣，返回一个较差的值
    if sum(mask) == 0
        pslr = -5;  % 一个较差的值
        return;
    end
    
    % 计算旁瓣峰值和主瓣峰值
    sidelobe_peak = max(abs(corr_signal(mask)));
    mainlobe_peak = abs(corr_signal(peak_idx));
    
    % 计算PSLR（dB）
    if mainlobe_peak > 0
        pslr = 20*log10(sidelobe_peak / mainlobe_peak);
    else
        pslr = -inf;
    end
end
function bw_3db = compute_3db_width_corrected(corr_signal, peak_idx, fs, B)
    % 修正的3dB主瓣宽度计算函数
    corr_mag = abs(corr_signal);
    peak_value = corr_mag(peak_idx);
    
    % 找到3dB点（峰值下降3dB）
    threshold_3db = peak_value / sqrt(2);  % -3dB点
    
    % 向左搜索3dB点
    left_idx = peak_idx;
    step_count = 0;
    max_steps = 100;  % 防止无限循环
    
    while left_idx > 1 && corr_mag(left_idx) >= threshold_3db && step_count < max_steps
        left_idx = left_idx - 1;
        step_count = step_count + 1;
    end
    
    % 向右搜索3dB点
    right_idx = peak_idx;
    step_count = 0;
    
    while right_idx < length(corr_mag) && corr_mag(right_idx) >= threshold_3db && step_count < max_steps
        right_idx = right_idx + 1;
        step_count = step_count + 1;
    end
    
    % 计算宽度（秒）
    width_samples = right_idx - left_idx;
    bw_3db = width_samples / fs * 1e6;  % 转换为微秒
    
    % 如果找不到3dB点或宽度异常，返回理论值
    if bw_3db <= 0 || bw_3db > 100
        bw_3db = 0.886 / B * 1e6;  % 修正后的理论公式
    end
    
    % 确保宽度不会太小
    if bw_3db < 0.01
        bw_3db = 0.886 / B * 1e6;
    end
end

