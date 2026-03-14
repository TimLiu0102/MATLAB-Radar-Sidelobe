clc;clear;close all;

%% 发射端LFM预补偿算法
clear; close all; clc;

%% 步骤0: 仿真参数设置
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
f_center_idx = find(abs(freq)<=B/2);

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

%% 步骤3: 频域窗/带宽约束 - 使用矩形窗
% 创建海明窗：带宽内为1，带宽外为0
f_idx = find(abs(freq)<=B);
%创建海明窗
hamming_window = zeros(N_fft, 1);
% 生成对应带宽长度的海明窗（仅在LFM信号带宽内施加海明窗，带宽外为0）
hamming_win_local = hamming(length(f_idx));  % 生成局部海明窗
hamming_window(f_idx) = hamming_win_local;   % 填充到频域对应位置
W_k = fftshift(hamming_window);      % 转换为FFT顺序
% 应用海明窗
G_tx_k_windowed = G_tx_k .* W_k;

%% 步骤4: 形成发射频谱并IFFT回时域
% 应用预补偿：S_tx[k] = S_LFM[k] * G_tx[k] * W[k]
S_tx_k = S_LFM_k .* G_tx_k_windowed;

% IFFT得到时域信号
s_tx_time = ifft(S_tx_k, N_fft);

% 截取有效部分（保留原始脉冲长度）
s_tx_pulse = s_tx_time(1:N_pulse);

% 能量归一化（而不是幅度归一化）
s_tx_pulse = s_tx_pulse / sqrt(sum(abs(s_tx_pulse).^2));

%% 步骤5: 准备对比信号（统一能量归一化）
% 情况1: 理想LFM（作为参考基准）
s_ideal = s_lfm / sqrt(sum(abs(s_lfm).^2));

% 情况2: 原始LFM经过系统H（无预补偿）
S_LFM_full = fft(s_lfm_padded, N_fft);
S_with_H = S_LFM_full .* H_k;  % 经过系统响应
s_with_H_time = ifft(S_with_H, N_fft);
s_with_H = s_with_H_time(1:N_pulse);
s_with_H = s_with_H / sqrt(sum(abs(s_with_H).^2));

% 情况3: 预补偿LFM经过系统H（有预补偿）
S_tx_full = fft(s_tx_time, N_fft);  % 预补偿信号频谱
S_tx_with_H = S_tx_full .* H_k;  % 经过系统响应
s_tx_with_H_time = ifft(S_tx_with_H, N_fft);
s_tx_with_H = s_tx_with_H_time(1:N_pulse);
s_tx_with_H = s_tx_with_H / sqrt(sum(abs(s_tx_with_H).^2));

%% 步骤6: 自相关分析
% 计算自相关函数
t_corr = (-N_pulse+1:N_pulse-1) / fs * 1e6;  % 微秒

% 计算三种情况的自相关
auto_corr_ideal = xcorr(s_ideal, s_ideal);
auto_corr_no_comp = xcorr(s_with_H, s_with_H);
auto_corr_with_comp = xcorr(s_tx_with_H, s_tx_with_H);

% 归一化自相关（能量归一化）
auto_corr_ideal = auto_corr_ideal / max(abs(auto_corr_ideal));
auto_corr_no_comp = auto_corr_no_comp / max(abs(auto_corr_no_comp));
auto_corr_with_comp = auto_corr_with_comp / max(abs(auto_corr_with_comp));

% 转换为dB显示
auto_corr_ideal_db = 20*log10(abs(auto_corr_ideal) + 1e-10);
auto_corr_no_comp_db = 20*log10(abs(auto_corr_no_comp) + 1e-10);
auto_corr_with_comp_db = 20*log10(abs(auto_corr_with_comp) + 1e-10);

center_idx = ceil(length(t_corr)/2);
range_idx = center_idx-50:center_idx+50;
%% 步骤7: 综合性能分析
% 频域对比
figure(1);
s_ideal_padded = zeros(N_fft, 1);
s_ideal_padded(1:N_pulse) = s_ideal;
S_ideal_mag = fftshift(abs(fft(s_ideal_padded, N_fft)));

s_with_H_padded = zeros(N_fft, 1);
s_with_H_padded(1:N_pulse) = s_with_H;
S_no_comp_mag = fftshift(abs(fft(s_with_H_padded, N_fft)));

s_tx_with_H_padded = zeros(N_fft, 1);
s_tx_with_H_padded(1:N_pulse) = s_tx_with_H;
S_with_comp_mag = fftshift(abs(fft(s_tx_with_H_padded, N_fft)));

plot(freq/1e6, 20*log10(S_ideal_mag/max(S_ideal_mag)), 'k-', 'LineWidth', 1.5); hold on;
plot(freq/1e6, 20*log10(S_no_comp_mag/max(S_no_comp_mag)), 'r--', 'LineWidth', 1.5);
plot(freq/1e6, 20*log10(S_with_comp_mag/max(S_with_comp_mag)), 'b-.', 'LineWidth', 1.5);
xlim([-B/1e6*1.5, B/1e6*1.5]); ylim([-100, 5]);
xlabel('Frequency (MHz)'); ylabel('Amplitude (dB)');
legend('LFM', 'S_{out}(f)', 'S_{tx}(f)', 'Location', 'best');
grid on;

% 理想和无补偿自相关
% 归一化
figure(2);
plot(t_corr(range_idx), abs(auto_corr_ideal(range_idx)), 'k-', 'LineWidth', 1.5); hold on;
plot(t_corr(range_idx), abs(auto_corr_no_comp(range_idx)), 'r--', 'LineWidth', 1.5);
xlabel('Time Delay(μs)'); ylabel('Normalized Amplitude');
legend('LFM', 'S_{out}(f)', 'Location', 'best');
grid on;

% 对数
figure(3)
plot(t_corr, auto_corr_ideal_db, 'k-', 'LineWidth', 1.5); hold on;
plot(t_corr, auto_corr_no_comp_db, 'r--', 'LineWidth', 1.5);
xlabel('Time Delay (μs)'); ylabel('Amplitude (dB)');
legend('LFM', 'S_{out}(f)', 'Location', 'best');
grid on;
ylim([-120, 0]);

% 预补偿和无补偿自相关
% 归一化
figure(4);
plot(t_corr(range_idx), abs(auto_corr_no_comp(range_idx)), 'r--', 'LineWidth', 1.5);hold on;
plot(t_corr(range_idx), abs(auto_corr_with_comp(range_idx)), 'b-', 'LineWidth', 1.5);
xlabel('Time Delay(μs)'); ylabel('Normalized Amplitude');
legend( 'S_{out}(f)', 'S_{tx}(f)', 'Location', 'best');
grid on;

% 对数
figure(5);
plot(t_corr, auto_corr_no_comp_db, 'r--', 'LineWidth', 1.5); hold on;
plot(t_corr, auto_corr_with_comp_db, 'b-', 'LineWidth', 1.5);
xlabel('Time Delay (μs)'); ylabel('Amplitude (dB)');
legend(  'S_{out}(f)', 'S_{tx}(f)', 'Location', 'best');
grid on;
ylim([-120, 0]);

figure(6);
plot(t*1e6, real(s_ideal), 'k-', 'LineWidth', 1.5); hold on;
plot(t*1e6, real(s_with_H), 'r--', 'LineWidth', 1.5);
xlabel('Time (μs)'); ylabel('Amplitude');
legend('LFM', 'S_{out}(f)', 'Location', 'best');
grid on;

figure(7);
plot(freq/1e6, 20*log10(S_ideal_mag/max(S_ideal_mag)), 'k-', 'LineWidth', 1.5); hold on;
plot(freq/1e6, 20*log10(S_no_comp_mag/max(S_no_comp_mag)), 'r--', 'LineWidth', 1.5);
xlim([-B/1e6*1.5, B/1e6*1.5]);ylim([-50, 0]);
xlabel('Frequency (MHz)'); ylabel('Amplitude (dB)');
legend('LFM', 'S_{out}(f)', 'Location', 'best');
grid on;

figure(8);
plot(t_corr(range_idx), abs(auto_corr_ideal(range_idx)), 'k-', 'LineWidth', 1.5); hold on;
plot(t_corr(range_idx), abs(auto_corr_no_comp(range_idx)), 'r--', 'LineWidth', 1.5);
plot(t_corr(range_idx), abs(auto_corr_with_comp(range_idx)), 'b-.', 'LineWidth', 1.5);
xlabel('Time Delay(μs)'); ylabel('Normalized Amplitude');
legend('LFM', 'S_{out}(t)', 'S_{tx}(t)', 'Location', 'best');
grid on;

figure(9);
plot(freq/1e6, unwrap(angle(fftshift(fft(s_ideal_padded, N_fft)))), 'k-', 'LineWidth', 1.5); hold on;
plot(freq/1e6, unwrap(angle(fftshift(fft(s_with_H_padded, N_fft)))), 'r--', 'LineWidth', 1.5);
plot(freq/1e6, unwrap(angle(fftshift(fft(s_tx_with_H_padded, N_fft)))), 'b-.', 'LineWidth', 1.5);
xlim([-B/1e6*1.5, B/1e6*1.5]);
xlabel('Frequency (MHz)'); ylabel('Phase (rad)');
legend('LFM', 'S_{out}(f)','S_{tx}(f)', 'Location', 'best');
grid on;

figure(10);
plot(freq/1e6, unwrap(angle(fftshift(fft(s_lfm_padded, N_fft)))), 'k-', 'LineWidth', 1.5); hold on;
plot(freq/1e6, unwrap(angle(fftshift(fft(s_with_H_padded, N_fft)))), 'r--', 'LineWidth', 1.5);
xlim([-B/1e6*1.5, B/1e6*1.5]);
xlabel('Frequency (MHz)'); ylabel('Phase (rad)');
legend('LFM', 'S_{out}(f)', 'Location', 'best');
grid on;

%% 步骤8: 传统窗对比（矩形窗/汉宁窗/海明窗/布莱克曼窗）
window_names = {'Rectangular', 'Hann', 'Hamming', 'Blackman'};
num_windows = numel(window_names);

auto_corr_windows_db = zeros(2*N_pulse-1, num_windows);
PSLR_windows = zeros(num_windows, 1);
ISLR_windows = zeros(num_windows, 1);

for i = 1:num_windows
    % 生成传统窗（频域带宽固定为 B）
    f_idx_win = find(abs(freq) <= B/2);
    win_full = zeros(N_fft, 1);

    switch window_names{i}
        case 'Rectangular'
            local_win = rectwin(length(f_idx_win));
        case 'Hann'
            local_win = hann(length(f_idx_win));
        case 'Hamming'
            local_win = hamming(length(f_idx_win));
        case 'Blackman'
            local_win = blackman(length(f_idx_win));
    end

    win_full(f_idx_win) = local_win;
    W_win = fftshift(win_full);

    % 预补偿 + 加窗 + 系统响应
    S_tx_win = S_LFM_k .* (G_tx_k .* W_win);
    s_tx_win_time = ifft(S_tx_win, N_fft);
    S_with_H_win = fft(s_tx_win_time, N_fft) .* H_k;
    s_with_H_win = ifft(S_with_H_win, N_fft);
    s_with_H_win = s_with_H_win(1:N_pulse);
    s_with_H_win = s_with_H_win / sqrt(sum(abs(s_with_H_win).^2));

    % 自相关
    auto_corr_win = xcorr(s_with_H_win, s_with_H_win);
    auto_corr_win = auto_corr_win / max(abs(auto_corr_win));
    auto_corr_windows_db(:, i) = 20*log10(abs(auto_corr_win) + 1e-10);

    % 统计指标
    center_idx_win = ceil(length(auto_corr_win)/2);
    PSLR_windows(i) = compute_pslr_corrected(auto_corr_win, center_idx_win, fs, B);
    ISLR_windows(i) = compute_islr_corrected(auto_corr_win, center_idx_win, fs, B);
end

% 传统窗自相关对数对比
figure(11);
plot(t_corr, auto_corr_windows_db(:,1), 'k-', 'LineWidth', 1.3); hold on;
plot(t_corr, auto_corr_windows_db(:,2), 'b--', 'LineWidth', 1.3);
plot(t_corr, auto_corr_windows_db(:,3), 'r-.', 'LineWidth', 1.3);
plot(t_corr, auto_corr_windows_db(:,4), 'm:', 'LineWidth', 1.8);
xlabel('Time Delay (μs)'); ylabel('Amplitude (dB)');
legend(window_names, 'Location', 'best');
grid on; ylim([-120, 0]);
title('Traditional Window Comparison (Autocorrelation, dB)');

% 传统窗指标对比
figure(12);
plot(1:num_windows, PSLR_windows, 'o-b', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
plot(1:num_windows, ISLR_windows, 's-r', 'LineWidth', 1.5, 'MarkerSize', 7);
xticks(1:num_windows);
xticklabels(window_names);
xlabel('Traditional window type');
ylabel('Metric (dB)');
title('Traditional Window Metric Comparison');
legend('PSLR', 'ISLR', 'Location', 'best');
grid on;

fprintf('\n=== Traditional Window Comparison ===\n');
for i = 1:num_windows
    fprintf('%s -> PSLR: %.3f dB, ISLR: %.3f dB\n', window_names{i}, PSLR_windows(i), ISLR_windows(i));
end

%% 辅助函数
function pslr = compute_pslr_corrected(corr_signal, peak_idx, fs, B)
    mainlobe_width_samples = ceil(2 * fs / B);
    mainlobe_start = max(1, peak_idx - mainlobe_width_samples);
    mainlobe_end = min(length(corr_signal), peak_idx + mainlobe_width_samples);

    mask = true(length(corr_signal), 1);
    mask(mainlobe_start:mainlobe_end) = false;

    if sum(mask) == 0
        pslr = -5;
        return;
    end

    sidelobe_peak = max(abs(corr_signal(mask)));
    mainlobe_peak = abs(corr_signal(peak_idx));
    if mainlobe_peak > 0
        pslr = 20*log10(sidelobe_peak / mainlobe_peak);
    else
        pslr = -inf;
    end
end

function islr = compute_islr_corrected(corr_signal, peak_idx, fs, B)
    mainlobe_width_samples = ceil(2 * fs / B);
    mainlobe_start = max(1, peak_idx - mainlobe_width_samples);
    mainlobe_end = min(length(corr_signal), peak_idx + mainlobe_width_samples);

    mainlobe_energy = sum(abs(corr_signal(mainlobe_start:mainlobe_end)).^2);
    mask = true(length(corr_signal), 1);
    mask(mainlobe_start:mainlobe_end) = false;
    sidelobe_energy = sum(abs(corr_signal(mask)).^2);

    if mainlobe_energy > 0
        islr = 10*log10(sidelobe_energy / mainlobe_energy + eps);
    else
        islr = inf;
    end
end
