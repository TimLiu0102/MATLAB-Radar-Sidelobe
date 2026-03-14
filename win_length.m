clc; clear; close all;

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
alpha_reg = 1e-2;       % 正则化因子系数
A_max = 4.0;            % 最大增益限制

% 显示参数
f_center_idx = find(abs(freq) < B/2);

%% 步骤0.1: 生成系统频率响应 H[k]
H_mag = zeros(N_fft, 1);
H_mag(f_center_idx) = 0.7 + 0.4*cos(2*pi*freq(f_center_idx)/B*6) .* ...
                      exp(-(freq(f_center_idx)/(0.5*B)).^2) + ...
                      0.1*sin(2*pi*freq(f_center_idx)/B*3);

H_phase = zeros(N_fft, 1);
H_phase(f_center_idx) = 0.4*pi*(freq(f_center_idx)/B).^3 + ...
                        0.3*pi*sin(2*pi*freq(f_center_idx)/B*5) + ...
                        0.05*pi*(freq(f_center_idx)/B).^2;

H_k = H_mag .* exp(1j*H_phase);
H_k = fftshift(H_k);  % 转换为MATLAB的FFT顺序

%% 步骤1: 生成理想LFM信号及其频谱
% 生成时域LFM信号
t = (-N_pulse/2:N_pulse/2-1)' / fs;
k_chirp = B / T_pulse;
s_lfm = exp(1j * pi * k_chirp * t.^2);

% 过采样并补零到FFT长度
s_lfm_padded = zeros(N_fft, 1);
s_lfm_padded(1:N_pulse) = s_lfm;

% LFM频谱
S_LFM_k = fft(s_lfm_padded, N_fft);
R_k = S_LFM_k;  % 参考频谱

%% 步骤2: 计算预补偿谱
epsilon = alpha_reg * mean(abs(H_k .* S_LFM_k));
G_tx_k = R_k ./ (H_k .* S_LFM_k + epsilon);

% 幅度限制
G_tx_mag = abs(G_tx_k);
if max(G_tx_mag) > A_max
    G_tx_k = G_tx_k ./ max(1, G_tx_mag/A_max);
end

%% 步骤3: 窗宽灵敏度分析（1.0B, 1.5B, 2.0B, 2.5B）
width_factors = [1.0, 1.5, 2.0, 2.5];
num_cases = numel(width_factors);

% 结果向量
PSLR_vec = zeros(num_cases,1);
ISLR_vec = zeros(num_cases,1);
PAPR_vec = zeros(num_cases,1);
MLW_vec  = zeros(num_cases,1);   % 主瓣宽度(3dB), us

% 用于可视化
t_corr = (-N_pulse+1:N_pulse-1) / fs * 1e6;  % 微秒
auto_corr_db_all = zeros(length(t_corr), num_cases);

for i = 1:num_cases
    wf = width_factors(i);
    BW_win = wf * B;

    % 频域海明窗（窗宽 = wf*B）
    f_idx = find(abs(freq) <= BW_win/2);
    hamming_window = zeros(N_fft, 1);
    hamming_window(f_idx) = hamming(length(f_idx));
    W_k = fftshift(hamming_window);

    % 应用预补偿与窗
    G_tx_k_windowed = G_tx_k .* W_k;
    S_tx_k = S_LFM_k .* G_tx_k_windowed;

    % IFFT 到时域
    s_tx_time = ifft(S_tx_k, N_fft);

    % 通过系统H
    S_tx_full = fft(s_tx_time, N_fft);
    S_tx_with_H = S_tx_full .* H_k;
    s_tx_with_H_time = ifft(S_tx_with_H, N_fft);

    % 截取并归一化
    s_case = s_tx_with_H_time(1:N_pulse);
    s_case = s_case / sqrt(sum(abs(s_case).^2) + eps);

    % 自相关
    auto_corr = xcorr(s_case, s_case);
    auto_corr = auto_corr / (max(abs(auto_corr)) + eps);
    auto_corr_db = 20*log10(abs(auto_corr) + 1e-10);
    auto_corr_db_all(:, i) = auto_corr_db;

    center_idx = ceil(length(auto_corr)/2);

    % 指标
    PSLR_vec(i) = compute_pslr_corrected(auto_corr, center_idx, fs, B);
    ISLR_vec(i) = compute_islr_corrected(auto_corr, center_idx, fs, B);
    MLW_vec(i)  = compute_3db_width_corrected(auto_corr, center_idx, fs, B);
    PAPR_vec(i) = 10*log10(max(abs(s_case).^2) / (mean(abs(s_case).^2) + eps));
end

%% 步骤4: 输出结果
fprintf('\n=== Window Width Sensitivity Analysis ===\n');
for i = 1:num_cases
    fprintf('%.1fB -> PSLR: %.3f dB, ISLR: %.3f dB, PAPR: %.3f dB, Main-lobe width: %.3f us\n', ...
        width_factors(i), PSLR_vec(i), ISLR_vec(i), PAPR_vec(i), MLW_vec(i));
end

%% 步骤5: 绘图
% 自相关对数图（可选）
figure(1);
for i = 1:num_cases
    plot(t_corr, auto_corr_db_all(:, i), 'LineWidth', 1.2); hold on;
end
xlabel('Time Delay (\mus)');
ylabel('Amplitude (dB)');
legend('Window Width:1.0B', 'Window Width:1.5B', 'Window Width:2.0B', 'Window Width:2.5B', 'Location', 'best');
grid on;
ylim([-120, 0]);

% 你要求的灵敏度分析双纵轴图
figure(2);
yyaxis left;
plot(width_factors, PSLR_vec, 'o-b', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
plot(width_factors, ISLR_vec, 's-r', 'LineWidth', 1.5, 'MarkerSize', 7);
ylabel('PSLR / ISLR (dB)');

yyaxis right;
plot(width_factors, PAPR_vec, 'd-k', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
plot(width_factors, MLW_vec, '^-m', 'LineWidth', 1.5, 'MarkerSize', 7);
ylabel('PAPR (dB) / main-lobe width (\mus)');

xlabel('window width');
legend('PSLR', 'ISLR', 'PAPR', 'main-lobe width', 'Location', 'best');
grid on;

%% 辅助函数
function pslr = compute_pslr_corrected(corr_signal, peak_idx, fs, B)
    % 主瓣宽度估计（样点）
    mainlobe_width_samples = ceil(2 * fs / B);

    % 主瓣区间
    mainlobe_start = max(1, peak_idx - mainlobe_width_samples);
    mainlobe_end = min(length(corr_signal), peak_idx + mainlobe_width_samples);

    % 排除主瓣
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
    % 主瓣宽度估计（样点）
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

function bw_3db = compute_3db_width_corrected(corr_signal, peak_idx, fs, B)
    corr_mag = abs(corr_signal);
    peak_value = corr_mag(peak_idx);

    threshold_3db = peak_value / sqrt(2);

    left_idx = peak_idx;
    while left_idx > 1 && corr_mag(left_idx) >= threshold_3db
        left_idx = left_idx - 1;
    end

    right_idx = peak_idx;
    while right_idx < length(corr_mag) && corr_mag(right_idx) >= threshold_3db
        right_idx = right_idx + 1;
    end

    width_samples = right_idx - left_idx;
    bw_3db = width_samples / fs * 1e6;  % us

    if bw_3db <= 0 || bw_3db > 100
        bw_3db = 0.886 / B * 1e6;
    end

    if bw_3db < 0.01
        bw_3db = 0.886 / B * 1e6;
    end
end
