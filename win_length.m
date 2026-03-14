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
auto_corr_abs_all = zeros(length(t_corr), num_cases);
s_tx_case_all = zeros(N_pulse, num_cases);

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

    % 保存不同窗宽下的时域信号（用于figure(4)按补零-FFT方式计算频谱）
    s_tx_case_all(:, i) = s_tx_time(1:N_pulse);

    % 截取并归一化
    s_case = s_tx_with_H_time(1:N_pulse);
    s_case = s_case / sqrt(sum(abs(s_case).^2) + eps);

    % 自相关
    auto_corr = xcorr(s_case, s_case);
    auto_corr = auto_corr / (max(abs(auto_corr)) + eps);
    auto_corr_abs_all(:, i) = abs(auto_corr);
    auto_corr_db = 20*log10(abs(auto_corr) + 1e-10);
    auto_corr_db_all(:, i) = auto_corr_db;

    center_idx = ceil(length(auto_corr)/2);

    % 指标
    PSLR_vec(i) = compute_pslr_corrected(auto_corr, center_idx, fs, B);
    ISLR_vec(i) = compute_islr_corrected(auto_corr, center_idx, fs, B);
    MLW_vec(i)  = compute_3db_width_corrected(auto_corr, center_idx, fs, B);
    PAPR_vec(i) = compute_papr(s_case);
end

%% 步骤4: 输出结果
fprintf('\n=== Window Width Sensitivity Analysis ===\n');
for i = 1:num_cases
    fprintf('%.1fB -> PSLR: %.3f dB, ISLR: %.3f dB, PAPR: %.3f dB, Main-lobe width: %.3f us\n', ...
        width_factors(i), PSLR_vec(i), ISLR_vec(i), PAPR_vec(i), MLW_vec(i));
end

%% 步骤4.1: 正则化因子灵敏度分析（restoration error）
alpha_reg_candidates = [1e-4, 1e-3, 1e-2, 1e-1];
num_alpha = numel(alpha_reg_candidates);
restoration_error_vec = zeros(num_alpha, 1);
time_error_vec = zeros(num_alpha, 1);

for i = 1:num_alpha
    alpha_i = alpha_reg_candidates(i);

    % 预补偿（不加窗）
    epsilon_i = alpha_i * mean(abs(H_k .* S_LFM_k));
    G_tx_i = R_k ./ (H_k .* S_LFM_k + epsilon_i);

    % 幅度限制
    G_tx_i_mag = abs(G_tx_i);
    if max(G_tx_i_mag) > A_max
        G_tx_i = G_tx_i ./ max(1, G_tx_i_mag/A_max);
    end

    % 发射频谱 + 系统响应后频谱
    S_tx_i = S_LFM_k .* G_tx_i;
    S_rx_i = S_tx_i .* H_k;

    % 转到时域后计算参考误差（含 time_nmse / spec_nmse）
    s_cmp_i = ifft(S_rx_i, N_fft);
    s_cmp_i = s_cmp_i(1:N_pulse);
    err_i = evaluate_reference_error(s_cmp_i, S_LFM_k, N_pulse, N_fft, freq, B);

    % restoration error 使用 spec_nmse
    restoration_error_vec(i) = err_i.spec_nmse;
    time_error_vec(i) = err_i.time_nmse;
end

fprintf('\n=== Regularization Factor Sensitivity Analysis (Restoration Error) ===\n');
for i = 1:num_alpha
    fprintf('alpha = %.4g -> restoration error(spec_nmse): %.6f, time_nmse: %.6f\n', ...
        alpha_reg_candidates(i), restoration_error_vec(i), time_error_vec(i));
end

%% 步骤5: 绘图
legend_labels = arrayfun(@(wf) sprintf('%.1fB', wf), width_factors, 'UniformOutput', false);

% 自相关对数图（可选）
figure(1);
for i = 1:num_cases
    plot(t_corr, auto_corr_db_all(:, i), 'LineWidth', 1.2); hold on;
end
xlabel('Time Delay (\mus)');
ylabel('Amplitude (dB)');
legend(legend_labels, 'Location', 'best');
grid on;
ylim([-120, 0]);

% figure(4): 不同窗宽下信号频谱对比（参考原 figure(3) 画法）
figure(4);
line_styles = {'k-', 'r--', 'b-.', 'm:'};
for i = 1:num_cases
    s_case_padded = zeros(N_fft, 1);
    s_case_padded(1:N_pulse) = s_tx_case_all(:, i);
    S_case_mag = fftshift(abs(fft(s_case_padded, N_fft)));
    plot(freq/1e6, 20*log10(S_case_mag / max(S_case_mag)), line_styles{i}, 'LineWidth', 1.5); hold on;
end
xlabel('Frequency (MHz)'); ylabel('Amplitude (dB)');
legend(legend_labels{:}, 'Location', 'best');
grid on;

% figure(8) 代码样式：自相关主瓣对比
center_idx_plot = ceil(length(t_corr)/2);
range_idx = center_idx_plot-50:center_idx_plot+50;
figure(5);
plot(t_corr(range_idx), auto_corr_abs_all(range_idx,1), 'k-', 'LineWidth', 1.5); hold on;
plot(t_corr(range_idx), auto_corr_abs_all(range_idx,2), 'r--', 'LineWidth', 1.5);
plot(t_corr(range_idx), auto_corr_abs_all(range_idx,3), 'b-.', 'LineWidth', 1.5);
plot(t_corr(range_idx), auto_corr_abs_all(range_idx,4), 'm:', 'LineWidth', 1.5);
xlabel('Time Delay(\mus)'); ylabel('Normalized Amplitude');
legend(legend_labels{:}, 'Location', 'best');
grid on;

% 正则化因子灵敏度图（restoration error）
figure(3);
semilogx(alpha_reg_candidates, restoration_error_vec, 'o-b', 'LineWidth', 1.5, 'MarkerSize', 7);
xlabel('regularization factor \alpha');
ylabel('restoration error (normalized in-band spectrum error)');
title('Sensitivity Analysis vs Regularization Factor');
grid on;

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
title('Sensitivity Analysis vs Window Width');
legend('PSLR', 'ISLR', 'PAPR', 'main-lobe width', 'Location', 'best');
grid on;

%% 修正后的辅助函数
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
    % 修正的积分旁瓣比计算函数
    % 主瓣宽度 ≈ 1/B，转换为采样点数
    mainlobe_width_samples = ceil(2 * fs / B);  % 取2倍作为安全余量
    
    % 确保边界有效
    mainlobe_start = max(1, peak_idx - mainlobe_width_samples);
    mainlobe_end = min(length(corr_signal), peak_idx + mainlobe_width_samples);
    
    % 主瓣能量
    mainlobe_energy = sum(abs(corr_signal(mainlobe_start:mainlobe_end)).^2);
    
    % 总能量
    total_energy = sum(abs(corr_signal).^2);
    
    % 旁瓣能量 = 总能量 - 主瓣能量
    sidelobe_energy = total_energy - mainlobe_energy;
    
    % 计算ISLR（dB）
    if mainlobe_energy > 0 && sidelobe_energy > 0
        islr = 10*log10(sidelobe_energy / mainlobe_energy);
    else
        islr = -inf;
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

function papr_db = compute_papr(x)
p = abs(x).^2;
papr_db = 10 * log10(max(p) / mean(p));
end

function E_spec = compute_spectrum_error(S_ideal_k, S_out_k, freq, B)
    % 方案A：归一化幅度均方误差（稳健）
    band_idx = abs(freq)<=B/2; % 在有效带宽内评价
    A = abs(S_ideal_k(band_idx));
    Bv = abs(S_out_k(band_idx));
    % 先在带内做L2归一化，抑制“仅幅度尺度不同”导致的失真误判
    A = A / (norm(A,2) + eps);
    Bv = Bv / (norm(Bv,2) + eps);
    E_spec = (norm(Bv - A, 2)^2) / (norm(A, 2)^2 + eps);
end


function err = evaluate_reference_error(s_cmp, S_ideal_k, N_pulse, N_fft, freq, B)
    s_ideal = ifft(S_ideal_k, N_fft);
    s_ideal = s_ideal(1:N_pulse);
    s_ideal = s_ideal / sqrt(sum(abs(s_ideal).^2) + eps);
    s_cmp = s_cmp / sqrt(sum(abs(s_cmp).^2) + eps);

    err.time_nmse = norm(s_cmp - s_ideal, 2)^2 / (norm(s_ideal,2)^2 + eps);

    s_cmp_pad = zeros(N_fft,1); s_cmp_pad(1:N_pulse) = s_cmp;
    S_cmp = fft(s_cmp_pad, N_fft);
    err.spec_nmse = compute_spectrum_error(S_ideal_k, S_cmp, freq, B);
end
