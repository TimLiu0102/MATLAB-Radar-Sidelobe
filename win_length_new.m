clc; clear; close all;

%% 参数设置
B = 20e6;               % 基准带宽: 20 MHz
T_pulse = 50e-6;        % 脉宽: 50 μs
fs = 60e6;              % 采样率: 60 MHz

N_pulse = round(T_pulse * fs);
N_fft = 2^nextpow2(N_pulse * 8);
freq = (-N_fft/2:N_fft/2-1)' * (fs/N_fft);

alpha_reg = 1e-2;       % 正则化因子
A_max = 4.0;            % 最大增益限制

%% 生成系统频率响应 H[k]
f_center_idx = find(abs(freq) < B/2);

H_mag = zeros(N_fft, 1);
H_mag(f_center_idx) = 0.7 + 0.4*cos(2*pi*freq(f_center_idx)/B*6) .* ...
                      exp(-(freq(f_center_idx)/(0.5*B)).^2) + ...
                      0.1*sin(2*pi*freq(f_center_idx)/B*3);

H_phase = zeros(N_fft, 1);
H_phase(f_center_idx) = 0.4*pi*(freq(f_center_idx)/B).^3 + ...
                        0.3*pi*sin(2*pi*freq(f_center_idx)/B*5) + ...
                        0.05*pi*(freq(f_center_idx)/B).^2;

H_k = fftshift(H_mag .* exp(1j*H_phase));

%% 理想LFM信号
k_chirp = B / T_pulse;
t = (-N_pulse/2:N_pulse/2-1)' / fs;
s_lfm = exp(1j * pi * k_chirp * t.^2);

s_lfm_padded = zeros(N_fft, 1);
s_lfm_padded(1:N_pulse) = s_lfm;
S_LFM_k = fft(s_lfm_padded, N_fft);

%% 预补偿谱
R_k = S_LFM_k;
epsilon = alpha_reg * mean(abs(H_k .* S_LFM_k));
G_tx_k = R_k ./ (H_k .* S_LFM_k + epsilon);

G_tx_mag = abs(G_tx_k);
if max(G_tx_mag) > A_max
    G_tx_k = G_tx_k ./ max(1, G_tx_mag/A_max);
end

%% 窗函数带宽对比：B, 1.5B, 2B, 2.5B
bandwidth_factors = [1.0, 1.5, 2.0, 2.5];
num_cases = numel(bandwidth_factors);
colors = lines(num_cases);

% 结果缓存
labels = strings(num_cases, 1);
auto_corr_db_all = zeros(2*N_pulse-1, num_cases);
auto_corr_all = zeros(2*N_pulse-1, num_cases);
S_mag_all = zeros(N_fft, num_cases);
S_phase_all = zeros(N_fft, num_cases);

pslr_all = zeros(num_cases, 1);
bw3db_all = zeros(num_cases, 1);
papr_all = zeros(num_cases, 1);

t_corr = (-N_pulse+1:N_pulse-1) / fs * 1e6;
center_idx = ceil(length(t_corr)/2);
range_idx = center_idx-50:center_idx+50;

for i = 1:num_cases
    bw_factor = bandwidth_factors(i);
    window_bw = bw_factor * B;
    labels(i) = sprintf('Window Width: %.1fB', bw_factor);

    % 频域海明窗
    f_idx = find(abs(freq) <= window_bw/2);
    hamming_window = zeros(N_fft, 1);
    hamming_window(f_idx) = hamming(length(f_idx));
    W_k = fftshift(hamming_window);

    % 发射信号（预补偿+窗函数）
    S_tx_k = S_LFM_k .* G_tx_k .* W_k;
    s_tx_time = ifft(S_tx_k, N_fft);

    % 通过系统响应
    S_tx_with_H = fft(s_tx_time, N_fft) .* H_k;
    s_tx_with_H_time = ifft(S_tx_with_H, N_fft);
    s_tx_with_H_pulse = s_tx_with_H_time(1:N_pulse);
    s_tx_with_H_pulse = s_tx_with_H_pulse / sqrt(sum(abs(s_tx_with_H_pulse).^2));

    % 自相关
    auto_corr = xcorr(s_tx_with_H_pulse, s_tx_with_H_pulse);
    auto_corr = auto_corr / max(abs(auto_corr));
    auto_corr_db = 20*log10(abs(auto_corr) + 1e-10);

    auto_corr_all(:, i) = auto_corr;
    auto_corr_db_all(:, i) = auto_corr_db;

    % 频谱
    s_pad = zeros(N_fft, 1);
    s_pad(1:N_pulse) = s_tx_with_H_pulse;
    S_plot = fftshift(fft(s_pad, N_fft));
    S_mag_all(:, i) = 20*log10(abs(S_plot) / max(abs(S_plot)) + 1e-10);
    S_phase_all(:, i) = unwrap(angle(S_plot));

    % 指标
    pslr_all(i) = compute_pslr_corrected(auto_corr, ceil(length(auto_corr)/2), fs, B);
    bw3db_all(i) = compute_3db_width_corrected(auto_corr, ceil(length(auto_corr)/2), fs, B);
    papr_all(i) = 10*log10(max(abs(s_tx_with_H_pulse).^2) / mean(abs(s_tx_with_H_pulse).^2));
end

%% 图1：全延迟自相关（dB）
figure(1);
hold on;
for i = 1:num_cases
    plot(t_corr, auto_corr_db_all(:, i), 'LineWidth', 1.5, 'Color', colors(i, :));
end
xlabel('Time Delay (μs)'); ylabel('Amplitude (dB)');
legend(labels, 'Location', 'best');
grid on; ylim([-120, 0]);

%% 图2：主瓣区域（线性）
figure(2);
hold on;
for i = 1:num_cases
    plot(t_corr(range_idx), abs(auto_corr_all(range_idx, i)), 'LineWidth', 1.5, 'Color', colors(i, :));
end
xlabel('Time Delay (μs)'); ylabel('Normalized Amplitude');
legend(labels, 'Location', 'best');
grid on;

%% 图3：幅度频谱
figure(3);
hold on;
for i = 1:num_cases
    plot(freq/1e6, S_mag_all(:, i), 'LineWidth', 1.5, 'Color', colors(i, :));
end
xlabel('Frequency (MHz)'); ylabel('Amplitude (dB)');
legend(labels, 'Location', 'best');
grid on;

%% 图4：相位频谱
figure(4);
hold on;
for i = 1:num_cases
    plot(freq/1e6, S_phase_all(:, i), 'LineWidth', 1.5, 'Color', colors(i, :));
end
xlabel('Frequency (MHz)'); ylabel('Phase (rad)');
legend(labels, 'Location', 'best');
grid on;

%% 输出指标
fprintf('================ 窗函数带宽对比 ================\n');
for i = 1:num_cases
    fprintf('%.1fB -> PSLR: %.3f dB, 3dB主瓣宽度: %.3f μs, PAPR: %.3f dB\n', ...
        bandwidth_factors(i), pslr_all(i), bw3db_all(i), papr_all(i));
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
    bw_3db = width_samples / fs * 1e6;

    if bw_3db <= 0 || bw_3db > 100 || bw_3db < 0.01
        bw_3db = 0.886 / B * 1e6;
    end
end
