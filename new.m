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

%% 步骤3: 频域窗函数设计（Baseline + NEW）
f_idx = find(abs(freq)<=B);
L = length(f_idx);
min_width_ratio = 0.6;  % NEW: 优化窗宽下限（相对B带宽）

% Baseline: 原有海明窗
hamming_window = zeros(N_fft, 1);
hamming_win_local = hamming(L);
hamming_window(f_idx) = hamming_win_local;
W_hamming_k = fftshift(hamming_window);

% 先计算无补偿和理想信号（用于基准指标）
s_ideal = s_lfm / sqrt(sum(abs(s_lfm).^2));
S_with_H = S_LFM_k .* H_k;
s_with_H_time = ifft(S_with_H, N_fft);
s_with_H = s_with_H_time(1:N_pulse);
s_with_H = s_with_H / sqrt(sum(abs(s_with_H).^2));

% Baseline链路: 预补偿 + Hamming窗
S_tx_hamming_k = S_LFM_k .* (G_tx_k .* W_hamming_k);
s_tx_hamming_time = ifft(S_tx_hamming_k, N_fft);
s_tx_hamming_pulse = s_tx_hamming_time(1:N_pulse);
s_tx_hamming_pulse = s_tx_hamming_pulse / sqrt(sum(abs(s_tx_hamming_pulse).^2));
S_tx_hamming_full = fft(s_tx_hamming_time, N_fft);
S_tx_hamming_with_H = S_tx_hamming_full .* H_k;
s_tx_hamming_with_H_time = ifft(S_tx_hamming_with_H, N_fft);
s_tx_hamming_with_H = s_tx_hamming_with_H_time(1:N_pulse);
s_tx_hamming_with_H = s_tx_hamming_with_H / sqrt(sum(abs(s_tx_hamming_with_H).^2));

% Baseline指标
auto_corr_hamming = xcorr(s_tx_hamming_with_H, s_tx_hamming_with_H);
auto_corr_hamming = auto_corr_hamming / max(abs(auto_corr_hamming));
PSLR_hamming_linear = compute_pslr(auto_corr_hamming);
PSLR_hamming = 20*log10(PSLR_hamming_linear + 1e-12);
mainlobe_width_hamming = compute_mainlobe_width(auto_corr_hamming);
PAPR_hamming = compute_papr(s_tx_hamming_with_H);

%% NEW: 优化广义余弦窗设计
%% NEW: 萤火虫算法优化
rng(1);
lambda1 = 10;
lambda2 = 1;
fa_opt.pop_size = 20;
fa_opt.max_iter = 30;
fa_opt.beta0 = 1;
fa_opt.gamma = 1;
fa_opt.alpha = 0.2;

best_param = optimize_generalized_cosine_fa(...
    G_tx_k, S_LFM_k, H_k, N_fft, N_pulse, f_idx, ...
    mainlobe_width_hamming, PAPR_hamming, ...
    lambda1, lambda2, min_width_ratio, fa_opt);

best_coeff = best_param(1:3);
best_width_ratio = best_param(4);
opt_gc_local = build_generalized_cosine_window(L, best_coeff, best_width_ratio);
opt_gc_window = zeros(N_fft, 1);
opt_gc_window(f_idx) = opt_gc_local;
W_opt_k = fftshift(opt_gc_window);

% 优化窗链路
S_tx_opt_k = S_LFM_k .* (G_tx_k .* W_opt_k);
s_tx_opt_time = ifft(S_tx_opt_k, N_fft);
s_tx_opt_pulse = s_tx_opt_time(1:N_pulse);
s_tx_opt_pulse = s_tx_opt_pulse / sqrt(sum(abs(s_tx_opt_pulse).^2));
S_tx_opt_full = fft(s_tx_opt_time, N_fft);
S_tx_opt_with_H = S_tx_opt_full .* H_k;
s_tx_opt_with_H_time = ifft(S_tx_opt_with_H, N_fft);
s_tx_opt_with_H = s_tx_opt_with_H_time(1:N_pulse);
s_tx_opt_with_H = s_tx_opt_with_H / sqrt(sum(abs(s_tx_opt_with_H).^2));

%% 步骤4: 自相关与指标计算
% 理想、失真、Hamming、优化窗
auto_corr_ideal = xcorr(s_ideal, s_ideal);
auto_corr_no_comp = xcorr(s_with_H, s_with_H);
auto_corr_opt = xcorr(s_tx_opt_with_H, s_tx_opt_with_H);

auto_corr_ideal = auto_corr_ideal / max(abs(auto_corr_ideal));
auto_corr_no_comp = auto_corr_no_comp / max(abs(auto_corr_no_comp));
auto_corr_opt = auto_corr_opt / max(abs(auto_corr_opt));

auto_corr_ideal_db = 20*log10(abs(auto_corr_ideal) + 1e-10);
auto_corr_no_comp_db = 20*log10(abs(auto_corr_no_comp) + 1e-10);
auto_corr_hamming_db = 20*log10(abs(auto_corr_hamming) + 1e-10);
auto_corr_opt_db = 20*log10(abs(auto_corr_opt) + 1e-10);

t_corr = (-N_pulse+1:N_pulse-1) / fs * 1e6;

PSLR_ideal = 20*log10(compute_pslr(auto_corr_ideal) + 1e-12);
PSLR_no_comp = 20*log10(compute_pslr(auto_corr_no_comp) + 1e-12);
PSLR_opt = 20*log10(compute_pslr(auto_corr_opt) + 1e-12);

mainlobe_width_ideal = compute_mainlobe_width(auto_corr_ideal);
mainlobe_width_no_comp = compute_mainlobe_width(auto_corr_no_comp);
mainlobe_width_opt = compute_mainlobe_width(auto_corr_opt);

PAPR_ideal = compute_papr(s_ideal);
PAPR_no_comp = compute_papr(s_with_H);
PAPR_opt = compute_papr(s_tx_opt_with_H);

ISLR_ideal = 10*log10(compute_islr(auto_corr_ideal) + 1e-12);
ISLR_no_comp = 10*log10(compute_islr(auto_corr_no_comp) + 1e-12);
ISLR_hamming = 10*log10(compute_islr(auto_corr_hamming) + 1e-12);
ISLR_opt = 10*log10(compute_islr(auto_corr_opt) + 1e-12);

%% 步骤5: 关键图输出（仅保留关键对比图）
% 频谱准备
s_ideal_padded = zeros(N_fft, 1); s_ideal_padded(1:N_pulse) = s_ideal;
s_with_H_padded = zeros(N_fft, 1); s_with_H_padded(1:N_pulse) = s_with_H;
s_hamming_padded = zeros(N_fft, 1); s_hamming_padded(1:N_pulse) = s_tx_hamming_with_H;
s_opt_padded = zeros(N_fft, 1); s_opt_padded(1:N_pulse) = s_tx_opt_with_H;

S_ideal_mag = fftshift(abs(fft(s_ideal_padded, N_fft)));
S_no_comp_mag = fftshift(abs(fft(s_with_H_padded, N_fft)));
S_hamming_mag = fftshift(abs(fft(s_hamming_padded, N_fft)));
S_opt_mag = fftshift(abs(fft(s_opt_padded, N_fft)));

% 图1：窗形对比
figure(1);
plot(0:L-1, hamming_win_local, 'b-', 'LineWidth', 1.5); hold on;
plot(0:L-1, opt_gc_local, 'r--', 'LineWidth', 1.5);
xlabel('Window Sample Index'); ylabel('Amplitude');
legend('Hamming', 'Optimized generalized cosine', 'Location', 'best');
grid on;

% 图2：自相关对比图
figure(2);
plot(t_corr, auto_corr_ideal_db, 'k-', 'LineWidth', 1.5); hold on;
plot(t_corr, auto_corr_no_comp_db, 'r--', 'LineWidth', 1.5);
plot(t_corr, auto_corr_hamming_db, 'b-.', 'LineWidth', 1.5);
plot(t_corr, auto_corr_opt_db, 'm-', 'LineWidth', 1.5);
xlabel('Time Delay (\mus)'); ylabel('Autocorrelation (dB)');
legend('Ideal LFM', 'Distorted output', 'Hamming window', 'Optimized generalized cosine', 'Location', 'best');
ylim([-120, 0]); grid on;

% 图3：频谱幅度对比图
figure(3);
plot(freq/1e6, 20*log10(S_ideal_mag/max(S_ideal_mag) + 1e-12), 'k-', 'LineWidth', 1.5); hold on;
plot(freq/1e6, 20*log10(S_no_comp_mag/max(S_no_comp_mag) + 1e-12), 'r--', 'LineWidth', 1.5);
plot(freq/1e6, 20*log10(S_hamming_mag/max(S_hamming_mag) + 1e-12), 'b-.', 'LineWidth', 1.5);
plot(freq/1e6, 20*log10(S_opt_mag/max(S_opt_mag) + 1e-12), 'm-', 'LineWidth', 1.5);
xlabel('Frequency (MHz)'); ylabel('Magnitude (dB)');
legend('Ideal LFM', 'Distorted output', 'Hamming window', 'Optimized generalized cosine', 'Location', 'best');
xlim([-B/1e6*1.5, B/1e6*1.5]); ylim([-100, 5]); grid on;

%% 步骤6: 命令行输出指标表
fprintf('\n%-35s %-12s %-12s %-22s %-12s\n', 'Method', 'PSLR (dB)', 'ISLR (dB)', '3-dB Mainlobe Width', 'PAPR (dB)');
fprintf('%s\n', repmat('-', 1, 102));
fprintf('%-35s %-12.3f %-12.3f %-22d %-12.3f\n', 'Ideal LFM', PSLR_ideal, ISLR_ideal, mainlobe_width_ideal, PAPR_ideal);
fprintf('%-35s %-12.3f %-12.3f %-22d %-12.3f\n', 'Distorted output', PSLR_no_comp, ISLR_no_comp, mainlobe_width_no_comp, PAPR_no_comp);
fprintf('%-35s %-12.3f %-12.3f %-22d %-12.3f\n', 'Hamming window', PSLR_hamming, ISLR_hamming, mainlobe_width_hamming, PAPR_hamming);
fprintf('%-35s %-12.3f %-12.3f %-22d %-12.3f\n', 'Optimized generalized cosine window', PSLR_opt, ISLR_opt, mainlobe_width_opt, PAPR_opt);
fprintf('Optimized [a0, a1, a2] = [%.4f, %.4f, %.4f], width ratio = %.4f\n', best_coeff(1), best_coeff(2), best_coeff(3), best_width_ratio);

%% ===== local functions =====
function best_param = optimize_generalized_cosine_fa(G_tx_k, S_LFM_k, H_k, N_fft, N_pulse, f_idx, mlw_ham, papr_ham, lambda1, lambda2, min_width_ratio, fa_opt)
pop_size = fa_opt.pop_size;
max_iter = fa_opt.max_iter;
beta0 = fa_opt.beta0;
gamma = fa_opt.gamma;
alpha = fa_opt.alpha;

pop = rand(pop_size, 4);
pop(:,4) = min_width_ratio + (1-min_width_ratio)*pop(:,4);
for i = 1:pop_size
    pop(i, :) = project_coeffs(pop(i, :), min_width_ratio);
end

J = zeros(pop_size, 1);
for i = 1:pop_size
    J(i) = evaluate_window_objective(pop(i, :), G_tx_k, S_LFM_k, H_k, N_fft, N_pulse, f_idx, mlw_ham, papr_ham, lambda1, lambda2);
end

for iter = 1:max_iter
    for i = 1:pop_size
        for j = 1:pop_size
            if J(j) < J(i)
                r2 = sum((pop(i, :) - pop(j, :)).^2);
                beta = beta0 * exp(-gamma * r2);
                step = beta * (pop(j, :) - pop(i, :)) + alpha * ([rand(1, 3)-0.5, rand-0.5]);
                pop(i, :) = project_coeffs(pop(i, :) + step, min_width_ratio);
                J(i) = evaluate_window_objective(pop(i, :), G_tx_k, S_LFM_k, H_k, N_fft, N_pulse, f_idx, mlw_ham, papr_ham, lambda1, lambda2);
            end
        end
    end
end

[~, best_idx] = min(J);
best_param = pop(best_idx, :);
end

function J = evaluate_window_objective(param, G_tx_k, S_LFM_k, H_k, N_fft, N_pulse, f_idx, mlw_ham, papr_ham, lambda1, lambda2)
L = length(f_idx);
coeff = param(1:3);
width_ratio = param(4);
win_local = build_generalized_cosine_window(L, coeff, width_ratio);
win = zeros(N_fft, 1);
win(f_idx) = win_local;
W_k = fftshift(win);

S_tx_k = S_LFM_k .* (G_tx_k .* W_k);
s_tx_time = ifft(S_tx_k, N_fft);
S_tx_full = fft(s_tx_time, N_fft);
S_tx_with_H = S_tx_full .* H_k;
s_out_time = ifft(S_tx_with_H, N_fft);
s_out = s_out_time(1:N_pulse);
s_out = s_out / sqrt(sum(abs(s_out).^2));

auto_corr = xcorr(s_out, s_out);
auto_corr = auto_corr / max(abs(auto_corr));

pslr_linear = compute_pslr(auto_corr);
mainlobe_width = compute_mainlobe_width(auto_corr);
papr_db = compute_papr(s_out);
islr_linear = compute_islr(auto_corr);

J = pslr_linear + islr_linear ...
    + lambda1 * max(0, mainlobe_width - mlw_ham)^2 ...
    + lambda2 * max(0, papr_db - papr_ham)^2;
end

function w = build_generalized_cosine_window(L, coeff, width_ratio)
a0 = coeff(1); a1 = coeff(2); a2 = coeff(3);
Lw = max(3, min(L, round(width_ratio * L)));

w = zeros(L, 1);
idx0 = floor((L - Lw)/2) + 1;
idx1 = idx0 + Lw - 1;
n = (0:Lw-1)';
if Lw == 1
    w_local = 1;
else
    w_local = a0 - a1*cos(2*pi*n/(Lw-1)) + a2*cos(4*pi*n/(Lw-1));
end
w_local(w_local < 0) = 0;
if max(w_local) > 0
    w_local = w_local / max(w_local);
end
w(idx0:idx1) = w_local;
end

function param = project_coeffs(param, min_width_ratio)
coeff = param(1:3);
width_ratio = param(4);

coeff = max(0, min(1, coeff));
s = sum(coeff);
if s <= eps
    coeff = [1, 0, 0];
else
    coeff = coeff / s;
end

width_ratio = max(min_width_ratio, min(1.0, width_ratio));
param = [coeff, width_ratio];
end

function pslr_linear = compute_pslr(auto_corr_norm)
% draw_figure.m口径：基于归一化自相关，取中心主峰与其余点最大旁瓣比
% auto_corr_norm 已在主流程按 max(abs(.)) 归一化

a = abs(auto_corr_norm(:));
center_idx = ceil(length(a)/2);
peak_val = a(center_idx);

a_no_peak = a;
a_no_peak(center_idx) = 0;
max_sidelobe = max(a_no_peak);

pslr_linear = max_sidelobe / max(peak_val, eps);
end

function width = compute_mainlobe_width(auto_corr_norm)
% draw_figure.m口径：以中心主峰为参考，统计3-dB主瓣连续宽度

a = abs(auto_corr_norm(:));
center_idx = ceil(length(a)/2);
peak_val = a(center_idx);
th = peak_val / sqrt(2);

left = center_idx;
while left > 1 && a(left-1) >= th
    left = left - 1;
end
right = center_idx;
while right < length(a) && a(right+1) >= th
    right = right + 1;
end

width = right - left + 1;
end

function islr_linear = compute_islr(auto_corr_norm)
a = abs(auto_corr_norm(:));
center_idx = ceil(length(a)/2);
peak_val = a(center_idx);
th = peak_val / sqrt(2);

left = center_idx;
while left > 1 && a(left-1) >= th
    left = left - 1;
end
right = center_idx;
while right < length(a) && a(right+1) >= th
    right = right + 1;
end

main_energy = sum(a(left:right).^2);
a(left:right) = 0;
side_energy = sum(a.^2);

islr_linear = side_energy / max(main_energy, eps);
end

function papr_db = compute_papr(x)
p = abs(x).^2;
papr_db = 10 * log10(max(p) / mean(p));
end
