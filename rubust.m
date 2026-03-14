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

% 鲁棒优化参数（按给定思路：多场景均值 + 最坏场景项）
K_robust = 20;          % 扰动场景数（推荐10或20）
mu_robust = 0.3;        % 最坏场景权重
delta_a = 0.05;         % 幅度扰动上界（±5%）
delta_phi_deg = 5;      % 相位扰动上界（度）
delta_phi = delta_phi_deg*pi/180;

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

% 构造鲁棒优化场景集合 Ω={H^(1),...,H^(K)}
H_scenarios = build_perturbed_channels(H_k, f_center_idx, K_robust, delta_a, delta_phi);

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
min_width_B_multiple = 0.8;  % NEW: 优化窗宽下限（以B为单位）
max_width_B_multiple = 2.0;  % NEW: 优化窗宽上限（受当前频带截断）
f_local = freq(f_idx);

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
peak_idx_hamming = ceil(length(auto_corr_hamming)/2);
PSLR_hamming = compute_pslr_corrected(auto_corr_hamming, peak_idx_hamming, fs, B);
ISLR_hamming = compute_islr_corrected(auto_corr_hamming, peak_idx_hamming, fs, B);
mainlobe_width_hamming = compute_3db_width_corrected(auto_corr_hamming, peak_idx_hamming, fs, B);
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
fa_opt.verbose = true;

best_param = optimize_generalized_cosine_fa(...
    G_tx_k, S_LFM_k, H_scenarios, N_fft, N_pulse, f_idx, f_local, ...
    mainlobe_width_hamming, PAPR_hamming, ...
    lambda1, lambda2, mu_robust, min_width_B_multiple, max_width_B_multiple, fa_opt, fs, B);

best_coeff = best_param(1:3);
best_width_B_multiple = best_param(4);
opt_gc_local = build_generalized_cosine_window(L, f_local, best_coeff, best_width_B_multiple, B);
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

peak_idx_ideal = ceil(length(auto_corr_ideal)/2);
peak_idx_no_comp = ceil(length(auto_corr_no_comp)/2);
peak_idx_opt = ceil(length(auto_corr_opt)/2);

PSLR_ideal = compute_pslr_corrected(auto_corr_ideal, peak_idx_ideal, fs, B);
PSLR_no_comp = compute_pslr_corrected(auto_corr_no_comp, peak_idx_no_comp, fs, B);
PSLR_opt = compute_pslr_corrected(auto_corr_opt, peak_idx_opt, fs, B);

ISLR_ideal = compute_islr_corrected(auto_corr_ideal, peak_idx_ideal, fs, B);
ISLR_no_comp = compute_islr_corrected(auto_corr_no_comp, peak_idx_no_comp, fs, B);
ISLR_opt = compute_islr_corrected(auto_corr_opt, peak_idx_opt, fs, B);

mainlobe_width_ideal = compute_3db_width_corrected(auto_corr_ideal, peak_idx_ideal, fs, B);
mainlobe_width_no_comp = compute_3db_width_corrected(auto_corr_no_comp, peak_idx_no_comp, fs, B);
mainlobe_width_opt = compute_3db_width_corrected(auto_corr_opt, peak_idx_opt, fs, B);

PAPR_ideal = compute_papr(s_ideal);
PAPR_no_comp = compute_papr(s_with_H);
PAPR_opt = compute_papr(s_tx_opt_with_H);

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
fprintf('\n%-35s %-12s %-12s %-22s %-12s\n', 'Method', 'PSLR (dB)', 'ISLR (dB)', '3-dB Mainlobe Width (us)', 'PAPR (dB)');
fprintf('%s\n', repmat('-', 1, 102));
fprintf('%-35s %-12.3f %-12.3f %-22.4f %-12.3f\n', 'Ideal LFM', PSLR_ideal, ISLR_ideal, mainlobe_width_ideal, PAPR_ideal);
fprintf('%-35s %-12.3f %-12.3f %-22.4f %-12.3f\n', 'Distorted output', PSLR_no_comp, ISLR_no_comp, mainlobe_width_no_comp, PAPR_no_comp);
fprintf('%-35s %-12.3f %-12.3f %-22.4f %-12.3f\n', 'Hamming window', PSLR_hamming, ISLR_hamming, mainlobe_width_hamming, PAPR_hamming);
fprintf('%-35s %-12.3f %-12.3f %-22.4f %-12.3f\n', 'Optimized generalized cosine window', PSLR_opt, ISLR_opt, mainlobe_width_opt, PAPR_opt);
fprintf('Optimized [a0, a1, a2] = [%.4f, %.4f, %.4f], window width = %.4f * B\n', best_coeff(1), best_coeff(2), best_coeff(3), best_width_B_multiple);
fprintf('Robust settings: K=%d, mu=%.3f, delta_a=%.3f, delta_phi=%.2f deg\n', K_robust, mu_robust, delta_a, delta_phi_deg);

%% ===== local functions =====
function best_param = optimize_generalized_cosine_fa(G_tx_k, S_LFM_k, H_scenarios, N_fft, N_pulse, f_idx, f_local, mlw_ham, papr_ham, lambda1, lambda2, mu_robust, min_width_B_multiple, max_width_B_multiple, fa_opt, fs, B)
pop_size = fa_opt.pop_size;
max_iter = fa_opt.max_iter;
beta0 = fa_opt.beta0;
gamma = fa_opt.gamma;
alpha = fa_opt.alpha;
if isfield(fa_opt, 'verbose')
    verbose = fa_opt.verbose;
else
    verbose = true;
end

pop = rand(pop_size, 4);
pop(:,4) = min_width_B_multiple + (max_width_B_multiple-min_width_B_multiple)*pop(:,4);
for i = 1:pop_size
    pop(i, :) = project_coeffs(pop(i, :), min_width_B_multiple, max_width_B_multiple);
end

J = zeros(pop_size, 1);
for i = 1:pop_size
    J(i) = evaluate_window_objective(pop(i, :), G_tx_k, S_LFM_k, H_scenarios, N_fft, N_pulse, f_idx, f_local, mlw_ham, papr_ham, lambda1, lambda2, mu_robust, fs, B);
end

for iter = 1:max_iter
    for i = 1:pop_size
        for j = 1:pop_size
            if J(j) < J(i)
                r2 = sum((pop(i, :) - pop(j, :)).^2);
                beta = beta0 * exp(-gamma * r2);
                step = beta * (pop(j, :) - pop(i, :)) + alpha * ([rand(1, 3)-0.5, rand-0.5]);
                pop(i, :) = project_coeffs(pop(i, :) + step, min_width_B_multiple, max_width_B_multiple);
                J(i) = evaluate_window_objective(pop(i, :), G_tx_k, S_LFM_k, H_scenarios, N_fft, N_pulse, f_idx, f_local, mlw_ham, papr_ham, lambda1, lambda2, mu_robust, fs, B);
            end
        end
    end

    [best_J, best_idx_iter] = min(J);
    if verbose
        [~, metrics] = evaluate_window_objective(pop(best_idx_iter, :), G_tx_k, S_LFM_k, H_scenarios, N_fft, N_pulse, f_idx, f_local, mlw_ham, papr_ham, lambda1, lambda2, mu_robust, fs, B);
        fprintf('FA iter %02d/%02d | J=%.4f (mean %.4f + mu*max %.4f) | PSLR=%.3f dB | ISLR=%.3f dB | Mainlobe=%.4f us | PAPR=%.3f dB | Width=%.3f*B\n', ...
            iter, max_iter, best_J, metrics.mean_J, metrics.max_J, metrics.pslr_db, metrics.islr_db, metrics.mainlobe_width, metrics.papr_db, pop(best_idx_iter, 4));
    end
end

[~, best_idx] = min(J);
best_param = pop(best_idx, :);
end

function [J, metrics] = evaluate_window_objective(param, G_tx_k, S_LFM_k, H_scenarios, N_fft, N_pulse, f_idx, f_local, mlw_ham, papr_ham, lambda1, lambda2, mu_robust, fs, B)
L = length(f_idx);
coeff = param(1:3);
width_B_multiple = param(4);
win_local = build_generalized_cosine_window(L, f_local, coeff, width_B_multiple, B);
win = zeros(N_fft, 1);
win(f_idx) = win_local;
W_k = fftshift(win);

K = size(H_scenarios, 2);
J_list = zeros(K,1);
metric_list = zeros(K,4); % [pslr, islr, mainlobe, papr]

for k = 1:K
    H_k = H_scenarios(:, k);
    S_tx_k = S_LFM_k .* (G_tx_k .* W_k);
    s_tx_time = ifft(S_tx_k, N_fft);
    S_tx_full = fft(s_tx_time, N_fft);
    S_tx_with_H = S_tx_full .* H_k;
    s_out_time = ifft(S_tx_with_H, N_fft);
    s_out = s_out_time(1:N_pulse);
    s_out = s_out / sqrt(sum(abs(s_out).^2));

    auto_corr = xcorr(s_out, s_out);
    auto_corr = auto_corr / max(abs(auto_corr));

    peak_idx = ceil(length(auto_corr)/2);
    pslr_db = compute_pslr_corrected(auto_corr, peak_idx, fs, B);
    islr_db = compute_islr_corrected(auto_corr, peak_idx, fs, B);
    mainlobe_width = compute_3db_width_corrected(auto_corr, peak_idx, fs, B);
    papr_db = compute_papr(s_out);

    J_k = pslr_db + islr_db ...
        + lambda1 * max(0, mainlobe_width - mlw_ham)^2 ...
        + lambda2 * max(0, papr_db - papr_ham)^2;

    J_list(k) = J_k;
    metric_list(k, :) = [pslr_db, islr_db, mainlobe_width, papr_db];
end

mean_J = mean(J_list);
max_J = max(J_list);
J = mean_J + mu_robust * max_J;

[~, best_case_idx] = max(J_list);
metrics = struct('pslr_db', metric_list(best_case_idx,1), ...
                 'islr_db', metric_list(best_case_idx,2), ...
                 'mainlobe_width', metric_list(best_case_idx,3), ...
                 'papr_db', metric_list(best_case_idx,4), ...
                 'mean_J', mean_J, 'max_J', max_J);
end

function w = build_generalized_cosine_window(L, f_local, coeff, width_B_multiple, B)
a0 = coeff(1); a1 = coeff(2); a2 = coeff(3);
w = zeros(L, 1);
target_bw = min(2*B, max(0.1*B, width_B_multiple * B));  % 窗宽 = k*B，并限制在当前可用频带
active_idx = find(abs(f_local) <= target_bw/2);
if numel(active_idx) < 3
    [~, sorted_idx] = sort(abs(f_local), 'ascend');
    active_idx = sort(sorted_idx(1:min(3, L)));
end

idx0 = min(active_idx);
idx1 = max(active_idx);
Lw = idx1 - idx0 + 1;
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

function param = project_coeffs(param, min_width_B_multiple, max_width_B_multiple)
coeff = param(1:3);
width_B_multiple = param(4);

coeff = max(0, min(1, coeff));
s = sum(coeff);
if s <= eps
    coeff = [1, 0, 0];
else
    coeff = coeff / s;
end

width_B_multiple = max(min_width_B_multiple, min(max_width_B_multiple, width_B_multiple));
param = [coeff, width_B_multiple];
end


function H_scenarios = build_perturbed_channels(H_nominal, inband_idx, K, delta_a, delta_phi)
N = numel(H_nominal);
H_scenarios = repmat(H_nominal, 1, K);

for k = 1:K
    eps_a = zeros(N,1);
    eps_phi = zeros(N,1);
    eps_a(inband_idx) = -delta_a + 2*delta_a*rand(numel(inband_idx),1);
    eps_phi(inband_idx) = -delta_phi + 2*delta_phi*rand(numel(inband_idx),1);

    H_scenarios(:,k) = H_nominal .* (1 + eps_a) .* exp(1j*eps_phi);
end
end

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

function papr_db = compute_papr(x)
p = abs(x).^2;
papr_db = 10 * log10(max(p) / mean(p));
end
