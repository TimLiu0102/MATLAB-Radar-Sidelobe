%% 联合优化版发射端LFM预补偿算法（保留 test.m 指标与绘图风格）
% 在 test.m 物理建模与评价逻辑基础上，改写为：
% “正则化因子 alpha_reg + 广义余弦窗(a1,a2) 联合优化”

clear; close all; clc; clear functions;

%% 步骤0: 仿真参数设置
cfg = struct();
% 雷达参数
cfg.B = 20e6;               % 带宽: 20 MHz
cfg.T_pulse = 50e-6;        % 脉宽: 50 μs
cfg.fs = 60e6;              % 采样率: 60 MHz

% 处理参数
cfg.N_pulse = round(cfg.T_pulse * cfg.fs);
cfg.N_fft = 2^nextpow2(cfg.N_pulse * 8);
cfg.freq = (-cfg.N_fft/2:cfg.N_fft/2-1)' * (cfg.fs/cfg.N_fft);

% 补偿参数
cfg.A_max = 4.0;
cfg.alpha_baseline = 1e-2;

% 优化约束
cfg.bounds.alpha = [1e-5, 1e-2];
cfg.bounds.a1 = [0.01, 1];
cfg.bounds.a2 = [0.01, 1];
cfg.enable_a0_nonnegative = true;  % a0 = 1-a1-a2 >=0

% 频谱误差有效带宽（按题意使用有效带宽）
cfg.band_limit = cfg.B/2;

% 目标函数权重与目标阈值
cfg.weights = struct('pslr',20.0,'islr',1.0,'bw',1.5,'papr',1.5,'spec',10);
cfg.targets = struct('pslr',-30,'islr',-20,'papr',0,'spec',0);
cfg.spec_guard_factor = 5;   % 频谱不劣化约束权重因子

% 显示参数
f_center_idx = find(abs(cfg.freq)<=cfg.B/2);

%% 步骤0.1: 生成增强系统频率响应 H[k]
H_k = build_system_response(cfg.freq, cfg.B, cfg.N_fft);

% 显示系统响应（沿用 test.m 风格）
figure('Position', [100, 100, 1200, 400]);
subplot(1,2,1);
plot(cfg.freq/1e6, abs(H_k));
xlim([-cfg.B/1e6, cfg.B/1e6]*1.5);
xlabel('频率 (MHz)'); ylabel('幅度');
title('增强的系统幅度响应 H(f)');
grid on;

subplot(1,2,2);
plot(cfg.freq/1e6, unwrap(angle(H_k))*180/pi);
xlim([-cfg.B/1e6, cfg.B/1e6]*1.5);
xlabel('频率 (MHz)'); ylabel('相位 (度)');
title('增强的系统相位响应 ∠H(f)');
grid on;

%% 步骤1: 生成理想LFM信号及其频谱
[s_lfm, t, S_LFM_k] = generate_lfm_signal(cfg.B, cfg.T_pulse, cfg.fs, cfg.N_pulse, cfg.N_fft);
cfg.spec_baseline_no_comp = compute_spectrum_error(S_LFM_k, S_LFM_k, cfg.freq, cfg.band_limit);

% 显示LFM信号及频谱（沿用 test.m 风格）
figure('Position', [100, 100, 1200, 400]);
subplot(1,2,1);
plot(t*1e6, real(s_lfm));
xlabel('时间 (μs)'); ylabel('幅度');
title('原始LFM时域信号（实部）');
grid on;

subplot(1,2,2);
S_LFM_mag = fftshift(abs(S_LFM_k));
plot(cfg.freq/1e6, 20*log10(S_LFM_mag/max(S_LFM_mag)+1e-12));
xlim([-cfg.B/1e6, cfg.B/1e6]*1.5);
xlabel('频率 (MHz)'); ylabel('幅度 (dB)');
title('原始LFM频谱（矩形频谱）');
grid on;

%% 步骤2: 先计算基线（固定 alpha + Hamming），用于优化前后对比
base_params = struct('alpha_reg', cfg.alpha_baseline, 'a1', 0.46, 'a2', 0, 'a0', 0.54);
base_out = simulate_transmit_signal(base_params, S_LFM_k, H_k, cfg, 'hamming');
base_metrics = evaluate_metrics(base_out.s_ideal, base_out.s_no_comp, base_out.s_with_comp, cfg);

%% 步骤3: 联合优化（差分进化 DE，全局无导数优化）
[best_params, best_obj, best_out] = optimize_joint_parameters(S_LFM_k, H_k, cfg);

fprintf('\n========== 联合优化结果 ==========' );
fprintf('\n最优 alpha_reg = %.6g', best_params.alpha_reg);
fprintf('\n最优 a1 = %.4f, a2 = %.4f, a0 = %.4f', best_params.a1, best_params.a2, best_params.a0);
fprintf('\n最优目标函数值 J* = %.6f\n', best_obj);

%% 步骤4: 按 test.m 风格准备三种对比信号
% 理想 / 无预补偿 / 联合优化预补偿
s_ideal = best_out.s_ideal;
s_with_H = best_out.s_no_comp;
s_tx_with_H = best_out.s_with_comp;

fprintf('\n=== 三种情况对比（统一能量归一化） ===\n');
fprintf('1. 理想LFM（无系统失真）\n');
fprintf('2. 原始LFM + 系统失真（无预补偿）\n');
fprintf('3. 联合优化预补偿LFM + 系统失真（有预补偿）\n\n');

%% 步骤5: 自相关分析（沿用 test.m 计算方式）
t_corr = (-cfg.N_pulse+1:cfg.N_pulse-1)' / cfg.fs * 1e6;
auto_corr_ideal = xcorr(s_ideal, s_ideal);
auto_corr_no_comp = xcorr(s_with_H, s_with_H);
auto_corr_with_comp = xcorr(s_tx_with_H, s_tx_with_H);

auto_corr_ideal = auto_corr_ideal / max(abs(auto_corr_ideal));
auto_corr_no_comp = auto_corr_no_comp / max(abs(auto_corr_no_comp));
auto_corr_with_comp = auto_corr_with_comp / max(abs(auto_corr_with_comp));

auto_corr_ideal_db = 20*log10(abs(auto_corr_ideal) + 1e-10);
auto_corr_no_comp_db = 20*log10(abs(auto_corr_no_comp) + 1e-10);
auto_corr_with_comp_db = 20*log10(abs(auto_corr_with_comp) + 1e-10);

%% 步骤6: 综合性能分析图（严格沿用 test.m 样式）
figure('Position', [50, 50, 1400, 800]);

% 6.1 时域波形对比
subplot(3,3,1);
plot(t*1e6, real(s_ideal), 'k-', 'LineWidth', 1.5); hold on;
plot(t*1e6, real(s_with_H), 'r--', 'LineWidth', 1.5);
plot(t*1e6, real(s_tx_with_H), 'b-.', 'LineWidth', 1.5);
xlabel('时间 (μs)'); ylabel('幅度');
title('时域信号对比（实部）');
legend('理想LFM', '无预补偿', '有预补偿', 'Location', 'best');
grid on;

% 6.2 频谱对比
subplot(3,3,2);
s_ideal_padded = zeros(cfg.N_fft, 1); s_ideal_padded(1:cfg.N_pulse) = s_ideal;
S_ideal_mag = fftshift(abs(fft(s_ideal_padded, cfg.N_fft)));

s_with_H_padded = zeros(cfg.N_fft, 1); s_with_H_padded(1:cfg.N_pulse) = s_with_H;
S_no_comp_mag = fftshift(abs(fft(s_with_H_padded, cfg.N_fft)));

s_tx_with_H_padded = zeros(cfg.N_fft, 1); s_tx_with_H_padded(1:cfg.N_pulse) = s_tx_with_H;
S_with_comp_mag = fftshift(abs(fft(s_tx_with_H_padded, cfg.N_fft)));

plot(cfg.freq/1e6, 20*log10(S_ideal_mag/max(S_ideal_mag)+1e-12), 'k-', 'LineWidth', 1.5); hold on;
plot(cfg.freq/1e6, 20*log10(S_no_comp_mag/max(S_no_comp_mag)+1e-12), 'r--', 'LineWidth', 1.5);
plot(cfg.freq/1e6, 20*log10(S_with_comp_mag/max(S_with_comp_mag)+1e-12), 'b-.', 'LineWidth', 1.5);
xlim([-cfg.B/1e6*1.5, cfg.B/1e6*1.5]); ylim([-100, 5]);
xlabel('频率 (MHz)'); ylabel('幅度 (dB)');
title('经过系统后的频谱对比');
legend('理想LFM', '无预补偿', '有预补偿', 'Location', 'best');
grid on;

% 6.3 完整自相关函数（线性）
subplot(3,3,4);
plot(t_corr, abs(auto_corr_ideal), 'k-', 'LineWidth', 1.5); hold on;
plot(t_corr, abs(auto_corr_no_comp), 'r--', 'LineWidth', 1.5);
plot(t_corr, abs(auto_corr_with_comp), 'b-.', 'LineWidth', 1.5);
xlabel('时延 (μs)'); ylabel('幅度');
title('自相关函数（线性）');
legend('理想LFM', '无预补偿', '有预补偿', 'Location', 'best');
grid on;

% 6.4 完整自相关函数（对数）
subplot(3,3,5);
plot(t_corr, auto_corr_ideal_db, 'k-', 'LineWidth', 1.5); hold on;
plot(t_corr, auto_corr_no_comp_db, 'r--', 'LineWidth', 1.5);
plot(t_corr, auto_corr_with_comp_db, 'b-.', 'LineWidth', 1.5);
xlabel('时延 (μs)'); ylabel('幅度 (dB)');
title('自相关函数（对数）');
legend('理想LFM', '无预补偿', '有预补偿', 'Location', 'best');
grid on; ylim([-120, 0]);

% 6.5 自相关主瓣区域（线性）
subplot(3,3,7);
center_idx = ceil(length(t_corr)/2);
range_idx = center_idx-50:center_idx+50;
plot(t_corr(range_idx), abs(auto_corr_ideal(range_idx)), 'k-', 'LineWidth', 1.5); hold on;
plot(t_corr(range_idx), abs(auto_corr_no_comp(range_idx)), 'r--', 'LineWidth', 1.5);
plot(t_corr(range_idx), abs(auto_corr_with_comp(range_idx)), 'b-.', 'LineWidth', 1.5);
xlabel('时延 (μs)'); ylabel('幅度');
title('自相关主瓣区域');
legend('理想LFM', '无预补偿', '有预补偿', 'Location', 'best');
grid on;

% 6.6 自相关旁瓣区域（对数）
subplot(3,3,8);
mainlobe_width = ceil(2*cfg.fs/cfg.B);
sidelobe_idx = [1:center_idx-mainlobe_width, center_idx+mainlobe_width:length(t_corr)];
plot(t_corr(sidelobe_idx), auto_corr_ideal_db(sidelobe_idx), 'k-', 'LineWidth', 1.5); hold on;
plot(t_corr(sidelobe_idx), auto_corr_no_comp_db(sidelobe_idx), 'r--', 'LineWidth', 1.5);
plot(t_corr(sidelobe_idx), auto_corr_with_comp_db(sidelobe_idx), 'b-.', 'LineWidth', 1.5);
xlabel('时延 (μs)'); ylabel('幅度 (dB)');
title('自相关旁瓣区域');
legend('理想LFM', '无预补偿', '有预补偿', 'Location', 'best');
grid on; ylim([-120, 0]);

%% 步骤7: 计算并显示性能指标（PAPR/PSLR/主瓣宽度计算保留 test.m 逻辑）
pslr_ideal = compute_pslr_corrected(auto_corr_ideal, center_idx, cfg.fs, cfg.B);
pslr_no_comp = compute_pslr_corrected(auto_corr_no_comp, center_idx, cfg.fs, cfg.B);
pslr_with_comp = compute_pslr_corrected(auto_corr_with_comp, center_idx, cfg.fs, cfg.B);

islr_ideal = compute_islr_corrected(auto_corr_ideal, center_idx, cfg.fs, cfg.B);
islr_no_comp = compute_islr_corrected(auto_corr_no_comp, center_idx, cfg.fs, cfg.B);
islr_with_comp = compute_islr_corrected(auto_corr_with_comp, center_idx, cfg.fs, cfg.B);

bw3db_ideal = compute_3db_width_corrected(auto_corr_ideal, center_idx, cfg.fs, cfg.B);
bw3db_no_comp = compute_3db_width_corrected(auto_corr_no_comp, center_idx, cfg.fs, cfg.B);
bw3db_with_comp = compute_3db_width_corrected(auto_corr_with_comp, center_idx, cfg.fs, cfg.B);

theory_width = 0.886 / cfg.B * 1e6;
theory_pslr = -13.26;

% PAPR 按 test.m 方式
PAPR_ideal = 10*log10(max(abs(s_ideal).^2) / mean(abs(s_ideal).^2));
PAPR_no_comp = 10*log10(max(abs(s_with_H).^2) / mean(abs(s_with_H).^2));
PAPR_with_comp = 10*log10(max(abs(s_tx_with_H).^2) / mean(abs(s_tx_with_H).^2));

% 频谱恢复误差（定义：预补偿发射频谱 vs 原始LFM频谱）
spec_with_comp = compute_spectrum_error(best_out.S_tx_k, S_LFM_k, cfg.freq, cfg.band_limit);

% 误差对比（经过系统后）：无预补偿 vs 有预补偿
spec_no_comp_sys = compute_spectrum_error(best_out.S_no_comp_out_k, S_LFM_k, cfg.freq, cfg.band_limit);
spec_with_comp_sys = compute_spectrum_error(best_out.S_out_k, S_LFM_k, cfg.freq, cfg.band_limit);

% 右侧文字汇总区（test.m 样式）
subplot(3,3,[3,6,9]);
axis off;
text_str = sprintf('=== 性能指标对比 ===\n\n');
text_str = [text_str sprintf('理论主瓣宽度: %.3f μs\n', theory_width)];
text_str = [text_str sprintf('理论PSLR: %.2f dB\n\n', theory_pslr)];
text_str = [text_str sprintf('%-18s %-10s %-10s %-12s\n', '情况', 'PSLR(dB)', 'ISLR(dB)', '3dB宽度(μs)')];
text_str = [text_str sprintf('%-18s %-10.2f %-10.2f %-12.3f\n', '理想LFM', pslr_ideal, islr_ideal, bw3db_ideal)];
text_str = [text_str sprintf('%-18s %-10.2f %-10.2f %-12.3f\n', '无预补偿', pslr_no_comp, islr_no_comp, bw3db_no_comp)];
text_str = [text_str sprintf('%-18s %-10.2f %-10.2f %-12.3f\n', '联合优化有预补偿', pslr_with_comp, islr_with_comp, bw3db_with_comp)];

text_str = [text_str sprintf('\n=== 改善效果 ===\n')];
text_str = [text_str sprintf('PSLR改善: %.2f dB\n', pslr_no_comp - pslr_with_comp)];
text_str = [text_str sprintf('ISLR改善: %.2f dB\n', islr_no_comp - islr_with_comp)];
text_str = [text_str sprintf('主瓣宽度变化: %.3f μs\n', bw3db_with_comp - bw3db_ideal)];

text_str = [text_str sprintf('\n=== PAPR对比（能量归一化后） ===\n')];
text_str = [text_str sprintf('理想LFM: %.2f dB\n', PAPR_ideal)];
text_str = [text_str sprintf('无预补偿: %.2f dB\n', PAPR_no_comp)];
text_str = [text_str sprintf('联合优化有预补偿: %.2f dB\n', PAPR_with_comp)];
text_str = [text_str sprintf('PAPR增加: %.2f dB\n', PAPR_with_comp - PAPR_ideal)];

text_str = [text_str sprintf('\n=== 频谱恢复误差(NMSE, 方案A) ===\n')];
text_str = [text_str sprintf('预补偿发射频谱 vs 原始LFM: %.4e\n', spec_with_comp)];
text_str = [text_str sprintf('系统后无预补偿 vs 原始LFM: %.4e\n', spec_no_comp_sys)];
text_str = [text_str sprintf('系统后有预补偿 vs 原始LFM: %.4e\n', spec_with_comp_sys)];
text_str = [text_str sprintf('系统后误差改善: %.4e\n', spec_no_comp_sys - spec_with_comp_sys)];

text(0.05, 0.5, text_str, 'FontName', 'FixedWidth', 'FontSize', 10, 'VerticalAlignment', 'middle');
title('性能指标汇总（修正后）');

%% 步骤8: 性能改善可视化（沿用 test.m 样式）
figure('Position', [100, 100, 1200, 400]);

subplot(1,2,1);
mainlobe_exclude = ceil(cfg.fs/cfg.B);
sidelobe_idx_detailed = [1:center_idx-mainlobe_exclude, center_idx+mainlobe_exclude:length(t_corr)];
plot(t_corr(sidelobe_idx_detailed), auto_corr_ideal_db(sidelobe_idx_detailed), 'k-', 'LineWidth', 1.5); hold on;
plot(t_corr(sidelobe_idx_detailed), auto_corr_no_comp_db(sidelobe_idx_detailed), 'r--', 'LineWidth', 1.5);
plot(t_corr(sidelobe_idx_detailed), auto_corr_with_comp_db(sidelobe_idx_detailed), 'b-.', 'LineWidth', 1.5);
xlabel('时延 (μs)'); ylabel('幅度 (dB)');
title('自相关旁瓣对比');
legend('理想LFM', '无预补偿', '有预补偿', 'Location', 'best');
grid on; ylim([-120, 0]);

subplot(1,2,2);
improvement_data = [pslr_ideal, pslr_no_comp, pslr_with_comp; ...
                    islr_ideal, islr_no_comp, islr_with_comp; ...
                    bw3db_ideal, bw3db_no_comp, bw3db_with_comp; ...
                    PAPR_ideal, PAPR_no_comp, PAPR_with_comp; ...
                    0, 0, spec_with_comp];
bar_labels = {'理想LFM', '无预补偿', '有预补偿'};
bar(improvement_data');
set(gca, 'XTickLabel', bar_labels);
ylabel('数值');
legend('PSLR(dB)', 'ISLR(dB)', '3dB宽度(μs)', 'PAPR(dB)', '频谱误差', 'Location', 'best');
title('性能指标对比');
grid on;

%% 步骤9: 控制台输出总结
fprintf('\n=== 性能总结 ===\n');
fprintf('理论PSLR（矩形窗）: %.2f dB\n', theory_pslr);
fprintf('实测PSLR - 理想LFM: %.2f dB\n', pslr_ideal);
fprintf('实测PSLR - 无预补偿: %.2f dB（恶化 %.2f dB）\n', pslr_no_comp, pslr_no_comp - pslr_ideal);
fprintf('实测PSLR - 有预补偿: %.2f dB（改善 %.2f dB）\n', pslr_with_comp, pslr_no_comp - pslr_with_comp);
fprintf('\n理论主瓣宽度: %.3f μs\n', theory_width);
fprintf('实测主瓣宽度 - 理想LFM: %.3f μs\n', bw3db_ideal);
fprintf('实测主瓣宽度 - 无预补偿: %.3f μs\n', bw3db_no_comp);
fprintf('实测主瓣宽度 - 有预补偿: %.3f μs\n', bw3db_with_comp);
fprintf('\nPAPR - 理想LFM: %.2f dB\n', PAPR_ideal);
fprintf('PAPR - 无预补偿: %.2f dB\n', PAPR_no_comp);
fprintf('PAPR - 有预补偿: %.2f dB（增加 %.2f dB）\n', PAPR_with_comp, PAPR_with_comp - PAPR_ideal);
fprintf('\n频谱恢复误差(预补偿发射频谱 vs 原始LFM): %.4e\n', spec_with_comp);
fprintf('频谱恢复误差(系统后无预补偿 vs 原始LFM): %.4e\n', spec_no_comp_sys);
fprintf('频谱恢复误差(系统后有预补偿 vs 原始LFM): %.4e\n', spec_with_comp_sys);
fprintf('系统后误差改善: %.4e\n', spec_no_comp_sys - spec_with_comp_sys);

fprintf('\n=== 与固定Hamming基线对比（优化前后） ===\n');
fprintf('基线PSLR/ISLR/BW/PAPR/SPEC: %.2f / %.2f / %.3f / %.2f / %.4e\n', ...
    base_metrics.with_comp.pslr, base_metrics.with_comp.islr, base_metrics.with_comp.bw3db, base_metrics.with_comp.papr, compute_spectrum_error(base_out.S_tx_k, S_LFM_k, cfg.freq, cfg.band_limit));
fprintf('联合PSLR/ISLR/BW/PAPR/SPEC: %.2f / %.2f / %.3f / %.2f / %.4e\n', ...
    pslr_with_comp, islr_with_comp, bw3db_with_comp, PAPR_with_comp, spec_with_comp);

%% ======================= 局部函数定义 =======================
function [s_lfm, t, S_LFM_k] = generate_lfm_signal(B, T_pulse, fs, N_pulse, N_fft)
    t = (-N_pulse/2:N_pulse/2-1)' / fs;
    k_chirp = B / T_pulse;
    s_lfm = exp(1j * pi * k_chirp * t.^2);

    s_lfm_padded = zeros(N_fft, 1);
    s_lfm_padded(1:N_pulse) = s_lfm;
    S_LFM_k = fft(s_lfm_padded, N_fft);
end

function H_k = build_system_response(freq, B, N_fft)
    f_center_idx = find(abs(freq)<=B/2);
    H_mag = zeros(N_fft, 1);
    H_mag(f_center_idx) = 0.7 + 0.4*cos(2*pi*freq(f_center_idx)/B*6) .* ...
                          exp(-(freq(f_center_idx)/(0.5*B)).^2) + ...
                          0.1*sin(2*pi*freq(f_center_idx)/B*3);

    H_phase = zeros(N_fft, 1);
    H_phase(f_center_idx) = 0.4*pi*(freq(f_center_idx)/B).^3 + ...
                            0.3*pi*sin(2*pi*freq(f_center_idx)/B*5) + ...
                            0.05*pi*(freq(f_center_idx)/B).^2;

    H_k = H_mag .* exp(1j*H_phase);
    H_k = fftshift(H_k);
end

function G_tx_k = compute_precomp_filter(S_LFM_k, H_k, alpha_reg, A_max)
    %#ok<INUSD> % S_LFM_k 保留为参数，便于后续扩展
    % 正则化逆滤波（Wiener/Tikhonov 形式）
    G_tx_k = conj(H_k) ./ (abs(H_k).^2 + alpha_reg);

    G_tx_mag = abs(G_tx_k);
    max_G = max(G_tx_mag);
    if max_G > A_max
        G_tx_k = G_tx_k ./ max(1, G_tx_mag/A_max);
    end
end

function W_k = generate_generalized_cosine_window(freq, B, N_fft, a1, a2)
    a0 = 1 - a1 - a2;
    f_idx = find(abs(freq)<=B/2);

    gc_window = zeros(N_fft,1);
    Nw = length(f_idx);
    n = (0:Nw-1)';
    if Nw > 1
        w_local = a0 - a1*cos(2*pi*n/(Nw-1)) + a2*cos(4*pi*n/(Nw-1));
    else
        w_local = 1;
    end
    gc_window(f_idx) = w_local;
    W_k = fftshift(gc_window);
end

function out = simulate_transmit_signal(params, S_LFM_k, H_k, cfg, mode)
    G_tx_k = compute_precomp_filter(S_LFM_k, H_k, params.alpha_reg, cfg.A_max);
    if any(~isfinite(G_tx_k))
        out = struct('valid',false);
        return;
    end

    if strcmpi(mode, 'hamming')
        f_idx = find(abs(cfg.freq)<=cfg.B/2);
        hamming_window = zeros(cfg.N_fft, 1);
        hamming_win_local = hamming(length(f_idx));
        hamming_window(f_idx) = hamming_win_local;
        W_k = fftshift(hamming_window);
    else
        W_k = generate_generalized_cosine_window(cfg.freq, cfg.B, cfg.N_fft, params.a1, params.a2);
    end

    G_tx_k_windowed = G_tx_k .* W_k;
    S_tx_k = S_LFM_k .* G_tx_k_windowed;
    s_tx_time = ifft(S_tx_k, cfg.N_fft);
    s_tx_pulse = s_tx_time(1:cfg.N_pulse);
    s_tx_pulse = s_tx_pulse / sqrt(sum(abs(s_tx_pulse).^2));

    s_lfm_time = ifft(S_LFM_k, cfg.N_fft);
    s_ideal = s_lfm_time(1:cfg.N_pulse);
    s_ideal = s_ideal / sqrt(sum(abs(s_ideal).^2));

    S_no_comp_out_k = S_LFM_k .* H_k;
    s_no_comp_time = ifft(S_no_comp_out_k, cfg.N_fft);
    s_no_comp = s_no_comp_time(1:cfg.N_pulse);
    s_no_comp = s_no_comp / sqrt(sum(abs(s_no_comp).^2));

    S_out_k = fft(s_tx_time, cfg.N_fft) .* H_k;
    s_out_time = ifft(S_out_k, cfg.N_fft);
    s_with_comp = s_out_time(1:cfg.N_pulse);
    s_with_comp = s_with_comp / sqrt(sum(abs(s_with_comp).^2));

    out = struct();
    out.valid = all(isfinite([real(s_with_comp); imag(s_with_comp)]));
    out.G_tx_k = G_tx_k;
    out.W_k = W_k;
    out.G_tx_k_windowed = G_tx_k_windowed;
    out.S_tx_k = S_tx_k;
    out.S_out_k = S_out_k;
    out.S_no_comp_out_k = S_no_comp_out_k;
    out.s_ideal = s_ideal;
    out.s_no_comp = s_no_comp;
    out.s_with_comp = s_with_comp;
end

function metrics = evaluate_metrics(s_ideal, s_no_comp, s_with_comp, cfg)
    auto_corr_ideal = xcorr(s_ideal, s_ideal);
    auto_corr_no = xcorr(s_no_comp, s_no_comp);
    auto_corr_yes = xcorr(s_with_comp, s_with_comp);

    auto_corr_ideal = auto_corr_ideal / max(abs(auto_corr_ideal));
    auto_corr_no = auto_corr_no / max(abs(auto_corr_no));
    auto_corr_yes = auto_corr_yes / max(abs(auto_corr_yes));

    center_idx = ceil(length(auto_corr_ideal)/2);

    metrics.ideal.pslr = compute_pslr_corrected(auto_corr_ideal, center_idx, cfg.fs, cfg.B);
    metrics.no_comp.pslr = compute_pslr_corrected(auto_corr_no, center_idx, cfg.fs, cfg.B);
    metrics.with_comp.pslr = compute_pslr_corrected(auto_corr_yes, center_idx, cfg.fs, cfg.B);

    metrics.ideal.islr = compute_islr_corrected(auto_corr_ideal, center_idx, cfg.fs, cfg.B);
    metrics.no_comp.islr = compute_islr_corrected(auto_corr_no, center_idx, cfg.fs, cfg.B);
    metrics.with_comp.islr = compute_islr_corrected(auto_corr_yes, center_idx, cfg.fs, cfg.B);

    metrics.ideal.bw3db = compute_3db_width_corrected(auto_corr_ideal, center_idx, cfg.fs, cfg.B);
    metrics.no_comp.bw3db = compute_3db_width_corrected(auto_corr_no, center_idx, cfg.fs, cfg.B);
    metrics.with_comp.bw3db = compute_3db_width_corrected(auto_corr_yes, center_idx, cfg.fs, cfg.B);

    metrics.ideal.papr = 10*log10(max(abs(s_ideal).^2) / mean(abs(s_ideal).^2));
    metrics.no_comp.papr = 10*log10(max(abs(s_no_comp).^2) / mean(abs(s_no_comp).^2));
    metrics.with_comp.papr = 10*log10(max(abs(s_with_comp).^2) / mean(abs(s_with_comp).^2));
end

function E_spec = compute_spectrum_error(S_out_k, S_ideal_k, freq, band_limit)
    % 方案A：幅度谱归一化均方误差（有效带宽内）
    % 注意：freq 采用中心化频率轴，因此需要先 fftshift 到中心化顺序再按 freq 索引。
    idx = find(abs(freq)<=band_limit);

    S_out_shift = fftshift(S_out_k);
    S_ideal_shift = fftshift(S_ideal_k);

    S_out_band = S_out_shift(idx);
    S_ideal_band = S_ideal_shift(idx);

    E_spec = norm(abs(S_out_band) - abs(S_ideal_band), 2)^2 / ...
             (norm(abs(S_ideal_band), 2)^2 + eps);
end

function J = objective_function(params, S_LFM_k, H_k, cfg)
    big_penalty = 1e8;
    alpha_reg = params(1);
    a1 = params(2);
    a2 = params(3);
    a0 = 1 - a1 - a2;

    if alpha_reg <= 0 || a1 < cfg.bounds.a1(1) || a1 > cfg.bounds.a1(2) || a2 < cfg.bounds.a2(1) || a2 > cfg.bounds.a2(2)
        J = big_penalty; return;
    end
    if cfg.enable_a0_nonnegative && a0 < 0
        J = big_penalty; return;
    end

    p = struct('alpha_reg',alpha_reg, 'a1',a1, 'a2',a2, 'a0',a0);
    sim = simulate_transmit_signal(p, S_LFM_k, H_k, cfg, 'generalized');
    if ~sim.valid || any(~isfinite(sim.G_tx_k))
        J = big_penalty; return;
    end

    M = evaluate_metrics(sim.s_ideal, sim.s_no_comp, sim.s_with_comp, cfg);
    spec_error = compute_spectrum_error(sim.S_tx_k, S_LFM_k, cfg.freq, cfg.band_limit);

    if any(~isfinite([M.with_comp.pslr, M.with_comp.islr, M.with_comp.bw3db, M.with_comp.papr, spec_error]))
        J = big_penalty; return;
    end

    pslr_penalty = max(0, M.with_comp.pslr - cfg.targets.pslr)^2;
    islr_penalty = max(0, M.with_comp.islr - cfg.targets.islr)^2;
    bw_penalty = max(0, (M.with_comp.bw3db - M.ideal.bw3db) / M.ideal.bw3db)^2;
    papr_penalty = max(0, M.with_comp.papr - cfg.targets.papr)^2;
    spec_penalty = max(0, spec_error - cfg.targets.spec)^2;

    % 向后兼容：若旧版配置中仍有 spec_guard_factor，则允许以 0 惩罚安全运行。
    spec_guard_penalty = 0;
    if isfield(cfg, 'spec_guard_factor')
        spec_guard_factor = cfg.spec_guard_factor;
    else
        spec_guard_factor = 0;
    end

    J = cfg.weights.pslr * pslr_penalty + ...
        cfg.weights.islr * islr_penalty + ...
        cfg.weights.bw   * bw_penalty + ...
        cfg.weights.papr * papr_penalty + ...
        cfg.weights.spec * spec_penalty + ...
        cfg.weights.spec * spec_guard_factor * spec_guard_penalty;

    if ~isfinite(J)
        J = big_penalty;
    end
end

function [best_params, best_obj, best_out] = optimize_joint_parameters(S_LFM_k, H_k, cfg)
    fprintf('\n=== 使用差分进化(DE)进行联合优化 ===\n');

    % 参数向量: x = [alpha_reg, a1, a2]
    lb = [cfg.bounds.alpha(1), cfg.bounds.a1(1), cfg.bounds.a2(1)];
    ub = [cfg.bounds.alpha(2), cfg.bounds.a1(2), cfg.bounds.a2(2)];

    pop_size = 36;
    max_gen = 70;
    F = 0.75;      % mutation factor
    CR = 0.90;     % crossover rate

    pop = zeros(pop_size, 3);
    obj = inf(pop_size, 1);

    for i = 1:pop_size
        pop(i,:) = sample_feasible_point(lb, ub, cfg.enable_a0_nonnegative);
        obj(i) = objective_function(pop(i,:), S_LFM_k, H_k, cfg);
    end

    [best_obj, best_idx] = min(obj);
    best_vec = pop(best_idx,:);

    for g = 1:max_gen
        for i = 1:pop_size
            idx = randperm(pop_size, 3);
            while any(idx == i)
                idx = randperm(pop_size, 3);
            end

            x1 = pop(idx(1),:); x2 = pop(idx(2),:); x3 = pop(idx(3),:);
            v = x1 + F * (x2 - x3);

            % binomial crossover
            u = pop(i,:);
            j_rand = randi(3);
            for j = 1:3
                if rand <= CR || j == j_rand
                    u(j) = v(j);
                end
            end

            u = project_feasible_point(u, lb, ub, cfg.enable_a0_nonnegative);
            fu = objective_function(u, S_LFM_k, H_k, cfg);

            if fu < obj(i)
                pop(i,:) = u;
                obj(i) = fu;
                if fu < best_obj
                    best_obj = fu;
                    best_vec = u;
                end
            end
        end
        fprintf('DE进度: %d/%d, 当前最优 J=%.6f\n', g, max_gen, best_obj);
    end

    best_params = struct('alpha_reg', best_vec(1), 'a1', best_vec(2), 'a2', best_vec(3), 'a0', 1-best_vec(2)-best_vec(3));
    best_out = simulate_transmit_signal(best_params, S_LFM_k, H_k, cfg, 'generalized');
end

function x = sample_feasible_point(lb, ub, enable_a0_nonnegative)
    x = lb + rand(1,3) .* (ub - lb);
    x = project_feasible_point(x, lb, ub, enable_a0_nonnegative);
end

function x = project_feasible_point(x, lb, ub, enable_a0_nonnegative)
    x = max(lb, min(ub, x));

    if enable_a0_nonnegative
        a1 = x(2); a2 = x(3);
        if a1 + a2 > 1
            s = a1 + a2;
            if s <= 0
                a1 = lb(2); a2 = lb(3);
            else
                a1 = a1 / s;
                a2 = a2 / s;
            end
            a1 = max(lb(2), min(ub(2), a1));
            a2 = max(lb(3), min(ub(3), a2));
            if a1 + a2 > 1
                scale = 1 / (a1 + a2 + eps);
                a1 = a1 * scale;
                a2 = a2 * scale;
            end
            x(2) = a1; x(3) = a2;
        end
    end
end

% ===== 以下三个函数保持 test.m 版本逻辑 =====
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
