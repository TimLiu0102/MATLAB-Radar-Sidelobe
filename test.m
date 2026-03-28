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

% 显示系统响应
figure('Position', [100, 100, 1200, 400]);
subplot(1,2,1);
plot(freq/1e6, abs(H_k));
xlim([-B/1e6, B/1e6]*1.5);
xlabel('频率 (MHz)'); ylabel('幅度');
title('增强的系统幅度响应 H(f)');
grid on;

subplot(1,2,2);
plot(freq/1e6, unwrap(angle(H_k))*180/pi);
xlim([-B/1e6, B/1e6]*1.5);
xlabel('频率 (MHz)'); ylabel('相位 (度)');
title('增强的系统相位响应 ∠H(f)');
grid on;

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

% 显示LFM信号及频谱
figure('Position', [100, 100, 1200, 400]);
subplot(1,2,1);
plot(t*1e6, real(s_lfm));
xlabel('时间 (μs)'); ylabel('幅度');
title('原始LFM时域信号（实部）');
grid on;

subplot(1,2,2);
S_LFM_mag = fftshift(abs(S_LFM_k));
plot(freq/1e6, 20*log10(S_LFM_mag/max(S_LFM_mag)));
xlim([-B/1e6, B/1e6]*1.5);
xlabel('频率 (MHz)'); ylabel('幅度 (dB)');
title('原始LFM频谱（矩形频谱）');
grid on;

%% 步骤2: 计算发射端预补偿谱（幅度+相位全补偿）
% 计算正则化项：使用均值而不是最大值，更稳定
epsilon = alpha_reg / mean(abs(H_k .* S_LFM_k));

% 核心补偿公式：G_tx[k] = R[k] / (H[k] * S_LFM[k] + ε)
G_tx_k = R_k ./ (H_k .* S_LFM_k + epsilon);

% 幅度限制：防止过高的增益导致PA饱和
G_tx_mag = abs(G_tx_k);
max_G = max(G_tx_mag);
if max_G > A_max
    fprintf('应用幅度限制: 最大增益 %.2f > 限制值 %.2f\n', max_G, A_max);
    G_tx_k = G_tx_k ./ max(1, G_tx_mag/A_max);
end

% 显示补偿滤波器
figure('Position', [100, 100, 1200, 400]);
subplot(1,2,1);
G_tx_mag_plot = fftshift(abs(G_tx_k));
plot(freq/1e6, 20*log10(G_tx_mag_plot/max(G_tx_mag_plot)));
xlim([-B/1e6, B/1e6]*1.5);
xlabel('频率 (MHz)'); ylabel('增益 (dB)');
title('预补偿滤波器幅度响应');
grid on;

subplot(1,2,2);
plot(freq/1e6, fftshift(unwrap(angle(G_tx_k)))*180/pi);
xlim([-B/1e6, B/1e6]*1.5);
xlabel('频率 (MHz)'); ylabel('相位 (度)');
title('预补偿滤波器相位响应');
grid on;

%% 步骤3: 频域窗/带宽约束 - 使用海明窗
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

% 情况4: 仅预补偿不加窗后经过系统H
S_tx_k_no_window = S_LFM_k .* G_tx_k;
s_tx_time_no_window = ifft(S_tx_k_no_window, N_fft);
S_tx_no_window_with_H = fft(s_tx_time_no_window, N_fft) .* H_k;
s_tx_no_window_with_H_time = ifft(S_tx_no_window_with_H, N_fft);
s_tx_no_window_with_H = s_tx_no_window_with_H_time(1:N_pulse);
s_tx_no_window_with_H = s_tx_no_window_with_H / sqrt(sum(abs(s_tx_no_window_with_H).^2));

fprintf('\n=== 四种情况对比（统一能量归一化） ===\n');
fprintf('1. 理想LFM（无系统失真）\n');
fprintf('2. 原始LFM + 系统失真（无预补偿）\n');
fprintf('3. 预补偿LFM + 系统失真（有预补偿）\n\n');
fprintf('4. 预补偿LFM + 系统失真（仅预补偿不加窗）\n\n');

%% 步骤6: 自相关分析
% 计算自相关函数
t_corr = (-N_pulse+1:N_pulse-1) / fs * 1e6;  % 微秒

% 计算四种情况的自相关
auto_corr_ideal = xcorr(s_ideal, s_ideal);
auto_corr_no_comp = xcorr(s_with_H, s_with_H);
auto_corr_with_comp = xcorr(s_tx_with_H, s_tx_with_H);
auto_corr_precomp_only = xcorr(s_tx_no_window_with_H, s_tx_no_window_with_H);

% 归一化自相关（能量归一化）
auto_corr_ideal = auto_corr_ideal / max(abs(auto_corr_ideal));
auto_corr_no_comp = auto_corr_no_comp / max(abs(auto_corr_no_comp));
auto_corr_with_comp = auto_corr_with_comp / max(abs(auto_corr_with_comp));
auto_corr_precomp_only = auto_corr_precomp_only / max(abs(auto_corr_precomp_only));

% 转换为dB显示
auto_corr_ideal_db = 20*log10(abs(auto_corr_ideal) + 1e-10);
auto_corr_no_comp_db = 20*log10(abs(auto_corr_no_comp) + 1e-10);
auto_corr_with_comp_db = 20*log10(abs(auto_corr_with_comp) + 1e-10);
auto_corr_precomp_only_db = 20*log10(abs(auto_corr_precomp_only) + 1e-10);

%% 步骤7: 综合性能分析
figure('Position', [50, 50, 1400, 800]);

% 7.1 时域波形对比（实部）
subplot(3,3,1);
plot(t*1e6, real(s_ideal), 'k-', 'LineWidth', 1.5); hold on;
plot(t*1e6, real(s_with_H), 'r--', 'LineWidth', 1.5);
plot(t*1e6, real(s_tx_with_H), 'b-.', 'LineWidth', 1.5);
xlabel('时间 (μs)'); ylabel('幅度');
title('时域信号对比（实部）');
legend('理想LFM', '无预补偿', '有预补偿', 'Location', 'best');
grid on;

% 7.2 频谱对比
subplot(3,3,2);
% 重新计算频谱（从时域信号计算，确保正确）
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
xlabel('频率 (MHz)'); ylabel('幅度 (dB)');
title('经过系统后的频谱对比');
legend('理想LFM', '无预补偿', '有预补偿', 'Location', 'best');
grid on;

% 7.3 完整自相关函数（线性）
subplot(3,3,4);
plot(t_corr, abs(auto_corr_ideal), 'k-', 'LineWidth', 1.5); hold on;
plot(t_corr, abs(auto_corr_no_comp), 'r--', 'LineWidth', 1.5);
plot(t_corr, abs(auto_corr_with_comp), 'b-.', 'LineWidth', 1.5);
xlabel('时延 (μs)'); ylabel('幅度');
title('自相关函数（线性）');
legend('理想LFM', '无预补偿', '有预补偿', 'Location', 'best');
grid on;

% 7.4 完整自相关函数（对数）
subplot(3,3,5);
plot(t_corr, auto_corr_ideal_db, 'k-', 'LineWidth', 1.5); hold on;
plot(t_corr, auto_corr_no_comp_db, 'r--', 'LineWidth', 1.5);
plot(t_corr, auto_corr_with_comp_db, 'b-.', 'LineWidth', 1.5);
xlabel('时延 (μs)'); ylabel('幅度 (dB)');
title('自相关函数（对数）');
legend('理想LFM', '无预补偿', '有预补偿', 'Location', 'best');
grid on;
ylim([-120, 0]);

% 7.5 自相关主瓣区域（线性）
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

% 7.6 自相关旁瓣区域（对数）
subplot(3,3,8);
% 选择旁瓣区域（排除主瓣）
mainlobe_width = ceil(2*fs/B);  % 根据主瓣理论宽度设定
sidelobe_idx = [1:center_idx-mainlobe_width, center_idx+mainlobe_width:length(t_corr)];
plot(t_corr(sidelobe_idx), auto_corr_ideal_db(sidelobe_idx), 'k-', 'LineWidth', 1.5); hold on;
plot(t_corr(sidelobe_idx), auto_corr_no_comp_db(sidelobe_idx), 'r--', 'LineWidth', 1.5);
plot(t_corr(sidelobe_idx), auto_corr_with_comp_db(sidelobe_idx), 'b-.', 'LineWidth', 1.5);
xlabel('时延 (μs)'); ylabel('幅度 (dB)');
title('自相关旁瓣区域');
legend('理想LFM', '无预补偿', '有预补偿', 'Location', 'best');
grid on;
ylim([-120, 0]);

%% 步骤8: 计算和显示性能指标
% 峰值旁瓣比（PSLR）
center_idx = ceil(length(auto_corr_ideal)/2);
pslr_ideal = compute_pslr_corrected(auto_corr_ideal, center_idx, fs, B);
pslr_no_comp = compute_pslr_corrected(auto_corr_no_comp, center_idx, fs, B);
pslr_with_comp = compute_pslr_corrected(auto_corr_with_comp, center_idx, fs, B);
pslr_precomp_only = compute_pslr_corrected(auto_corr_precomp_only, center_idx, fs, B);

% 积分旁瓣比（ISLR）
islr_ideal = compute_islr_corrected(auto_corr_ideal, center_idx, fs, B);
islr_no_comp = compute_islr_corrected(auto_corr_no_comp, center_idx, fs, B);
islr_with_comp = compute_islr_corrected(auto_corr_with_comp, center_idx, fs, B);
islr_precomp_only = compute_islr_corrected(auto_corr_precomp_only, center_idx, fs, B);

% 主瓣宽度（3dB）
bw3db_ideal = compute_3db_width_corrected(auto_corr_ideal, center_idx, fs, B);
bw3db_no_comp = compute_3db_width_corrected(auto_corr_no_comp, center_idx, fs, B);
bw3db_with_comp = compute_3db_width_corrected(auto_corr_with_comp, center_idx, fs, B);
bw3db_precomp_only = compute_3db_width_corrected(auto_corr_precomp_only, center_idx, fs, B);

% 理论主瓣宽度
theory_width = 0.886 / B * 1e6;  % 微秒

% 理论PSLR（矩形窗）
theory_pslr = -13.26;  % dB

% 计算PAPR（峰均功率比） - 使用能量归一化后的信号
PAPR_ideal = 10*log10(max(abs(s_ideal).^2) / mean(abs(s_ideal).^2));
PAPR_no_comp = 10*log10(max(abs(s_with_H).^2) / mean(abs(s_with_H).^2));
PAPR_with_comp = 10*log10(max(abs(s_tx_with_H).^2) / mean(abs(s_tx_with_H).^2));
PAPR_precomp_only = 10*log10(max(abs(s_tx_no_window_with_H).^2) / mean(abs(s_tx_no_window_with_H).^2));

% 显示性能指标表格
subplot(3,3,[3,6,9]);
axis off;

% 创建性能对比表格
text_str = sprintf('=== 性能指标对比 ===\n\n');
text_str = [text_str sprintf('理论主瓣宽度: %.3f μs\n', theory_width)];
text_str = [text_str sprintf('理论PSLR: %.2f dB\n\n', theory_pslr)];
text_str = [text_str sprintf('%-15s %-12s %-12s %-12s\n', '情况', 'PSLR(dB)', 'ISLR(dB)', '3dB宽度(μs)')];
text_str = [text_str sprintf('%-15s %-12.2f %-12.2f %-12.3f\n', '理想LFM', pslr_ideal, islr_ideal, bw3db_ideal)];
text_str = [text_str sprintf('%-15s %-12.2f %-12.2f %-12.3f\n', '无预补偿', pslr_no_comp, islr_no_comp, bw3db_no_comp)];
text_str = [text_str sprintf('%-15s %-12.2f %-12.2f %-12.3f\n', '有预补偿', pslr_with_comp, islr_with_comp, bw3db_with_comp)];
text_str = [text_str sprintf('%-15s %-12.2f %-12.2f %-12.3f\n', '仅预补偿不加窗', pslr_precomp_only, islr_precomp_only, bw3db_precomp_only)];

text_str = [text_str sprintf('\n=== 改善效果 ===\n')];
text_str = [text_str sprintf('PSLR改善: %.2f dB\n', pslr_no_comp - pslr_with_comp)];
text_str = [text_str sprintf('ISLR改善: %.2f dB\n', islr_no_comp - islr_with_comp)];
text_str = [text_str sprintf('主瓣宽度变化: %.3f μs\n', bw3db_with_comp - bw3db_ideal)];

text_str = [text_str sprintf('\n=== PAPR对比（能量归一化后） ===\n')];
text_str = [text_str sprintf('理想LFM: %.2f dB\n', PAPR_ideal)];
text_str = [text_str sprintf('无预补偿: %.2f dB\n', PAPR_no_comp)];
text_str = [text_str sprintf('有预补偿: %.2f dB\n', PAPR_with_comp)];
text_str = [text_str sprintf('仅预补偿不加窗: %.2f dB\n', PAPR_precomp_only)];
text_str = [text_str sprintf('PAPR增加: %.2f dB\n', PAPR_with_comp - PAPR_ideal)];

text(0.1, 0.5, text_str, 'FontName', 'FixedWidth', 'FontSize', 10, 'VerticalAlignment', 'middle');
title('性能指标汇总（修正后）');

%% 步骤9: 性能改善可视化
figure('Position', [100, 100, 1200, 400]);

% 9.1 旁瓣对比
subplot(1,2,1);
mainlobe_exclude = ceil(fs/B);  % 根据主瓣宽度排除
sidelobe_idx_detailed = [1:center_idx-mainlobe_exclude, center_idx+mainlobe_exclude:length(t_corr)];

plot(t_corr(sidelobe_idx_detailed), auto_corr_ideal_db(sidelobe_idx_detailed), 'k-', 'LineWidth', 1.5); hold on;
plot(t_corr(sidelobe_idx_detailed), auto_corr_no_comp_db(sidelobe_idx_detailed), 'r--', 'LineWidth', 1.5);
plot(t_corr(sidelobe_idx_detailed), auto_corr_with_comp_db(sidelobe_idx_detailed), 'b-.', 'LineWidth', 1.5);
xlabel('时延 (μs)'); ylabel('幅度 (dB)');
title('自相关旁瓣对比');
legend('理想LFM', '无预补偿', '有预补偿', 'Location', 'best');
grid on;
ylim([-120, 0]);

% 9.2 性能改善柱状图
subplot(1,2,2);
improvement_data = [pslr_ideal, pslr_no_comp, pslr_with_comp; ...
                    islr_ideal, islr_no_comp, islr_with_comp; ...
                    bw3db_ideal, bw3db_no_comp, bw3db_with_comp];

bar_labels = {'理想LFM', '无预补偿', '有预补偿'};
bar(improvement_data');
set(gca, 'XTickLabel', bar_labels);
ylabel('数值');
legend('PSLR(dB)', 'ISLR(dB)', '3dB宽度(μs)', 'Location', 'best');
title('性能指标对比');
grid on;

% 在柱状图上添加数值标签
for i = 1:3
    for j = 1:3
        if j == 3  % 3dB宽度
            text_str = sprintf('%.3f', improvement_data(j,i));
        else  % PSLR和ISLR
            text_str = sprintf('%.2f', improvement_data(j,i));
        end
        text(i, improvement_data(j,i), text_str, ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
            'FontSize', 8);
    end
end


%% 控制台输出总结
fprintf('\n=== 性能总结 ===\n');
fprintf('理论PSLR（矩形窗）: %.2f dB\n', theory_pslr);
fprintf('实测PSLR - 理想LFM: %.2f dB\n', pslr_ideal);
fprintf('实测PSLR - 无预补偿: %.2f dB（恶化 %.2f dB）\n', pslr_no_comp, pslr_no_comp - pslr_ideal);
fprintf('实测PSLR - 有预补偿: %.2f dB（改善 %.2f dB）\n', pslr_with_comp, pslr_no_comp - pslr_with_comp);
fprintf('实测PSLR - 仅预补偿不加窗: %.2f dB\n', pslr_precomp_only);
fprintf('\n理论主瓣宽度: %.3f μs\n', theory_width);
fprintf('实测主瓣宽度 - 理想LFM: %.3f μs\n', bw3db_ideal);
fprintf('实测主瓣宽度 - 无预补偿: %.3f μs\n', bw3db_no_comp);
fprintf('实测主瓣宽度 - 有预补偿: %.3f μs\n', bw3db_with_comp);
fprintf('实测主瓣宽度 - 仅预补偿不加窗: %.3f μs\n', bw3db_precomp_only);
fprintf('\nPAPR - 理想LFM: %.2f dB\n', PAPR_ideal);
fprintf('PAPR - 无预补偿: %.2f dB\n', PAPR_no_comp);
fprintf('PAPR - 有预补偿: %.2f dB（增加 %.2f dB）\n', PAPR_with_comp, PAPR_with_comp - PAPR_ideal);
fprintf('PAPR - 仅预补偿不加窗: %.2f dB（增加 %.2f dB）\n', PAPR_precomp_only, PAPR_precomp_only - PAPR_ideal);
fprintf('\nISLR - 仅预补偿不加窗: %.2f dB\n', islr_precomp_only);


%% 步骤10: 引入rubust.m扰动并按win_length.m步骤4.1计算相位误差
% 采用rubust.m中的建模方式：H_pert = H_nominal .* (1+eps_a) .* exp(1j*eps_phi)
K_perturb = 20;
delta_a = 0.05;          % 幅度扰动上限（±5%，与rubust.m一致）
delta_phi = 5*pi/180;    % 相位扰动上限（±5°，与rubust.m一致）

H_scenarios = build_perturbed_channels(H_k, f_center_idx, K_perturb, delta_a, delta_phi);

restoration_error_vec = zeros(K_perturb,1);
time_error_vec = zeros(K_perturb,1);
precomp_spec_error_vec = zeros(K_perturb,1);
precomp_time_error_vec = zeros(K_perturb,1);
pslr_perturb_vec = zeros(K_perturb,1);
islr_perturb_vec = zeros(K_perturb,1);
papr_perturb_vec = zeros(K_perturb,1);
mlw_perturb_vec = zeros(K_perturb,1);

for k = 1:K_perturb
    H_case = H_scenarios(:,k);

    % test.m自身的预补偿发射链路
    S_tx_case = S_LFM_k .* G_tx_k_windowed;
    s_tx_case_time = ifft(S_tx_case, N_fft);

    % 预补偿后发射信号（未经过扰动信道）
    s_tx_case = s_tx_case_time(1:N_pulse);
    s_tx_case = s_tx_case / sqrt(sum(abs(s_tx_case).^2) + eps);

    % 扰动信道后接收
    S_rx_case = fft(s_tx_case_time, N_fft) .* H_case;
    s_rx_case_time = ifft(S_rx_case, N_fft);
    s_rx_case = s_rx_case_time(1:N_pulse);
    s_rx_case = s_rx_case / sqrt(sum(abs(s_rx_case).^2) + eps);

    % 用win_length.m步骤4.1方法计算相对误差：
    % (1) 预补偿后信号 vs 原始LFM
    err_precomp = evaluate_reference_error(s_tx_case, S_LFM_k, N_pulse, N_fft, freq, B);
    precomp_spec_error_vec(k) = err_precomp.spec_nmse;
    precomp_time_error_vec(k) = err_precomp.time_nmse;

    % (2) 加扰动后信号 vs 原始LFM
    err_case = evaluate_reference_error(s_rx_case, S_LFM_k, N_pulse, N_fft, freq, B);
    restoration_error_vec(k) = err_case.spec_nmse;
    time_error_vec(k) = err_case.time_nmse;

    % 用test.m已有方式计算四个指标
    auto_corr_case = xcorr(s_rx_case, s_rx_case);
    auto_corr_case = auto_corr_case / (max(abs(auto_corr_case)) + eps);
    center_idx_case = ceil(length(auto_corr_case)/2);

    pslr_perturb_vec(k) = compute_pslr_corrected(auto_corr_case, center_idx_case, fs, B);
    islr_perturb_vec(k) = compute_islr_corrected(auto_corr_case, center_idx_case, fs, B);
    mlw_perturb_vec(k) = compute_3db_width_corrected(auto_corr_case, center_idx_case, fs, B);
    papr_perturb_vec(k) = 10*log10(max(abs(s_rx_case).^2) / mean(abs(s_rx_case).^2));
end

fprintf('\n=== 扰动信道下指标统计（K=%d）===\n', K_perturb);
fprintf('预补偿后 vs 原始LFM: spec\_nmse均值=%.6f, time\_nmse均值=%.6f\n', mean(precomp_spec_error_vec), mean(precomp_time_error_vec));
fprintf('加扰动后 vs 原始LFM: spec\_nmse均值=%.6f, 最大=%.6f\n', mean(restoration_error_vec), max(restoration_error_vec));
fprintf('加扰动后 vs 原始LFM: time\_nmse均值=%.6f, 最大=%.6f\n', mean(time_error_vec), max(time_error_vec));
fprintf('PSLR(dB): 均值=%.4f, 最差=%.4f\n', mean(pslr_perturb_vec), max(pslr_perturb_vec));
fprintf('ISLR(dB): 均值=%.4f, 最差=%.4f\n', mean(islr_perturb_vec), max(islr_perturb_vec));
fprintf('PAPR(dB): 均值=%.4f, 最大=%.4f\n', mean(papr_perturb_vec), max(papr_perturb_vec));
fprintf('主瓣宽度(us): 均值=%.4f, 最大=%.4f\n', mean(mlw_perturb_vec), max(mlw_perturb_vec));

% 可视化：五个指标箱型图（相对频谱误差 + PSLR + ISLR + PAPR + 主瓣宽度）
metrics_matrix = [restoration_error_vec, pslr_perturb_vec, islr_perturb_vec, papr_perturb_vec, mlw_perturb_vec];
figure('Position', [120, 120, 1200, 420]);
boxplot(metrics_matrix, 'Labels', {'spec NMSE (perturbed)','PSLR (dB)','ISLR (dB)','PAPR (dB)','Mainlobe Width (us)'});
ylabel('Metric Value');
title('扰动场景下五个指标箱型图（误差按win_length步骤4.1）');
grid on;


%% 步骤10: 不同B、T_pulse（fs=60e6）鲁棒性分析（1x2子图）
fs_robust = 60e6;
B_sweep_MHz = [10, 15, 20, 25, 30,35, 40];
Tp_sweep_us = [10, 15, 20, 25, 30,35, 40];

% 图A: 扫描B，固定T_pulse
T_fix = T_pulse;
num_B = numel(B_sweep_MHz);
robust_B = zeros(num_B, 5); % [B_MHz, PSLR, ISLR, MLW, PAPR]
for i = 1:num_B
    B_case = B_sweep_MHz(i) * 1e6;
    case_metrics = run_single_case_metrics_test(B_case, T_fix, fs_robust, alpha_reg, A_max);
    robust_B(i, :) = [B_sweep_MHz(i), case_metrics.pslr_with_comp, case_metrics.islr_with_comp, ...
        case_metrics.mlw_with_comp, case_metrics.papr_with_comp];
end

% 图B: 扫描T_pulse，固定B
B_fix = B;
num_T = numel(Tp_sweep_us);
robust_T = zeros(num_T, 5); % [Tp_us, PSLR, ISLR, MLW, PAPR]
for i = 1:num_T
    T_case = Tp_sweep_us(i) * 1e-6;
    case_metrics = run_single_case_metrics_test(B_fix, T_case, fs_robust, alpha_reg, A_max);
    robust_T(i, :) = [Tp_sweep_us(i), case_metrics.pslr_with_comp, case_metrics.islr_with_comp, ...
        case_metrics.mlw_with_comp, case_metrics.papr_with_comp];
end

figure;
set(gcf, 'Position', [150, 120, 1280, 460], 'Color', [1 1 1]);
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

% 图A：不同B下性能变化
nexttile;
yyaxis left;
p1 = plot(robust_B(:,1), robust_B(:,2), 'o-b', 'LineWidth', 1.5, 'MarkerSize', 6); hold on;
p2 = plot(robust_B(:,1), robust_B(:,3), 's-r', 'LineWidth', 1.5, 'MarkerSize', 6);
ylabel('PSLR / ISLR (dB)');

yyaxis right;
p3 = plot(robust_B(:,1), robust_B(:,4), '^-m', 'LineWidth', 1.5, 'MarkerSize', 6); hold on;
p4 = plot(robust_B(:,1), robust_B(:,5), 'd-k', 'LineWidth', 1.5, 'MarkerSize', 6);
ylabel('Mainlobe width (us) / PAPR (dB)');

xlabel('B (MHz)');
title('A: Performance vs B');
grid on;
set(gca, 'XTick', B_sweep_MHz);
legend([p1 p2 p3 p4], {'PSLR','ISLR','Mainlobe width','PAPR'}, 'Location', 'best');

% 图B：不同T_pulse下性能变化
nexttile;
yyaxis left;
p1 = plot(robust_T(:,1), robust_T(:,2), 'o-b', 'LineWidth', 1.5, 'MarkerSize', 6); hold on;
p2 = plot(robust_T(:,1), robust_T(:,3), 's-r', 'LineWidth', 1.5, 'MarkerSize', 6);
ylabel('PSLR / ISLR (dB)');

yyaxis right;
p3 = plot(robust_T(:,1), robust_T(:,4), '^-m', 'LineWidth', 1.5, 'MarkerSize', 6); hold on;
p4 = plot(robust_T(:,1), robust_T(:,5), 'd-k', 'LineWidth', 1.5, 'MarkerSize', 6);
ylabel('Mainlobe width (us) / PAPR (dB)');

xlabel('T_{pulse} (us)');
title('B: Performance vs T_{pulse}');
grid on;
set(gca, 'XTick', Tp_sweep_us);
legend([p1 p2 p3 p4], {'PSLR','ISLR','Mainlobe width','PAPR'}, 'Location', 'best');

%% 修正后的辅助函数

function metrics = run_single_case_metrics_test(B, T_pulse, fs, alpha_reg, A_max)
N_pulse = round(T_pulse * fs);
N_fft = 2^nextpow2(N_pulse * 8);
freq = (-N_fft/2:N_fft/2-1)' * (fs/N_fft);
f_center_idx = find(abs(freq)<=B/2);

H_mag = zeros(N_fft, 1);
H_mag(f_center_idx) = 0.7 + 0.4*cos(2*pi*freq(f_center_idx)/B*6) .* ...
                      exp(-(freq(f_center_idx)/(0.5*B)).^2) + ...
                      0.1*sin(2*pi*freq(f_center_idx)/B*3);
H_phase = zeros(N_fft, 1);
H_phase(f_center_idx) = 0.4*pi*(freq(f_center_idx)/B).^3 + ...
                        0.3*pi*sin(2*pi*freq(f_center_idx)/B*5) + ...
                        0.05*pi*(freq(f_center_idx)/B).^2;
H_k = fftshift(H_mag .* exp(1j*H_phase));

t = (-N_pulse/2:N_pulse/2-1)' / fs;
k_chirp = B / T_pulse;
s_lfm = exp(1j * pi * k_chirp * t.^2);
s_lfm_padded = zeros(N_fft, 1);
s_lfm_padded(1:N_pulse) = s_lfm;
S_LFM_k = fft(s_lfm_padded, N_fft);
R_k = S_LFM_k;

epsilon = alpha_reg / mean(abs(H_k .* S_LFM_k));
G_tx_k = R_k ./ (H_k .* S_LFM_k + epsilon);
G_tx_mag = abs(G_tx_k);
if max(G_tx_mag) > A_max
    G_tx_k = G_tx_k ./ max(1, G_tx_mag/A_max);
end

f_idx = find(abs(freq)<=B);
hamming_window = zeros(N_fft, 1);
hamming_window(f_idx) = hamming(length(f_idx));
W_k = fftshift(hamming_window);

S_tx_k = S_LFM_k .* (G_tx_k .* W_k);
s_tx_time = ifft(S_tx_k, N_fft);
S_tx_with_H = fft(s_tx_time, N_fft) .* H_k;
s_tx_with_H_time = ifft(S_tx_with_H, N_fft);
s_tx_with_H_pulse = s_tx_with_H_time(1:N_pulse);
s_tx_with_H_pulse = s_tx_with_H_pulse / sqrt(sum(abs(s_tx_with_H_pulse).^2));

auto_corr = xcorr(s_tx_with_H_pulse, s_tx_with_H_pulse);
auto_corr = auto_corr / max(abs(auto_corr));
peak_idx = ceil(length(auto_corr)/2);

metrics.pslr_with_comp = compute_pslr_corrected(auto_corr, peak_idx, fs, B);
metrics.islr_with_comp = compute_islr_corrected(auto_corr, peak_idx, fs, B);
metrics.mlw_with_comp = compute_3db_width_corrected(auto_corr, peak_idx, fs, B);
metrics.papr_with_comp = 10*log10(max(abs(s_tx_with_H_pulse).^2) / mean(abs(s_tx_with_H_pulse).^2));
end

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

function E_spec = compute_spectrum_error(S_ideal_k, S_out_k, freq, B)
% win_length.m步骤4.1中使用的带内归一化频谱相对误差
band_idx = abs(freq)<=B/2;
A = abs(S_ideal_k(band_idx));
Bv = abs(S_out_k(band_idx));
A = A / (norm(A,2) + eps);
Bv = Bv / (norm(Bv,2) + eps);
E_spec = (norm(Bv - A, 2)^2) / (norm(A, 2)^2 + eps);
end

function err = evaluate_reference_error(s_cmp, S_ideal_k, N_pulse, N_fft, freq, B)
% win_length.m步骤4.1中用于restoration error和time error的计算方式
s_ideal = ifft(S_ideal_k, N_fft);
s_ideal = s_ideal(1:N_pulse);
s_ideal = s_ideal / sqrt(sum(abs(s_ideal).^2) + eps);
s_cmp = s_cmp / sqrt(sum(abs(s_cmp).^2) + eps);

err.time_nmse = norm(s_cmp - s_ideal, 2)^2 / (norm(s_ideal,2)^2 + eps);

s_cmp_pad = zeros(N_fft,1);
s_cmp_pad(1:N_pulse) = s_cmp;
S_cmp = fft(s_cmp_pad, N_fft);
err.spec_nmse = compute_spectrum_error(S_ideal_k, S_cmp, freq, B);
end

