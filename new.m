%% 联合优化版发射端 LFM 预补偿（alpha_reg + 广义余弦窗参数）
% 说明：
% 1) 保留原始系统建模和指标分析物理意义
% 2) 将“固定 Hamming 窗 + 固定 alpha_reg”改为“alpha_reg, a1, a2 联合优化”
% 3) 默认采用两阶段网格搜索；可选 fmincon 精细优化

clear; close all; clc;

%% ======================= 0. 参数设置 =======================
cfg = struct();
% 雷达与采样参数
cfg.B = 20e6;                 % 带宽 (Hz)
cfg.T_pulse = 50e-6;          % 脉宽 (s)
cfg.fs = 60e6;                % 采样率 (Hz)
cfg.N_pulse = round(cfg.T_pulse * cfg.fs);
cfg.N_fft = 2^nextpow2(cfg.N_pulse * 8);
cfg.freq = (-cfg.N_fft/2:cfg.N_fft/2-1)' * (cfg.fs/cfg.N_fft);

% 预补偿参数
cfg.A_max = 4.0;              % 增益限制
cfg.alpha_baseline = 1e-2;    % 基线方案固定 alpha

% 约束与优化设置
cfg.bounds.alpha = [1e-6, 1e-1];
cfg.bounds.a1    = [0, 1];
cfg.bounds.a2    = [0, 1];
cfg.enable_a0_nonnegative = true;  % 强制 a0 = 1-a1-a2 >= 0
cfg.enable_fmincon = true;          % 是否尝试 fmincon（二阶段后）

% 指标目标与权重（可按论文需求调节）
cfg.targets.pslr = -25;       % dB，越负越好
cfg.targets.islr = -15;       % dB，越负越好
cfg.targets.papr = 5.5;       % dB，参考阈值
cfg.weights = struct('pslr',1.0,'islr',1.0,'bw',0.8,'papr',0.5,'spec',1.2);

% 定义“有效带宽”用于谱误差
cfg.band_limit = cfg.B/2;

%% ======================= 1. 系统模型与理想 LFM =======================
H_k = build_system_response(cfg.freq, cfg.B, cfg.N_fft);
[s_lfm, t_pulse, S_ideal_k] = generate_lfm_signal(cfg.B, cfg.T_pulse, cfg.fs, cfg.N_pulse, cfg.N_fft);

% 系统响应图
figure('Position',[100,100,1200,420]);
subplot(1,2,1);
plot(cfg.freq/1e6, abs(H_k), 'LineWidth',1.2); grid on;
xlabel('频率 (MHz)'); ylabel('|H(f)|'); title('系统幅频响应 H(f)'); xlim([-1.5 1.5]*cfg.B/1e6);
subplot(1,2,2);
plot(cfg.freq/1e6, unwrap(angle(H_k))*180/pi, 'LineWidth',1.2); grid on;
xlabel('频率 (MHz)'); ylabel('相位 (deg)'); title('系统相频响应 \angleH(f)'); xlim([-1.5 1.5]*cfg.B/1e6);

%% ======================= 2. 基线方案（固定 alpha + Hamming） =======================
base_params = struct('alpha_reg',cfg.alpha_baseline,'a1',0.46,'a2',0,'a0',0.54,'window_type','hamming');
base_out = simulate_transmit_signal(base_params, S_ideal_k, H_k, cfg, 'hamming');
base_metrics = evaluate_metrics(base_out.s_ideal, base_out.s_no_comp, base_out.s_with_comp, cfg);
base_spec_err = compute_spectrum_error(base_out.S_out_k, S_ideal_k, cfg.freq, cfg.band_limit);
base_metrics.spec_error = base_spec_err;

%% ======================= 3. 联合优化（网格搜索 + 可选 fmincon） =======================
[best_params, best_obj, best_out, best_metrics] = optimize_joint_parameters(S_ideal_k, H_k, cfg);

%% ======================= 4. 输出最优结果与对比表 =======================
fprintf('\n========== 联合优化结果 ==========' );
fprintf('\n最优 alpha_reg = %.6g', best_params.alpha_reg);
fprintf('\n最优 a1 = %.4f, a2 = %.4f, a0 = %.4f', best_params.a1, best_params.a2, best_params.a0);
fprintf('\n最优目标函数值 J* = %.6f\n', best_obj);

opt_spec_err = compute_spectrum_error(best_out.S_out_k, S_ideal_k, cfg.freq, cfg.band_limit);
best_metrics.spec_error = opt_spec_err;

result_names = {'PSLR(dB)','ISLR(dB)','3dB宽度(us)','PAPR(dB)','频谱恢复误差'};
ideal_row = [best_metrics.ideal.pslr, best_metrics.ideal.islr, best_metrics.ideal.bw3db, best_metrics.ideal.papr, 0];
no_comp_row = [best_metrics.no_comp.pslr, best_metrics.no_comp.islr, best_metrics.no_comp.bw3db, best_metrics.no_comp.papr, ...
               compute_spectrum_error(best_out.S_no_comp_out_k, S_ideal_k, cfg.freq, cfg.band_limit)];
base_row = [base_metrics.with_comp.pslr, base_metrics.with_comp.islr, base_metrics.with_comp.bw3db, base_metrics.with_comp.papr, base_metrics.spec_error];
opt_row  = [best_metrics.with_comp.pslr, best_metrics.with_comp.islr, best_metrics.with_comp.bw3db, best_metrics.with_comp.papr, best_metrics.spec_error];

T = array2table([ideal_row; no_comp_row; base_row; opt_row], ...
    'VariableNames', matlab.lang.makeValidName(result_names), ...
    'RowNames', {'IdealLFM','NoComp','Baseline_Hamming','JointOptimized'});

disp('--- 优化前后性能对比表 ---');
disp(T);

%% ======================= 5. 图形输出扩展 =======================
% (1) 最优补偿滤波器幅相
figure('Position',[120,120,1200,420]);
subplot(1,2,1);
plot(cfg.freq/1e6, 20*log10(fftshift(abs(best_out.G_tx_k_windowed))/max(abs(best_out.G_tx_k_windowed)+eps)), 'LineWidth',1.2);
grid on; xlabel('频率 (MHz)'); ylabel('归一化幅度 (dB)'); title('优化后补偿滤波器幅频响应'); xlim([-1.5 1.5]*cfg.B/1e6);
subplot(1,2,2);
plot(cfg.freq/1e6, fftshift(unwrap(angle(best_out.G_tx_k_windowed)))*180/pi, 'LineWidth',1.2);
grid on; xlabel('频率 (MHz)'); ylabel('相位 (deg)'); title('优化后补偿滤波器相频响应'); xlim([-1.5 1.5]*cfg.B/1e6);

% (2) 频谱对比：理想 / 无预补偿输出 / 联合优化输出
figure('Position',[140,140,1200,420]);
S_ideal_mag = fftshift(abs(S_ideal_k));
S_no_comp_mag = fftshift(abs(best_out.S_no_comp_out_k));
S_opt_mag = fftshift(abs(best_out.S_out_k));
plot(cfg.freq/1e6, 20*log10(S_ideal_mag/max(S_ideal_mag)+eps), 'k-', 'LineWidth',1.3); hold on;
plot(cfg.freq/1e6, 20*log10(S_no_comp_mag/max(S_no_comp_mag)+eps), 'r--', 'LineWidth',1.2);
plot(cfg.freq/1e6, 20*log10(S_opt_mag/max(S_opt_mag)+eps), 'b-.', 'LineWidth',1.2);
grid on; xlabel('频率 (MHz)'); ylabel('幅度 (dB)'); xlim([-1.5 1.5]*cfg.B/1e6);
legend('Ideal LFM', 'Distorted output without compensation', 'Jointly optimized output', 'Location','best');
title('频谱对比');

% (3) 自相关对比（线性与 dB）
[corr_ideal, corr_nocomp, corr_opt, tau_us] = best_out.corr_pack{:};
center_idx = (numel(corr_ideal)+1)/2;
figure('Position',[160,160,1250,460]);
subplot(1,2,1);
plot(tau_us, abs(corr_ideal), 'k-', 'LineWidth',1.2); hold on;
plot(tau_us, abs(corr_nocomp), 'r--', 'LineWidth',1.2);
plot(tau_us, abs(corr_opt), 'b-.', 'LineWidth',1.2);
grid on; xlabel('时延 (us)'); ylabel('|自相关|'); title('自相关函数（线性）');
legend('Ideal LFM','No Compensation','Joint Optimized','Location','best');

subplot(1,2,2);
plot(tau_us, 20*log10(abs(corr_ideal)/max(abs(corr_ideal))+eps), 'k-', 'LineWidth',1.2); hold on;
plot(tau_us, 20*log10(abs(corr_nocomp)/max(abs(corr_nocomp))+eps), 'r--', 'LineWidth',1.2);
plot(tau_us, 20*log10(abs(corr_opt)/max(abs(corr_opt))+eps), 'b-.', 'LineWidth',1.2);
grid on; xlabel('时延 (us)'); ylabel('归一化幅度 (dB)'); title('自相关函数（dB）'); ylim([-80 5]);
legend('Ideal LFM','No Compensation','Joint Optimized','Location','best');

% (4) 指标柱状图
figure('Position',[180,180,1250,460]);
metrics_mat = [no_comp_row; base_row; opt_row];
bar(metrics_mat);
set(gca,'XTickLabel',{'NoComp','Baseline(Hamming)','JointOptimized'});
legend(result_names,'Location','bestoutside');
grid on; title('PSLR/ISLR/3dB宽度/PAPR/频谱恢复误差 对比'); ylabel('指标值');

% 单独显示 PSLR/ISLR 主瓣排除区（检查合理性）
mainlobe_exclude = ceil(2 * cfg.fs / cfg.B);
idx_side = [1:center_idx-mainlobe_exclude, center_idx+mainlobe_exclude:numel(corr_opt)];
figure('Position',[200,200,1200,420]);
plot(tau_us(idx_side), 20*log10(abs(corr_ideal(idx_side))/max(abs(corr_ideal))+eps),'k-','LineWidth',1.2); hold on;
plot(tau_us(idx_side), 20*log10(abs(corr_nocomp(idx_side))/max(abs(corr_nocomp))+eps),'r--','LineWidth',1.2);
plot(tau_us(idx_side), 20*log10(abs(corr_opt(idx_side))/max(abs(corr_opt))+eps),'b-.','LineWidth',1.2);
grid on; xlabel('时延 (us)'); ylabel('旁瓣幅度 (dB)');
title(sprintf('旁瓣对比（主瓣排除区 = \pm%d 点）', mainlobe_exclude));
legend('Ideal','NoComp','JointOptimized','Location','best');

%% ======================= 局部函数定义 =======================
function [s_lfm, t, S_ideal_k] = generate_lfm_signal(B, T_pulse, fs, N_pulse, N_fft)
    t = (-N_pulse/2:N_pulse/2-1)'/fs;
    k_chirp = B/T_pulse;
    s_lfm = exp(1j*pi*k_chirp*t.^2);
    s_pad = zeros(N_fft,1);
    s_pad(1:N_pulse) = s_lfm;
    S_ideal_k = fft(s_pad, N_fft);
end

function H_k = build_system_response(freq, B, N_fft)
    idx = abs(freq)<=B/2;
    H_mag = zeros(N_fft,1);
    H_mag(idx) = 0.7 + 0.4*cos(2*pi*freq(idx)/B*6).*exp(-(freq(idx)/(0.5*B)).^2) + 0.1*sin(2*pi*freq(idx)/B*3);
    H_phase = zeros(N_fft,1);
    H_phase(idx) = 0.4*pi*(freq(idx)/B).^3 + 0.3*pi*sin(2*pi*freq(idx)/B*5) + 0.05*pi*(freq(idx)/B).^2;
    H_k = fftshift(H_mag .* exp(1j*H_phase));
end

function G_tx_k = compute_precomp_filter(S_ideal_k, H_k, alpha_reg, A_max)
    denom_scale = mean(abs(H_k .* S_ideal_k)) + eps;
    epsilon = alpha_reg / denom_scale;
    G_tx_k = S_ideal_k ./ (H_k .* S_ideal_k + epsilon);

    mag = abs(G_tx_k);
    if any(~isfinite(mag))
        G_tx_k(:) = NaN;
        return;
    end
    if max(mag) > A_max
        G_tx_k = G_tx_k ./ max(1, mag/A_max);
    end
end

function W_k = generate_generalized_cosine_window(N_fft, freq, B, a1, a2)
    % 广义余弦窗：a0 = 1-a1-a2, w(n) = a0 - a1*cos(...) + a2*cos(...)
    a0 = 1 - a1 - a2;
    idx = find(abs(freq)<=B);
    Nw = numel(idx);
    n = (0:Nw-1)';
    if Nw <= 1
        w_local = ones(Nw,1);
    else
        w_local = a0 - a1*cos(2*pi*n/(Nw-1)) + a2*cos(4*pi*n/(Nw-1));
    end
    w_local = max(0, real(w_local));
    W_unshift = zeros(N_fft,1);
    W_unshift(idx) = w_local;
    W_k = fftshift(W_unshift);
end

function out = simulate_transmit_signal(params, S_ideal_k, H_k, cfg, window_mode)
    % 生成预补偿滤波器
    G_tx_k = compute_precomp_filter(S_ideal_k, H_k, params.alpha_reg, cfg.A_max);
    if any(~isfinite(G_tx_k))
        out = struct('valid',false);
        return;
    end

    % 窗函数
    switch lower(window_mode)
        case 'hamming'
            idx = find(abs(cfg.freq)<=cfg.B);
            w = zeros(cfg.N_fft,1);
            w(idx) = hamming(numel(idx));
            W_k = fftshift(w);
        otherwise
            W_k = generate_generalized_cosine_window(cfg.N_fft, cfg.freq, cfg.B, params.a1, params.a2);
    end

    G_tx_k_windowed = G_tx_k .* W_k;
    S_tx_k = S_ideal_k .* G_tx_k_windowed;
    s_tx_time = ifft(S_tx_k, cfg.N_fft);
    s_tx_pulse = s_tx_time(1:cfg.N_pulse);
    s_tx_pulse = s_tx_pulse / sqrt(sum(abs(s_tx_pulse).^2)+eps);

    % 理想与无补偿输出
    s_ideal_time = ifft(S_ideal_k, cfg.N_fft);
    s_ideal = s_ideal_time(1:cfg.N_pulse);
    s_ideal = s_ideal / sqrt(sum(abs(s_ideal).^2)+eps);

    S_no_comp_out_k = S_ideal_k .* H_k;
    s_no_comp_time = ifft(S_no_comp_out_k, cfg.N_fft);
    s_no_comp = s_no_comp_time(1:cfg.N_pulse);
    s_no_comp = s_no_comp / sqrt(sum(abs(s_no_comp).^2)+eps);

    % 有补偿+系统输出
    S_out_k = fft(s_tx_time, cfg.N_fft) .* H_k;
    s_out_time = ifft(S_out_k, cfg.N_fft);
    s_with_comp = s_out_time(1:cfg.N_pulse);
    s_with_comp = s_with_comp / sqrt(sum(abs(s_with_comp).^2)+eps);

    corr_ideal = xcorr(s_ideal, s_ideal);
    corr_nocomp = xcorr(s_no_comp, s_no_comp);
    corr_comp = xcorr(s_with_comp, s_with_comp);
    tau_us = (-cfg.N_pulse+1:cfg.N_pulse-1)'/cfg.fs*1e6;

    out = struct();
    out.valid = all(isfinite([real(s_tx_pulse); imag(s_tx_pulse)]));
    out.G_tx_k = G_tx_k;
    out.W_k = W_k;
    out.G_tx_k_windowed = G_tx_k_windowed;
    out.S_tx_k = S_tx_k;
    out.s_tx_pulse = s_tx_pulse;
    out.s_ideal = s_ideal;
    out.s_no_comp = s_no_comp;
    out.s_with_comp = s_with_comp;
    out.S_out_k = S_out_k;
    out.S_no_comp_out_k = S_no_comp_out_k;
    out.corr_pack = {corr_ideal, corr_nocomp, corr_comp, tau_us};
end

function metrics = evaluate_metrics(s_ideal, s_no_comp, s_with_comp, cfg)
    corr_ideal = xcorr(s_ideal, s_ideal);
    corr_no = xcorr(s_no_comp, s_no_comp);
    corr_yes = xcorr(s_with_comp, s_with_comp);
    peak_idx = (numel(corr_ideal)+1)/2;

    metrics.ideal.pslr = compute_pslr_corrected(corr_ideal, peak_idx, cfg.fs, cfg.B);
    metrics.no_comp.pslr = compute_pslr_corrected(corr_no, peak_idx, cfg.fs, cfg.B);
    metrics.with_comp.pslr = compute_pslr_corrected(corr_yes, peak_idx, cfg.fs, cfg.B);

    metrics.ideal.islr = compute_islr_corrected(corr_ideal, peak_idx, cfg.fs, cfg.B);
    metrics.no_comp.islr = compute_islr_corrected(corr_no, peak_idx, cfg.fs, cfg.B);
    metrics.with_comp.islr = compute_islr_corrected(corr_yes, peak_idx, cfg.fs, cfg.B);

    metrics.ideal.bw3db = compute_3db_width_corrected(corr_ideal, peak_idx, cfg.fs, cfg.B);
    metrics.no_comp.bw3db = compute_3db_width_corrected(corr_no, peak_idx, cfg.fs, cfg.B);
    metrics.with_comp.bw3db = compute_3db_width_corrected(corr_yes, peak_idx, cfg.fs, cfg.B);

    metrics.ideal.papr = 10*log10(max(abs(s_ideal).^2)/(mean(abs(s_ideal).^2)+eps));
    metrics.no_comp.papr = 10*log10(max(abs(s_no_comp).^2)/(mean(abs(s_no_comp).^2)+eps));
    metrics.with_comp.papr = 10*log10(max(abs(s_with_comp).^2)/(mean(abs(s_with_comp).^2)+eps));
end

function E_spec = compute_spectrum_error(S_out_k, S_ideal_k, freq, band_limit)
    % 方案A：有效带宽内幅度归一化均方误差（稳健，避免相位包裹影响）
    idx = abs(freq)<=band_limit;
    A = abs(S_out_k(idx));
    B = abs(S_ideal_k(idx));
    denom = norm(B,2)^2 + eps;
    E_spec = norm(A-B,2)^2 / denom;
end

function J = objective_function(params, S_ideal_k, H_k, cfg)
    big_penalty = 1e6;
    alpha_reg = params(1); a1 = params(2); a2 = params(3);
    a0 = 1 - a1 - a2;

    % 参数约束
    if alpha_reg <= 0 || a1<cfg.bounds.a1(1) || a1>cfg.bounds.a1(2) || a2<cfg.bounds.a2(1) || a2>cfg.bounds.a2(2)
        J = big_penalty; return;
    end
    if cfg.enable_a0_nonnegative && a0 < 0
        J = big_penalty; return;
    end

    p = struct('alpha_reg',alpha_reg,'a1',a1,'a2',a2,'a0',a0);
    sim = simulate_transmit_signal(p, S_ideal_k, H_k, cfg, 'generalized');
    if ~sim.valid || any(~isfinite(sim.G_tx_k))
        J = big_penalty; return;
    end

    M = evaluate_metrics(sim.s_ideal, sim.s_no_comp, sim.s_with_comp, cfg);
    spec_err = compute_spectrum_error(sim.S_out_k, S_ideal_k, cfg.freq, cfg.band_limit);

    if any(~isfinite([M.with_comp.pslr, M.with_comp.islr, M.with_comp.bw3db, M.with_comp.papr, spec_err]))
        J = big_penalty; return;
    end

    % 惩罚定义（均转为“越小越好”的正值）
    pslr_pen = max(0, M.with_comp.pslr - cfg.targets.pslr)^2; % PSLR越负越好
    islr_pen = max(0, M.with_comp.islr - cfg.targets.islr)^2; % ISLR越负越好
    bw_pen = max(0, (M.with_comp.bw3db - M.ideal.bw3db)/(M.ideal.bw3db+eps))^2;
    papr_pen = max(0, M.with_comp.papr - cfg.targets.papr)^2;
    spec_pen = spec_err;

    J = cfg.weights.pslr*pslr_pen + cfg.weights.islr*islr_pen + cfg.weights.bw*bw_pen + ...
        cfg.weights.papr*papr_pen + cfg.weights.spec*spec_pen;

    if ~isfinite(J)
        J = big_penalty;
    end
end

function [best_params, best_obj, best_out, best_metrics] = optimize_joint_parameters(S_ideal_k, H_k, cfg)
    fprintf('\n=== 开始两阶段网格搜索 ===\n');

    % ---------- 第一阶段：粗搜索 ----------
    alpha_grid_1 = logspace(log10(cfg.bounds.alpha(1)), log10(cfg.bounds.alpha(2)), 8);
    a1_grid_1 = linspace(cfg.bounds.a1(1), cfg.bounds.a1(2), 8);
    a2_grid_1 = linspace(cfg.bounds.a2(1), cfg.bounds.a2(2), 8);

    [best_obj, best_vec] = grid_search(alpha_grid_1, a1_grid_1, a2_grid_1, S_ideal_k, H_k, cfg);

    % ---------- 第二阶段：细搜索 ----------
    alpha_c = best_vec(1); a1_c = best_vec(2); a2_c = best_vec(3);
    alpha_grid_2 = logspace(log10(max(cfg.bounds.alpha(1), alpha_c/3)), log10(min(cfg.bounds.alpha(2), alpha_c*3)), 8);
    a1_grid_2 = linspace(max(cfg.bounds.a1(1), a1_c-0.15), min(cfg.bounds.a1(2), a1_c+0.15), 8);
    a2_grid_2 = linspace(max(cfg.bounds.a2(1), a2_c-0.15), min(cfg.bounds.a2(2), a2_c+0.15), 8);

    [best_obj_2, best_vec_2] = grid_search(alpha_grid_2, a1_grid_2, a2_grid_2, S_ideal_k, H_k, cfg);
    if best_obj_2 < best_obj
        best_obj = best_obj_2;
        best_vec = best_vec_2;
    end

    % ---------- 可选：fmincon 精细优化 ----------
    if cfg.enable_fmincon && exist('fmincon','file') == 2
        fprintf('=== 尝试 fmincon 精细优化 ===\n');
        x0 = best_vec(:)';
        lb = [cfg.bounds.alpha(1), cfg.bounds.a1(1), cfg.bounds.a2(1)];
        ub = [cfg.bounds.alpha(2), cfg.bounds.a1(2), cfg.bounds.a2(2)];
        nonlcon = [];
        if cfg.enable_a0_nonnegative
            nonlcon = @(x) deal(x(2)+x(3)-1,[]); % c(x)<=0 => a1+a2<=1
        end
        opts = optimoptions('fmincon','Display','off','Algorithm','sqp','MaxFunctionEvaluations',300);
        try
            [x_fm, fval_fm, exitflag] = fmincon(@(x)objective_function(x,S_ideal_k,H_k,cfg), x0, [],[],[],[],lb,ub,nonlcon,opts);
            if exitflag > 0 && isfinite(fval_fm) && fval_fm < best_obj
                best_obj = fval_fm;
                best_vec = x_fm;
            end
        catch
            fprintf('fmincon 失败，回退至网格搜索最优结果。\n');
        end
    else
        fprintf('=== 未使用 fmincon（工具箱不可用或已禁用）===\n');
    end

    best_params = struct('alpha_reg',best_vec(1),'a1',best_vec(2),'a2',best_vec(3),'a0',1-best_vec(2)-best_vec(3));
    best_out = simulate_transmit_signal(best_params, S_ideal_k, H_k, cfg, 'generalized');
    best_metrics = evaluate_metrics(best_out.s_ideal, best_out.s_no_comp, best_out.s_with_comp, cfg);
end

function [best_obj, best_vec] = grid_search(alpha_grid, a1_grid, a2_grid, S_ideal_k, H_k, cfg)
    best_obj = inf;
    best_vec = [alpha_grid(1), a1_grid(1), a2_grid(1)];
    total = numel(alpha_grid)*numel(a1_grid)*numel(a2_grid);
    k = 0;

    for ia = 1:numel(alpha_grid)
        for i1 = 1:numel(a1_grid)
            for i2 = 1:numel(a2_grid)
                k = k + 1;
                x = [alpha_grid(ia), a1_grid(i1), a2_grid(i2)];
                J = objective_function(x, S_ideal_k, H_k, cfg);
                if J < best_obj
                    best_obj = J;
                    best_vec = x;
                end
            end
        end
        fprintf('网格搜索进度: %d/%d, 当前最优J=%.4f\n', k, total, best_obj);
    end
end

function pslr = compute_pslr_corrected(corr_signal, peak_idx, fs, B)
    mainlobe_width_samples = ceil(2 * fs / B);
    mainlobe_start = max(1, peak_idx - mainlobe_width_samples);
    mainlobe_end = min(length(corr_signal), peak_idx + mainlobe_width_samples);

    mask = true(length(corr_signal), 1);
    mask(mainlobe_start:mainlobe_end) = false;
    if ~any(mask)
        pslr = -5;
        return;
    end

    sidelobe_peak = max(abs(corr_signal(mask)));
    mainlobe_peak = abs(corr_signal(peak_idx));
    if mainlobe_peak > 0
        pslr = 20*log10(sidelobe_peak / mainlobe_peak + eps);
    else
        pslr = -inf;
    end
end

function islr = compute_islr_corrected(corr_signal, peak_idx, fs, B)
    mainlobe_width_samples = ceil(2 * fs / B);
    mainlobe_start = max(1, peak_idx - mainlobe_width_samples);
    mainlobe_end = min(length(corr_signal), peak_idx + mainlobe_width_samples);

    mainlobe_energy = sum(abs(corr_signal(mainlobe_start:mainlobe_end)).^2);
    total_energy = sum(abs(corr_signal).^2);
    sidelobe_energy = max(0, total_energy - mainlobe_energy);

    if mainlobe_energy > 0 && sidelobe_energy > 0
        islr = 10*log10(sidelobe_energy / mainlobe_energy + eps);
    else
        islr = -inf;
    end
end

function bw_3db = compute_3db_width_corrected(corr_signal, peak_idx, fs, B)
    corr_mag = abs(corr_signal);
    peak_value = corr_mag(peak_idx);
    thr = peak_value / sqrt(2);

    left = peak_idx;
    while left > 1 && corr_mag(left) >= thr
        left = left - 1;
    end

    right = peak_idx;
    while right < length(corr_mag) && corr_mag(right) >= thr
        right = right + 1;
    end

    bw_3db = (right - left) / fs * 1e6;
    if bw_3db <= 0 || bw_3db > 100
        bw_3db = 0.886 / B * 1e6;
    end
end
