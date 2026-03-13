clc; clear; close all;

%% 联合优化版：预补偿正则化因子 + 广义余弦窗参数
% 保留原始物理建模逻辑，并将“固定Hamming+固定alpha”改为联合优化框架。

%% 1) 基本参数
cfg.B = 20e6;                 % LFM带宽
cfg.T_pulse = 50e-6;          % 脉宽
cfg.fs = 60e6;                % 采样率
cfg.N_pulse = round(cfg.T_pulse * cfg.fs);
cfg.N_fft = 2^nextpow2(cfg.N_pulse * 8);
cfg.freq = (-cfg.N_fft/2:cfg.N_fft/2-1)' * (cfg.fs/cfg.N_fft);
cfg.A_max = 4.0;              % 增益上限，避免补偿滤波器爆炸

% 评价与优化配置
cfg.pslr_target = -25;        % dB，越小越好
cfg.islr_target = -15;        % dB，越小越好
cfg.papr_ref = 5.5;           % dB，参考上限
cfg.mainlobe_guard = [];      % 留空则自动按-3dB主瓣估计

% 权重（可调）
weights.pslr = 1.0;
weights.islr = 1.0;
weights.bw   = 0.8;
weights.papr = 0.5;
weights.spec = 1.2;

%% 2) 生成理想LFM与系统响应
[s_lfm, S_ideal_k, t] = generate_lfm_signal(cfg.B, cfg.T_pulse, cfg.fs, cfg.N_pulse, cfg.N_fft);
H_k = build_system_response(cfg.freq, cfg.B, cfg.N_fft);

% 理想匹配参考（用于BW展宽惩罚）
ideal_case = simulate_transmit_signal(S_ideal_k, H_k, ones(cfg.N_fft,1), cfg.N_pulse, cfg.N_fft);
metrics_ideal = evaluate_metrics(ideal_case.s_out, cfg.fs, cfg.mainlobe_guard);

%% 3) 基准方案：固定alpha + Hamming窗（与原流程一致）
baseline.alpha_reg = 1e-2;
baseline.a1 = 0.46;              % 近似Hamming
baseline.a2 = 0.00;
baseline.a0 = 1 - baseline.a1 - baseline.a2;

W_base = generate_generalized_cosine_window(cfg.N_fft, baseline.a1, baseline.a2, cfg.B, cfg.freq);
G_base = compute_precomp_filter(S_ideal_k, H_k, baseline.alpha_reg, cfg.A_max);
base_case = simulate_transmit_signal(S_ideal_k, H_k, G_base .* W_base, cfg.N_pulse, cfg.N_fft);
base_metrics = evaluate_metrics(base_case.s_out, cfg.fs, cfg.mainlobe_guard);
base_metrics.papr_db = 10*log10(max(abs(base_case.s_tx).^2) / mean(abs(base_case.s_tx).^2));
base_metrics.spec_error = compute_spectrum_error(S_ideal_k, base_case.S_out_k, cfg.freq, cfg.B);

%% 4) 联合优化：网格粗细两阶段搜索
opt = optimize_joint_parameters(S_ideal_k, H_k, cfg, weights, metrics_ideal);

% 可选：fmincon精修
opt_fmincon = [];
if exist('fmincon', 'file') == 2
    try
        x0 = [opt.alpha_reg, opt.a1, opt.a2];
        lb = [1e-6, 0, 0];
        ub = [1, 1, 1];
        A = [0, 1, 1]; b = 1;   % a1 + a2 <= 1 -> a0>=0
        fobj = @(x) objective_function(x, S_ideal_k, H_k, cfg, weights, metrics_ideal);
        options = optimoptions('fmincon','Display','none','Algorithm','sqp',...
            'MaxIterations',80,'OptimalityTolerance',1e-5,'StepTolerance',1e-7);
        [xopt, fval, exitflag] = fmincon(fobj, x0, A, b, [], [], lb, ub, [], options);
        if exitflag > 0
            opt_fmincon = opt;
            opt_fmincon.alpha_reg = xopt(1);
            opt_fmincon.a1 = xopt(2);
            opt_fmincon.a2 = xopt(3);
            opt_fmincon.a0 = 1 - xopt(2) - xopt(3);
            opt_fmincon.J = fval;
            [~, sim_best, met_best] = objective_function(xopt, S_ideal_k, H_k, cfg, weights, metrics_ideal);
            opt_fmincon.sim = sim_best;
            opt_fmincon.metrics = met_best;
        end
    catch
        % fmincon不可用或失败时直接保留网格搜索结果
    end
end

if ~isempty(opt_fmincon)
    final_opt = opt_fmincon;
    method_name = 'Grid + fmincon';
else
    final_opt = opt;
    method_name = 'Grid(粗细两阶段)';
end

%% 5) 无补偿情况（用于对比）
no_comp_case = simulate_transmit_signal(S_ideal_k, H_k, ones(cfg.N_fft,1), cfg.N_pulse, cfg.N_fft);
no_comp_metrics = evaluate_metrics(no_comp_case.s_out, cfg.fs, cfg.mainlobe_guard);
no_comp_metrics.papr_db = 10*log10(max(abs(no_comp_case.s_tx).^2) / mean(abs(no_comp_case.s_tx).^2));
no_comp_metrics.spec_error = compute_spectrum_error(S_ideal_k, no_comp_case.S_out_k, cfg.freq, cfg.B);

% 理想项补全便于表格输出
metrics_ideal.papr_db = 10*log10(max(abs(ideal_case.s_tx).^2)/mean(abs(ideal_case.s_tx).^2));
metrics_ideal.spec_error = compute_spectrum_error(S_ideal_k, ideal_case.S_out_k, cfg.freq, cfg.B);

%% 6) 结果打印与对比表
fprintf('\n=== 联合优化结果 (%s) ===\n', method_name);
fprintf('最优 alpha_reg = %.6g\n', final_opt.alpha_reg);
fprintf('最优 a1 = %.4f, a2 = %.4f, a0 = %.4f\n', final_opt.a1, final_opt.a2, final_opt.a0);
fprintf('最优目标函数 J* = %.6f\n', final_opt.J);

Row = {'Ideal LFM'; 'No Compensation'; 'Baseline(Hamming)'; 'Joint Optimized'};
PSLR = [metrics_ideal.pslr_db; no_comp_metrics.pslr_db; base_metrics.pslr_db; final_opt.metrics.pslr_db];
ISLR = [metrics_ideal.islr_db; no_comp_metrics.islr_db; base_metrics.islr_db; final_opt.metrics.islr_db];
BWus = [metrics_ideal.bw3db_s; no_comp_metrics.bw3db_s; base_metrics.bw3db_s; final_opt.metrics.bw3db_s] * 1e6;
PAPR = [metrics_ideal.papr_db; no_comp_metrics.papr_db; base_metrics.papr_db; final_opt.metrics.papr_db];
SPEC = [metrics_ideal.spec_error; no_comp_metrics.spec_error; base_metrics.spec_error; final_opt.metrics.spec_error];

T = table(Row, PSLR, ISLR, BWus, PAPR, SPEC, ...
    'VariableNames', {'Case','PSLR_dB','ISLR_dB','BW3dB_us','PAPR_dB','SpecErr_NMSE'});
disp(T);

%% 7) 绘图（保持与draw_figure.m相近风格并扩展）
S_ideal_mag = fftshift(abs(S_ideal_k));
S_no_mag = fftshift(abs(no_comp_case.S_out_k));
S_opt_mag = fftshift(abs(final_opt.sim.S_out_k));

figure(1);
plot(cfg.freq/1e6, 20*log10(abs(fftshift(H_k))+1e-12), 'LineWidth', 1.5); grid on;
xlabel('Frequency (MHz)'); ylabel('|H(f)| (dB)'); title('系统响应 H(f)');

figure(2);
subplot(2,1,1);
plot(cfg.freq/1e6, 20*log10(abs(fftshift(final_opt.sim.G_eff))+1e-12), 'b', 'LineWidth', 1.5); grid on;
ylabel('Magnitude (dB)'); title('优化后补偿滤波器幅相响应');
subplot(2,1,2);
plot(cfg.freq/1e6, unwrap(angle(fftshift(final_opt.sim.G_eff))), 'b', 'LineWidth', 1.5); grid on;
xlabel('Frequency (MHz)'); ylabel('Phase (rad)');

figure(3);
plot(cfg.freq/1e6, 20*log10(S_ideal_mag/max(S_ideal_mag)+1e-12), 'k-', 'LineWidth', 1.5); hold on;
plot(cfg.freq/1e6, 20*log10(S_no_mag/max(S_no_mag)+1e-12), 'r--', 'LineWidth', 1.5);
plot(cfg.freq/1e6, 20*log10(S_opt_mag/max(S_opt_mag)+1e-12), 'b-.', 'LineWidth', 1.5);
legend('Ideal LFM', 'Distorted(no compensation)', 'Joint optimized', 'Location','best');
xlabel('Frequency (MHz)'); ylabel('Amplitude (dB)'); xlim([-1.5*cfg.B,1.5*cfg.B]/1e6); ylim([-80 5]); grid on;
title('频谱对比');

% 自相关对比
[t_corr, ac_ideal, ac_no, ac_opt] = local_autocorr_compare(ideal_case.s_out, no_comp_case.s_out, final_opt.sim.s_out, cfg.fs);
center_idx = ceil(length(t_corr)/2); range_idx = center_idx-50:center_idx+50;

figure(4);
plot(t_corr(range_idx), abs(ac_ideal(range_idx)), 'k-', 'LineWidth', 1.5); hold on;
plot(t_corr(range_idx), abs(ac_no(range_idx)), 'r--', 'LineWidth', 1.5);
plot(t_corr(range_idx), abs(ac_opt(range_idx)), 'b-.', 'LineWidth', 1.5);
xlabel('Time Delay (\mus)'); ylabel('Normalized Amplitude'); grid on;
legend('Ideal LFM','No compensation','Joint optimized','Location','best');
title('自相关（线性）');

figure(5);
plot(t_corr, 20*log10(abs(ac_ideal)+1e-12), 'k-', 'LineWidth', 1.5); hold on;
plot(t_corr, 20*log10(abs(ac_no)+1e-12), 'r--', 'LineWidth', 1.5);
plot(t_corr, 20*log10(abs(ac_opt)+1e-12), 'b-.', 'LineWidth', 1.5);
xlabel('Time Delay (\mus)'); ylabel('Amplitude (dB)'); ylim([-120 0]); grid on;
legend('Ideal LFM','No compensation','Joint optimized','Location','best');
title('自相关（dB）');

figure(6);
metrics_bar = [no_comp_metrics.pslr_db, no_comp_metrics.islr_db, no_comp_metrics.bw3db_s*1e6, no_comp_metrics.papr_db, no_comp_metrics.spec_error; ...
               final_opt.metrics.pslr_db, final_opt.metrics.islr_db, final_opt.metrics.bw3db_s*1e6, final_opt.metrics.papr_db, final_opt.metrics.spec_error];
bar(metrics_bar); grid on;
set(gca, 'XTickLabel', {'No compensation','Joint optimized'});
legend({'PSLR(dB)','ISLR(dB)','BW3dB(\mus)','PAPR(dB)','SpecErr'}, 'Location', 'bestoutside');
title('关键性能指标柱状对比');


%% ======================= 本文件局部函数 =======================
function [s_lfm, S_lfm_k, t] = generate_lfm_signal(B, T_pulse, fs, N_pulse, N_fft)
    t = (-N_pulse/2:N_pulse/2-1)'/fs;
    k_chirp = B/T_pulse;
    s_lfm = exp(1j*pi*k_chirp*t.^2);
    s_pad = zeros(N_fft,1);
    s_pad(1:N_pulse) = s_lfm;
    S_lfm_k = fft(s_pad, N_fft);
end

function H_k = build_system_response(freq, B, N_fft)
    f_center_idx = find(abs(freq)<=B/2);
    H_mag = zeros(N_fft,1);
    H_mag(f_center_idx) = 0.7 + 0.4*cos(2*pi*freq(f_center_idx)/B*6) .* exp(-(freq(f_center_idx)/(0.5*B)).^2) ...
                          + 0.1*sin(2*pi*freq(f_center_idx)/B*3);
    H_phase = zeros(N_fft,1);
    H_phase(f_center_idx) = 0.4*pi*(freq(f_center_idx)/B).^3 + 0.3*pi*sin(2*pi*freq(f_center_idx)/B*5) ...
                            + 0.05*pi*(freq(f_center_idx)/B).^2;
    H_k = fftshift(H_mag .* exp(1j*H_phase));
end

function G_tx_k = compute_precomp_filter(S_ideal_k, H_k, alpha_reg, A_max)
    epsilon = alpha_reg * mean(abs(H_k .* S_ideal_k));
    G_tx_k = S_ideal_k ./ (H_k .* S_ideal_k + epsilon);

    if any(~isfinite(G_tx_k))
        G_tx_k = nan(size(G_tx_k));
        return;
    end

    G_mag = abs(G_tx_k);
    if any(G_mag > A_max)
        G_tx_k = G_tx_k ./ max(1, G_mag/A_max);
    end
end

function W_k = generate_generalized_cosine_window(N_fft, a1, a2, B, freq)
    a0 = 1 - a1 - a2;
    if a0 < 0
        W_k = nan(N_fft,1);
        return;
    end
    W = zeros(N_fft,1);
    idx = find(abs(freq)<=B); % 与原代码一致：补偿窗在[-B, B]内
    n = (0:length(idx)-1)';
    if length(idx) == 1
        w_local = a0;
    else
        w_local = a0 - a1*cos(2*pi*n/(length(idx)-1)) + a2*cos(4*pi*n/(length(idx)-1));
    end
    w_local = max(w_local, 0);
    W(idx) = w_local;
    W_k = fftshift(W);
end

function sim = simulate_transmit_signal(S_ideal_k, H_k, G_eff, N_pulse, N_fft)
    sim.G_eff = G_eff;
    sim.S_tx_k = S_ideal_k .* G_eff;
    s_tx_full = ifft(sim.S_tx_k, N_fft);
    s_tx = s_tx_full(1:N_pulse);
    s_tx = s_tx / sqrt(sum(abs(s_tx).^2) + eps);

    s_tx_pad = zeros(N_fft,1); s_tx_pad(1:N_pulse) = s_tx;
    S_tx_norm = fft(s_tx_pad, N_fft);

    sim.S_out_k = S_tx_norm .* H_k;
    s_out_full = ifft(sim.S_out_k, N_fft);
    s_out = s_out_full(1:N_pulse);
    s_out = s_out / sqrt(sum(abs(s_out).^2) + eps);

    sim.s_tx = s_tx;
    sim.s_out = s_out;
end

function metrics = evaluate_metrics(sig, fs, mainlobe_guard)
    ac = xcorr(sig, sig);
    ac = ac / (max(abs(ac)) + eps);
    mag = abs(ac);
    N = length(ac);
    c = ceil(N/2);

    % 用-3dB区域估计主瓣宽度，并用于PSLR/ISLR主瓣排除
    if isempty(mainlobe_guard)
        thr = 10^(-3/20);
        left = c; right = c;
        while left>1 && mag(left)>=thr, left=left-1; end
        while right<N && mag(right)>=thr, right=right+1; end
    else
        left = max(1, c-mainlobe_guard);
        right = min(N, c+mainlobe_guard);
    end
    main_idx = left:right;
    side_idx = [1:left-1, right+1:N];

    main_peak = max(mag(main_idx));
    side_peak = max(mag(side_idx));
    metrics.pslr_db = 20*log10(side_peak/(main_peak+eps) + eps);

    E_main = sum(mag(main_idx).^2);
    E_side = sum(mag(side_idx).^2);
    metrics.islr_db = 10*log10(E_side/(E_main+eps) + eps);

    metrics.bw3db_s = (right-left)/fs;
    metrics.ac = ac;
end

function E_spec = compute_spectrum_error(S_ideal_k, S_out_k, freq, B)
    % 方案A：归一化幅度均方误差（稳健）
    band_idx = abs(freq)<=B/2; % 在有效带宽内评价
    A = abs(S_ideal_k(band_idx));
    Bv = abs(S_out_k(band_idx));
    E_spec = (norm(Bv - A, 2)^2) / (norm(A, 2)^2 + eps);
end

function [J, sim, metrics] = objective_function(params, S_ideal_k, H_k, cfg, weights, metrics_ideal)
    huge = 1e9;
    alpha_reg = params(1); a1 = params(2); a2 = params(3);

    if ~(isfinite(alpha_reg) && isfinite(a1) && isfinite(a2)) || alpha_reg<=0 || a1<0 || a1>1 || a2<0 || a2>1 || (1-a1-a2)<0
        J = huge; sim = []; metrics = []; return;
    end

    G = compute_precomp_filter(S_ideal_k, H_k, alpha_reg, cfg.A_max);
    W = generate_generalized_cosine_window(cfg.N_fft, a1, a2, cfg.B, cfg.freq);
    if any(~isfinite(G)) || any(~isfinite(W))
        J = huge; sim = []; metrics = []; return;
    end

    sim = simulate_transmit_signal(S_ideal_k, H_k, G.*W, cfg.N_pulse, cfg.N_fft);
    if any(~isfinite(sim.s_tx)) || any(~isfinite(sim.s_out))
        J = huge; metrics = []; return;
    end

    metrics = evaluate_metrics(sim.s_out, cfg.fs, cfg.mainlobe_guard);
    metrics.papr_db = 10*log10(max(abs(sim.s_tx).^2) / (mean(abs(sim.s_tx).^2)+eps));
    metrics.spec_error = compute_spectrum_error(S_ideal_k, sim.S_out_k, cfg.freq, cfg.B);

    if any(~isfinite([metrics.pslr_db, metrics.islr_db, metrics.bw3db_s, metrics.papr_db, metrics.spec_error]))
        J = huge; return;
    end

    pslr_pen = max(0, metrics.pslr_db - cfg.pslr_target)^2;
    islr_pen = max(0, metrics.islr_db - cfg.islr_target)^2;
    bw_pen = max(0, (metrics.bw3db_s - metrics_ideal.bw3db_s)/(metrics_ideal.bw3db_s+eps))^2;
    papr_pen = max(0, metrics.papr_db - cfg.papr_ref)^2;
    spec_pen = metrics.spec_error;

    J = weights.pslr*pslr_pen + weights.islr*islr_pen + weights.bw*bw_pen + ...
        weights.papr*papr_pen + weights.spec*spec_pen;
end

function opt = optimize_joint_parameters(S_ideal_k, H_k, cfg, weights, metrics_ideal)
    % 阶段1：粗搜索
    alpha_grid_1 = logspace(-5, -1, 9);
    a1_grid_1 = 0:0.1:1;
    a2_grid_1 = 0:0.1:1;

    [bestJ, bestx] = grid_search(alpha_grid_1, a1_grid_1, a2_grid_1, S_ideal_k, H_k, cfg, weights, metrics_ideal);

    % 阶段2：细搜索（围绕粗搜索最优）
    a = bestx(1); u = bestx(2); v = bestx(3);
    alpha_grid_2 = logspace(log10(max(a/3,1e-6)), log10(min(a*3,1)), 11);
    a1_grid_2 = max(0,u-0.15):0.03:min(1,u+0.15);
    a2_grid_2 = max(0,v-0.15):0.03:min(1,v+0.15);

    [bestJ2, bestx2, bestsim, bestmet] = grid_search(alpha_grid_2, a1_grid_2, a2_grid_2, S_ideal_k, H_k, cfg, weights, metrics_ideal);

    if bestJ2 < bestJ
        bestJ = bestJ2;
        bestx = bestx2;
    else
        [~, bestsim, bestmet] = objective_function(bestx, S_ideal_k, H_k, cfg, weights, metrics_ideal);
    end

    opt.alpha_reg = bestx(1);
    opt.a1 = bestx(2);
    opt.a2 = bestx(3);
    opt.a0 = 1 - opt.a1 - opt.a2;
    opt.J = bestJ;
    opt.sim = bestsim;
    opt.metrics = bestmet;
end

function [bestJ, bestx, bestsim, bestmet] = grid_search(alpha_grid, a1_grid, a2_grid, S_ideal_k, H_k, cfg, weights, metrics_ideal)
    bestJ = inf; bestx = [1e-2, 0.46, 0]; bestsim = []; bestmet = [];
    for ia = 1:numel(alpha_grid)
        for i1 = 1:numel(a1_grid)
            for i2 = 1:numel(a2_grid)
                x = [alpha_grid(ia), a1_grid(i1), a2_grid(i2)];
                if x(2)+x(3) > 1
                    continue;
                end
                [J, sim, met] = objective_function(x, S_ideal_k, H_k, cfg, weights, metrics_ideal);
                if J < bestJ
                    bestJ = J;
                    bestx = x;
                    bestsim = sim;
                    bestmet = met;
                end
            end
        end
    end
end

function [t_corr, ac_i, ac_n, ac_o] = local_autocorr_compare(s_ideal, s_no, s_opt, fs)
    ac_i = xcorr(s_ideal, s_ideal); ac_i = ac_i/max(abs(ac_i)+eps);
    ac_n = xcorr(s_no, s_no);       ac_n = ac_n/max(abs(ac_n)+eps);
    ac_o = xcorr(s_opt, s_opt);     ac_o = ac_o/max(abs(ac_o)+eps);
    t_corr = (-length(s_ideal)+1:length(s_ideal)-1)/fs*1e6;
end
