% Generate convergence plots (psnr and ssim) for our 5 methods with the examples in Fig. 2 (eccv3 kernel) and Fig. 3 (Gauss kernel)
% Show metrics for the denoised images. 

close all;
clear;

% some settings for the plots
alw = 0.75;    % AxesLineWidth
fsz = 11;      % Fontsize

xs = im2double(imread('images/barbara_face.png'));

% Fig. eccv3
noise_mean = 0;
noise_var = 0.00001;

h = im2double(imread('./kernels/eccv3_blurred_kernel.png'));
h = h./sum(h(:));
N = size(xs,1); M = size(xs,2); C = size(xs,3); Hf = psf2otf(h, [N M C]);
f = @(x) real(ifft2(fft2(x(:,:,:)).*Hf));

F = @(x) imnoise(f(x),'gaussian',noise_mean,noise_var);

y = F(xs);

opts.maxiter = 50; 
opts.show_denoised = 1; 
[~, ~, psnr_lm, ssim_lm, ~, ~, ~, ~] = mLM_metrics(F, y, xs, opts); % modified LM
[~, ~, psnr_T, ssim_T, ~, ~, ~, ~] = pcVC_metrics(F, y, xs, opts); 
[~, ~, psnr_iW, ssim_iW, ~, ~, ~, ~] = mW_metrics(F, y, xs, opts);
[~, ~, psnr_dfL, ssim_dfL, ~, ~, ~, ~] = aL_metrics(F, y, xs, opts); % approximate Landweber
[~, ~, psnr_mRL, ssim_mRL, ~, ~, ~, ~] = mRL_metrics(F, y, xs, opts); % modified Richardson-Lucy

n = length(psnr_lm);
plot(1:n, [psnr_lm, psnr_T, psnr_iW, psnr_dfL, psnr_mRL], 'LineWidth', 2)
legend('Modified LM', 'Phase-corrected VC', 'Modified Wiener', 'Approx. Landweber', 'Modified RL')
set(gca, 'FontSize', fsz, 'LineWidth', alw);
print('fig_eccv3_psnrs_convergence_show_denoised', '-dpng');

% save the data 
csvwrite('fig_eccv3_psnrs_convergence_lm_show_denoised.csv', psnr_lm);
csvwrite('fig_eccv3_psnrs_convergence_corrected_tao_show_denoised.csv', psnr_T);
csvwrite('fig_eccv3_psnrs_convergence_modified_Wiener_show_denoised.csv', psnr_iW);
csvwrite('fig_eccv3_psnrs_convergence_dfL_show_denoised.csv', psnr_dfL);
csvwrite('fig_eccv3_psnrs_convergence_mRL_show_denoised.csv', psnr_mRL);

n = length(ssim_lm);
plot(1:n, [ssim_lm, ssim_T, ssim_iW, ssim_dfL, ssim_mRL], 'LineWidth', 2)
legend('Modified LM', 'Phase-corrected VC', 'Modified Wiener', 'Approx. Landweber', 'Modified RL')
set(gca, 'FontSize', fsz, 'LineWidth', alw);
print('fig_eccv3_ssims_convergence_show_denoised', '-dpng');

% save the data 
csvwrite('fig_eccv3_ssims_convergence_lm_show_denoised.csv', ssim_lm);
csvwrite('fig_eccv3_ssims_convergence_corrected_tao_show_denoised.csv', ssim_T);
csvwrite('fig_eccv3_ssims_convergence_modified_Wiener_show_denoised.csv', ssim_iW);
csvwrite('fig_eccv3_ssims_convergence_dfL_show_denoised.csv', ssim_dfL);
csvwrite('fig_eccv3_ssims_convergence_mRL_show_denoised.csv', ssim_mRL);


% Fig. Gauss
noise_mean = 0;
noise_var = 0.00001;

% default padding is 'replicate'
f = @(x) imgaussfilt(x, 3, 'Padding', 'circular');
F = @(x) imnoise(f(x),'gaussian',noise_mean,noise_var);

y = F(xs);

opts.maxiter = 50; 
opts.show_denoised = 1; 
[~, ~, psnr_lm, ssim_lm, ~, ~, ~, ~] = mLM_metrics(F, y, xs, opts); % modified LM
[~, ~, psnr_T, ssim_T, ~, ~, ~, ~] = pcVC_metrics(F, y, xs, opts); 
[~, ~, psnr_iW, ssim_iW, ~, ~, ~, ~] = mW_metrics(F, y, xs, opts);
[~, ~, psnr_dfL, ssim_dfL, ~, ~, ~, ~] = aL_metrics(F, y, xs, opts); % approximate Landweber
[~, ~, psnr_mRL, ssim_mRL, ~, ~, ~, ~] = mRL_metrics(F, y, xs, opts); % modified Richardson-Lucy

n = length(psnr_lm);
plot(1:n, [psnr_lm, psnr_T, psnr_iW, psnr_dfL, psnr_mRL], 'LineWidth', 2)
legend('Modified LM', 'Phase-corrected VC', 'Modified Wiener', 'Approx. Landweber', 'Modified RL')
set(gca, 'FontSize', fsz, 'LineWidth', alw);
print('fig_Gauss_psnrs_convergence_show_denoised', '-dpng');

% save the data 
csvwrite('fig_Gauss_psnrs_convergence_lm_show_denoised.csv', psnr_lm);
csvwrite('fig_Gauss_psnrs_convergence_corrected_tao_show_denoised.csv', psnr_T);
csvwrite('fig_Gauss_psnrs_convergence_modified_Wiener_show_denoised.csv', psnr_iW);
csvwrite('fig_Gauss_psnrs_convergence_dfL_show_denoised.csv', psnr_dfL);
csvwrite('fig_Gauss_psnrs_convergence_mRL_show_denoised.csv', psnr_mRL);

n = length(ssim_lm);
plot(1:n, [ssim_lm, ssim_T, ssim_iW, ssim_dfL, ssim_mRL], 'LineWidth', 2)
legend('Modified LM', 'Phase-corrected VC', 'Modified Wiener', 'Approx. Landweber', 'Modified RL')
set(gca, 'FontSize', fsz, 'LineWidth', alw);
print('fig_Gauss_ssims_convergence_show_denoised', '-dpng');

% save the data 
csvwrite('fig_Gauss_ssims_convergence_lm_show_denoised.csv', ssim_lm);
csvwrite('fig_Gauss_ssims_convergence_corrected_tao_show_denoised.csv', ssim_T);
csvwrite('fig_Gauss_ssims_convergence_modified_Wiener_show_denoised.csv', ssim_iW);
csvwrite('fig_Gauss_ssims_convergence_dfL_show_denoised.csv', ssim_dfL);
csvwrite('fig_Gauss_ssims_convergence_mRL_show_denoised.csv', ssim_mRL);
