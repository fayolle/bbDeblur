% Generate convergence plots (residual error) for our 5 methods with the examples in Fig. 2 (eccv3 kernel) and Fig. 3 (Gauss kernel)

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
[~, E_lm] = mLM(F, y, opts); % modified LM
[~, E_T] = pcVC(F, y, opts); 
[~, E_iW] = mW(F, y, opts);
[~, E_dfL] = aL(F, y, opts); % approximate Landweber
[~, E_mRL] = mRL(F, y, opts); % modified Richardson-Lucy

n = length(E_lm);
plot(1:n, [E_lm, E_T, E_iW, E_dfL, E_mRL], 'LineWidth', 2)
legend('Modified LM', 'Phase-corrected VC', 'Modified Wiener', 'Approx. Landweber', 'Modified RL')
set(gca, 'FontSize', fsz, 'LineWidth', alw);
print('fig_eccv3_err_convergence', '-dpng');

% save the data 
csvwrite('fig_eccv3_err_convergence_lm.csv', E_lm);
csvwrite('fig_eccv3_err_convergence_corrected_tao.csv', E_T);
csvwrite('fig_eccv3_err_convergence_modified_Wiener.csv', E_iW);
csvwrite('fig_eccv3_err_convergence_dfL.csv', E_dfL);
csvwrite('fig_eccv3_err_convergence_mRL.csv', E_mRL);


% Fig. Gauss
noise_mean = 0;
noise_var = 0.00001;

% default padding is 'replicate'
f = @(x) imgaussfilt(x, 3, 'Padding', 'circular');
F = @(x) imnoise(f(x),'gaussian',noise_mean,noise_var);

y = F(xs);

opts.maxiter = 50; 
[lm, E_lm] = mLM(F, y, opts); % modified LM
[T, E_T] = pcVC(F, y, opts); 
[iW, E_iW] = mW(F, y, opts);
[dfL, E_dfL] = aL(F, y, opts); % approximate Landweber
[mrl, E_mRL] = mRL(F, y, opts); % modified Richardson-Lucy

close all;
n = length(E_lm);
plot(1:n, [E_lm, E_T, E_iW, E_dfL, E_mRL], 'LineWidth', 2)
legend('Modified LM', 'Phase-corrected VC', 'Modified Wiener', 'Approx. Landweber', 'Modified RL')
set(gca, 'FontSize', fsz, 'LineWidth', alw);
print('fig_gauss_err_convergence', '-dpng');

% save the data 
csvwrite('fig_Gauss_err_convergence_lm.csv', E_lm);
csvwrite('fig_Gauss_err_convergence_corrected_tao.csv', E_T);
csvwrite('fig_Gauss_err_convergence_modified_Wiener.csv', E_iW);
csvwrite('fig_Gauss_err_convergence_dfL.csv', E_dfL);
csvwrite('fig_Gauss_err_convergence_mRL.csv', E_mRL);
