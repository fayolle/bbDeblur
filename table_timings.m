% Generate computational timings for our 5 methods with the examples in Fig. 2 (eccv3 kernel) and Fig. 3 (Gauss kernel) with Gaussian noise and variance 10^{-5}

close all;
clear;

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

lm = @() mLM(F, y); % modified LM
t_lm = timeit(lm);

T = @() pcVC(F, y); 
t_T = timeit(T);

iW = @() mW(F, y);
t_iW = timeit(iW);

dfl_opts.maxiter = 500; % follows the paper
dfL = @() aL(F, y, dfl_opts); % approximate Landweber
t_dfL = timeit(dfL);

mrl = @() mRL(F, y); % modified Richardson-Lucy
t_mRL = timeit(mrl);

fig_title = 'fig_eccv3';
method_names = {'modified\_LM', 'corrected\_Tao', 'modified\_Wiener', 'df\_Landweber', 'modified\_RL'};
times = [t_lm, t_T, t_iW, t_dfL, t_mRL];
prepare_Latex_table(fig_title, method_names, times);


% Fig. Gauss
noise_mean = 0;
noise_var = 0.00001;

% default padding is 'replicate'
f = @(x) imgaussfilt(x, 3, 'Padding', 'circular');
F = @(x) imnoise(f(x),'gaussian',noise_mean,noise_var);

y = F(xs);

lm = @() mLM(F, y); % modified LM
t_lm = timeit(lm);

T = @() pcVC(F, y); 
t_T = timeit(T);

iW = @() mW(F, y);
t_iW = timeit(iW);

dfl_opts.maxiter = 500; % follows the paper
dfL = @() aL(F, y, dfl_opts); % approximate Landweber
t_dfL = timeit(dfL);

mrl = @() mRL(F, y); % modified Richardson-Lucy
t_mRL = timeit(mrl);

fig_title = 'fig_Gauss';
method_names = {'modified\_LM', 'corrected\_Tao', 'modified\_Wiener', 'df\_Landweber', 'modified\_RL'};
times = [t_lm, t_T, t_iW, t_dfL, t_mRL];
prepare_Latex_table(fig_title, method_names, times);


% Utils
function prepare_Latex_table(fig_title, method_names, times)
file_name = [fig_title, '.tex'];
fid = fopen(file_name, 'w');

% preamble
fprintf(fid, '\\begin{table}\n');
fprintf(fid, '\\begin{tabular}{ccccc}\n');
fprintf(fid, '\\toprule\n');

n_times = length(times);
for i=1:(n_times-1)
    fprintf(fid, '%s & ', method_names{i});
end
fprintf(fid, '%s \\\\\n', method_names{n_times});

fprintf(fid, '\\midrule\n');

for i=1:(n_times-1)
    fprintf(fid, '%.2f & ', times(i));
end
fprintf(fid, '%.2f \\\\\n', times(n_times));

fprintf(fid, '\\bottomrule\n');
fprintf(fid, '\\end{tabular}\n');
fprintf(fid, '\\caption{%s}\n', fig_title);
fprintf(fid, '\\end{table}\n');

fclose(fid);
end
