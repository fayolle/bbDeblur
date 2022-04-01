close all; 
clear; 
clc;


% Starfish 
xs = im2double(imread('images/starfish.png'));


% noisy motion blurry
noise_mean = 0;
noise_var = 0.00001;

h = im2double(rgb2gray(imread('./kernels/testkernel2.bmp')));
h = h./sum(h(:));
N = size(xs,1); M = size(xs,2); Hf = psf2otf(h, [N M]);
f = @(x) real(ifft2(fft2(x(:,:)).*Hf));

F = @(x) imnoise(f(x),'gaussian',noise_mean,noise_var);

yr = F(xs(:,:,1));
yg = F(xs(:,:,2));
yb = F(xs(:,:,3));

lmr = mLM(F, yr); % modified LM
lmg = mLM(F, yg);
lmb = mLM(F, yb);

Wr = mW(F, yr);
Wg = mW(F, yg);
Wb = mW(F, yb);

dfl_opts.maxiter = 500; % follows the paper 
dfLr = aL(F, yr, dfl_opts); % approximate Landweber 
dfLg = aL(F, yg, dfl_opts);
dfLb = aL(F, yb, dfl_opts);

addpath('./export_fig');
figure();
imshow(xs);
export_fig starfish.png;

figure();
y = xs;
y(:,:,1) = yr;
y(:,:,2) = yg;
y(:,:,3) = yb;
imshow(y);
export_fig starfish_ker2_nvar10m5_per_channel.png;
fprintf("PSNR of blurred: %f\n", psnr(y, xs));

figure();
lm = xs; 
lm(:,:,1) = lmr;
lm(:,:,2) = lmg;
lm(:,:,3) = lmb;
imshow(lm);
export_fig starfish_LM_100_ker2_nvar10m5_per_channel.png;
fprintf("PSNR of modified LM: %f\n", psnr(lm, xs));

figure();
W = xs; 
W(:,:,1) = Wr; 
W(:,:,2) = Wg; 
W(:,:,3) = Wb; 
imshow(W);
export_fig starfish_iterWiener_100_ker2_nvar10m5_per_channel.png;
fprintf("PSNR of modified Wiener: %f\n", psnr(W, xs));

figure(); 
dfL = xs;
dfL(:,:,1) = dfLr;
dfL(:,:,2) = dfLg;
dfL(:,:,3) = dfLb;
imshow(dfL);
export_fig starfish_dfL_500_ker2_nvar10m5_per_channel.png;
fprintf("PSNR of approximate Landweber: %f\n", psnr(dfL, xs));


% Parrot
close all;
clear;
%clc;


xs = im2double(imread('images/parrots.png'));


% noisy motion blurry
noise_mean = 0;
noise_var = 0.00001;

h = im2double(rgb2gray(imread('./kernels/testkernel2.bmp')));
h = h./sum(h(:));
N = size(xs,1); M = size(xs,2); Hf = psf2otf(h, [N M]);
f = @(x) real(ifft2(fft2(x(:,:)).*Hf));

F = @(x) imnoise(f(x),'gaussian',noise_mean,noise_var);

yr = F(xs(:,:,1));
yg = F(xs(:,:,2));
yb = F(xs(:,:,3));

lmr = mLM(F, yr); % modified LM
lmg = mLM(F, yg);
lmb = mLM(F, yb);

Wr = mW(F, yr);
Wg = mW(F, yg);
Wb = mW(F, yb);

dfl_opts.maxiter = 500; % follows the paper 
dfLr = aL(F, yr, dfl_opts); % approximate Landweber 
dfLg = aL(F, yg, dfl_opts);
dfLb = aL(F, yb, dfl_opts);


addpath('./export_fig');
figure();
imshow(xs);
export_fig parrot.png;

figure();
y = xs;
y(:,:,1) = yr;
y(:,:,2) = yg;
y(:,:,3) = yb;
imshow(y);
export_fig parrot_ker2_nvar10m5_per_channel.png;
fprintf("PSNR of blurred: %f\n", psnr(y, xs));

figure();
lm = xs; 
lm(:,:,1) = lmr;
lm(:,:,2) = lmg;
lm(:,:,3) = lmb;
imshow(lm);
export_fig parrot_LM_100_ker2_nvar10m5_per_channel.png;
fprintf("PSNR of modified LM: %f\n", psnr(lm, xs));

figure();
W = xs; 
W(:,:,1) = Wr; 
W(:,:,2) = Wg; 
W(:,:,3) = Wb; 
imshow(W);
export_fig parrot_iterWiener_100_ker2_nvar10m5_per_channel.png;
fprintf("PSNR of modified Wiener: %f\n", psnr(W, xs));

figure();
dfL = xs;
dfL(:,:,1) = dfLr;
dfL(:,:,2) = dfLg;
dfL(:,:,3) = dfLb;
imshow(dfL);
export_fig parrot_dfL_500_ker2_nvar10m5_per_channel.png;
fprintf("PSNR of approximate Landweber: %f\n", psnr(dfL, xs));
