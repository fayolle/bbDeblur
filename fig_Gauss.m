close all;
clear;
clc;


disp('Gaussian blur with Gaussian noise');


xs = im2double(imread('images/barbara_face.png'));


% noisy motion blurry
noise_mean = 0;
noise_var = 0.00001;

% default padding is 'replicate'
f = @(x) imgaussfilt(x, 3, 'Padding', 'circular');
F = @(x) imnoise(f(x),'gaussian',noise_mean,noise_var);

y = F(xs);


% Sample the kernel for Wiener and deconv-tv
d = zeros(11, 11);
d(6,6) = 1.0;
h = f(d); 
h = h./sum(h(:)); 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Wiener
signal_var = var(y(:));
NSR = noise_var / signal_var;
g = @(x) deconvwnr(x,h,NSR);
W = g(y); 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% deconv tv
addpath('./sota_comparisons/non_blind/deconvtv_v1');
% parameters
opts.rho_r   = 2;
opts.beta    = [1 1 0];
opts.print   = true;
opts.alpha   = 0.7;
opts.method  = 'l2';
mu           = 35000;
% end of parameters
tvs = deconvtv(y, h, mu, opts);
tv = tvs.f;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Regular Tao
opts.maxiter = 20;
[T, ~] = Tao_orig(F, y, opts);
%T = T(ps+1:end-ps,ps+1:end-ps,:);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Phase corrected 
[Tc, ~] = pcVC(F, y);
%Tc = Tc(ps+1:end-ps,ps+1:end-ps,:);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% modified Wiener
[iW, ~] = mW(F, y);
%iW = iW(ps+1:end-ps,ps+1:end-ps,:);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LM
[lm, ~] = mLM(F, y);
%lm = lm(ps+1:end-ps,ps+1:end-ps,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dfl_opts.maxiter = 500; % follows the paper 
[dfL, ~] = aL(F, y, dfl_opts); % approximate Landweber 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[mrl, ~] = mRL(F, y); % modified Richardson-Lucy 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Print psnr 
% Note: the padding seems to affect the psnr
fprintf("PSNR of Wiener: %f\n", psnr(W, xs));
fprintf("PSNR of TV: %f\n", psnr(tv, xs));
fprintf("PSNR of (original) Tao: %f\n", psnr(T, xs));
fprintf("PSNR of phase-corrected VC: %f\n", psnr(Tc, xs));
fprintf("PSNR of modified Wiener: %f\n", psnr(iW, xs));
fprintf("PSNR of modified LM: %f\n", psnr(lm, xs));
fprintf("PSNR of approximate Landweber: %f\n", psnr(dfL, xs)); 
fprintf("PSNR of modified Richardson-Lucy: %f\n", psnr(mrl, xs));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save all images
%
addpath('./export_fig');
figure();
imshow(y);
export_fig -m2 barbara_face_g3_nvar10m5.png;

figure();
imshow(W);
export_fig -m2 barbara_face_Wiener_g3_nvar10m5.png;

figure();
imshow(tv);
export_fig -m2 barbara_face_tv_35K_g3_nvar10m5.png;

figure();
imshow(T);
export_fig -m2 barbara_face_T_20_g3_nvar10m5.png;

figure();
imshow(Tc);
export_fig -m2 barbara_face_Tpc_100_g3_nvar10m5.png;

figure();
imshow(iW);
export_fig -m2 barbara_face_iW_100_g3_nvar10m5.png;

figure();
imshow(lm);
export_fig -m2 barbara_face_LM_100_g3_nvar10m5.png;

figure();
imshow(dfL);
export_fig -m2 barbara_face_dfL_500_g3_nvar10m5.png;

figure();
imshow(mrl);
export_fig -m2 barbara_face_mRL_500_g3_nvar10m5.png;


% Chin
close all; 
clear; 

% read images saved with x2 resolution
y = im2double(imread('barbara_face_g3_nvar10m5.png'));
W = im2double(imread('barbara_face_Wiener_g3_nvar10m5.png'));
tv = im2double(imread('barbara_face_tv_35K_g3_nvar10m5.png'));
lm = im2double(imread('barbara_face_LM_100_g3_nvar10m5.png'));
iW = im2double(imread('barbara_face_iW_100_g3_nvar10m5.png'));
T = im2double(imread('barbara_face_T_20_g3_nvar10m5.png'));
Tc = im2double(imread('barbara_face_Tpc_100_g3_nvar10m5.png'));
mrl = im2double(imread('barbara_face_mRL_500_g3_nvar10m5.png'));
dfL = im2double(imread('barbara_face_dfL_500_g3_nvar10m5.png'));


chin = imcrop(y, [210 320 190 110]);
chinW = imcrop(W, [210 320 190 110]);
chinTV = imcrop(tv, [210 320 190 110]);
chinT = imcrop(T, [210 320 190 110]);
chinLM = imcrop(lm, [210 320 190 110]);
chinIW = imcrop(iW, [210 320 190 110]);
chinTc = imcrop(Tc, [210 320 190 110]);
chinDfL = imcrop(dfL, [210 320 190 110]);
chinMRL = imcrop(mrl, [210 320 190 110]);


figure();
imshow(chin);
export_fig barbara_face_chin_g3_nvar10m5.png;

figure();
imshow(chinW);
export_fig barbara_face_Wiener_chin_g3_nvar10m5.png;

figure();
imshow(chinTV);
export_fig barbara_face_chin_tv_35K_g3_nvar10m5.png;

figure();
imshow(chinT);
export_fig barbara_face_chin_T_20_g3_nvar10m5.png;

figure();
imshow(chinTc);
export_fig barbara_face_chin_Tpc_100_g3_nvar10m5.png;

figure();
imshow(chinIW);
export_fig barbara_face_chin_iW_100_g3_nvar10m5.png;

figure();
imshow(chinLM);
export_fig barbara_face_chin_LM_100_g3_nvar10m5.png;

figure();
imshow(chinDfL);
export_fig barbara_face_chin_dfL_500_g3_nvar10m5.png;

figure();
imshow(chinMRL);
export_fig barbara_face_chin_mRL_500_g3_nvar10m5.png;


% Eyes
eyes = imcrop(y, [200 130 190 110]);
eyesW = imcrop(W, [200 130 190 110]);
eyesT = imcrop(T, [200 130 190 110]);
eyesTV = imcrop(tv, [200 130 190 110]);
eyesLM = imcrop(lm, [200 130 190 110]);
eyesIW = imcrop(iW, [200 130 190 110]);
eyesTc = imcrop(Tc, [200 130 190 110]);
eyesDfL = imcrop(dfL, [200 130 190 110]);
eyesMRL = imcrop(mrl, [200 130 190 110]);
close all;

figure();
imshow(eyes);
export_fig barbara_face_eyes_g3_nvar10m5.png;

figure();
imshow(eyesW);
export_fig barbara_face_Wiener_eyes_g3_nvar10m5.png;

figure();
imshow(eyesTV);
export_fig barbara_face_eyes_tv_35K_g3_nvar10m5.png;

figure();
imshow(eyesT);
export_fig barbara_face_eyes_T_20_g3_nvar10m5.png;

figure();
imshow(eyesTc);
export_fig barbara_face_eyes_Tpc_100_g3_nvar10m5.png;

figure();
imshow(eyesIW);
export_fig barbara_face_eyes_iW_100_g3_nvar10m5.png;

figure();
imshow(eyesLM);
export_fig barbara_face_eyes_LM_100_g3_nvar10m5.png;

figure();
imshow(eyesDfL);
export_fig barbara_face_eyes_dfL_500_g3_nvar10m5.png;

figure();
imshow(eyesMRL);
export_fig barbara_face_eyes_mRL_500_g3_nvar10m5.png;
