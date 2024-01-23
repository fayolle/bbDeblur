close all; 
clear; 
clc;

xs = im2double(imread('images/barbara_face.png'));


% Motion blur with Poisson noise 
h = im2double(imread('./kernels/eccv3_blurred_kernel.png'));
h = h./sum(h(:));
N = size(xs,1); M = size(xs,2); C = size(xs,3); Hf = psf2otf(h, [N M C]);
f = @(x) real(ifft2(fft2(x(:,:,:)).*Hf));

peak = 1e4;
F = @(x) poissrnd(f(x)*peak)/peak;

y = F(xs);


% Wiener
if (size(y,3) ~= 1)
    ybw = rgb2gray(y);
    NSR = estimate_noise(ybw)^2 / var(ybw(:));
else
    NSR = estimate_noise(y)^2 / var(y(:));
end
g = @(x) deconvwnr(x,h,NSR);
W = g(y); 


% Our schemes 
lm = mLM(F, y); % modified LM
tc = pcVC(F, y); 
iw = mW(F, y);
dfl_opts.maxiter = 500; % follows the paper 
dfL = aL(F, y, dfl_opts); % approximate Landweber 
mRL = mRL(F, y); % modified Richardson-Lucy 


addpath('./export_fig');
figure();
imshow(y);
export_fig -m2 barbara_face_eccv3ker_p1e4.png;
fprintf("PSNR of blurred: %f\n", psnr(y, xs));

figure();
imshow(W);
export_fig -m2 barbara_face_Wiener_eccv3ker_p1e4.png;
fprintf("PSNR of Wiener: %f\n", psnr(W, xs));

figure();
imshow(lm);
export_fig -m2 barbara_face_LM_100_eccv3ker_p1e4.png;
fprintf("PSNR of modified LM: %f\n", psnr(lm, xs));

figure();
imshow(iw);
export_fig -m2 barbara_face_iW_100_eccv3ker_p1e4.png;
fprintf("PSNR of modified Wiener: %f\n", psnr(iw, xs));

figure();
imshow(tc);
export_fig -m2 barbara_face_Tpc_100_eccv3ker_p1e4.png;
fprintf("PSNR of phase corrected VC: %f\n", psnr(tc, xs));

figure();
imshow(dfL);
export_fig -m2 barbara_face_dfL_500_eccv3ker_p1e4.png;
fprintf("PSNR of approximate Landweber: %f\n", psnr(dfL, xs));

figure();
imshow(mRL);
export_fig -m2 barbara_face_mRL_500_eccv3ker_p1e4.png;
fprintf("PSNR of modified RL: %f\n", psnr(mRL, xs));


% Chin
close all;
clear;

% read images saved with x2 resolution
y = im2double(imread('barbara_face_eccv3ker_p1e4.png'));
W = im2double(imread('barbara_face_Wiener_eccv3ker_p1e4.png'));
lm = im2double(imread('barbara_face_LM_100_eccv3ker_p1e4.png'));
iW = im2double(imread('barbara_face_iW_100_eccv3ker_p1e4.png'));
Tc = im2double(imread('barbara_face_Tpc_100_eccv3ker_p1e4.png'));
dfL = im2double(imread('barbara_face_dfL_500_eccv3ker_p1e4.png'));
mRL = im2double(imread('barbara_face_mRL_500_eccv3ker_p1e4.png'));


chin = imcrop(y, [210 320 190 110]);
chinW = imcrop(W, [210 320 190 110]);
chinLM = imcrop(lm, [210 320 190 110]);
chinIW = imcrop(iW, [210 320 190 110]);
chinTc = imcrop(Tc, [210 320 190 110]);
chinDfL = imcrop(dfL, [210 320 190 110]);
chinMRL = imcrop(mRL, [210 320 190 110]);


figure();
imshow(chin);
export_fig barbara_face_chin_eccv3ker_p1e4.png;

figure();
imshow(chinW);
export_fig barbara_face_chin_Wiener_eccv3ker_p1e4.png;

figure();
imshow(chinLM);
export_fig barbara_face_chin_LM_100_eccv3ker_p1e4.png;

figure();
imshow(chinIW);
export_fig barbara_face_chin_iW_100_eccv3ker_p1e4.png;

figure();
imshow(chinTc);
export_fig barbara_face_chin_Tpc_100_eccv3ker_p1e4.png;

figure();
imshow(chinDfL);
export_fig barbara_face_chin_dfL_500_eccv3ker_p1e4.png;

figure();
imshow(chinMRL);
export_fig barbara_face_chin_mRL_500_eccv3ker_p1e4.png;


% Eyes
eyes = imcrop(y, [200 130 190 110]);
eyesW = imcrop(W, [200 130 190 110]);
eyesLM = imcrop(lm, [200 130 190 110]);
eyesIW = imcrop(iW, [200 130 190 110]);
eyesTc = imcrop(Tc, [200 130 190 110]);
eyesDfL = imcrop(dfL, [200 130 190 110]);
eyesMRL = imcrop(mRL, [200 130 190 110]);
close all;

figure();
imshow(eyes);
export_fig barbara_face_eyes_eccv3ker_p1e4.png;

figure();
imshow(eyesW);
export_fig barbara_face_eyes_Wiener_eccv3ker_p1e4.png;

figure();
imshow(eyesLM);
export_fig barbara_face_eyes_LM_100_eccv3ker_p1e4.png;

figure();
imshow(eyesIW);
export_fig barbara_face_eyes_iW_100_eccv3ker_p1e4.png;

figure();
imshow(eyesTc);
export_fig barbara_face_eyes_Tpc_100_eccv3ker_p1e4.png;

figure();
imshow(eyesDfL);
export_fig barbara_face_eyes_dfL_500_eccv3ker_p1e4.png;

figure();
imshow(eyesMRL);
export_fig barbara_face_eyes_mRL_500_eccv3ker_p1e4.png;
