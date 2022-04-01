close all; 
clear; 
clc;


xs = im2double(imread('images/barbara_face.png'));


% noisy motion blurry
noise_mean = 0;
noise_var = 0.0001;


h = im2double(rgb2gray(imread('./kernels/testkernel2.bmp')));
h = h./sum(h(:));
N = size(xs,1); M = size(xs,2); C = size(xs,3); Hf = psf2otf(h, [N M C]);
f = @(x) real(ifft2(fft2(x(:,:,:)).*Hf));

F = @(x) imnoise(f(x),'gaussian',noise_mean,noise_var);

y = F(xs);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Dong 
D = Dong(F, y);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% regular Tao 
% doesn't work for non-symmetric kernels
maxiter = 20;
T = F(y);
for i=1:maxiter
  T = T + (y-F(T));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Wiener 
signal_var = var(y(:));
NSR = noise_var / signal_var;
g = @(x) deconvwnr(x,h,NSR);
W = g(y); 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fortunato 
addpath('./sota_comparisons/non_blind/Fortunato/'); 
wev   = [0.001, 20, 0.033, 0.05]; 
fto = Fortunato(y, h, wev); 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% phase corrected VC
Tc = pcVC(F, y);


addpath('./export_fig');
figure();
imshow(xs); 
export_fig barbara_face.png

figure();
imshow(y);
export_fig barbara_face_ker2_nvar10m4.png

figure();
imshow(T);
export_fig barbara_face_Tao_ker2_nvar10m4.png;

figure();
imshow(D);
export_fig barbara_face_Dong_ker2_nvar10m4.png;

figure();
imshow(W);
export_fig barbara_face_Wiener_ker2_nvar10m4.png;

figure();
imshow(fto);
export_fig barbara_face_Fortunato_ker2_nvar10m4.png;

figure();
imshow(Tc);
export_fig barbara_face_Tao_corrected_ker2_nvar10m4.png;
