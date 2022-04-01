function im_out = Fortunato(im_blurred, kernel, wev)
KR = floor((size(kernel, 1) - 1)/2);
KC = floor((size(kernel, 2) - 1)/2);
pad_size = 2 * max(KR, KC);

[im_blurred_padded, mask_pad] = imPad(im_blurred, pad_size);
[R, C, ~] = size(im_blurred_padded);
big_kernel = getBigKernel(R, C, kernel);

im_out_padded = our_method_bifilter(im_blurred_padded, big_kernel, wev);
im_out        = imUnpad(im_out_padded, mask_pad, pad_size);

end


% ==========================================================================
%                           Aux. functions
% ==========================================================================
function kernel = getBigKernel(R, C, small_kernel)
% Resize kernel to match image size

kernel = zeros(R,C);
RC     = floor(R/2);
CC     = floor(C/2);

[RF,CF] = size(small_kernel);
RCF = floor(RF/2); CCF = floor(CF/2);

kernel(RC-RCF+1:RC-RCF+RF,CC-CCF+1:CC-CCF+CF) = small_kernel;
kernel = ifftshift(kernel);
kernel = kernel ./ sum(kernel(:));

end

% ==========================================================================

function [im_out, mask] = imPad(im_in, pad)
% Pad image to remove border ringing atifacts (see section 4.1 of our paper)

im_pad   = padarray(im_in, [pad, pad],'replicate','both');
[R,C,CH] = size(im_pad);

[X Y] = meshgrid (1:C, 1:R);

X0 = 1 + floor ( C / 2); Y0 = 1 + floor ( R / 2);
DX = abs( X - X0 )     ; DY = abs( Y - Y0 );
C0 = X0 - pad          ; R0 = Y0 - pad;

alpha = 0.01;
% force mask value at the borders aprox equal to alpha
% this makes the transition smoother for large kernels
nx = ceil(0.5 * log((1-alpha)/alpha) / log(X0 / C0));
ny = ceil(0.5 * log((1-alpha)/alpha) / log(Y0 / R0));

mX = 1 ./ ( 1 + ( DX ./ C0 ).^ (2 * nx));
mY = 1 ./ ( 1 + ( DY ./ R0 ).^ (2 * ny));
mask_0 = mX .* mY;

mask   = zeros(R,C,CH);
for ch = 1:CH
    mask(:,:,ch) = mask_0;
end;

im_out = zeros(R,C,CH);
im_out = im_pad .* mask;

end

% ==========================================================================
function im_out = imUnpad(im_in, mask_pad, pad)
% Remove padding (see section 4.1 of our paper)

im_out1 = im_in ./ mask_pad;
im_out = im_out1(pad+1:end-pad, pad+1:end-pad, :);

end

% ==========================================================================
% Blurr image and add noise for synthetic example (linear convolution)
function [im_blurred, im_in_valid, noise] = blurrAndNoiseLinear(im_in, small_kernel, sigma)
[R, C, CH] = size(im_in);

for ch = 1:CH
    im_blurred(:,:,ch) = conv2(im_in(:,:,ch), small_kernel, 'valid');
end;

[RB, CB, CH] = size(im_blurred);

noise = randn(RB, CB, CH) * sigma;
im_blurred = im_blurred + noise;
im_blurred = double(uint8(im_blurred .* 255))./255;

RB2 = floor((R-RB)/2); CB2 = floor((C-CB)/2);
im_in_valid = im_in(RB2+1:RB2+RB, CB2+1:CB2+CB, :);

end
