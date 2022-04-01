
images = {'barbara_face.png', 'cameraman.png', 'parrots.png', 'starfish.png', 'trui.png'};
kernels = {'testkernel1.bmp', 'testkernel2.bmp', 'eccv3_blurred_kernel.png'};
noise_vars = [0.00001, 0.0001];

fid_psnr = fopen('table_psnr.csv', 'w');
fprintf(fid_psnr, 'kernel,noise,image,method,psnr\n');
fid_psnr_denoise = fopen('table_psnr_denoise.csv', 'w');
fprintf(fid_psnr_denoise, 'kernel,noise,image,method,psnr\n');

fid_mse = fopen('table_mse.csv', 'w');
fprintf(fid_mse, 'kernel,noise,image,method,mse\n');
fid_mse_denoise = fopen('table_mse_denoise.csv', 'w');
fprintf(fid_mse_denoise, 'kernel,noise,image,method,mse\n');

fid_ssim = fopen('table_ssim.csv', 'w');
fprintf(fid_ssim, 'kernel,noise,image,method,ssim\n');
fid_ssim_denoise = fopen('table_ssim_denoise.csv', 'w');
fprintf(fid_ssim_denoise, 'kernel,noise,image,method,ssim\n');

fid_brisque = fopen('table_brisque.csv', 'w');
fprintf(fid_brisque, 'kernel,noise,image,method,brisque\n');
fid_brisque_denoise = fopen('table_brisque_denoise.csv', 'w');
fprintf(fid_brisque_denoise, 'kernel,noise,image,method,brisque\n');

fid_niqe = fopen('table_niqe.csv', 'w');
fprintf(fid_niqe, 'kernel,noise,image,method,niqe\n');
fid_niqe_denoise = fopen('table_niqe_denoise.csv', 'w');
fprintf(fid_niqe_denoise, 'kernel,noise,image,method,niqe\n');

fid_piqe = fopen('table_piqe.csv', 'w');
fprintf(fid_piqe, 'kernel,noise,image,method,piqe\n');
fid_piqe_denoise = fopen('table_piqe_denoise.csv', 'w');
fprintf(fid_piqe_denoise, 'kernel,noise,image,method,piqe\n');


for i = 1:length(images)
    xs = im2double(imread(['./images/', images{i}]));
    
    for j = 1:length(kernels)
        kern_image = imread(['./kernels/', kernels{j}]);
        if size(kern_image, 3) == 3
            h = im2double(rgb2gray(kern_image));    
        else
            h = im2double(kern_image);
        end
        h = h./sum(h(:));
        N = size(xs,1); M = size(xs,2); C = size(xs,3); Hf = psf2otf(h, [N M C]);
        f = @(x) real(ifft2(fft2(x(:,:,:)).*Hf));
        
        for k = 1:length(noise_vars)
            noise_mean = 0;
            noise_var = noise_vars(k);
            
            F = @(x) imnoise(f(x), 'gaussian', noise_mean, noise_var);
            
            y = F(xs); 
            
            % Dong
            D = Dong(F, y);
            
            % Tao
            tao_opts.maxiter = 20;
            T = Tao_orig(F, y, tao_opts);
            
            % our methods
            lm = mLM(F, y); % modified LM
            Tpc = pcVC(F, y); 
            iW = mW(F, y);
            dfl_opts.maxiter = 500; % follows the paper
            dfL = aL(F, y, dfl_opts); % approximate Landweber
            mrl = mRL(F, y); % modified Richardson-Lucy
            
            % denoise 
            if has_a_nan(D)==0, Dd = denoise(D); end
            if has_a_nan(T)==0, Td = denoise(T); end 
            if has_a_nan(lm)==0, lmd = denoise(lm); end
            if has_a_nan(Tpc)==0, Tpcd = denoise(Tpc); end
            if has_a_nan(iW)==0, iWd = denoise(iW); end
            if has_a_nan(dfL)==0, dfLd = denoise(dfL); end
            if has_a_nan(mrl)==0, mRLd = denoise(mrl); end
            
            % kernel, noise, image, method, psnr
            if has_a_nan(D)==0, fprintf(fid_psnr, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'Dong', psnr(D, xs)); end
            if has_a_nan(T)==0, fprintf(fid_psnr, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'Tao', psnr(T, xs)); end
            if has_a_nan(lm)==0, fprintf(fid_psnr, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'modified_LM', psnr(lm, xs)); end
            if has_a_nan(Tpc)==0, fprintf(fid_psnr, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'corrected_Tao', psnr(Tpc, xs)); end
            if has_a_nan(iW)==0, fprintf(fid_psnr, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'modified_Wiener', psnr(iW, xs)); end
            if has_a_nan(dfL)==0, fprintf(fid_psnr, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'df_Landweber', psnr(dfL, xs)); end
            if has_a_nan(mrl)==0, fprintf(fid_psnr, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'modified_RL', psnr(mrl, xs)); end
            
            if has_a_nan(D)==0, fprintf(fid_psnr_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'Dong', psnr(Dd, xs)); end
            if has_a_nan(T)==0, fprintf(fid_psnr_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'Tao', psnr(Td, xs)); end
            if has_a_nan(lm)==0, fprintf(fid_psnr_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'modified_LM', psnr(lmd, xs)); end
            if has_a_nan(Tpc)==0, fprintf(fid_psnr_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'corrected_Tao', psnr(Tpcd, xs)); end
            if has_a_nan(iW)==0, fprintf(fid_psnr_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'modified_Wiener', psnr(iWd, xs)); end
            if has_a_nan(dfL)==0, fprintf(fid_psnr_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'df_Landweber', psnr(dfLd, xs)); end
            if has_a_nan(mrl)==0, fprintf(fid_psnr_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'modified_RL', psnr(mRLd, xs)); end
            
            % kernel, noise, image, method, mse
            if has_a_nan(D)==0, fprintf(fid_mse, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'Dong', immse(D, xs)); end
            if has_a_nan(T)==0, fprintf(fid_mse, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'Tao', immse(T, xs)); end
            if has_a_nan(lm)==0, fprintf(fid_mse, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'modified_LM', immse(lm, xs)); end
            if has_a_nan(Tpc)==0, fprintf(fid_mse, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'corrected_Tao', immse(Tpc, xs)); end
            if has_a_nan(iW)==0, fprintf(fid_mse, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'modified_Wiener', immse(iW, xs)); end
            if has_a_nan(dfL)==0, fprintf(fid_mse, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'df_Landweber', immse(dfL, xs)); end
            if has_a_nan(mrl)==0, fprintf(fid_mse, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'modified_RL', immse(mrl, xs)); end
            
            if has_a_nan(D)==0, fprintf(fid_mse_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'Dong', immse(Dd, xs)); end
            if has_a_nan(T)==0, fprintf(fid_mse_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'Tao', immse(Td, xs)); end
            if has_a_nan(lm)==0, fprintf(fid_mse_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'modified_LM', immse(lmd, xs)); end
            if has_a_nan(Tpc)==0, fprintf(fid_mse_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'corrected_Tao', immse(Tpcd, xs)); end
            if has_a_nan(iW)==0, fprintf(fid_mse_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'modified_Wiener', immse(iWd, xs)); end
            if has_a_nan(dfL)==0, fprintf(fid_mse_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'df_Landweber', immse(dfLd, xs)); end
            if has_a_nan(mrl)==0, fprintf(fid_mse_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'modified_RL', immse(mRLd, xs)); end
            
            % kernel, noise, image, method, ssim
            if has_a_nan(D)==0, fprintf(fid_ssim, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'Dong', ssim(D, xs)); end
            if has_a_nan(T)==0, fprintf(fid_ssim, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'Tao', ssim(T, xs)); end
            if has_a_nan(lm)==0, fprintf(fid_ssim, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'modified_LM', ssim(lm, xs)); end
            if has_a_nan(Tpc)==0, fprintf(fid_ssim, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'corrected_Tao', ssim(Tpc, xs)); end
            if has_a_nan(iW)==0, fprintf(fid_ssim, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'modified_Wiener', ssim(iW, xs)); end
            if has_a_nan(dfL)==0, fprintf(fid_ssim, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'df_Landweber', ssim(dfL, xs)); end
            if has_a_nan(mrl)==0, fprintf(fid_ssim, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'modified_RL', ssim(mrl, xs)); end
            
            if has_a_nan(D)==0, fprintf(fid_ssim_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'Dong', ssim(Dd, xs)); end
            if has_a_nan(T)==0, fprintf(fid_ssim_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'Tao', ssim(Td, xs)); end
            if has_a_nan(lm)==0, fprintf(fid_ssim_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'modified_LM', ssim(lmd, xs)); end
            if has_a_nan(Tpc)==0, fprintf(fid_ssim_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'corrected_Tao', ssim(Tpcd, xs)); end
            if has_a_nan(iW)==0, fprintf(fid_ssim_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'modified_Wiener', ssim(iWd, xs)); end
            if has_a_nan(dfL)==0, fprintf(fid_ssim_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'df_Landweber', ssim(dfLd, xs)); end
            if has_a_nan(mrl)==0, fprintf(fid_ssim_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'modified_RL', ssim(mRLd, xs)); end
            
            % kernel, noise, image, method, brisque
            if has_a_nan(D)==0, fprintf(fid_brisque, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'Dong', brisque(D)); end
            if has_a_nan(T)==0, fprintf(fid_brisque, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'Tao', brisque(T)); end
            if has_a_nan(lm)==0, fprintf(fid_brisque, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'modified_LM', brisque(lm)); end
            if has_a_nan(Tpc)==0, fprintf(fid_brisque, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'corrected_Tao', brisque(Tpc)); end
            if has_a_nan(iW)==0, fprintf(fid_brisque, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'modified_Wiener', brisque(iW)); end
            if has_a_nan(dfL)==0, fprintf(fid_brisque, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'df_Landweber', brisque(dfL)); end
            if has_a_nan(mrl)==0, fprintf(fid_brisque, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'modified_RL', brisque(mrl)); end
            
            if has_a_nan(D)==0, fprintf(fid_brisque_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'Dong', brisque(Dd)); end
            if has_a_nan(T)==0, fprintf(fid_brisque_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'Tao', brisque(Td)); end
            if has_a_nan(lm)==0, fprintf(fid_brisque_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'modified_LM', brisque(lmd)); end
            if has_a_nan(Tpc)==0, fprintf(fid_brisque_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'corrected_Tao', brisque(Tpcd)); end
            if has_a_nan(iW)==0, fprintf(fid_brisque_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'modified_Wiener', brisque(iWd)); end
            if has_a_nan(dfL)==0, fprintf(fid_brisque_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'df_Landweber', brisque(dfLd)); end
            if has_a_nan(mrl)==0, fprintf(fid_brisque_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'modified_RL', brisque(mRLd)); end
            
            % kernel, noise, image, method, niqe
            if has_a_nan(D)==0, fprintf(fid_niqe, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'Dong', niqe(D)); end
            if has_a_nan(T)==0, fprintf(fid_niqe, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'Tao', niqe(T)); end
            if has_a_nan(lm)==0, fprintf(fid_niqe, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'modified_LM', niqe(lm)); end
            if has_a_nan(Tpc)==0, fprintf(fid_niqe, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'corrected_Tao', niqe(Tpc)); end
            if has_a_nan(iW)==0, fprintf(fid_niqe, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'modified_Wiener', niqe(iW)); end
            if has_a_nan(dfL)==0, fprintf(fid_niqe, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'df_Landweber', niqe(dfL)); end
            if has_a_nan(mrl)==0, fprintf(fid_niqe, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'modified_RL', niqe(mrl)); end
            
            if has_a_nan(D)==0, fprintf(fid_niqe_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'Dong', niqe(Dd)); end
            if has_a_nan(T)==0, fprintf(fid_niqe_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'Tao', niqe(Td)); end
            if has_a_nan(lm)==0, fprintf(fid_niqe_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'modified_LM', niqe(lmd)); end
            if has_a_nan(Tpc)==0, fprintf(fid_niqe_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'corrected_Tao', niqe(Tpcd)); end
            if has_a_nan(iW)==0, fprintf(fid_niqe_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'modified_Wiener', niqe(iWd)); end
            if has_a_nan(dfL)==0, fprintf(fid_niqe_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'df_Landweber', niqe(dfLd)); end
            if has_a_nan(mrl)==0, fprintf(fid_niqe_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'modified_RL', niqe(mRLd)); end
            
            % kernel, noise, image, method, piqe
            if has_a_nan(D)==0, fprintf(fid_piqe, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'Dong', piqe(D)); end
            if has_a_nan(T)==0, fprintf(fid_piqe, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'Tao', piqe(T)); end
            if has_a_nan(lm)==0, fprintf(fid_piqe, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'modified_LM', piqe(lm)); end
            if has_a_nan(Tpc)==0, fprintf(fid_piqe, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'corrected_Tao', piqe(Tpc)); end
            if has_a_nan(iW)==0, fprintf(fid_piqe, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'modified_Wiener', piqe(iW)); end
            if has_a_nan(dfL)==0, fprintf(fid_piqe, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'df_Landweber', piqe(dfL)); end
            if has_a_nan(mrl)==0, fprintf(fid_piqe, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'modified_RL', piqe(mrl)); end
            
            if has_a_nan(D)==0, fprintf(fid_piqe_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'Dong', piqe(Dd)); end
            if has_a_nan(T)==0, fprintf(fid_piqe_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'Tao', piqe(Td)); end
            if has_a_nan(lm)==0, fprintf(fid_piqe_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'modified_LM', piqe(lmd)); end
            if has_a_nan(Tpc)==0, fprintf(fid_piqe_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'corrected_Tao', piqe(Tpcd)); end
            if has_a_nan(iW)==0, fprintf(fid_piqe_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'modified_Wiener', piqe(iWd)); end
            if has_a_nan(dfL)==0, fprintf(fid_piqe_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'df_Landweber', piqe(dfLd)); end
            if has_a_nan(mrl)==0, fprintf(fid_piqe_denoise, '%s,%f,%s,%s,%f\n', kernels{j}, noise_var, images{i}, 'modified_RL', piqe(mRLd)); end
            
        end
        
    end
    
    % deal with Gaussian blur differently?
    f = @(x) imgaussfilt(x, 3, 'Padding', 'circular');

        for k = 1:length(noise_vars)
            noise_mean = 0;
            noise_var = noise_vars(k);
            
            F = @(x) imnoise(f(x), 'gaussian', noise_mean, noise_var);
            
            y = F(xs); 
            
            % Dong
            D = Dong(F, y);
            
            % Tao
            tao_opts.maxiter = 20;
            T = Tao_orig(F, y, tao_opts);
            
            % our methods
            lm = mLM(F, y); % modified LM
            Tpc = pcVC(F, y); 
            iW = mW(F, y);
            dfl_opts.maxiter = 500; % follows the paper
            dfL = aL(F, y, dfl_opts); % approximate Landweber
            mrl = mRL(F, y); % modified Richardson-Lucy
            
            % denoise 
            if has_a_nan(D)==0, Dd = denoise(D); end
            if has_a_nan(T)==0, Td = denoise(T); end 
            if has_a_nan(lm)==0, lmd = denoise(lm); end
            if has_a_nan(Tpc)==0, Tpcd = denoise(Tpc); end
            if has_a_nan(iW)==0, iWd = denoise(iW); end
            if has_a_nan(dfL)==0, dfLd = denoise(dfL); end
            if has_a_nan(mrl)==0, mRLd = denoise(mrl); end
            
            % kernel, noise, image, method, psnr
            if has_a_nan(D)==0, fprintf(fid_psnr, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'Dong', psnr(D, xs)); end
            if has_a_nan(T)==0, fprintf(fid_psnr, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'Tao', psnr(T, xs)); end
            if has_a_nan(lm)==0, fprintf(fid_psnr, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'modified_LM', psnr(lm, xs)); end
            if has_a_nan(Tpc)==0, fprintf(fid_psnr, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'corrected_Tao', psnr(Tpc, xs)); end
            if has_a_nan(iW)==0, fprintf(fid_psnr, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'modified_Wiener', psnr(iW, xs)); end
            if has_a_nan(dfL)==0, fprintf(fid_psnr, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'df_Landweber', psnr(dfL, xs)); end
            if has_a_nan(mrl)==0, fprintf(fid_psnr, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'modified_RL', psnr(mrl, xs)); end
            
            if has_a_nan(D)==0, fprintf(fid_psnr_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'Dong', psnr(Dd, xs)); end
            if has_a_nan(T)==0, fprintf(fid_psnr_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'Tao', psnr(Td, xs)); end
            if has_a_nan(lm)==0, fprintf(fid_psnr_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'modified_LM', psnr(lmd, xs)); end
            if has_a_nan(Tpc)==0, fprintf(fid_psnr_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'corrected_Tao', psnr(Tpcd, xs)); end
            if has_a_nan(iW)==0, fprintf(fid_psnr_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'modified_Wiener', psnr(iWd, xs)); end
            if has_a_nan(dfL)==0, fprintf(fid_psnr_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'df_Landweber', psnr(dfLd, xs)); end
            if has_a_nan(mrl)==0, fprintf(fid_psnr_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'modified_RL', psnr(mRLd, xs)); end
            
            % kernel, noise, image, method, immse
            if has_a_nan(D)==0, fprintf(fid_mse, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'Dong', immse(D, xs)); end
            if has_a_nan(T)==0, fprintf(fid_mse, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'Tao', immse(T, xs)); end
            if has_a_nan(lm)==0, fprintf(fid_mse, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'modified_LM', immse(lm, xs)); end
            if has_a_nan(Tpc)==0, fprintf(fid_mse, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'corrected_Tao', immse(Tpc, xs)); end
            if has_a_nan(iW)==0, fprintf(fid_mse, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'modified_Wiener', immse(iW, xs)); end
            if has_a_nan(dfL)==0, fprintf(fid_mse, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'df_Landweber', immse(dfL, xs)); end
            if has_a_nan(mrl)==0, fprintf(fid_mse, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'modified_RL', immse(mrl, xs)); end
            
            if has_a_nan(D)==0, fprintf(fid_mse_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'Dong', immse(Dd, xs)); end
            if has_a_nan(T)==0, fprintf(fid_mse_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'Tao', immse(Td, xs)); end
            if has_a_nan(lm)==0, fprintf(fid_mse_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'modified_LM', immse(lmd, xs)); end
            if has_a_nan(Tpc)==0, fprintf(fid_mse_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'corrected_Tao', immse(Tpcd, xs)); end
            if has_a_nan(iW)==0, fprintf(fid_mse_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'modified_Wiener', immse(iWd, xs)); end
            if has_a_nan(dfL)==0, fprintf(fid_mse_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'df_Landweber', immse(dfLd, xs)); end
            if has_a_nan(mrl)==0, fprintf(fid_mse_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'modified_RL', immse(mRLd, xs)); end
            
            % kernel, noise, image, method, ssim
            if has_a_nan(D)==0, fprintf(fid_ssim, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'Dong', ssim(D, xs)); end
            if has_a_nan(T)==0, fprintf(fid_ssim, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'Tao', ssim(T, xs)); end
            if has_a_nan(lm)==0, fprintf(fid_ssim, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'modified_LM', ssim(lm, xs)); end
            if has_a_nan(Tpc)==0, fprintf(fid_ssim, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'corrected_Tao', ssim(Tpc, xs)); end
            if has_a_nan(iW)==0, fprintf(fid_ssim, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'modified_Wiener', ssim(iW, xs)); end
            if has_a_nan(dfL)==0, fprintf(fid_ssim, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'df_Landweber', ssim(dfL, xs)); end
            if has_a_nan(mrl)==0, fprintf(fid_ssim, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'modified_RL', ssim(mrl, xs)); end
            
            if has_a_nan(D)==0, fprintf(fid_ssim_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'Dong', ssim(Dd, xs)); end
            if has_a_nan(T)==0, fprintf(fid_ssim_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'Tao', ssim(Td, xs)); end
            if has_a_nan(lm)==0, fprintf(fid_ssim_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'modified_LM', ssim(lmd, xs)); end
            if has_a_nan(Tpc)==0, fprintf(fid_ssim_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'corrected_Tao', ssim(Tpcd, xs)); end
            if has_a_nan(iW)==0, fprintf(fid_ssim_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'modified_Wiener', ssim(iWd, xs)); end
            if has_a_nan(dfL)==0, fprintf(fid_ssim_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'df_Landweber', ssim(dfLd, xs)); end
            if has_a_nan(mrl)==0, fprintf(fid_ssim_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'modified_RL', ssim(mRLd, xs)); end
            
            % kernel, noise, image, method, brisqe
            if has_a_nan(D)==0, fprintf(fid_brisque, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'Dong', brisque(D)); end
            if has_a_nan(T)==0, fprintf(fid_brisque, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'Tao', brisque(T)); end
            if has_a_nan(lm)==0, fprintf(fid_brisque, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'modified_LM', brisque(lm)); end
            if has_a_nan(Tpc)==0, fprintf(fid_brisque, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'corrected_Tao', brisque(Tpc)); end
            if has_a_nan(iW)==0, fprintf(fid_brisque, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'modified_Wiener', brisque(iW)); end
            if has_a_nan(dfL)==0, fprintf(fid_brisque, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'df_Landweber', brisque(dfL)); end
            if has_a_nan(mrl)==0, fprintf(fid_brisque, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'modified_RL', brisque(mrl)); end
            
            if has_a_nan(D)==0, fprintf(fid_brisque_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'Dong', brisque(Dd)); end
            if has_a_nan(T)==0, fprintf(fid_brisque_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'Tao', brisque(Td)); end
            if has_a_nan(lm)==0, fprintf(fid_brisque_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'modified_LM', brisque(lmd)); end
            if has_a_nan(Tpc)==0, fprintf(fid_brisque_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'corrected_Tao', brisque(Tpcd)); end
            if has_a_nan(iW)==0, fprintf(fid_brisque_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'modified_Wiener', brisque(iWd)); end
            if has_a_nan(dfL)==0, fprintf(fid_brisque_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'df_Landweber', brisque(dfLd)); end
            if has_a_nan(mrl)==0, fprintf(fid_brisque_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'modified_RL', brisque(mRLd)); end
            
            % kernel, noise, image, method, niqe
            if has_a_nan(D)==0, fprintf(fid_niqe, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'Dong', niqe(D)); end
            if has_a_nan(T)==0, fprintf(fid_niqe, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'Tao', niqe(T)); end
            if has_a_nan(lm)==0, fprintf(fid_niqe, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'modified_LM', niqe(lm)); end
            if has_a_nan(Tpc)==0, fprintf(fid_niqe, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'corrected_Tao', niqe(Tpc)); end
            if has_a_nan(iW)==0, fprintf(fid_niqe, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'modified_Wiener', niqe(iW)); end
            if has_a_nan(dfL)==0, fprintf(fid_niqe, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'df_Landweber', niqe(dfL)); end
            if has_a_nan(mrl)==0, fprintf(fid_niqe, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'modified_RL', niqe(mrl)); end
            
            if has_a_nan(D)==0, fprintf(fid_niqe_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'Dong', niqe(Dd)); end
            if has_a_nan(T)==0, fprintf(fid_niqe_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'Tao', niqe(Td)); end
            if has_a_nan(lm)==0, fprintf(fid_niqe_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'modified_LM', niqe(lmd)); end
            if has_a_nan(Tpc)==0, fprintf(fid_niqe_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'corrected_Tao', niqe(Tpcd)); end
            if has_a_nan(iW)==0, fprintf(fid_niqe_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'modified_Wiener', niqe(iWd)); end
            if has_a_nan(dfL)==0, fprintf(fid_niqe_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'df_Landweber', niqe(dfLd)); end
            if has_a_nan(mrl)==0, fprintf(fid_niqe_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'modified_RL', niqe(mRLd)); end
            
            % kernel, noise, image, method, piqe
            if has_a_nan(D)==0, fprintf(fid_piqe, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'Dong', piqe(D)); end
            if has_a_nan(T)==0, fprintf(fid_piqe, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'Tao', piqe(T)); end
            if has_a_nan(lm)==0, fprintf(fid_piqe, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'modified_LM', piqe(lm)); end
            if has_a_nan(Tpc)==0, fprintf(fid_piqe, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'corrected_Tao', piqe(Tpc)); end
            if has_a_nan(iW)==0, fprintf(fid_piqe, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'modified_Wiener', piqe(iW)); end
            if has_a_nan(dfL)==0, fprintf(fid_piqe, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'df_Landweber', piqe(dfL)); end
            if has_a_nan(mrl)==0, fprintf(fid_piqe, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'modified_RL', piqe(mrl)); end
            
            if has_a_nan(D)==0, fprintf(fid_piqe_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'Dong', piqe(Dd)); end
            if has_a_nan(T)==0, fprintf(fid_piqe_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'Tao', piqe(Td)); end
            if has_a_nan(lm)==0, fprintf(fid_piqe_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'modified_LM', piqe(lmd)); end
            if has_a_nan(Tpc)==0, fprintf(fid_piqe_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'corrected_Tao', piqe(Tpcd)); end
            if has_a_nan(iW)==0, fprintf(fid_piqe_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'modified_Wiener', piqe(iWd)); end
            if has_a_nan(dfL)==0, fprintf(fid_piqe_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'df_Landweber', piqe(dfLd)); end
            if has_a_nan(mrl)==0, fprintf(fid_piqe_denoise, '%s,%f,%s,%s,%f\n', 'Gauss_blur', noise_var, images{i}, 'modified_RL', piqe(mRLd)); end
            
        end
    
end

fclose(fid_psnr); 
fclose(fid_mse); 
fclose(fid_ssim); 
fclose(fid_brisque); 
fclose(fid_niqe); 
fclose(fid_piqe); 

fclose(fid_psnr_denoise); 
fclose(fid_mse_denoise); 
fclose(fid_ssim_denoise); 
fclose(fid_brisque_denoise); 
fclose(fid_niqe_denoise); 
fclose(fid_piqe_denoise); 

