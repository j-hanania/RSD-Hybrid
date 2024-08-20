function RSD_Hybrid(img_fname, t, td, mask, cer_mask, blmethod, outfldr, saveall)
% img_fname: full filename for 4D nifti (preferably denoised w/ IHYPR4D)
% t: timing vector in mins (1 x num_frames)
% td: start time of task in minutes
% mask: 3D volume of striatal mask
% cer_mask: 3D volume of cerebellum mask
% blmethod: configuration for RSD, use either "RSD_Hybrid_IMRTM" or
%           "RSD_Hybrid_removal"
% outfldr: output directory name
% saveall: 0 or 1, set to 1 if you want to save 4D movies of fits/tacs

if blmethod~="RSD_Hybrid_IMRTM" && blmethod~="RSD_Hybrid_removal"
    error("Double check method type; type 'help RSD_Hybrid' for info.")
end

img = niftiread(img_fname);
striatum = find(mask>0);
sz = size(img);
num_frames = sz(4);
img = reshape(img,[prod(sz(1:3)),num_frames]);
tacs = img(mask>0,:);
cerebellum = find(cer_mask);
ref_tac = mean(img(cerebellum,:));

mkdir(outfldr);

tm = t - 0.5*diff([0 t]);
del_t = diff([0 t]);
[~,start_idx0]=min(abs(t-td));     % task start index

%% Run lp-ntPET
% Find TACs that are significantly different from MRTM at p<0.05
fprintf("Running lp-ntPET\n")
tic
[WRSS_lpntpet, WRSS_mrtm, fit_lpntpet_baseline, fits_lpntpet, fits_mrtm, thetas_lpntpet, tD_lpntpet]  = fit_lpntpet1(img_fname, striatum, cerebellum, t, td);
Nf = length(t);
dof = 5.5;
tDs = tD_lpntpet/60;
F = ((Nf-dof)/(dof-3)) * (WRSS_mrtm./WRSS_lpntpet - 1);
toc
w_F05 = (1-fcdf(F,4,num_frames-7))>0.05;
w_F1 = (1-fcdf(F,4,num_frames-7))>0.1;
[~,start_idx_vox]=min(abs(t-tDs),[],2);     % task start index

%% RSD -- pre-task MRTM kinetics for clustering
fprintf("\nBeginning RSD analysis...\n")

fprintf("\nBaseline method: %s\n",blmethod)

% Pre-task analysis: get R1 k2 and k2a
% fit voxel TAC up to task start time with MRTM
t1s = 0:(max(t)*60); % one second resolution
frame_edges = [0 t*60];
delta_t = diff(frame_edges);
tm1s = (frame_edges(1:end-1) + frame_edges(2:end))/2; % frame midpoints

% mean reference TAC terms: TAC for R1, integrated TAC up to t_start 
% then resample back down to TAC framing
mean_ref_int = zeros(1,num_frames);
mean_ref_interp = interp1([0 tm1s 20000], [0 ref_tac 0], t1s);
mean_ref_interp_int = cumsum(mean_ref_interp);
for i=1:num_frames
    mean_ref_int(i) = sum(mean_ref_interp_int((frame_edges(i)+1):frame_edges(i+1)))/delta_t(i);
end

BPnd = zeros(1,length(striatum));
kinetics = zeros(3,length(striatum));
full_mrtm_fits_pre = zeros(length(striatum),length(t));
X_pretask = zeros(length(striatum),length(t),3);
for v=1:length(striatum)
    start_idx = start_idx_vox(v);

    % integrate the voxel TAC for the second term
    % tac_interp = zeros(size(t1s));
    tac_interp = interp1([0 tm1s 20000], [0 tacs(v,:) 0], t1s);
    tac_interp_int = cumsum(tac_interp);
    tac_int = zeros(1,num_frames);
    for i=1:num_frames
        tac_int(i) = sum(tac_interp_int((frame_edges(i)+1):frame_edges(i+1)))/delta_t(i);
    end

    tac = tacs(v,:);
    w = del_t ./ (tac .* exp((log(2)/20.364)*(tm/60)));
    w(tac<max(tac)/100) = 0;
    W = diag(w);
    Wpre = W(1:start_idx,1:start_idx);

    X_pretask(v,:,:) = [ref_tac' mean_ref_int' -tac_int'];
    thetas = (Wpre*squeeze(X_pretask(v,1:start_idx,:))) \ (Wpre*tacs(v,1:start_idx)');
    R1=thetas(1);k2=thetas(2);k2a=thetas(3);
    kinetics(:,v) = [R1 k2 k2a];
    BPnd(v) = k2/k2a - 1;

    full_mrtm_fits_pre(v,:) = squeeze(X_pretask(v,:,:))*kinetics(:,v);
end

%% RSD -- IMRTM TACs
% iteratively remove release signal from model-derived TACs 

fits_imrtm = zeros(length(striatum),length(t));
mean_ref_int = zeros(1,start_idx0);

% mean reference TAC terms: TAC for R1, integrated TAC up to t_start 
% then resample back down to TAC framing
mean_ref_interp = interp1([0 tm1s 20000], [0 ref_tac 0], t1s);
mean_ref_interp_int = cumsum(mean_ref_interp);
for i=1:num_frames
    mean_ref_int(i) = sum(mean_ref_interp_int((frame_edges(i)+1):frame_edges(i+1)))/delta_t(i);
end

for v=1:length(striatum)
    start_idx = start_idx_vox(v);

    tac = tacs(v,:);
    w = del_t ./ (tac .* exp((log(2)/20.364)*(tm/60)));
    w(tac<max(tac)/100) = 0;
    W = diag(w);
    Wpre = W(1:start_idx,1:start_idx);

    % integrate the voxel TAC for the second term
    % tac_interp = zeros(size(t1s));
    tac_interp = interp1([0 tm1s 20000], [0 tac 0], t1s);
    tac_interp_int = cumsum(tac_interp);
    tac_int = zeros(1,num_frames);
    for i=1:num_frames
        tac_int(i) = sum(tac_interp_int((frame_edges(i)+1):frame_edges(i+1)))/delta_t(i);
    end

    X_pretask = [ref_tac' mean_ref_int' -tac_int'];
    kin_pre = (Wpre*squeeze(X_pretask(1:start_idx,:))) \ (Wpre*tac(1:start_idx)');
    mrtm_fit_pre = squeeze(X_pretask)*kin_pre;

    % iterate by replacing k2a term
    for iter=1:25
        % integrate the voxel TAC for the second term
        % tac_interp = zeros(size(t1s));
        tac_interp = interp1([0 tm1s 20000], [0 mrtm_fit_pre' 0], t1s);
        tac_interp_int = cumsum(tac_interp);
        tac_int = zeros(1,num_frames);
        for i=1:num_frames
            tac_int(i) = sum(tac_interp_int((frame_edges(i)+1):frame_edges(i+1)))/delta_t(i);
        end
        X_iter = [ref_tac' mean_ref_int' -tac_int'];
        mrtm_fit_pre = squeeze(X_iter)*kin_pre;
    end
    fits_imrtm(v,:) = mrtm_fit_pre;
end

%% Kinetic clustering / NLM weights
[data, ~, ~] = zscore(kinetics');               % put R1,k2,k2a on equal footing
nlm_kernel_h2 = 0.5;
h = sqrt(nlm_kernel_h2);                        % NLM kernel (sigma; units of z)
d = zeros(length(striatum),length(striatum));   
for i = 1:length(striatum)                      % distance of one voxel to all others in kinetic z-space
    d(i,:) = sqrt(sum((data(i,:) - data).^2,2));
end
w_nlm = exp(-d.^2/(2*h^2));                         % weight all other voxels by Gaussian kernel
w_nlm = w_nlm - eye(length(striatum));                  % remove self-weighting

%% Extract residuals
full_res = zeros(size(tacs));
full_fits = zeros(size(tacs));
if blmethod == "RSD_Hybrid_removal"
    num_fit_fcns = 2;
    full_betas = zeros(size(tacs,1),num_fit_fcns);
    X_temp = tacs;
    for i=1:length(striatum)
        if w_F05(i)==0
            X_temp(i,:) = zeros(1,num_frames);
            w_nlm(:,i) = zeros(length(striatum),1);     % remove this axis because this impacts other voxel's weight avg
        end
    end
    fprintf("%d voxels (out of %d) were significant at p<0.05 in lp-ntPET and were removed from use in baseline determination.\n",nnz(w_F05==0),nnz(striatum))
    for i=1:length(striatum)
        if w_F05(i)==0
            start_idx = start_idx_vox(i);
        else
            start_idx = start_idx0;
        end
        Ctp = double(w_nlm(i,:)*X_temp ./ sum(w_nlm(i,:)));
        Ct = double([Ctp;ref_tac]);
        Xk = double(tacs(i,:));
        full_betas(i,:) =  lsqnonneg(Ct(:,1:start_idx)', Xk(1:start_idx)');
        full_fits(i,:) = full_betas(i,:)*Ct;
        full_res(i,:) = full_fits(i,:) - Xk;
    end
elseif blmethod == "RSD_Hybrid_IMRTM"
    num_fit_fcns = 2;
    full_betas = zeros(size(tacs,1),num_fit_fcns);
    X_temp = tacs;
    w_temp = w_nlm;
    for i=1:length(striatum)
        if w_F05(i)==0
            X_temp(i,:) = fits_imrtm(i,:);
        elseif w_F05(i)==1 && w_F1(i)==0
            X_temp(i,:) = zeros(1,num_frames);
            w_temp(:,i) = zeros(length(striatum),1);     % remove this axis because this impacts other voxel's weight avg
        end
    end
    fprintf("%d voxels (out of %d) were significant at p<0.05 in lp-ntPET and had their TAC replaced using IMRTM.\n",nnz(w_F05==0),nnz(striatum))
    fprintf("%d voxels (out of %d) were significant at 0.05<p<0.1 in lp-ntPET and were removed from use in baseline determination.\n",nnz(w_F05==1 & w_F1==0),nnz(striatum))
    for i=1:length(striatum)
        if w_F05(i)==0
            start_idx = start_idx_vox(i);
        else
            start_idx = start_idx0;
        end                    
        Ctp = double(w_temp(i,:)*X_temp ./ sum(w_temp(i,:)));
        Ct = double([Ctp;ref_tac]);
        Xk = double(tacs(i,:));
        full_betas(i,:) =  lsqnonneg(Ct(:,1:start_idx)', Xk(1:start_idx)');
        full_fits(i,:) = full_betas(i,:)*Ct;
        full_res(i,:) = full_fits(i,:) - Xk;
    end
end

%% Calculate Area Under Curve and rAUC
AUC = zeros(1,length(striatum));
rAUC = zeros(size(AUC));
for i=1:length(striatum)
    % start_idx = start_idx_vox(i);
    AUC(i) = trapz(t(start_idx0:end),full_res(i,start_idx0:end));
    rAUC(i) = AUC(i) / trapz(t(start_idx0:end),tacs(i,start_idx0:end));
end

%% Run GLM on residuals
% resample to 1 min
x = 0.5:1:ceil(tm(end));
rres = double(interp1(tm, full_res', tm));
rfitted = double(interp1(tm, full_fits', tm));
norm_rres = 100*rres./rfitted;

% load in cognitive/motor and reward events
single_task_events=zeros(size(1,5000));
single_task_events(td:td+15)=3;

% convolve events with transfer function
kernel = (x/10).*exp(-x/10);
predicted_response_min = conv(single_task_events, kernel);
predicted_response_min = predicted_response_min(1:length(x));

% resample to framing
predicted_response = zeros(1,num_frames);
for i=1:num_frames
    predicted_response(i) = predicted_response_min(ceil(tm(i)));
end

% GLM with model P(t)
betas_norm=zeros(1,length(striatum));
predictors_norm=zeros(size(rres));
for i=1:length(striatum)
    % Normalize predictor to % signal change shape
    predictors_norm(:,i) = (predicted_response./rfitted(:,i)')'/max(predicted_response./rfitted(:,i)');
    % Fit predictor curve to % signal change residuals
    betas_norm(i) = predictors_norm(:,i) \ norm_rres(:,i);
end

% Second pass
beta_sig=zeros(1,length(striatum));
betas_norm_2ndpass=zeros(1,length(striatum));
new_predictors = norm_rres ./ max(norm_rres);
betas_p2uncorr = betas_norm;
betas_norm(betas_norm<0)=0;
predictors_norm_pass2 = (betas_norm*new_predictors' ./ max(betas_norm*new_predictors',[],2))';
predictors_norm_pass2(1:start_idx0-5,1)=0;
for i=1:length(striatum)
    % Fit predictor curve to % signal change residuals
    betas_norm_2ndpass(i) = predictors_norm_pass2 \ norm_rres(:,i);

    mdl=fitlm(predictors_norm_pass2, norm_rres(:,i));
    alpha_sig = 0.05;
    beta_sig(i) = mdl.Coefficients.pValue(2)<alpha_sig;
end

%% Save prediction maps
fprintf("\nSaving 3D volumes...\n")
info = niftiinfo(img_fname);
info.ImageSize = info.ImageSize(1:3);
info.PixelDimensions = info.PixelDimensions(1:3);
map = zeros(size(mask));
map(striatum) = betas_norm_2ndpass;
niftiwrite(single(map), sprintf('%s/betamap_glm_%s.nii', outfldr, blmethod), info); % beta values after updating P(t) with data
map = zeros(size(mask));
map(striatum(beta_sig>0)) = betas_norm_2ndpass(beta_sig>0);
niftiwrite(single(map), sprintf('%s/betamap_glm_sig_%s.nii', outfldr, blmethod), info); % beta values after updating P(t) with data, passing significance test
map = zeros(size(mask));
map(striatum) = betas_p2uncorr;
niftiwrite(single(map), sprintf('%s/betamap_modelP_%s.nii', outfldr, blmethod), info); % beta values with model-based P(t)
map = zeros(size(mask));
map(striatum) = rAUC;
niftiwrite(single(map), sprintf('%s/rAUC_%s.nii', outfldr, blmethod), info);    % relative area under the curve (i.e. % decrease from baseline)
map = zeros(size(mask));
map(striatum) = F;
niftiwrite(single(map), sprintf('%s/F_map_lpntPET.nii', outfldr), info);    % F statistics from lp-ntPET fitting
map = zeros(size(mask));
map(striatum) = ~w_F05;
niftiwrite(single(map), sprintf('%s/sig_lpnt.nii', outfldr), info); % voxels with F statistics passing p<0.05
map = zeros(size(mask));
map(striatum) = thetas_lpntpet(:,4);
niftiwrite(single(map), sprintf('%s/gamma_lpntpet.nii', outfldr), info);    % Gamma values from lp-ntPET

% 4D images
BL_fit = full_fits;
R=betas_norm_2ndpass'*predictors_norm_pass2';
GLM_fit = full_fits.*(1-R/100);
% 
if saveall
    fprintf("\nSaving 4D volumes...\n")
    map = zeros([size(mask(:)),length(x)]);
    sz4=[size(mask),length(x)];
    info4d = niftiinfo(img_fname);
    info4d.ImageSize = sz4;
    map(striatum,:) = double(interp1(tm, BL_fit', x))';
    niftiwrite(single(reshape(map,sz4)), sprintf('%s/fit_%s_baseline.nii', outfldr, blmethod), info4d); % RSD Baseline TACs
    map = zeros([size(mask(:)),length(x)]);
    map(striatum,:) = double(interp1(tm, GLM_fit', x))';
    niftiwrite(single(reshape(map,sz4)), sprintf('%s/fit_GLM.nii', outfldr), info4d);   % RSD voxel fit TAC
    map = zeros([size(mask(:)),length(x)]);
    map(striatum,:) = double(interp1(tm, fits_mrtm', x))';
    niftiwrite(single(reshape(map,sz4)), sprintf('%s/fits_mrtm.nii', outfldr), info4d); % MRTM fits
    map = zeros([size(mask(:)),length(x)]);
    map(striatum,:) = double(interp1(tm, fits_lpntpet', x))';
    niftiwrite(single(reshape(map,sz4)), sprintf('%s/fits_lpntpet.nii', outfldr), info4d);  % lp-ntPET Fits
    map = zeros([size(mask(:)),length(x)]);
    map(striatum,:) = double(interp1(tm, fit_lpntpet_baseline', x))';
    niftiwrite(single(reshape(map,sz4)), sprintf('%s/fits_lpntpet_baseline.nii', outfldr), info4d); % lp-ntPET first 3 terms    
    map = zeros([size(mask(:)),length(x)]);
    map(striatum,:) = double(interp1(tm, tacs', x))';
    niftiwrite(single(reshape(map,sz4)), sprintf('%s/PET_img.nii', outfldr), info4d);   % Voxel TACs
end
