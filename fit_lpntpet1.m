function [WRSS_lpntpet, WRSS_mrtm, fits_lpntpet_baseline, fits_lpntpet, fits_mrtm, thetas_lpntpet, tD_lpntpet] = fit_lpntpet1(img_fname, striatum, cerebellum, t, td)

% determine number of voxels to fit over
num_vox = length(striatum);

% load in image and determine number of frames
img = niftiread(img_fname);
num_frames = size(img,4);
del_t = diff([0 t]);

% compute decay correction for TAC weighting
tm = t - 0.5*diff([0 t]);
lam = log(2)/20.364;
dc = exp(lam*tm);

% extract regional cerebellum TAC and striatal voxel TACs
cer_tac = zeros(1,num_frames);
striatal_tacs = zeros(num_vox,num_frames);
for frame=1:num_frames
    temp = squeeze(img(:,:,:,frame));
    cer_tac(frame) = mean(temp(cerebellum));
    striatal_tacs(:,frame) = temp(striatum);
end


% lp-ntPET has four terms: R1*cer_tac; k2*(integral of cer_tac); 
% k2a*(integral of striatal TAC); gamma*[integral of (striatal TAC)*
% (response function)]. The response functions are actually a library of
% pre-specified responses, so we fit separately using each function and
% pick the one with the lowest WRSS.

% let's create our response 1 functions
t_start = td*60; % start of task, in seconds
t_end = (td+10)*60; % end of scan, in seconds
tDs = (t_start-5*60):60:(t_start+5*60);
% tDs = t_start;
alphas = [0.25 1 4];
response_fn1_params = [];
% loop through all possible permutations of parameters alpha, tD, tP
for i=1:length(alphas)
    temp_alpha = alphas(i);
    for j=1:length(tDs)
        temp_tD = tDs(j);
        temp_tP = temp_tD + 60;
        while temp_tP <= (t_end+5*60) %temp_tP <= t_end 
            response_fn1_params = [response_fn1_params; temp_alpha temp_tD temp_tP];
            temp_tP = temp_tP + 60;
        end
    end
end

% let's create our response 2 functions
t_start = 49*60; % start of task, in seconds
t_end = 79*60; % end of scan, in seconds
%tDs = (t_start-5*60):60:(t_start+5*60);
tDs = t_start;
alphas = [0.25 1 4];
response_fn2_params = [];
% loop through all possible permutations of parameters alpha, tD, tP
for i=1:length(alphas)
    temp_alpha = alphas(i);
    for j=1:length(tDs)
        temp_tD = tDs(j);
        temp_tP = temp_tD + 60;
        while temp_tP <= t_start + 10*60 %temp_tP <= (t_end+5*60)
            response_fn2_params = [response_fn2_params; temp_alpha temp_tD temp_tP];
            temp_tP = temp_tP + 90;
        end
    end
end

% build up response 1 functions with one second resolution
num_response_fn1s = size(response_fn1_params,1);
t1s = 0:(max(t)*60); % one second resolution
h1 = zeros(num_response_fn1s,length(t1s));
for i=1:num_response_fn1s
    temp_alpha = response_fn1_params(i,1);
    temp_tD = response_fn1_params(i,2);
    temp_tP = response_fn1_params(i,3);
    temp_h = ((t1s-temp_tD)/(temp_tP-temp_tD)).^temp_alpha ...
        .* exp(temp_alpha*(1 - (t1s-temp_tD)/(temp_tP-temp_tD)));
    temp_h(1:temp_tD) = 0;
    h1(i,:) = temp_h;
end

% build up response 2 functions with one second resolution
num_response_fn2s = size(response_fn2_params,1);
t1s = 0:(max(t)*60); % one second resolution
h2 = zeros(num_response_fn2s,length(t1s));
for i=1:num_response_fn2s
    temp_alpha = response_fn2_params(i,1);
    temp_tD = response_fn2_params(i,2);
    temp_tP = response_fn2_params(i,3);
    temp_h = ((t1s-temp_tD)/(temp_tP-temp_tD)).^temp_alpha ...
        .* exp(temp_alpha*(1 - (t1s-temp_tD)/(temp_tP-temp_tD)));
    temp_h(1:temp_tD) = 0;
    h2(i,:) = temp_h;
end
% Override second task predictor for single task scans
num_response_fn2s = 1;
h2 = zeros(num_response_fn2s,length(t1s));

% Next we can integrate the cerebellum TAC for the second term. We
% interpolate to one second resolution, perform the integral, then resample
% to the correct framing
cer_tac_int = size(cer_tac);
frame_edges = [0 t*60];
delta_t = diff(frame_edges);
tm = (frame_edges(1:end-1) + frame_edges(2:end))/2; % frame midpoints
cer_tac_interp = interp1([0 tm 20000], [0 cer_tac 0], t1s);
cer_tac_interp_int = cumsum(cer_tac_interp);
for i=1:num_frames
    cer_tac_int(i) = sum(cer_tac_interp_int((frame_edges(i)+1):frame_edges(i+1)))/delta_t(i);
end


% Now it's time to fit! Remember we are fitting for each response
% function, for each voxel. Then we pick the best fit amongst the set of
% response function fits to determine the best model fit for each voxel.
% This can take a while, so we use parallel computing to speed things up.
WRSS_lpntpet = zeros(1,num_vox);
WRSS_mrtm = zeros(1,num_vox);
fits_lpntpet_baseline = zeros(num_vox,num_frames);
fits_lpntpet = zeros(num_vox,num_frames);
fits_mrtm = zeros(num_vox,num_frames);
thetas_lpntpet = zeros(num_vox,4);
tD_lpntpet = zeros(num_vox,1);
parfor n=1:num_vox
    % extract a single TAC
    tac = striatal_tacs(n,:);
    % compute weights
    %{
    w = del_t./(tac.*dc);
    W = diag(w);
    %}
    w = del_t ./ (tac .* exp((log(2)/20.364)*(tm/60)));
    %w = 1 ./ exp((log(2)/20.364)*tm);
    w(tac<max(tac)/100) = 0;
    W = diag(w);
    % W = eye(size(W));

    % integrate the TAC for the third term
    tac_interp = interp1([0 tm 20000], [0 tac 0], t1s);
    tac_interp_int = cumsum(tac_interp);
    tac_int = zeros(1,num_frames);
    for i=1:num_frames
        tac_int(i) = sum(tac_interp_int((frame_edges(i)+1):frame_edges(i+1)))/delta_t(i);
    end
    
    % Now let's fit with each response function and compute the WRSS for
    % each fit. The weights are uniform in this case, since HYPR-denoised
    % images contain data across all frames.
    WRSS_temp = zeros(num_response_fn1s,num_response_fn2s);
    fits_temp = zeros(num_response_fn1s,num_response_fn2s,num_frames);
    fits_first3_temp = zeros(num_response_fn1s,num_response_fn2s,num_frames);
    thetas = zeros(num_response_fn1s,num_response_fn2s,4);
    for j=1:num_response_fn1s
        for k=1:num_response_fn2s
            h_temp = h1(j,:);
            product1_interp = tac_interp .* h_temp;
            product1_interp_int = cumsum(product1_interp);
            product1_int = zeros(1,num_frames);
            for i=1:num_frames
                product1_int(i) = sum(product1_interp_int((frame_edges(i)+1):frame_edges(i+1)))/delta_t(i);
            end

            h_temp = h2(k,:);
            product2_interp = tac_interp .* h_temp;
            product2_interp_int = cumsum(product2_interp);
            product2_int = zeros(1,num_frames);
            for i=1:num_frames
                product2_int(i) = sum(product2_interp_int((frame_edges(i)+1):frame_edges(i+1)))/delta_t(i);
            end

            % build up X matrix and fit
            X = [cer_tac' cer_tac_int' -tac_int' -product1_int'];% -product2_int'];
            thetas_temp = lsqnonneg((W*X), (W*tac'));
%             thetas_temp = (W*X) \ (W*tac');
            fit_fn = X*thetas_temp;
            fits_temp(j,k,:) = fit_fn';
            fits_first3_temp(j,k,:) = X(:,1:3)*thetas_temp(1:3);
            WRSS_temp(j,k) = sum(w.*(fit_fn'-tac).^2);
            thetas(j,k,:) = thetas_temp;
        end
    end
    
    % Now find the minimum WRSS over all the response functions
    index = find(WRSS_temp(:)==min(WRSS_temp(:)));
    index = index(1);
    [index1, index2] = ind2sub(size(WRSS_temp), index);
    WRSS_lpntpet(n) = WRSS_temp(index1, index2);
    fits_lpntpet(n,:) = fits_temp(index1,index2,:);
    fits_lpntpet_baseline(n,:) = fits_first3_temp(index1,index2,:);
    thetas_lpntpet(n,:) = thetas(index1,index2,:);
    tD_lpntpet(n,:) = response_fn1_params(index1,2);

    % now MRTM fit
    X = [cer_tac' cer_tac_int' -tac_int'];
    thetas = (W*X) \ (W*tac');
    fit_fn = X*thetas;
    fits_mrtm(n,:) = fit_fn';
    WRSS_mrtm(n) = sum(w.*(fit_fn'-tac).^2);
end