%% Example Run of RSD 
% Storage requirements: 
% - with "saveall" (i.e. 4D outputs): for input volume 256x256x89, storage
%   needed is 11GB
% - without "saveall": for input volume 256x256x89, storage needed is 163MB
% - specific images can be commented out within RSD_Hybrid.m

cwd = pwd;
td = 32;   % task start time in mins
t = [1:4 6:2:10 15:5:25 27:2:65 69 74 79]; % framing; one-task, two-minute frames

input_dir = cwd;
example_data = "noisy"; % included in example are "noisy" and "noise-free"

% Read in images
if example_data == "noise-free"
    img_fname = sprintf('%s/RAC_DA_simulation_finger_grad_nf_noPSF.nii',input_dir); % noise free
    outfldr = sprintf('%s/Results_nf',cwd);
elseif example_data == "noisy"
    img_fname = sprintf('%s/RAC_DA_simulation_finger_nr1_all_frames_IH4D_7p2mm_7p2mm_7p2mm_2f_it1.nii',input_dir); % noisy data
    outfldr = sprintf('%s/Results_noisy',cwd);
end
mkdir(outfldr);

mask = niftiread(sprintf("%s/stri_mask_noedge.nii",input_dir)); % striatal mask
atlas = niftiread(sprintf("%s/simplified_seg.nii",input_dir));
cer_mask = atlas==4;    % cerebellar mask

%% Run
saveall = 1;
method_type = "RSD_Hybrid_IMRTM";
RSD_Hybrid(img_fname, t, td, mask, cer_mask, method_type, outfldr, saveall)