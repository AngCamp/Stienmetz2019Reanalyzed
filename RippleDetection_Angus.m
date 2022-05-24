clearvars
clc
%%
% cd 'C:\Users\angus\Desktop\SteinmetzLab\Analysis'
% loading the lfp data from steinmetz
% the timining information is in the main dataset will allow you to correct
% for the clock drift in the probes
% Each LFP file has 385 channels of int16 data at 2500Hz. 
% They can be loaded directly into software like Neuroscope, and can be
% read in matlab with this code, note [385 inf] in fread will load the full
% session for the probe this is not recomended

%Neuroinformatics class from Imperial College London, has a lecture on LFP
% https://www.ucl.ac.uk/cortexlab/neuroinformatics-class-page
% Session is Tatum-12-09
fid = fopen('C:\Users\angus\Desktop\SteinmetzLab\LFP\Tatum_2017-12-09_K1_g0_t0.imec.lf.bin', 'r');
dat = fread(fid, [385 Inf], '*int16'); %385 specifies channels
fclose(fid);

% We will need this for 

dat = dat(349:383, :); % these are the channels in CA3 for this session according to the allen ontology file
% we need to match the samples to the time on the behavioural data, remove
% the sections near movement 
dat = dat(20, :); % picking one channel at random from the CA3 ones so I can test this script

% animal_ID = num2str(184858);
date = '2021_06_17'; 
%root_path_save = ['\\mohajerani-nas.uleth.ca\storage\scratch\treadmill_widefield\lfp\'];
%root_path_load = root_path_save; 
% root_directory = fullfile(root_path_load, animal_ID);
%root_directory = root_path_save;
% ripples_path = fullfile(root_path_save,animal_ID,date,'ripples');
%LFP_path = fullfile(root_directory, date,'LFP');
%LFP_path = root_directory;

session_number = {'01'};
% session_NumofFrames = [30000]; %left this here because its where it
% originally was in the script, but value needed adjusting for my needs
DHPC_artifact_threshold_multiplier = 8;
DHPC_ripple_threshold_multiplier_values = [3, 4];
DHPC_duration_threshold_multiplier_values = [0,0];
camera_srate = 100; %this is a pointless parameter, it is never used in the script
srate = 2500; % I think this is sample rate, so for us should be 2500

win = round(0.008*srate); %window, this window matches the one from steinmetz
session_NumofFrames = [round(length(dat)/win)]; 
interripple_interval_min = 0.1*srate;
asd = 0.75;
bsd = 0.75; % fraction of thresh for spread of spindle in right direction

% unsure what these values do
trim = 0;
rows = 128;
columns = 128;

before_event = 1; %1 second
after_event = 1;  %1 second
% this is a dictionary
%this will creat a vector of ripples each with all this information
Ripples.parameters = struct('experiment_date', date, 'camera_srate', camera_srate, ...
    'LFP_srate', srate, 'RMS_smoothing_win_size', win, ...
    'minimum_interripple_interval', interripple_interval_min, ...
    'asd', asd, 'bsd', bsd, ...
    'artifact_free_radius_inseconds', after_event, ...
    'DHPC_artifact_threshold_multiplier', DHPC_artifact_threshold_multiplier, ...
    'minimum_ripple_duration_multiplier', DHPC_duration_threshold_multiplier_values);

%%
%for session_counter = 1:length(session_number)
session_counter = 1; % just here to fill in the for loop index, may remove loop later after debugging
    Ripples.(['session_' session_number{session_counter}]).NumberofFrames = ...
        session_NumofFrames(session_counter);
    LFP_name = [date '_00' session_number{session_counter}]; %this should be changed to session name from steinmetz
    nFrames = session_NumofFrames(session_counter);
    
    % loading the LFP data, we can switch this out
    %[eeg,si] = abfload(fullfile(LFP_path, [LFP_name, '.abf']), 'start', 0, 'stop', 'e'); 
    %srate = 1e6/si; % sampling rate, may need to be changed to 2500 to match hertz of neuropixels
    % we already set the srate variable not sure why it comes up again and
    % is recalculated
    decimation_factor = srate/1000;
    decimation_factor = round(decimation_factor);
    %eeg = eeg(1:decimation_factor:end,:);
    eeg = dat(1:decimation_factor:end,:); % my version
    %srate = 1000; % needs to be cahnged to 2500 possibly, need to check
    camera_clock = eeg(:,5); % these lines may need to be replaced by first and last sample from the alyx files
    camera_function = eeg(:,8);
   
   % reading in the intervals file we are using 
   T = readtable('myfile.csv');
 
%That's just linearly interpolating between the times given for the first and last samples.
%So, to get the time at which every sample occurred, use this line in matlab:
%include a line here that can autodetect which session we are in so we can
%call the file

tsData = readNPY('...lf.timestamps.npy'); allTS = interp1(tsData(:,1), tsData(:,2), tsData(1,1):tsData(2,1));


    
    clock_indx = []; % this may require information from the Alyx files
    for ii = 1:size(camera_function,1)
        if camera_function(ii,1) > 2 && camera_clock(ii,1) < 2 && camera_clock(ii+1,1) > 2
            clock_indx = [clock_indx ii];
        end
    end
    
    if size(clock_indx,2) > nFrames
        clock_indx = clock_indx(1,1:nFrames);
    end
    
    %these are the timing we need to correspond to the times in the .times
    %and .intervals npy-objects, also we will need it to match the
    %intervals we supply
    first_clock = min(clock_indx);
    last_clock = max(clock_indx);
    
    clock_indx = clock_indx - first_clock + 1; %counting camera clock from 1
    clock_indx_trimed = clock_indx(1,trim+1:nFrames-trim);
    % this  to remove signal from 60, 120, and 180 Hz this will remove
    % ambiant electricity from AC wiring
    DHPC_LFP_bipolar = eeg(first_clock:last_clock,1);   
    DHPC_LFP_bipolar = DHPC_LFP_bipolar - mean(DHPC_LFP_bipolar); % this is really important Ahbjit says keep this
    
    %     d = designfilt('bandstopiir','FilterOrder',2, ...
    %         'HalfPowerFrequency1',59,'HalfPowerFrequency2',61, ...
    %         'DesignMethod','butter','SampleRate',srate);
    %     DHPC_LFP_bipolar =  filtfilt(d,DHPC_LFP_bipolar);
    %
    %     d = designfilt('bandstopiir','FilterOrder',2, ...
    %         'HalfPowerFrequency1',119,'HalfPowerFrequency2',121, ...
    %         'DesignMethod','butter','SampleRate',srate);
    %     DHPC_LFP_bipolar =  filtfilt(d,DHPC_LFP_bipolar);
    %
    %     d = designfilt('bandstopiir','FilterOrder',2, ...
    %         'HalfPowerFrequency1',179,'HalfPowerFrequency2',181, ...
    %         'DesignMethod','butter','SampleRate',srate);
    %     DHPC_LFP_bipolar =  filtfilt(d,DHPC_LFP_bipolar);
    
    tvec =0:1/srate: 1/srate * (size(DHPC_LFP_bipolar,1)-1); % converting time to seconds
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%% Check for LFP artifacts %%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Checks for aberantly high activity as defined as being X standard
    % (set by this variable DHPC_articfact_threshold_multiplier)
    % deviations from the mean, then makring the index in the second line
    % DHPC_artifacts_indx, and then plotting it for visual inspection
    % 
    DHPC_artifact_threshold = [mean(DHPC_LFP_bipolar) + DHPC_artifact_threshold_multiplier*std(DHPC_LFP_bipolar),...
        mean(DHPC_LFP_bipolar) - DHPC_artifact_threshold_multiplier*std(DHPC_LFP_bipolar)];
    DHPC_artifacts_indx = find(DHPC_LFP_bipolar > DHPC_artifact_threshold(1) | DHPC_LFP_bipolar < DHPC_artifact_threshold(2));
    figure; plot(0:1/srate:(length(DHPC_LFP_bipolar)-1)/srate,DHPC_LFP_bipolar,DHPC_artifacts_indx/srate,DHPC_LFP_bipolar(DHPC_artifacts_indx),'.r'); grid on; shg
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     % filtering method for calculaing ripple power signal
%     filter_coefficients = load('RippleFilter_100_250_2000Hz.txt');
%     DHPC_LFP_monopolar_filtered_100_250 = filtfilt(filter_coefficients,1,DHPC_LFP_bipolar);
%     DHPC_rms = conv(DHPC_LFP_monopolar_filtered_100_250 .^2, ones(win,1), 'same');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % wavelt method for calculaing ripple power signal
    NumVoices = 64;
    a0 = 2^(1/NumVoices);
    wavCenterFreq = centfrq('morl'); % requires Wavelet Toolbox, 
    wavCenterFreq = 6/(2*pi); %this replaces the first one, check which is useful
    minfreq = 110;
    maxfreq = 250;
    f0 = 6/(2*pi);
    dt = 1/srate;
    scales = helperCWTTimeFreqVector(minfreq,maxfreq,wavCenterFreq,dt,NumVoices);
    DHPC_LFP_monopolar_filtered_100_250 = cwtft({DHPC_LFP_bipolar,dt},...
        'scales',scales,'wavelet','morl'); %cwtft is continuous wavelet transform
    % documentation at mathworks.com/help/wavelet/ref/cwt.html
    
    % this next line is possibly averaging across channels (electrodes),
    % root mean squared ( but looks like its just the mean)
    % win is defined on line 35, these lines are smoothing over an 8ms
    % window, may need to be changed to the steinmetz method
    DHPC_rms = conv(mean(abs(DHPC_LFP_monopolar_filtered_100_250.cfs).^2,1), ones(win,1), 'same');
    DHPC_LFP_monopolar_filtered_100_250 = mean(real(DHPC_LFP_monopolar_filtered_100_250.cfs));            
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for threshold_counter = 1:length(DHPC_ripple_threshold_multiplier_values)
        DHPC_ripple_threshold_multiplier = DHPC_ripple_threshold_multiplier_values(threshold_counter);
        DHPC_duration_threshold_multiplier = DHPC_duration_threshold_multiplier_values(threshold_counter);
        
        DHPC_thresh = mean(DHPC_rms(~ismember(1:length(DHPC_rms),...
            [DHPC_artifacts_indx-srate/200; DHPC_artifacts_indx; DHPC_artifacts_indx+srate/200])))...
            + DHPC_ripple_threshold_multiplier*std(DHPC_rms(~ismember(1:length(DHPC_rms),...
            [DHPC_artifacts_indx-srate/200; DHPC_artifacts_indx; DHPC_artifacts_indx+srate/200])));
        
        lsd = DHPC_thresh*asd;
        rsd = DHPC_thresh*bsd;
        
        [p,ip] = findpeaks(DHPC_rms,'minpeakheight',DHPC_thresh); %this could be the  value we need
        len = length(p);
        
        clear S E S_HPC E_HPC
        S = [];
        E = [];
        
        for ii = 1:len
            jj = ip(ii);
            while jj > 0 && DHPC_rms(jj) > lsd
                jj = jj - 1;
            end
            if jj
                S =union(S, jj);
            else
                S = union(S, ip(1));
            end
            jj = ip(ii);
            while jj < length(tvec) && DHPC_rms(jj) > rsd
                jj = jj + 1;
            end
            if jj < length(tvec)
                E = union(E, jj);
            else
                E = union(E, ip(end));
            end
        end
        
        DHPC_ripples_indx =[S', E'];
        
        minimum_ripple_duration = mean(DHPC_ripples_indx(:,2) - DHPC_ripples_indx(:,1)) + ...
            DHPC_duration_threshold_multiplier*std(DHPC_ripples_indx(:,2) - DHPC_ripples_indx(:,1));
        DHPC_ripples_indx(DHPC_ripples_indx(:,2) - DHPC_ripples_indx(:,1) < minimum_ripple_duration,:) = [];
        ripples_midpoints = mean(DHPC_ripples_indx, 2);
        a = find((ripples_midpoints(2:end) - ripples_midpoints(1:end-1,:)) < interripple_interval_min);
        DHPC_ripples_indx(a,2) = DHPC_ripples_indx(a+1,2);
        DHPC_ripples_indx(a+1,:) = [];
        clear a
        
        DHPC_ripples_centers_indx = zeros(size(DHPC_ripples_indx,1),1);
        for ii =1:size(DHPC_ripples_indx,1)
            [x, y] = findpeaks( -DHPC_LFP_monopolar_filtered_100_250(DHPC_ripples_indx(ii,1):DHPC_ripples_indx(ii,2)) ); %detecting ripple troughs
            DHPC_ripples_centers_indx(ii) = y(x == max(x) )+ DHPC_ripples_indx(ii,1) - 1; %index of largest trough
        end
        
        DHPC_ripples_centers_frames = []; %the frame corresponding to a negative peak
        for ii = 1:length(DHPC_ripples_centers_indx)
            [a, b] = min(abs(clock_indx_trimed - DHPC_ripples_centers_indx(ii)));
            DHPC_ripples_centers_frames = [DHPC_ripples_centers_frames; b];
        end
        
        non_empty = [];
        for ii= 1:length(DHPC_ripples_centers_indx)
            if DHPC_ripples_centers_indx(ii)-before_event*srate > 0 &&...
                    DHPC_ripples_centers_indx(ii)+after_event*srate < length(DHPC_LFP_bipolar) && ...
                    isempty(intersect(DHPC_ripples_centers_indx(ii)-before_event*srate:DHPC_ripples_centers_indx(ii)+after_event*srate, ...
                    DHPC_artifacts_indx)) && ...
                    isempty(intersect(DHPC_ripples_centers_indx(ii)-before_event*srate:DHPC_ripples_centers_indx(ii)+after_event*srate, ...
                    DHPC_artifacts_indx))
                
                non_empty = [non_empty ii];
            end
        end
        DHPC_ripples_centers_indx = DHPC_ripples_centers_indx(non_empty);
        DHPC_ripples_centers_frames = DHPC_ripples_centers_frames(non_empty);
        DHPC_ripples_indx = DHPC_ripples_indx(non_empty,:);
        
              
        figure;
        ax1 = subplot(2,1,1);
        plot(0:1/srate:(length(DHPC_LFP_bipolar)-1)/srate, DHPC_LFP_bipolar); grid on; hold on;
        plot(DHPC_ripples_centers_indx/srate, DHPC_LFP_bipolar(DHPC_ripples_centers_indx),'ok');
        plot(DHPC_ripples_indx(:,1)/srate, DHPC_LFP_bipolar(DHPC_ripples_indx(:,1)),'*r',DHPC_ripples_indx(:,2)/srate, DHPC_LFP_bipolar(DHPC_ripples_indx(:,2)), '*c');
        title({['sesseion ', session_number{session_counter}]; ...
            ['thrshld multiplier ' num2str(DHPC_ripple_threshold_multiplier)]; ...
            ['dur thrshld multiplier ' num2str(DHPC_duration_threshold_multiplier)]});
        ax2 = subplot(2,1,2);
        plot(0:1/srate:(length(DHPC_LFP_bipolar)-1)/srate, DHPC_rms); grid on; hold on;
        plot(DHPC_ripples_indx(:,1)/srate, DHPC_rms(DHPC_ripples_indx(:,1)),'*r',DHPC_ripples_indx(:,2)/srate, DHPC_rms(DHPC_ripples_indx(:,2)), '*c');
        plot(DHPC_ripples_centers_indx/srate, DHPC_rms(DHPC_ripples_centers_indx),'ok');
        line([0 (length(DHPC_LFP_bipolar)-1)/srate],[DHPC_thresh DHPC_thresh],'color', 'r');
        linkaxes([ax1 ax2], 'x');
        
        Ripples.(['session_' session_number{session_counter}]).DHPCThresholdMultiplier(threshold_counter) = ...
            DHPC_ripple_threshold_multiplier;
        Ripples.(['session_' session_number{session_counter}]).DHPCRippleCentersIndx{threshold_counter} = ...
            DHPC_ripples_centers_indx;
        Ripples.(['session_' session_number{session_counter}]).DHPCRippleCentersFrames{threshold_counter} = ...
            DHPC_ripples_centers_frames;
    end
end
%%
save(fullfile(ripples_path, 'Ripples_screened.mat'),'Ripples','-v7.3');



