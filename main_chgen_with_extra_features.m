% =========================================================================
% GENERATE COMPREHENSIVE WIRELESS CHANNEL DATASET WITH QUADRIGA
%
% Description:
% This script uses the Quadriga channel simulator to generate a dataset of
% narrowband MIMO channel matrices. In addition to the channel data (H_set)
% and receiver positions (rx_positions), it extracts and saves key wireless
% channel parameters to be used as conditioning variables for an AI model.
%
% Saved Features:
% 1. H_set: Channel matrices [num_samples x Nr x Nt]
% 2. rx_positions: 3D coordinates of the receiver [num_samples x 3]
% 3. los_conditions: Line-of-Sight flag (1 for LoS, 0 for NLoS)
% 4. path_losses_dB: Overall path loss in dB
% 5. k_factors_dB: Ricean K-factor in dB
% 6. rms_delay_spreads_s: RMS delay spread in seconds
% 7. Nt: Number of transmitter antennas (saved as a scalar)
% 8. Nr: Number of receiver antennas (saved as a scalar)
%
% Instructions:
% 1. Ensure the Quadriga toolbox is installed and added to your MATLAB path.
% 2. Modify the parameters in the "CONFIGURATION" section as needed.
% 3. Run the script. A .mat file will be saved in a structured directory.
% 4. To generate data for a different antenna setup, change Nt and Nr and
%    rerun the script.
%
% =========================================================================

clear
clc

% =========================================================================
% I. CONFIGURATION
% =========================================================================
% --- Antenna Configuration ---
% Modify these values to generate different datasets
Nt = 32; % Number of transmitter antennas
Nr = 4;  % Number of receiver antennas

% --- Simulation Parameters ---
num_samples = 100000;       % Total number of channel samples to generate
center_frequency = 28e9;  % Carrier frequency in Hz (e.g., 28 GHz)
scenario = 'BERLIN_UMa_LOS'; % Quadriga scenario (e.g., Urban Macro LoS)

% --- File Storage ---
basepath = "./Data_folder"; % Main directory to store generated data

% =========================================================================
% II. SIMULATION SETUP
% =========================================================================
fprintf('Starting channel generation for Nt=%d, Nr=%d...\n', Nt, Nr);

% Setup basic simulation parameters
s = qd_simulation_parameters;
s.show_progress_bars = 1;
s.center_frequency = center_frequency;

% Create the layout
l = qd_layout(s);
l.no_rx = num_samples;
l.randomize_rx_positions(100, 1.2, 2.0, 0, [], 10, 0); % Randomize UE positions

% Set the propagation scenario
l.set_scenario(scenario);

% Configure antenna arrays for Transmitter (Tx) and Receiver (Rx)
l.tx_array = qd_arrayant('3gpp-3d', 1, Nt, s.center_frequency(1));
l.rx_array = qd_arrayant('3gpp-3d', 1, Nr, s.center_frequency(1));

% Set a single transmitter at a fixed position
l.no_tx = 1;
l.tx_position = [0 0 25]';

% Build and generate channel coefficients
p = l.init_builder;
p.gen_parameters;
c = p.get_channels;

fprintf('Channel objects generated. Starting feature extraction...\n');
tic;

% =========================================================================
% III. DATA AND FEATURE EXTRACTION
% =========================================================================
% Initialize arrays to store the dataset
H_set = zeros(num_samples, Nr, Nt, 'single'); % Use single precision to save space
rx_positions = zeros(num_samples, 3, 'single');
los_conditions = zeros(num_samples, 1, 'logical');
path_losses_dB = zeros(num_samples, 1, 'single');
k_factors_dB = zeros(num_samples, 1, 'single');
rms_delay_spreads_s = zeros(num_samples, 1, 'single');

for ue_idx = 1:num_samples
    % Extract narrowband frequency-domain channel response
    num_subcarrier = 1;
    h_t = c(ue_idx).fr(20e6, num_subcarrier);
    H_set(ue_idx, :, :) = h_t;

    % 1. Receiver Position
    rx_positions(ue_idx, :) = c(ue_idx).rx_position;
    
    % 2. Line-of-Sight (LoS) Condition
    los_conditions(ue_idx) = c(ue_idx).los_condition;
    
    % 3. Path Loss in dB
    path_gain = sum(c(ue_idx).par.pg, 'all');
    path_losses_dB(ue_idx) = -10 * log10(path_gain);
    
    % 4. K-factor in dB (will be -Inf for pure NLoS)
    k_factors_dB(ue_idx) = c(ue_idx).par.K_factor_dB;
    
    % 5. RMS Delay Spread in seconds
    delays = c(ue_idx).par.delay;
    powers = c(ue_idx).par.pg;
    mean_delay = sum(delays .* powers) / sum(powers);
    mean_square_delay = sum((delays.^2) .* powers) / sum(powers);
    rms_delay_spreads_s(ue_idx) = sqrt(mean_square_delay - mean_delay^2);

    % Progress indicator
    if mod(ue_idx, num_samples / 10) == 0
        toc_temp = toc;
        fprintf('Processing: %d%% completed. Elapsed time: %.2f seconds.\n', ...
            round(ue_idx / num_samples * 100), toc_temp);
    end
end

total_time = toc;
fprintf('Feature extraction complete. Total time: %.2f seconds.\n', total_time);

% =========================================================================
% IV. SAVE DATASET TO FILE
% =========================================================================
% Create a descriptive directory name based on antenna config
storage_path = fullfile(basepath, sprintf('Data_Nt%d_Nr%d', Nt, Nr));
if ~isfolder(storage_path)
    mkdir(storage_path);
    fprintf('Created directory: %s\n', storage_path);
end

% Create a unique filename with timestamp
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
filename = fullfile(storage_path, sprintf('Channels_NumUEs%d_%s.mat', num_samples, timestamp));

fprintf('Saving dataset to: %s\n', filename);

% Save all relevant variables to the .mat file
% The '-v7.3' flag is important for saving files >2GB
save(filename, ...
    "H_set", ...
    "rx_positions", ...
    "los_conditions", ...
    "path_losses_dB", ...
    "k_factors_dB", ...
    "rms_delay_spreads_s", ...
    "Nt", ...
    "Nr", ...
    "-v7.3");

fprintf('Dataset saved successfully.\n');
% =========================================================================
% END OF SCRIPT
% =========================================================================
