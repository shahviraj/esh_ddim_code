clear
clc

Nt = 32; Nr = 4;
At = []; Ar = [];
anglegrid_tx = zeros(Nt, 1);
anglegrid_rx = zeros(Nr, 1);

% Creating a grid for the sin values
for a = 1:Nt
    anglegrid_tx(a) = 2 / Nt * (a - 1) - 1;
end
for a = 1:Nr
    anglegrid_rx(a) = 2 / Nr * (a - 1) - 1;
end

% Creating the DFT matrix
for itx = 1:Nt
    at = 1 / sqrt(Nt) * exp(1i * pi * (0:(Nt - 1))' * anglegrid_tx(itx));
    At = [At, at];
end
for irx = 1:Nr
    ar = 1 / sqrt(Nr) * exp(1i * pi * (0:(Nr - 1))' * anglegrid_rx(irx));
    Ar = [Ar, ar];
end

% Channel model setup and coefficient generation
s = qd_simulation_parameters;
s.show_progress_bars = 1;
s.center_frequency = 28e9;
num_samples = 100000;

% Layout and Channel Generation
l = qd_layout(s);
l.no_rx = num_samples;
l.randomize_rx_positions(100, 1.2, 2.0, 0, [], 10, 0);

% Scenario selection
BerUMaL = 'BERLIN_UMa_LOS';
l.set_scenario(BerUMaL);

% Antenna arrays (Tx and Rx)
l.tx_array = qd_arrayant('3gpp-3d', 1, 32, s.center_frequency(1));
l.rx_array = qd_arrayant('3gpp-3d', 1, 4, s.center_frequency(1));
l.no_tx = 1;
l.tx_position = [0 0 25]';

% Build and generate channel coefficients
p = l.init_builder;
p.gen_parameters;
c = p.get_channels;

tic;

H_set = zeros(num_samples, l.rx_array(1).no_elements, l.tx_array(1).no_elements);

for ue_idx = 1:num_samples
    num_subcarrier = 1;  % Narrowband setting
    h_t = c(ue_idx).fr(20e6, num_subcarrier);  % Frequency-domain channel
    H_set(ue_idx, :, :) = h_t;

    toc_temp = toc;
    if mod(ue_idx, num_samples / 10) == 0
        disp(['Processing: ', num2str(ue_idx / num_samples * 100), '% completed, Elapsed time: ', num2str(toc_temp)]);
    end
end

% Collect Receiver Positions
rx_positions = zeros(num_samples, 3);
for ue_idx = 1:num_samples
    rx_positions(ue_idx, :) = c(ue_idx).rx_position;
end

% Save the results
basepath = "./Data_folder";
storage_path = strcat(basepath, "/Data_Narrowband(CH+UEposition)_Nt", string(l.tx_array(1).no_elements), "_Nr", string(l.rx_array(1).no_elements));
if ~isfolder(storage_path)
    mkdir(storage_path);
end

DIRNAME = datestr(now, 'yyyymmdd');
filename = strcat(storage_path, "/NumUEs_", string(num_samples), "_date", DIRNAME, datestr(now, 'HHMMSS'), "multiuser_ULA.mat");

save(filename, "H_set", "rx_positions");
