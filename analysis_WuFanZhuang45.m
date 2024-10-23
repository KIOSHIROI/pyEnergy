clc;
clear;

addpath('functions');
% addpath('C:\Users\liyum\Dropbox\myFunctions\Plot_Utility\Cbrewer');
rng default  % For reproducibility

%% read data [ok]

path_to_file = '../data/WuFanZhuang-45-139040-values 20181001-20190214.csv';

raw_data = read_energy_data(path_to_file);

%% get all events and the duration of each event [ok]

[events_all, deltaT] = get_all_events(raw_data, 1, 10);

%% get single pumping events [ok]

threshold_val = 1; % [A]

[idx_monotype, monotype_events, other_events] = find_monotype_events(...
    threshold_val, events_all );

for i = 1: length(monotype_events)
    [real_P_proxy, react_P_proxy] = estimate_total_power(monotype_events{i});
    
    monotype_events{i}.real_P_proxy  = real_P_proxy;
    monotype_events{i}.react_P_proxy = react_P_proxy;
end

%% compute feature lists [ok]

[ feature_list, feature_info, idx_used_events ] = compute_feature( monotype_events );

% %% ranking features
%
% first normalize features; dimensionless features are not normalized, e.g.,
% slope estimate
max_val = repmat( max(feature_list), length(feature_list), 1);
min_val = repmat( min(feature_list), length(feature_list), 1);

feature_list_norm = (feature_list - min_val ) ./ (max_val - min_val);
feature_list_norm(:, [3,7,11]) = feature_list(:, [3,7,11]);

% perf_opt = feature_ranking(feature_list_norm, feature_info);
clear deltaT i max_val min_val path_to_file react_P_proxy real_P_proxy threshold_val
%% compute clusters

idx_feature = [2, 10];

% apply clustering algorithm
% [grps, PC] = kmeans_opt( feature_list_norm(:, idx_feature) );
grps = kmeans_elbow( feature_list_norm(:, idx_feature) );

figure; hold on;
for i = 1: length(unique(grps))
    idx = grps == i;
    scatter(feature_list(idx, idx_feature(1)), feature_list(idx, idx_feature(2)));
end

xlabel(feature_info{idx_feature(1)});
ylabel(feature_info{idx_feature(2)});

nr_clusters = length(unique(grps));
realP_perClster = splitapply(@mean, feature_list(:, 2), grps);

%%

eventSelected = other_events{5}; % 9, 5

thr_val = 5;
% signals = eventSelected.curnt_B;
signals = eventSelected.realP_tot;

[signal_reduced, idx_pairs] = reduce_signals(signals, thr_val);

%% solve a MILP problem; min|targetP - CX|

sols = signal_composition_opt1(realP_perClster, signal_reduced);

%% plot the estimated signal composite

plot_ts_estimate(sols, idx_pairs, realP_perClster, eventSelected.time_num,...
    signals, signal_reduced)

%% TODO: matching pursuit
%
% %% compute the performance using different cluster nr.
%
% S_score = zeros(10, 2);
%
% for i = 2: 10
%
%   c = kmeans(feature_list_norm(:, [2, 6, 10]), i);
%   S_coeff = silhouette(feature_list, c);
%
%   S_score(i, 1) = mean( S_coeff );
%
%   std_tmp = zeros(length(unique(c)), 1);
%   for j = 1: length(unique(c))
%     idx = c == j;
%     std_tmp(j) = std(S_coeff(idx));
%   end
%
%   S_score(i, 2) = mean(std_tmp);
% end
%
% figure; hold on;
% plot(S_score(:,1));
% plot(S_score(:,2));

% %% plot clustering sequence
%
% cc = brewermap(max(grps), 'Set1');
%
% figure; hold on;
% tmp_val = [1:length(grps)]';
%
% for i = 1: max(grps)
%   x = tmp_val(grps==i);
%   y = ones(size(x));
%
%   hf = plot(x, y, 'o');
%   set(hf, 'MarkerFaceColor', cc(i,:), 'MarkerEdgeColor', 'White');
% end
%
% %% plot time-series with highlight of pumps' type
%
% cc = brewermap(max(grps), 'Set1');
%
% ctr = 1;
%
% varName = 'real_P_proxy';
% ave_real_P = splitapply(@mean, feature_list(:,2), grps);
%
% figure; hold on;
% for i = 1: length(monotype_events)
%   tmp_event = monotype_events{i};
%
%   hf = plot(tmp_event.time_num, tmp_event.(varName));
%
%   if ( idx_used_events(i) == 1 )
%     set(hf, 'Color', cc(grps(ctr),:));
%     ctr = ctr + 1;
%   else
%     [~, idx_opt] = min(abs(ave_real_P - mean(tmp_event.(varName))));
%     set(hf, 'Color', cc(idx_opt,:));
%   end
% end