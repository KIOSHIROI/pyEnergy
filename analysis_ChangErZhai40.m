clc;
clear;

addpath('functions');
% addpath('C:\Users\liyum\Dropbox\myFunctions\Plot_Utility\Cbrewer');

%% read data [ok]

path_to_file = '../data/ChangErZhai-40-139079-values 20180101-20181031.csv';
raw_data = read_energy_data(path_to_file);

% path_to_file = '../data/Hengyuan1-301 data/Hengyuan1-301-values201703-13-20181031_complete.csv';
path_to_file = '../data/Hengyuan1-301 data/Hengyuan1-301-values20180101-20181031.csv';
hengyuan_data = readtable(path_to_file);
hengyuan_data.workingPower = 10*cat(1, 0, diff(hengyuan_data.Cum_Quant) ./ hours(diff(hengyuan_data.UTCTime)));

%% compare hengyuan data and raw data

datetimeRange_start = datetime(datevec(['2018-03-01 23:00:00'; '2018-03-29 23:00:00'; '2018-05-29 23:00:00'; '2018-07-06 18:00:00'; '2018-08-08 23:00:00'; '2018-08-24 23:00:00'; '2018-10-16 18:00:00']));
datetimeRange_end = datetime(datevec(['2018-03-18 23:00:00'; '2018-04-13 23:00:00'; '2018-06-12 23:00:00'; '2018-07-07 23:00:00'; '2018-08-12 23:00:00'; '2018-08-26 23:00:00'; '2018-10-17 23:00:00']));

close all ;

figure;
for i = 1: length(datetimeRange_end)
    subplot(4, 2, i);  hold on;
    h2 = area(datetime(datevec(raw_data.time_num)), raw_data.realP_tot);
    h1 = plot(hengyuan_data.UTCTime, hengyuan_data.workingPower);

    set(h2, 'FaceColor', [191 191 191]/255, 'EdgeColor', [191 191 191]/255, 'FaceAlpha', 0.6);
    set(h1, 'Color', [228, 35, 32]/255, 'LineWidth', 1.5);
    ylim([0, 20]);
    xlim([datetimeRange_start(i), datetimeRange_end(i)]);
    ylabel('Working Power [kW]');
    box on; grid on;
end

%% get all events and the duration of each event [ok]
thre_value = 1;  % a minimal current to be treated as a pump;
thre_time = 10;  % a minimal operating time for a pump;

[events_all, ~] = get_all_events(raw_data, thre_value, thre_time);


%% get single pumping events [ok]

% threshold value used to distinguish the single pumping instances from multi-pumping ones; [A]
threshold_val = 1;

[idx_monotype, monotype_events, other_events] = find_monotype_events(...
    threshold_val, events_all);


for i = 1: length(monotype_events)
    [real_P_proxy, react_P_proxy] = estimate_total_power(monotype_events{i});

    monotype_events{i}.real_P_proxy  = real_P_proxy;
    monotype_events{i}.react_P_proxy = react_P_proxy;
end

plot_ts_area(monotype_events{1}, 'curnt_B');
plot_ts_area(other_events{1}, 'curnt_B');

%% boxplot for the variability of monotype and compisite events

varName = 'realP_tot';
val = [];
G = [];
for i = 1: length(monotype_events)
    val = cat(1, val, monotype_events{i}.(varName));
    G = cat(1, G, ones(length( monotype_events{i}.(varName)), 1 )*i );
end

val2 = [];
G2 = [];
for i = 1: length(other_events)
    val2 = cat(1, val2, other_events{i}.(varName));
    G2 = cat(1, G2, ones(length( other_events{i}.(varName)), 1 )*i );
end

figure;
subplot(2,1,1);
hold on; box on; grid on;
h1 = boxplot(val, G);
set(h1(7,:), 'Visible', 'off');
set(gca, 'XTickLabelRotation', 90);
ylabel('Power [kW]');
title('monotype events');

subplot(2,1,2);
hold on; box on; grid on;
h2 = boxplot(val2, G2);
set(h2(7,:), 'Visible', 'off');
set(gca, 'XTickLabelRotation', 90);
ylabel('Power [kW]');
title('composite events');

% Set width, height, and orientation for A4 landscape
set(gcf,'PaperPositionMode','auto');
set(gcf,'PaperOrientation','landscape');

% Save the figure as a PDF
filename = 'boxplot_events.pdf';
print(gcf, filename, '-dpdf', '-bestfit');

%% compare raw signal and filtered (reduced) signals for a monotype event
close all;

thr_val = 5;
figure;

ax1 = subplot(1,2,1);
eventSelected = monotype_events{22};
signals = eventSelected.realP_tot;
% smooth signals
[signal_reduced, idx_pairs] = reduce_signals(signals, thr_val);
plot_ts_filter(ax1, idx_pairs,  eventSelected.time_num, signals, signal_reduced);

ax2 = subplot(1,2,2);
eventSelected = monotype_events{24};
signals = eventSelected.realP_tot;
% smooth signals
[signal_reduced, idx_pairs] = reduce_signals(signals, thr_val);
plot_ts_filter(ax2, idx_pairs,  eventSelected.time_num, signals, signal_reduced);

print(gcf, 'event_filtering.pdf', '-dpdf', '-bestfit');

%% Identify possible total nr. of pumps

thr_val = 5;
maxPumpNrs = 0;
diff_thrVal = 1; % if the difference of reduced signal for two events is less than 1A, we consider the two events coming from one pump

for i = 1: length(other_events)
    signals = other_events{i}.curnt_B;
    [signal_reduced, ~] = reduce_signals(signals, thr_val);

    maxPumpNrs_new = sum(diff(sort(signal_reduced)) > diff_thrVal) + 1;
    maxPumpNrs = max(maxPumpNrs, maxPumpNrs_new);
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
% feature_list_norm(:, [3,7,11]) = feature_list(:, [3,7,11]); % keep the
% trend values without normlizing them

% perf_opt = feature_ranking(feature_list_norm, feature_info);
clear deltaT i max_val min_val path_to_file react_P_proxy real_P_proxy threshold_val

%% TODO: add PCA analysis ??

[coeff, score, latent, tsquared, explained] = pca(feature_list_norm);

figure;
biplot(coeff(:,1:2),'scores',score(:,1:2),'varlabels',feature_info);


% correlation matrix
cor_matrix = corr(feature_list_norm);

figure;
yticks = 1.5: 11.5;
xticks = 1.5: 11.5;

hf = imagesc(cor_matrix);
colormap(flipud(parula));

% Draw the grid lines manually
hold on;
for x = xticks
    plot([x, x], ylim, 'Color', [1 1 1], 'LineWidth', 2);
end
for y = yticks
    plot(xlim, [y, y], 'Color', [1 1 1], 'LineWidth', 2);
end
hold off;

set(gca, 'TickLength', [0 0], ...
    'XTick', 1:12, 'XTickLabel', feature_info, ...
    'XTickLabelRotation', 90, ...
    'YTick', 1:12, 'YTickLabel', feature_info);

axis square
colorbar;
% clim([-1, 1]);

figure;
hf = boxplot(feature_list_norm);
ylabel('normalized values');
set(gca, 'XTickLabels', feature_info, 'XTicklabelRotation', 45);

h = my_parallelCoord(feature_list_norm, nan(12, 1), feature_info);

%% compute clusters
close all ;
rng default  % For reproducibility

idx_feature = [2, 10]; % 生成9个clusters
% idx_feature = [10, 6]; % 生成9个clusters
% idx_feature = [1,2,3,4];

% apply clustering algorithm
% [grps, PC] = kmeans_opt( feature_list_norm(:, idx_feature) );
% todo: include the maximum possible cluster numbers as an upper bound in
% the algorithmn
[grps, scores, nr_clusters, perf_iter] = kmeans_elbow( feature_list_norm(:, idx_feature) );

cc = [  0.4000    0.7608    0.6471;
        0.9882    0.5529    0.3843;...
        0.5529    0.6275    0.7961;...
        0.9059    0.5412    0.7647;...
        0.6510    0.8471    0.3294;...
        1.0000    0.8510    0.1843;...
        0.8980    0.7686    0.5804;...
        0.4510    0.1020    0.4549;...
        0.7804    0.2706    0.2745];
hf = cell(nr_clusters, 1);
clusterName = cell(nr_clusters, 1);

figure;
hold on;
for i = 1: nr_clusters
    idx = grps == i;
    hf{i} = scatter(feature_list(idx, idx_feature(1)), feature_list(idx, idx_feature(2)));
    set(hf{i}, 'MarkerFaceAlpha', 0.6, 'MarkerFaceColor', cc(i,:), 'MarkerEdgeColor', [0.45098, 0.45098, 0.45098], 'SizeData', 90)
    clusterName{i} = ['Cluster Nr.', num2str(i)];
end
box on; grid on;
set(gca, 'FontSize', 16);

xlabel(feature_info{idx_feature(1)});
ylabel(feature_info{idx_feature(2)});
legend([hf{:}], clusterName, 'Location', 'NorthWest');

realP_perClster = splitapply(@mean, feature_list(:, 10), grps);
workingP_perClster = splitapply(@mean, feature_list(:, 2), grps);

perf_iter(:, 3) = perf_iter(:, 3) * -1;

figure;
hf = bar(perf_iter);
cc = [0.36881, 0.53485, 0.66277;
    0.83243, 0.51044, 0.54496;
    0.94933, 0.78724, 0.44369];
for i = 1: 3
    set(hf(i), 'FaceColor', cc(i,:));
end

xlabel('Iteration');
legend('weighted score', 'mean', 'std.', 'Location', 'NorthWest');
grid on; box on;
set(gca, 'FontSize', 16);

%% solve a MILP problem; min|targetP - CX| for all composite event

residues_final = zeros(length(other_events), 1);
thr_val = 5;

for ii = 1: length(other_events)
    eventSelected = other_events{ii};

    signals = eventSelected.curnt_B;
    % signals = eventSelected.realP_tot;

    % smooth signals
    [signal_reduced, ~] = reduce_signals(signals, thr_val);

    sols = signal_composition_opt1(realP_perClster, signal_reduced);

    residue_perPulse_norm = zeros(size(sols));

    nr_types = size(realP_perClster, 1);

    for j = 1: length(residue_perPulse_norm)
        residue_perPulse_norm(j) = abs( (sols{j}(1:nr_types)' * realP_perClster - signal_reduced(j))./ signal_reduced(j));
    end
    residues_final(ii) = mean(residue_perPulse_norm) ;
end


figure; hf = plot(residues_final*100, 'LineWidth', 1.5, 'Color', [0 108 155]/255);
box on; grid on;
xlim([0 length(residues_final)+1]);
ylim([0 100]);
ylabel('Absolute error percentage [%]'); xlabel('Event ID');


%% Complex case: solve a MILP problem; min|targetP - CX| for a given composite event

close all;
eventSelected = other_events{9}; % 7, 9, 18, 25

thr_val = 5;
signals = eventSelected.curnt_B;
% signals = eventSelected.realP_tot;

% smooth signals
[signal_reduced, idx_pairs] = reduce_signals(signals, thr_val);

sols = signal_composition_opt1(realP_perClster, signal_reduced);

residue_perPulse = zeros(size(sols));

nr_types = size(realP_perClster, 1);

for j = 1: length(residue_perPulse)
    residue_perPulse(j) = abs( sols{j}(1:nr_types)' * realP_perClster - signal_reduced(j) );
end

% plot the estimated signal composite

hf = plot_ts_estimate(sols, idx_pairs, realP_perClster, eventSelected.time_num, signals, signal_reduced);

hf2 = plot_multi_ts(sols, idx_pairs, realP_perClster, eventSelected.time_num);

%% Simple case: solve a MILP problem; min|targetP - CX| for a given composite event

close all;
eventSelected = other_events{24}; % 7, 9, 18, 25

thr_val = 5;
signals = eventSelected.curnt_B;
% signals = eventSelected.realP_tot;

% smooth signals
[signal_reduced, idx_pairs] = reduce_signals(signals, thr_val);

sols = signal_composition_opt1(realP_perClster, signal_reduced);

residue_perPulse = zeros(size(sols));

nr_types = size(realP_perClster, 1);

for j = 1: length(residue_perPulse)
    residue_perPulse(j) = abs( sols{j}(1:nr_types)' * realP_perClster - signal_reduced(j) );
end

% plot the estimated signal composite

hf = plot_ts_estimate(sols, idx_pairs, realP_perClster, eventSelected.time_num, signals, signal_reduced);

hf2 = plot_multi_ts(sols, idx_pairs, realP_perClster, eventSelected.time_num);

%% get complete time-series per each pump type

thr_val = 5;
ts_list = reconstruct_clusterTS(realP_perClster, workingP_perClster, monotype_events(idx_used_events), grps, other_events, thr_val);

% map back to the original raw signals
for i = 1: length(realP_perClster)
    tmp = ts_list{i};
    tmp2 = cat(2, raw_data.time_num, zeros(size(raw_data, 1), 1));
    idx = ismember(raw_data.time_num, tmp(:,1));
    tmp2(idx, 2) = tmp(:, 2);
    ts_list{i} = tmp2;
end

%% make plot

close all ;
cc = [  0.4000    0.7608    0.6471;
        0.9882    0.5529    0.3843;...
        0.5529    0.6275    0.7961;...
        0.9059    0.5412    0.7647;...
        0.6510    0.8471    0.3294;...
        1.0000    0.8510    0.1843;...
        0.8980    0.7686    0.5804;...
        0.4510    0.1020    0.4549;...
        0.7804    0.2706    0.2745];

datetimeRange_start = datetime(datevec(['2018-03-01 23:00:00'; '2018-03-29 23:00:00'; '2018-05-29 23:00:00'; '2018-07-06 18:00:00'; '2018-08-08 23:00:00'; '2018-08-24 23:00:00'; '2018-10-16 18:00:00']));
datetimeRange_end = datetime(datevec(['2018-03-18 23:00:00'; '2018-04-13 23:00:00'; '2018-06-12 23:00:00'; '2018-07-07 23:00:00'; '2018-08-12 23:00:00'; '2018-08-26 23:00:00'; '2018-10-17 23:00:00']));

figure;
for ii = 1: length(datetimeRange_end)

    subplot(4,2,ii); hold on;
    h1 = area(hengyuan_data.UTCTime, hengyuan_data.workingPower);

    h2 = [];
    %     legendStr = cell(length(ts_list)+1, 1);
    %     legendStr{1} = 'Hengyuan';

    for i = 1: length(ts_list)
        cluster_idx = i;
        pred = ts_list{cluster_idx};
        [~, idx] = sort(pred(:,1));
        pred = pred(idx, :);

        h2(i) = plot(datetime(datevec(pred(:,1))), pred(:,2), 'Color', cc(i,:), 'LineWidth', 1.5);
        %         legendStr{i+1} = ['Nr.: ', num2str(i)];
    end

    set(h1, 'FaceColor', [100 100 100]/255, 'EdgeColor', [191 191 191]/255, 'FaceAlpha', 0.6);
    xlim([datetimeRange_start(ii), datetimeRange_end(ii)]);
    ylim([0, 20]);
    ylabel('Working Power [kW]');
    grid on; box on;
    % legend([h1, h2], legendStr);
end

cluster_idx = 8;
pred = ts_list{cluster_idx};
[~, idx] = sort(pred(:,1));
pred = pred(idx, :);

figure; hold on;
h1 = area(hengyuan_data.UTCTime, hengyuan_data.workingPower);
h2 = plot(datetime(datevec(pred(:,1))), pred(:,2));
set(h1, 'FaceColor', [191 191 191]/255, 'EdgeColor', [191 191 191]/255, 'FaceAlpha', 0.9);

legend([h1, h2], {'hengyuan', 'NLSM'});

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