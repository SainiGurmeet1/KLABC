%==========================================================================
%                      
% KLABC: Knowledge Learning Artificial Bee Colony Algorithm source codes
%  Developed in MATLAB R2023b   
%  Author and programmer: GURMEET SAINI 
%  e-Mail: gurmeetsaini3397@gmail.com                                 
%          gurmeet.phd@reck.ac.in
% Date: April 10, 2025
%
%==========================================================================

clear all;
clc;

% Add current directory to path to access cec17_func
addpath(fileparts(mfilename('fullpath')));

% Problem Settings
D = 10;                 % Dimension of the problem (10, 30, 50, 100)
LB = -100 * ones(1, D); % CEC 2017 bound
UB = 100 * ones(1, D);  % CEC 2017 bound
NP = 50;                % Population size (number of food sources)
MaxFEs = 10000* D;      % Maximum function evaluations per CEC 2017 guidelines
limit = NP * D / 2;     % Scout limit adjusted for population and dimension
lr_base = 0.2;          % Base learning rate for KLM (tuned for CEC 2017)
numRuns =50;            % Number of independent runs (increase for statistical significance)
funcNum = 1;            % Function number (1-30 for CEC 2017)

% Set function handle for CEC 2017
fhd = str2func('cec19_func'); % Assumes cec17_func is in the same directory

% KLM Settings (Feedforward Neural Network)
inputSize = D;         % Input layer: position dimension
hiddenSize1 = 16;      % First hidden layer nodes
hiddenSize2 = 16;      % Second hidden layer nodes
outputSize = D;        % Output layer: direction dimension
ep = 10   ;            % Epochs for KLM training

% Estimate iterations based on MaxFEs and population size
maxIterations = ceil(MaxFEs / (2 * NP));

% Preallocate storage for results
iterationResults = zeros(maxIterations, numRuns);   % Best fitness per iteration per run
runResults = zeros(numRuns, 1);                     % Best fitness per run

% Get current script directory and set filename
scriptDir = fileparts(mfilename('fullpath'));
filename = fullfile(scriptDir, sprintf('KLABC_CEC2019_F%d_Results.xls', funcNum));

% Multiple Runs
for run = 1:numRuns
    fprintf('\n=== CEC 2019 Function %d, Run %d/%d ===\n', funcNum, run, numRuns);
    
    % Reset FEs counter for each run
    FEs = 0;
    iteration = 0;
    
    % Initialize KLM weights for each run
    W1 = randn(hiddenSize1, inputSize) * 0.01;      % Normal distribution for better initialization
    b1 = zeros(hiddenSize1, 1)* 0.1;                % Zero bias initially
    W2 = randn(hiddenSize2, hiddenSize1) * 0.01;
    b2 = zeros(hiddenSize2, 1)* 0.1;
    W3 = randn(outputSize, hiddenSize2) * 0.01;
    b3 = zeros(outputSize, 1)* 0.1;
    
    % Initialize Population (full range with slight central bias)
    P = LB + rand(NP, D) .* (UB - LB);      % Random within [-100, 100]
    fitness = zeros(NP, 1);
    trial = zeros(NP, 1);
    % Evaluate initial population using cec17_func
    fitness = feval(fhd, P', funcNum)';    % Transpose P to DxNP, result is NPx1
    FEs = FEs + NP;
    
    % Experience List Q
    Q_positions = [];
    Q_directions = [];
    
    % Track best solution for this run
    [bestFitnessRun, idx] = min(fitness);
    bestSolutionRun = P(idx, :);
    
    % Main Loop
    while FEs <= MaxFEs
        iteration = iteration + 1;
        
        % Dynamic learning rate (balanced for CEC 2017)
        lr = lr_base * (iteration / maxIterations); % Quadratic decay
        %lr = 0.2
        % Employed Bee Phase
        for i = 1:NP
            temp_i = P(i, :);
            if rand < lr && ~isempty(Q_positions)
                d_i = utilizeKnowledge(P(i, :), W1, b1, W2, b2, W3, b3);
                v_i = applyDirection(P(i, :), d_i, LB, UB);
            else
                k = randi([1, NP]);
                while k == i
                    k = randi([1, NP]);
                end
                phi = -1 + 2 * rand(1, D);
                v_i = P(i, :) + phi .* (P(i, :) - P(k, :));
                v_i = max(min(v_i, UB), LB);
            end
            
            fitness_v = feval(fhd, v_i', funcNum);
            FEs = FEs + 1;
            
            if fitness_v < fitness(i)
                Q_positions = [Q_positions; temp_i];
                Q_directions = [Q_directions; (v_i - temp_i)];
                P(i, :) = v_i;
                fitness(i) = fitness_v;
                trial(i) = 0;
            else
                trial(i) = trial(i) + 1;
            end
        end
        
        % Onlooker Bee Phase
        fit_sum = sum(1 ./ (1 + fitness));
        prob = (1 ./ (1 + fitness)) / fit_sum;
        
        onlooker_count = NP;
        t = 0; i = 1;
        while t < onlooker_count && FEs <= MaxFEs
            if rand < prob(i)
                temp_i = P(i, :);
                if rand < lr && ~isempty(Q_positions)
                    d_i = utilizeKnowledge(P(i, :), W1, b1, W2, b2, W3, b3);
                    v_i = applyDirection(P(i, :), d_i, LB, UB);
                else
                    k = randi([1, NP]);
                    while k == i
                        k = randi([1, NP]);
                    end
                    phi = -1 + 2 * rand(1, D);
                    v_i = P(i, :) + phi .* (P(i, :) - P(k, :));
                    v_i = max(min(v_i, UB), LB);
                end
                
                fitness_v = feval(fhd, v_i', funcNum);
                FEs = FEs + 1;
                
                if fitness_v < fitness(i)
                    Q_positions = [Q_positions; temp_i];
                    Q_directions = [Q_directions; (v_i - temp_i)];
                    P(i, :) = v_i;
                    fitness(i) = fitness_v;
                    trial(i) = 0;
                else
                    trial(i) = trial(i) + 1;
                end
                t = t + 1;
            end
            i = mod(i, NP) + 1;
        end
        
        % Scout Bee Phase
        for i = 1:NP
            if trial(i) > limit
                P(i, :) = LB + rand(1, D) .* (UB - LB);
                fitness(i) = feval(fhd, P(i, :)', funcNum);
                FEs = FEs + 1;
                trial(i) = 0;
            end
        end
        
        % Update KLM
        if ~isempty(Q_positions)
            [W1, b1, W2, b2, W3, b3] = learningKnowledge(Q_positions, Q_directions, ...
                W1, b1, W2, b2, W3, b3, ep);
            Q_positions = [];
            Q_directions = [];
        end
        
        % Store and Display iteration-wise best fitness
        [bestFitness, idx] = min(fitness);
        if iteration <= maxIterations
            iterationResults(iteration, run) = bestFitness;
            fprintf('CEC2019 F%d, Run %d, Iteration %d: Best Fitness = %.4e\n', ...
                funcNum, run, iteration, bestFitness);
        end
        
        % Update run-wise best
        if bestFitness < bestFitnessRun
            bestFitnessRun = bestFitness;
            bestSolutionRun = P(idx, :);
        end
    end
    
    % Store run-wise result
    runResults(run) = bestFitnessRun;
    
    % Fill remaining iterations with final best fitness
    if iteration < maxIterations
        iterationResults(iteration+1:end, run) = bestFitnessRun;
    end
    
    % Display final result for this run
    fprintf('CEC2019 F%d, Run %d Completed - Best Fitness: %.4e\n', ...
        funcNum, run, bestFitnessRun);
end

% Display Run-wise Statistics
fprintf('\n=== CEC 2019 Function %d Final Run-wise Statistics ===\n', funcNum);
for run = 1:numRuns
    fprintf('Run %d: Best Fitness = %.4e\n', run, runResults(run));
end
fprintf('Mean Best Fitness across runs: %.4e\n', mean(runResults));
fprintf('Std Dev of Best Fitness: %.4e\n', std(runResults));

% Write Results to Excel
colHeaders = arrayfun(@(x) sprintf('Run%d', x), 1:numRuns, 'UniformOutput', false);
rowHeaders = arrayfun(@(x) sprintf('Iter%d', x), 1:maxIterations, 'UniformOutput', false);
iterationData = num2cell(iterationResults);
iterationData = [rowHeaders iterationData];
headerRow = [{'Iteration'} colHeaders];
iterationData = [headerRow; iterationData];
xlswrite(filename, iterationData, 'IterationWise');

runHeaders = {'Run', 'BestFitness'};
runData = [(1:numRuns)' runResults];
xlswrite(filename, [runHeaders; num2cell(runData)], 'RunWise');

fprintf('\nResults saved to: %s\n', filename);

% Helper Functions for KLABC
function direction = utilizeKnowledge(position, W1, b1, W2, b2, W3, b3)
    h1 = tanh(W1 * position' + b1);
    h2 = tanh(W2 * h1 + b2);
    direction = W3 * h2 + b3;
    direction = direction';
end

function newPosition = applyDirection(position, direction, LB, UB)
    newPosition = position + 2 * rand * direction; 
    newPosition = max(min(newPosition, UB), LB);
end

function [W1, b1, W2, b2, W3, b3] = learningKnowledge(positions, directions, ...
    W1, b1, W2, b2, W3, b3, epochs)
    learningRate = 0.2; 
    numExperiences = size(positions, 1);
    
    for ep = 1:epochs
        idx = randperm(numExperiences); % Shuffle experiences
        for i = 1:numExperiences
            x = positions(idx(i), :)';
            t = directions(idx(i), :)';
            h1 = tanh(W1 * x + b1);
            h2 = tanh(W2 * h1 + b2);
            y = W3 * h2 + b3;
            
            delta3 = y - t;
            delta2 = (W3' * delta3) .* (1 - h2.^2);
            delta1 = (W2' * delta2) .* (1 - h1.^2);
            
            W3 = W3 - learningRate * delta3 * h2';
            b3 = b3 - learningRate * delta3;
            W2 = W2 - learningRate * delta2 * h1';
            b2 = b2 - learningRate * delta2;
            W1 = W1 - learningRate * delta1 * x';
            b1 = b1 - learningRate * delta1;
        end
    end
end