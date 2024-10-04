%% Model definition for square geometry
rng('default'); % for reproducibility
model = createpde();
R1 = [3,4,-1,1,1,-1,-1,-1,1,1]'; % Rectangle definition (square)
gdm = [R1];
g = decsg(gdm, 'R1', ('R1')');
geometryFromEdges(model,g);
applyBoundaryCondition(model,"dirichlet", ...
                             "Edge",1:model.Geometry.NumEdges, ...
                             "u",0);

% Plot the geometry and display
figure 
pdegplot(model,"EdgeLabels","on"); 
axis equal

%% Structural array of coefficients (unchanged)
pdeCoeffs.c = .5;
pdeCoeffs.m = 0;
pdeCoeffs.d = 0;
pdeCoeffs.a = 0;
pdeCoeffs.f = 1;

specifyCoefficients(model,"m",pdeCoeffs.m,"d",pdeCoeffs.d,"c",pdeCoeffs.c,"a",pdeCoeffs.a,"f",pdeCoeffs.f);
Hmax = 0.05; Hgrad = 2; Hedge = Hmax/10;
msh = generateMesh(model,"Hmax",Hmax,"Hgrad",Hgrad,"Hedge",{1:model.Geometry.NumEdges,Hedge});

% Make coefficients dlarrays for gradient computation
pdeCoeffs = structfun(@dlarray,pdeCoeffs,'UniformOutput',false); 

%% Continue with the same training setup...


%%geneate the data for tranning
boundaryNodes = findNodes(msh,"region","Edge",1:model.Geometry.NumEdges);
domainNodes = setdiff(1:size(msh.Nodes,2),boundaryNodes);
domainCollocationPoints = msh.Nodes(:,domainNodes)';



%%define the deep learning
numLayers = 3;
numNeurons = 50;
layers = featureInputLayer(2);
for i = 1:numLayers-1
    layers = [
        layers
        fullyConnectedLayer(numNeurons)
        tanhLayer];%#ok<AGROW>
end
layers = [
    layers
    fullyConnectedLayer(1)];
pinn = dlnetwork(layers);


%%train the PINN using ADAM solver
numEpochs = 1500;
miniBatchSize = 2^12;
initialLearnRate = 0.01;
learnRateDecay = 0.001;
averageGrad = []; % for pinn updates
averageSqGrad = [];
pAverageGrad = []; % for parameter updates
pAverageSqGrad = []; 


%%store for tranning loop
ds = arrayDatastore(domainCollocationPoints);
mbq = minibatchqueue(ds, MiniBatchSize = miniBatchSize, MiniBatchFormat="BC");
%Calculate the total number of iterations for the training progress monitor and initialize the monitor. 
numIterations = numEpochs * ceil(size(domainCollocationPoints,1)/miniBatchSize);
monitor = trainingProgressMonitor(Metrics="Loss",Info="Epoch",XLabel="Iteration");


%%tranning loop
iteration = 0;
epoch = 0;
learningRate = initialLearnRate;
lossFcn = dlaccelerate(@modelLoss);
while epoch < numEpochs && ~monitor.Stop
    epoch = epoch + 1;
    reset(mbq);
    while hasdata(mbq) && ~monitor.Stop
        iteration = iteration + 1;
        XY = next(mbq);
        % Evaluate the model loss and gradients using dlfeval.
        [loss,gradients] = dlfeval(lossFcn,model,pinn,XY,pdeCoeffs);
        % Update the network parameters using the adamupdate function.
        [pinn,averageGrad,averageSqGrad] = adamupdate(pinn,gradients{1},averageGrad,...
                                               averageSqGrad,iteration,learningRate);
        % Update the c coefficient using the adamupdate function. Defer
        % updating until 1/10 of epochs are finished.
        if epoch > numEpochs/10
            [pdeCoeffs.c,pAverageGrad,pAverageSqGrad] = adamupdate(pdeCoeffs.c,gradients{2},pAverageGrad,...
                                               pAverageSqGrad,iteration,learningRate);
        end
    end
    % Update learning rate.
    learningRate = initialLearnRate / (1+learnRateDecay*iteration);
    % Update the training progress monitor.
    recordMetrics(monitor,iteration,Loss=loss);
    updateInfo(monitor,Epoch=epoch + " of " + numEpochs);
    monitor.Progress = 100 * iteration/numIterations;
end




%%visualize Data
nodesDLarry = dlarray(msh.Nodes,"CB");
Upinn = gather(extractdata(predict(pinn,nodesDLarry)));
figure;
pdeplot(model,"XYData",Upinn);
title(sprintf("Solution with c = %.4f",double(pdeCoeffs.c)));




%%model loss function
function [loss,gradients] = modelLoss(model,pinn,XY,pdeCoeffs)
dim = 2;
U = forward(pinn,XY);

% Loss for difference in data taken at mesh nodes. 
Utrue = getSolutionData(XY);
lossData = l2loss(U,Utrue);

% Compute gradients of U and Laplacian of U.
gradU = dlgradient(sum(U,"all"),XY,EnableHigherDerivatives=true);
Laplacian = 0;
for i=1:dim
    % Add each term of the Laplacian
    gradU2 = dlgradient(sum(pdeCoeffs.c.*gradU(i,:),"all"),XY,EnableHigherDerivatives=true);
    Laplacian = gradU2(i,:)+Laplacian;
end

% Enforce PDE. Calculate lossF.
res = -pdeCoeffs.f - Laplacian + pdeCoeffs.a.*U;
lossF = mean(sum(res.^2,1),2);

% Enforce boundary conditions. Calculate lossU.
actualBC = []; % contains the actual boundary information
BC_XY = []; % boundary coordinates
% Loop over the boundary edges and find boundary coordinates and actual BC
% assigned to PDE model.
numBoundaries = model.Geometry.NumEdges;
for i=1:numBoundaries
    BCi = findBoundaryConditions(model.BoundaryConditions,'Edge',i);
    BCiNodes = findNodes(model.Mesh,"region","Edge",i);
    BC_XY = [BC_XY, model.Mesh.Nodes(:,BCiNodes)]; %#ok<AGROW> 
    actualBCi = ones(1,numel(BCiNodes))*BCi.u;
    actualBC = [actualBC actualBCi]; %#ok<AGROW>
end
BC_XY = dlarray(BC_XY,"CB"); % format the coordinates
predictedBC = forward(pinn,BC_XY);
lossBC = mse(predictedBC,actualBC);

% Combine weighted losses. 
lambdaPDE  = 0.4; % weighting factor
lambdaBC   = 0.6;
lambdaData = 0.5;
loss = lambdaPDE*lossF + lambdaBC*lossBC + lambdaData*lossData;

% Calculate gradients with respect to the learnable parameters and
% C-coefficient. Pass back cell array to update pinn and coef separately.
gradients = dlgradient(loss,{pinn.Learnables,pdeCoeffs.c}); 
end
%% Initialize training loop variables
%iteration = 0;
epoch = 0;
learningRate = initialLearnRate;
lossFcn = dlaccelerate(@modelLoss);
lossHistory = [];  % Array to store loss values for plotting

%% Training loop
while epoch < numEpochs && ~monitor.Stop
    epoch = epoch + 1;
    reset(mbq);
    while hasdata(mbq) && ~monitor.Stop
        iteration = iteration + 1;
        XY = next(mbq);
        
        % Evaluate the model loss and gradients using dlfeval.
        [loss,gradients] = dlfeval(lossFcn,model,pinn,XY,pdeCoeffs);
        
        % Store the current loss value
        lossHistory = [lossHistory gather(extractdata(loss))];  %#ok<AGROW>

        % Update the network parameters using the adamupdate function.
        [pinn,averageGrad,averageSqGrad] = adamupdate(pinn,gradients{1},averageGrad,...
                                               averageSqGrad,iteration,learningRate);
        % Update the c coefficient using the adamupdate function. Defer
        % updating until 1/10 of epochs are finished.
        if epoch > numEpochs/10
            [pdeCoeffs.c,pAverageGrad,pAverageSqGrad] = adamupdate(pdeCoeffs.c,gradients{2},pAverageGrad,...
                                               pAverageSqGrad,iteration,learningRate);
        end
    end
    
    % Update learning rate.
    learningRate = initialLearnRate / (1+learnRateDecay*iteration);
    
    % Update the training progress monitor.
    recordMetrics(monitor,iteration,Loss=loss);
    updateInfo(monitor,Epoch=epoch + " of " + numEpochs);
    monitor.Progress = 100 * iteration/numIterations;
end

%% Plot the loss function
figure;
plot(1:iteration, lossHistory, 'LineWidth', 2);
xlabel('Iteration');
ylabel('Loss');
title('Training Loss vs Iteration');
grid on;


%data function
function UD = getSolutionData(XY)
    UD = (1-XY(1,:).^2-XY(2,:).^2)/4;
end
