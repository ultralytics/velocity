function [t, R, inliers, residuals, imagePointsProjected] = estimatePlatePosition(cameraParams,imagePoints,worldPoints,I)
n = size(imagePoints,1);
imagePoints=double(imagePoints);
worldPoints=double(worldPoints);

%CLOSE SQUARE
if n==4
    imagePoints = imagePoints(repmat(1:4,[1 2]),:);
    worldPoints = worldPoints(repmat(1:4,[1 2]),:);
end
%if n<5 %interpolate license plate outline
%    wpr = cumsum([0 fcnrange(worldPoints(2:end,:),worldPoints(1:end-1,:))']); %worldPointRange (m)
%    imagePoints=interp1(wpr,imagePoints,linspace(0,wpr(end),30));
%    worldPoints=interp1(wpr,worldPoints,linspace(0,wpr(end),30));
%end

if size(worldPoints,2)==2
    worldPoints3 = padarray(worldPoints,[0 1],0,'post');
else
    worldPoints3 = worldPoints;
end

%METHOD1 (POOR RESULT, use as initial guess to NLS)
%[R, t] = extrinsics(imagePoints, worldPoints, cameraParams);

% if norm(t)>100
%     fprintf('Warning: extrinsics() producing unstable outputs in estimatePlatePosition()\n')
%     %METHOD2 (GOOD RESULTS, but assumes outliers exist, so throws away good points)
%     [worldOrientation, worldLocation, inliers] = estimateWorldCameraPose(imagePoints, worldPoints3, cameraParams);%,'MaxReprojectionError',2,'Confidence',99.999,'MaxNumTrials',9000);
%     R=worldOrientation';  t=-worldLocation*R;  %[R,t] = cameraPoseToExtrinsics(worldOrientation,worldLocation);
% end


%METHOD3 MATLAB fitter (GOOD RESULTS, 10x slower than NLS though)
%x0 = [fcnB2WDCM2RPY(R) t];
%[x, fx]=fminunc(@(x) licensePlateCostFunction(imagePoints,worldPoints3,cameraParams,x),x0,optimoptions(@fminunc,'Display','notify','Algorithm','quasi-newton'));
%[x fx]=fminsearch(@(x) licensePlateCostFunction(imagePoints,worldPoints3,cameraParams,x),x0,optimset('MaxFunEvals',3000)); 
%[x fx]=lsqnonlin(@(x) licensePlateCostFunction(imagePoints,worldPoints3,cameraParams,x),x0,[],[],optimoptions(@lsqnonlin,'Display','off','Algorithm','Levenberg-Marquardt'));
%R=fcnRPY2DCM_B2W(x(1:3));  t=x(4:6);

%METHOD4 custom NLS (BEST RESULTS. Fastest and most accurate)
x0 = [0 0 0 0 0 1];
[R, t] = fcnNLScamera2world(imagePoints,worldPoints3,cameraParams,x0);

%RESIDUALS
imagePointsProjected = worldToImage(cameraParams, R, t, worldPoints3);
residuals = fcnrange(imagePointsProjected,imagePoints);
inliers = true(n,1);

%PLOT
% fig; imshow(I);
% plot(imagePoints(:,1),imagePoints(:,2),'g.','Markersize',10,'Display','Handpicked License Plate');
% plot(imagePointsProjected(:,1),imagePointsProjected(:,2),'ro','Markersize',6,'Display','Projected License Plate');
end

function fx=licensePlateCostFunction(imagePoints,worldPoints3,cameraParams,x)
R=fcnRPY2DCM_B2W(x(1:3));  t=x(4:6);
zhat = worldToImage(cameraParams, R, t, worldPoints3);  residuals=imagePoints-zhat;
fx=sum3(residuals.^2);
end


function [R, t] = fcnNLScamera2world(imagePoints,worldPoints,cameraParams,x0)
cameraRotationMatrix=eye(3);
cameraTranslationVector=[0 0 0];
camMatrix = cameraMatrix(cameraParams,cameraRotationMatrix,cameraTranslationVector);
%https://la.mathworks.com/help/vision/ref/cameramatrix.html
%Using the camera matrix and homogeneous coordinates, you can project a world point onto the image.
%w * [x,y,1] = [X,Y,Z,1] * camMatrix
%(X,Y,Z): world coordinates of a point
%(x,y): coordinates of the corresponding image point
%w: arbitrary scale factor

%x = 6 x 1 for 6 parameters
%J = n x 6
%z = n x 1 for n measurements

dx = 1E-6; %for numerical derivatives
x = [0 0 0 x0(4:6)]';  %nx=numel(x);
z = imagePoints(:);  %nz=numel(z);
maxIter = 100;
mdm = eye(6) * 1;
for i=1:maxIter
    zhat = fcnzhat(x,worldPoints,camMatrix);
    J = [fcnzhat(x+[dx 0 0 0 0 0]',worldPoints,camMatrix), fcnzhat(x+[0 dx 0 0 0 0]',worldPoints,camMatrix), fcnzhat(x+[0 0 dx 0 0 0]',worldPoints,camMatrix), ...
         fcnzhat(x+[0 0 0 dx 0 0]',worldPoints,camMatrix), fcnzhat(x+[0 0 0 0 dx 0]',worldPoints,camMatrix), fcnzhat(x+[0 0 0 0 0 dx]',worldPoints,camMatrix)];
    J = (J-zhat)/dx;
    delta = (J'*J + mdm)^-1*J'*(z-zhat) * min((i * .2)^2, 1);
    x = x + delta;
    if rms(delta)<1E-9; break; end
end
if i==maxIter; disp('WARNING: fcnNLScamera2world() reaching max iterations!'); end
R=fcnRPY2DCM_B2W(x(1:3));  t=x(4:6)';
%R=quat32rotm(x(1:3));  t=x(4:6)';


end

function zhat = fcnzhat(x,worldPoints,camMatrix)
zhat = [worldPoints*fcnRPY2DCM_B2W(x(1:3))+x(4:6)' ones(size(worldPoints,1),1)] * camMatrix;
zhat=zhat(:,1:2)./zhat(:,3);  zhat=zhat(:);
end

function  R = quat32rotm(q)
r = norm(q);
R = fcnRPY2DCM_B2W([r, asin(-q(3) / r ), atan(q(2) / q(1))]);
end

