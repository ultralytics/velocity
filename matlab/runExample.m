# Ultralytics YOLO 🚀, AGPL-3.0 License

function [] = runExample()
%this file imports a short movie of cars in Viña del Mar to track speeds
clc; clear; close all; 
% pname=fcnpathm1(mfilename('fullpath')); cd(pname); 
format short g

%VIDEO --------------------------------------------------------------------
%[fname, pname] = uigetfile({'*.mov';'*.m4v'},'Select images','MultiSelect','off');
%startFrame = 8;

addpath(genpath('/Users/glennjocher/Downloads/DATA'))
%load IMG_4119.MOV.mat; %frame 42 20km/h
load IMG_4134.MOV.mat; %frame 19 40km/h
%load IMG_4238.m4v.mat; %frame 8 60km/h


%IMAGES -------------------------------------------------------------------
%[fname, pname] = uigetfile({'*.jpg','JPG Files (*.jpg)'},'Select images','MultiSelect','on');  
%load IMG_4110.JPG.mat
%load IMG_4122.JPG.mat

n = 20;
startClock=clock;
cam = getCameraParams(fname);
if cam.videoFlag
    cam.V.CurrentTime=startFrame/30;
    fprintf('Starting image processing on ''%s''...\n',fname)
else
    n = min(numel(fname),n);
    fprintf('Starting image processing on ''%s'' through ''%s''...\n',fname{1},fname{end}) %#ok<*USENS>
end
fprintf('\n%13s%13s%13s%13s%13s%13s%13s%13s%13s\n%13s%13s%13s%13s%13s%13s%13s%13s%13s\n','image','procTime','pointTracks','metric','dt','time','dx','distance','speed','#','#','#','(pixels)','(s)','(s)','(m)','(m)','(km/h)');
A=zeros(n,13); B=A; z=zeros(n,1); dx=z; dt=z; speed=z;
zc=cell(n,1); residuals=zc; P=zc; Pp=zc; Pa=zc;

% A = [xyz, rpy, xyz_ecef, lla, t]; %camera information
% B = [xyz, rpy, xyz_ecef, lla, t]; %car information

for i=1:n
    tic
    if cam.videoFlag
        rgb=readFrame(cam.V);
        I = images.internal.rgb2graymex(rgb);
        A(i,13) = cam.V.CurrentTime;
    else
        I = images.internal.rgb2graymex(imread(fname{i}));
        EXIF = imfinfo(fname{i});
        A(i,10:13)=fcnEXIF2LLAT(EXIF);
        declination =  2.56; %WMM(A(1,10),A(1,11),'Declination'); %(deg), True = Magnetic + Declination
        A(i,6) = EXIF.GPSInfo.GPSImgDirection + declination; %(deg) yaw clockwise from magnetic north
    end
    
    if i==1
        if ~exist('q','var')
            fig(1,1,4,6); imshow(I); axis tight; pause %click four points for plate outline clockwise
            [xi, yi] = ginput(4); q=[xi,yi];  
            if cam.videoFlag
                save([fname '.mat'],'q','fname','startFrame');
            else
                save([fname{1} '.mat'],'q','fname');
            end
        end 
        border = 0;
        bbox = boundingRect(q, [cam.Width, cam.Height], border);
        %pc = detectBRISKFeatures(I,'ROI',bbox);
        %pc = detectSURFFeatures(I,'ROI',bbox);
        pc = detectHarrisFeatures(I,'ROI',bbox);
        %pc = detectMinEigenFeatures(I,'ROI',bbox);
        
        %pb=pb.selectStrongest(300);
        %pb=selectUniform(pb,200,bbox(3:4));
        p=[q; pc.Location]; %MinEigen, SURF, FAST, Harris, MSER, BRISK features
        
        cam.kltBlockSize = [1 1]*15;
        KLTa=vision.PointTracker('MaxBidirectionalError',.5,'NumPyramidLevels',11,'BlockSize',[15 15],'MaxIterations',20);  initialize(KLTa,p,I);
        [t,R] = estimatePlatePosition(cam.params,p(1:4,:),worldPointsLicensePlate,I);
        worldPoints = pointsToWorld(cam.params, R, t, p);   valid=true(size(p,1),1);
        worldPoints = padarray(worldPoints,[0 1],0,'post');
        worldPoints = worldPoints * R;
        %a=worldToImage(cam.params,R,t,worldPoints)
                
        I0=I; range=0; t0=A(1,13);
        
        %DEM = getDEM([],A(1,10:12));
        DEM.centerlla = A(1,10:12);
        DEM.centerecef = lla2ecef(DEM.centerlla);
        DEM.DCM_ECEF2NED = fcnLLA2DCM_ECEF2NED(DEM.centerlla*d2r);
        
        pa = detectHarrisFeatures(I);  pa=pa.selectStrongest(5000);  pa=pa.Location;
        KLTb=vision.PointTracker('MaxBidirectionalError',.5,'NumPyramidLevels',2,'BlockSize',[15 15],'MaxIterations',20);  initialize(KLTb,pa,I);
    else
        %[pa_, v] = KLTb(I);
        %tform = estimateGeometricTransform(pa_(v,:),pa(v,:),'affine');
        %I = fcnimwarp(I,cam.Ixy,inv(tform.T));
        %fig; showMatchedFeatures(Im1,I,pa(v,:),pa_(v,:));
        
        %[p, valid] = KLTa(I);
        [p, valid] = KLTwarp(KLTa,cam.Ixy,I,Im1,pm1,validm1,'reducedImage');
         
        if cam.videoFlag
            dt(i) = 1/cam.V.FrameRate;
        else %images
            dt(i) = A(i,13)-A(i-1,13);
        end
    end
    [t,R,~,residuals{i},pProj] = estimatePlatePosition(cam.params,p(valid,:),worldPoints(valid,:),I);  B_cam = t;
    
    P_cam = worldPoints*R + t;
    P_ned{i} = P_cam * cam2ned';
    
    %INVALIDATE HIGH RESIDUAL POINTS
    if i<30
        %fig(1,1,5,8); showMatchedFeatures(Im1,I,p(valid,:),pProj); axis tight;
        %[~, inliers] = fcnsigmarejection(residuals{i},3,3);  valid(valid)=inliers;  setPoints(KLTa,p,valid)
        %scatter(pProj(:,1),pProj(:,2),(residuals{i}(:)*10).^3,'filled')
    end
    
    R_cam2ned = fcnRPY2DCM_B2W(A(i,4:6)*d2r);
    B_ned = (B_cam*cam2ned')*R_cam2ned';
    B(i,1:3) = B_cam; %B_ned;
    Pa{i}=p; P{i}=p(valid,:); Pp{i}=pProj;
    
    
%     %N Vector Intercept
%     if i==4
%         ux1 = zeros(i,sum(valid)); uy1=ux1; uz1=ux1;
%         for j=1:i
%             [~, u] = pixel2angle(cam.params.IntrinsicMatrix, Pa{j}(valid,:));
%             ux1(j,:)=u(:,1);  uy1(j,:)=u(:,2);  uz1(j,:)=u(:,3); 
%         end
%         u0 = -(B(1:i,1:3) - B(1,1:3));
%         C2 = fcn2vintercept(u0,ux1,uy1,uz1);% * cam2ned';
%         CN = fcnNvintercept(u0,ux1,uy1,uz1);% * cam2ned';
%         C2-CN
%     end
% 
%     if i>1
%         %TRY POSE ESTIMATION
%         E = estimateEssentialMatrix(pm1(valid,:),p(valid,:),cam.params);
%         F = estimateFundamentalMatrix(pm1(valid,:),p(valid,:));
%         [~, that] = relativeCameraPose(E,cam.params,pm1(valid,:),p(valid,:));
%         [~, that] = relativeCameraPose(F,cam.params,pm1(valid,:),p(valid,:));
%         %[Rhat, that] = extrinsics(p(valid,:), worldPoints(valid,:), cam.params)
%         %[~, that] = estimateWorldCameraPose(p(valid,:), worldPoints(valid,:), cam.params)
%         fcnvec2uvec(B(i,1:3) - B(i-1,1:3))
%     end

    dx(i) = 0; if i>1; dx(i)=fcnrange(B(i,1:3),B(i-1,1:3)); end
    range = range+dx(i);
    speed(i) = convvel(dx(i)/dt(i),'m/s','km/h');
    info = [i-1 toc sum(valid) mean(residuals{i}) dt(i) A(i,13)-t0 dx(i) range speed(i)];
    fprintf('\n%13g%13.3f%13g%13.3f%13.3f%13.3f%13.2f%13.2f%13.1f',info)
    pm1=p; Im1=I; validm1=valid;
end
t=etime(clock,startClock);
ecef = lla2ecef(A(:,10:12));        A(:,7:9)=ecef; %(km) camera
ned = ecef2ned(DEM,ecef)*1E3;       A(:,1:3)=ned; %(m) camera
ecef = ned2ecef(DEM,B(:,1:3)/1E3);  B(:,7:9)=ecef; %(km) car
lla = ecef2lla(ecef);               B(:,10:12)=lla; %(m) car
x=speed(2:end);                         s=sprintf('\nSpeed = %.2f +/- %.2f km/h',mean(x),std(x));  fprintf('\n%s\n',s)
x=cellfun(@mean,residuals(2:end));      fprintf('Residuals = %.3f +/- %.3f pixels\n',mean(x),std(x))

%PLOT RESULTS
fig(1,1,2,3); 
P=cell2mat(P); Pp=cell2mat(Pp);
imshow(I0/2+I/2); axis on tight; if cam.Orientation==6; view(90,90); end; title([str_(cam.FileName) '  ' s])
plot(P([1:4 1],1),P([1:4 1],2),'y.-','Markersize',20,'Linewidth',2,'Display','License Outline')
plot(P(:,1),P(:,2),'g.','Markersize',10,'Display','Tracked Points');
plot(Pp(:,1),Pp(:,2),'ro','Markersize',6,'Display','Projected Points');
%plotBoundingBox(bbox,'Display','License Bounding Box');

ha=fig(1,3,1.5); 
h=sca; 
try
    fcnplot3(B(:,1:3),'.-','Markersize',21); xyzlabel('X (North)','Y (East)','Z (Down'); axis on equal tight vis3d
    for j=1:numel(P_ned)
        fcnplot3(P_ned{j},'r.');
    end
    fcnplot3(C0,'go')
end

plotCamera('Location',[0 0 0],'Orientation',cam2ned'*fcnRPY2DCM_W2B(A(1,4:6)*d2r),'Size',range/25);
str=cellfun(@(x) sprintf(' %g',x),num2cell(1:n),'UniformOutput',false); text(B(:,1),B(:,2),B(:,3),str)
h.CameraViewAngleMode = 'Auto'; view(90,-90)
sca;  x=1:n; cdx=cumsum(dx);  deg=min(n-1,2);
plot(x,cdx,'.');  P=polyfit(x,cdx',deg);
plot(x,polyval(P,x),'-','Display','Polyfit');  xyzlabel('image','distance (m)','',sprintf('''%s'' %g images',str_(cam.FileName),n));
sca;  x=2:n;
plot(x,speed(x),'.-'); xyzlabel('image','velocity (km/h)')
if n>3; plot(1:n,polyval(polyfit(x,speed(x)',2),1:n),'-','Display','Polyfit'); end
fcntight(ha(2:3),'y0')

fprintf('Processed %g images in %.2fs (%.2ffps)\n',n,t,n/t)
end

function [ea, u] = pixel2angle(K, x)
f = K(1,1);  % focal length (pixels)
y = padarray(x - K(3,1:2),[0 1],f,'post');
u = fcnvec2uvec(y);
ea = fcnelaz(y * cam2ned');  % (rad) angle

% % EXAMPLE
% j = linspace(-1,1,1000)';
% x = [j * 1000, j*0+100, j*0+30];
% xi = worldToImage(cam.params,eye(3),[0 0 0],x);
% ea = pixel2angle(cam.params.IntrinsicMatrix, xi) * r2d;
% fig(3,1,2,4);
% sca; scatter(xi(:,1),xi(:,2),20,1:1000,'filled'); plot([0, 0, cam.Width, cam.Width, 0],[0, cam.Height cam.Height, 0 0],'k.-'); axis equal
% sca; scatter(1:1000,ea(:,1),20,1:1000,'filled')
% sca; scatter(1:1000,ea(:,2),20,1:1000,'filled')
end

function bbox = boundingRect(x, imshape, border)
Width = imshape(1);
Height = imshape(2);

%bbox = [xmin ymin width height]
bbox = round([min(x) max(x)-min(x)] + [-1 -1 2 2]*border);
bbox(1:2) = max(bbox(1:2),1);  

if bbox(1)+bbox(3) > Width
    bbox(3) = Width-bbox(1);
end
if bbox(2)+bbox(4) > Height
    bbox(4) = Height-bbox(2);
end
end

function DEM = getDEM(cam,lla,geoidstr)
%lla in degrees and meters
if nargin==2; geoidstr = 'ellipsoid'; end %elevations off ellipsoid instead of geoid

ni=20; %DEM points along x and y
r=0.200; %(km) DEM expansion past about lla edges
kmperdeg = fcnmperLLAdeg(mean(lla,1))/1000;  dext = r./kmperdeg; % degrees of extension
latv = linspace(min(lla(:,1)) - dext(1), max(lla(:,1)) + dext(1), ni);
lngv = linspace(min(lla(:,2)) - dext(2), max(lla(:,2)) + dext(2), ni);

if isfield(cam,'DEM') && isfield(cam.DEM,'Fned')
    fprintf('Discovered Existing DEM in getDEM()\n')
    DEM = cam.DEM;
else %get DEM
    DEM = fcnGoogleElevationAPIDEM(latv,lngv,geoidstr);
end
end
