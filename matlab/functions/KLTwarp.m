function [p, v] = KLTwarp(KLT,Ixy,I,Im1,pm1,vm1,mode)
%KLTwarp Steps vision.PointTracker KLT tracks 
% This function uses the 2-stage KLT warp tracking method to propagate 
% vision.Pointracker points from image Im1 to image I
%
% INPUTS ------------------------------------------------------------------
% KLT: KLT object
% Ixy: predefined interpolant XY matrix to pass to fcnimwarp()
% I: current image
% Im1: previous image
% pm1: previous KLT points (nx2 single or double)
% vm1: previous 'PointValididty' vector for pm1 (nx1 boolean)
% mode: reduced or full warping. Reduced is faster
%
% OUTPUTS -----------------------------------------------------------------
% p: current KLT points (nx2 single or double)
% p: current valid vector (nx1 boolean)

pvm1=pm1(vm1,:);
if nargin<6; mode = 'reducedImage'; end % 'reducedImage' (default) or 'fullImage'

switch mode
    case 'reducedImage'
        %1. GET COARSE TRACKING
        [p, v] = KLT(I);  n=sum(v);
        %fig; showMatchedFeatures(Im1,I,pm1(v,:),p(v,:));

        %RETRY COARSE WITH TRANSLATION CORRECTION
        if n>1 %&& n<30
            tform=eye(3); tform(3,1:2)=mean(p(v,:)-pm1(v,:),1);
            K = vision.PointTracker('MaxBidirectionalError',0.5,'NumPyramidLevels',11,'BlockSize',[15 15],'MaxIterations',20);
            [p,v]=regionalKLT(K,I,Im1,pm1,vm1,tform);  n=sum(v);
        end

        if n>=10
            tform = fitgeotransSigmaRejection(pm1(v,:),p(v,:),3);
        else %SURF Matching from scratch
            fprintf('  WARNING: Coarse KLT failure, SURF matching...')
            f = detectSURFFeatures(I);
            n=0; a=0;
            while n<10
                fm1 = detectSURFFeatures(Im1,'ROI',[min(pvm1) max(pvm1)-min(pvm1)] + [-1 -1 1 1]*a);
                i = matchFeatures(extractFeatures(Im1, fm1), extractFeatures(I, f));  n=size(i,1);  a=a+10;
            end
            tform = fitgeotransSigmaRejection(fm1.Location(i(:,1),:),f.Location(i(:,2),:),3);
        end

        %2. GET FINE TRACKING
        K = vision.PointTracker('MaxBidirectionalError',0.1,'NumPyramidLevels',1,'BlockSize',[51 51],'MaxIterations',40); 
        [p,v]=regionalKLT(K,I,Im1,pm1,vm1,tform);
    case 'fullImage'
        KLTc = clone(KLT);
        [p, v] = KLT(I);
        tform = fitgeotransSigmaRejection(pm1(v,:),p(v,:),3);
        I_Im1 = fcnimwarp(I,Ixy,tform);  %current image warped to past image frame
        
        [p_Im1, v] = KLTc(I_Im1);
        x=[p_Im1(v,:) ones(sum(v),1)]*tform;  p(v,:)=x(:,1:2)./x(:,3);
end
setPoints(KLT,p,v)
end

function [pr,vr] = regionalKLT(K,I,Im1,pm1,vm1,tform)
a = round([min(pm1(vm1,:)) max(pm1(vm1,:))]) + [-1 -1 1 1]*50;  a=max(a,1);  [height, width]=size(I);  a(3)=min(a(3),width);  a(4)=min(a(4),height);
axy=a(1:2);
[X,Y] = meshgrid(a(1):a(3), a(2):a(4));  Ixys = [X(:) Y(:) ones(numel(X),1)];
I_Im1s = fcnimwarp(I,Ixys,tform);  I_Im1s=reshape(I_Im1s,size(X)); %fig; imshowpair(I_Im1s); %current image warped to past image frame
%i=sub2ind(size(I),Y(:),X(:)); Ib=Im1; Ib(i)=Ib(i)/2+I_Im1s(:)/2; fig; imshow(Ib); axis on tight;

initialize(K, pm1(vm1,:)-axy, imcrop(Im1,[axy a(3)-a(1) a(4)-a(2)]));

[p_Im1, vb] = K(I_Im1s);
x=[p_Im1+axy ones(size(p_Im1,1),1)]*tform;  pr=pm1;  pr(vm1,:)=x(:,1:2)./x(:,3);
vr=false(size(vm1));  vr(vm1)=vb;
end

% %HOUGH TRANSFORM
% fig; I=I_Im1s;
% BW = edge(I,'sobel');
% [H,T,R] = hough(BW);
% imshow(H,[],'XData',T,'YData',R,'InitialMagnification','fit');
% xlabel('\theta'), ylabel('\rho');
% axis on, axis normal, hold on;
% P  = houghpeaks(H,2,'threshold',ceil(0.3*max(H(:))));
% x = T(P(:,2));
% y = R(P(:,1));
% plot(x,y,'s','color','white');
% 
% % Find lines and plot them
% lines = houghlines(BW,T,R,P,'FillGap',50,'MinLength',50);
% figure, imshow(I), hold on
% max_len = 0;
% for k = 1:length(lines)
%     xy = [lines(k).point1; lines(k).point2];
%     plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');
%     
%     % plot beginnings and ends of lines
%     plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
%     plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');
%     
%     % determine the endpoints of the longest line segment
%     len = norm(lines(k).point1 - lines(k).point2);
%     if ( len > max_len)
%         max_len = len;
%         xy_long = xy;
%     end
% end
% 
% %highlight the longest line segment
% plot(xy_long(:,1),xy_long(:,2),'LineWidth',2,'Color','cyan');

% %RADON TRANSFORM
% fig;
% I = I_Im1s;
% theta = 0:179;
% [R,xp] = radon(edge(I,'sobel'),theta);
% imshow(R,[],'Xdata',theta,'Ydata',xp,'InitialMagnification','fit')
% xlabel('\theta (degrees)')
% ylabel('x''')
% colormap(gca,hot), colorbar; axis tight

% [F,Fpos,Fangles] = fanbeam(I_Im1s,258);
% figure
% imshow(F,[],'XData',Fangles,'YData',Fpos,'InitialMagnification','fit')
% axis normal
% xlabel('Rotation Angles (degrees)')
% ylabel('Sensor Positions (degrees)')
% colormap(gca,hot), colorbar
