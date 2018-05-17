function [tform, inliers] = fitgeotransSigmaRejection(xin,xout,srl)
%xin = [nx2] pixels in source image
%xout = [nx2] pixels in output image
%tform = [3x3] converts from input to output image [xout 1] = [xin 1]*tform;

dx = xout-xin;
dxr = sqrt(dx(:,1).^2 + dx(:,2).^2); %dx range
dxa = fcnatan2(dx(:,2),dx(:,1)); %dx angle
[~, inliersa] = fcnsigmarejection(dxr,srl,3,'onlyIndices');
[~, inliersb] = fcnsigmarejection(dxa,srl,3,'onlyIndices');
inliers = inliersa & inliersb;
%inliers = true(size(xin,1),1);

n = 4;
np0 = 0;  
for i=1:n
    np = sum(inliers);  if np==np0; break; end
    ov = ones(np,1);  xouti=xout(inliers,:);  xini=xin(inliers,:);
    H = [xini ov];
    %z = [xouti ov];
    %tform = (H'*H)\H'*z; %LLS
    tf=fitgeotrans(xini,xouti,'affine');
    %if rcond(tf.T)<1E-7; tf = fitgeotrans(xini,xouti,'affine'); end
    tform=tf.T; %MATLAB Built-In
    
    if i<n
        %TIE VECTOR METRICS
        [~, inliersa] = fcnsigmarejection(dxr(inliers),srl,3,'onlyIndices');
        [~, inliersb] = fcnsigmarejection(dxa(inliers),srl,3,'onlyIndices');
        
        %RESIDUAL METRICS
        zhat=H*tform;  zhat=zhat(:,1:2)./zhat(:,3);
        res = xouti - zhat; %xyz residual
        r = sqrt(sum(res.^2,2)); %range residual
        [~, inliersc] = fcnsigmarejection(r,srl,3,'onlyIndices');
        
        %fig(1,3); sca; histogram(dxr,30); sca; histogram(dxa,30); sca; histogram(r,30);
        inliers(inliers) = inliersa & inliersb & inliersc;
    end
    np0=np;
end

%Inverse Transform
%itform = tform^-1;  %itform = (z'*z)\z'*H; %LLS

