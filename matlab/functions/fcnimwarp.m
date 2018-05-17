function J = fcnimwarp(I,Ixy,tform)
%px = Ixy*tform(:,1);  py = Ixy*tform(:,2);  %faster than p=Ixy*tform;

p=Ixy*tform;  
pz=p(:,3);  px=p(:,1)./pz;  py=p(:,2)./pz;

if isinteger(I)
    %J = uint8(interp2(single(I),px,py)); %alternative method
    J = interp2mexchar(I,px,py);
else
    J = interp2mexsingle(I,px,py);
end

%RESHAPE
if numel(I) == numel(px)
    J = reshape(J, size(I)); 
end

%PLOT
%fig; imshowpair(I,J);
