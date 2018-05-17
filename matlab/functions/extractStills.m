function []=extractStills()
[fname, pname] = uigetfile({'*.mov','MOV Files (*.mov)'},'Select images','MultiSelect','off');
saverFolder=[fname(1:end-4) ' Stills' filesep];

V=VideoReader([pname fname]);
dt = .25;
t = 0;
i=0;

n=floor(V.Duration/dt);

mkdir(pname,saverFolder)
while t<(V.Duration-dt)
    i = i+1;  fprintf('%g/%g\n',i,n)
    t = t+dt;
    V.CurrentTime=t;
    I = images.internal.rgb2graymex(readFrame(V));
    imwrite(I,sprintf('%s_%g.jpg',[pname saverFolder fname(1:end-4)],i))
end

fprintf('Done. %g new images in %s\n',i,[pname saverFolder])