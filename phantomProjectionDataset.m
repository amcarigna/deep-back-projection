%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the code that generated our datasets.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function dataset = phantomProjectionDataset(pixels,number,angleRes)

testImage=phantom(pixels);
thIndex = 0:angleRes:180-angleRes;
[RImage,rIndex] = radon(testImage,thIndex); %Make an example image of the correct size to make sure that the dimensions of the tensor line up correctly.

%generate phantom Projections
for i = 1:number
	%make phantom: hacked together from code ripped from stack exchange

	%// Declare output image

	 out = zeros(pixels,pixels); 


	%// Declare maximum semimajor/semiminor axes

	a = 6;
	b = 6;


	%// Declare reference coordinates

	[X,Y] = meshgrid(1:pixels, 1:pixels);

	%// Declare Mask

	mask = (X-pixels/2).^2 + (Y-pixels/2).^2 < ((pixels-10)/2)^2;


	%// Declare Proportion of filled pixels. 

	maxprop = (pi*((pixels-10)/2)^2)/(pixels^2);

	prop = (maxprop - 0.05)*rand;


	while true
    
		%// Generate random a, b and centre
    
		x_axis = a*rand;
    
		y_axis = b*rand;

		angle = 180*rand;
    
		centre = [pixels*rand pixels*rand];

    
		%// Define coordinates with respect to this centre
    
		Xmove = X - centre(1);
    
		Ymove = Y - centre(2);

		Xturn = imrotate(Xmove,angle,'crop');

		Yturn = imrotate(Ymove,angle,'crop');

    
		%// Check ellipse equation and filter out those locations
    
		%// that satisfy result

		ind = ((Xturn.^2) / x_axis^2) + (((Yturn.^2) / y_axis^2)) <= 1;
		%mask = (X-pixels/2).^2 + (Y-pixels/2).^2 <= (pixels - 5)^2/4;
		ind = and(mask,ind);

		%// Set these locations in the output image to a random intensity
    
		intensity = floor(223*rand) + 32;        
    		%Nuke everything outside the mask
		out(ind) = intensity;

    
		%// Check proportion
    
		if sum((out(:) > 0) / numel(out)) >= prop
        
			break;
    
		end
	end


	%make phantom projection. Normalize so data is in [0,1]
	Rout = out./256;
	Rnormalize = max(max(radon(mask, thIndex)));

	[Rin,rIndex] = radon(Rout,thIndex);
    	imwrite(iradon(Rin,thIndex),strcat('test/ramp_back/ramp_back',int2str(i),'.png'));
	Rin = Rin./Rnormalize;
    imwrite(Rin,strcat('test/sinogram/sinogram',int2str(i),'.png'));
	%load phantom into dataset
	imwrite(Rout,strcat('test/orig/orig',int2str(i),'.png'));
    

	%Make BP Tensor
		pixels_desired = pixels+4;% + 4*4; %4 pixels burned off per layer, 4 layers
		mcb = MCB(Rin, rIndex, thIndex,pixels_desired);	
		fmcb = FMCB(Rin,rIndex,thIndex,pixels_desired);
	%load phantom's BP Tensor into dataset

		for j = 1:max(size(thIndex)) 
			imwrite(mcb(:,:,j),strcat('test/mcb/mcb',int2str(i),'channel',int2str(j),'.png'));
		end


		for j = 1:max(size(thIndex))
			imwrite(fmcb(:,:,j),strcat('test/fmcb/fmcb',int2str(i),'channel',int2str(j),'.png'));%write
		end

		disp(i);

		if mod(i, 100) == 0
			clc
		end
end

dataset=1;

end
