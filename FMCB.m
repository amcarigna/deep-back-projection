function fmcb = FMCB(Rin,rIndex,thIndex,pixels)

channels = max(size(thIndex));

fmcb = zeros(pixels,pixels,channels);

for i = 1:channels

	R = Rin(:,i);

	fmcb(:,:,i) = iradon([R R], [thIndex(i) thIndex(i)],'linear','Ram-Lak',1,pixels)/2;

end

end
