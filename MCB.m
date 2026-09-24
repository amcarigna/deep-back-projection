function mcb = MCB(Rin,rIndex,thIndex,pixels)

    channels = max(size(thIndex));

    mcb = zeros(pixels,pixels,channels);

    for i = 1:channels

	    R = Rin(:,i);

	    mcb(:,:,i) = iradon([R R], [thIndex(i) thIndex(i)],'linear','none',1,pixels)/2;

    end

end
