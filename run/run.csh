#!/usr/bin/csh

foreach i (`seq 1 1000`)
   echo "Image ${i}"
   make all
end
