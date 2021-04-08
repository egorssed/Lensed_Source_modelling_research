# These are examples several techniques of determining the borders of the galaxy

For explanation of the techniques read the end of the README

1. Galaxy 28193. It shows how unreliable may be Sersic fit for large complex structures. 95% of the Sersic flux region covers only the very center of the galaxy, whereas it spans much further. 
2. Galaxy 17269. Technique based on total flux and Increase curve tends to overestimate sizes of small galaxies, whereas our goal in performing the image cut is to conserve small galaxies in dataset and cast them to dots during the downscaling.
3. Galaxy 10614. For big values of Sersic index 'n'. The Sersic based technique also tends to overestimation of galaxy's size.
4. Galaxy 25882. For very thin galaxies Sersic technique may also choose to strict borders and cut off parts of the galaxy.

Summary:
 - Sersic technique sometimes overestimates, sometimes underestimates galaxy's size
 - Increase curve technique either gives big overestimation or plausible results.
 - Radial profile technique almost always gives good results and rarely slight overestimation.

Decision - use Radial profile cut. 



For all the techniques first step is similar:
We fit the image with elliptical Sersic profile (see the article) to obtain several important parameters. These parameters can be reliably determined with Sersic fitting, whereas the others are highly degenerate and unreliable.
 - x0,y0-positions of galaxy center
 - q - ratio of ellipsis axes
 - phi - angle of ellipsis rotation with respect to horizontal axis

With these parameters we build elliptical coordinates starting in the center of the galaxy and then calculate two radial characteristics:
 - Radial profile of the galaxy - dependence of average Flux in ring of radius R on this radius 
 - Increase curve of the galaxy - dependence of total Flux enclosed in circle of radius R on this radius

The three methods to determine galaxy borders radius:
 - Galaxy border is radius R such that on the elliptic ring of radius R the average Flux is 1/100 of the Flux in galaxy's center. It means that normalized Radial_profile(R)=0.01
 - Galaxy border is radius R such that in the ellipsis of radius R the total Flux is 95% of total image Flux. It means that normalised Increase_curve(R)=0.95
 -  The last technique is the same as the previous, but we don't integrate the image, and take the desired radius from integration of Sersic profile, namely inverse incomplete gamma function.

