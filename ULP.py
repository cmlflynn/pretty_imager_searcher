
import numpy as np
import matplotlib.pyplot as plt
import astropy
import astropy.units as units
import sys
from astropy.io import fits
from numpy.polynomial import polynomial as P
from astropy import wcs
from caspy_search.Boxcarer import Boxcar_and_threshold
from caspy_search.Cands_handler import Cands_handler

# DM dispersion time delay constant
DM_CONSTANT = 4.15 * units.millisecond

class plan:
    def __init__(self, npix, nt, tsamp_s, mjd, fmin, fmax, wcsheader):
        self.npix = npix
        self.nt = nt # Nsamples / block
        self.tsamp_s = tsamp_s # astropy object in seconds
        self.mjd = mjd # float in days
        self.fmin = fmin # float in Hz
        self.fmax = fmax # float in Hz
        self.wcsheader = wcsheader # astropy WCS object
    def __str__(self):
        string = str(self.npix)+" "+str(self.nt)+" "+str(self.tsamp_s)+" "\
            +str(self.mjd)+" "+str(self.fmin)+" "+str(self.fmax)+" "+str(self.wcsheader)
        return string

#NPIX = 256    
#def location2pix(location, npix=NPIX):
def location2pix(location, npix):
    npix_half = npix//2
    vpix = (location//npix)%npix - npix_half
    if (vpix<0):
        vpix = npix+vpix
    upix = location%npix - npix_half
    if (upix<0):
        upix = npix+upix
    return vpix, upix

def cand2str(candidate, npix, iblk, raw_noise_level):
    location = candidate['loc_2dfft']
    lpix, mpix = location2pix(location, npix)
    rawsn = candidate['snr']
    snr = float(candidate['snr'])#/(raw_noise_level*np.sqrt(candidate['boxcar'])))
    s = f"{snr:.2f}\t{lpix:4d}\t{mpix:4d}\t{candidate['boxc_width']:.2f}\t\t{candidate['time']:.2f}\t{candidate['dm']:.1f}\t{iblk:5d}\t{rawsn:.2f}"
    return s

def print_candidates(candidates, npix, iblk, plan=None):
    for candidate in candidates:
        print(cand2str(candidate, npix, iblk))

def cand2str_wcs(c, iblk, plan, raw_noise_level):
    s1 = cand2str(c, plan.npix, iblk, raw_noise_level)
    #total_sample = iblk*plan.nt + c['time']
    total_sample = c['total_sample']
    tsamp_s = plan.tsamp_s
    obstime_sec = total_sample*plan.tsamp_s
    #mjd = plan.tstart.utc.mjd + obstime_sec.value/3600/24
    mjd = plan.mjd + obstime_sec.value/3600/24
    dmdelay_ms = c['dm']*tsamp_s.to(units.millisecond)
    dm_pccm3 = dmdelay_ms / DM_CONSTANT / ((plan.fmin/1e9)**-2 - (plan.fmax/1e9)**-2)
    lpix, mpix = location2pix(c['loc_2dfft'], plan.npix)
    # the first arguments two are spatial, the third is the reference point in the wcsheader
    # CURRENTLY HARD WIRED
    coord = plan.wcsheader.pixel_to_world(lpix, mpix, (128, 128))
    ra_dec, junk = coord
    ra_dec = ra_dec[0]
    coord = ra_dec
    s2 = f'\t{total_sample:6d}\t\t{obstime_sec.value:0.4f}\t{mjd:0.2f}\t{dm_pccm3.value:0.2f}\t{coord.ra.deg:0.8f}\t{coord.dec.deg:0.6f}  final'
    return s1+s2

def print_candidates_with_wcs(candidates, iblk, plan, raw_noise_level):
    for c in candidates:
        print(cand2str_wcs(c, iblk, plan, raw_noise_level))


def search_cube(filename, threshold):
        
    # open the image plane - timeseries data cube    
    fits_file = fits.open(filename)    
    header = fits_file[0].header
    cube = fits_file[0].data

    # read 2 required cards from the fits header
    MJDREFF = header["MJDREFF"]
    MJDREFI = header["MJDREFI"]

    # reconstruct the MJD of the obs from these cards
    MJD = MJDREFI + MJDREFF

    # get the first channel's frequency 
    FCH1_HZ = header["FCH1_HZ"]
    FCH1_GHZ = FCH1_HZ/1.0E9

    # get the frequency channel width
    CH_BW_HZ = header["CH_BW_HZ"]

    # get the number of channels
    NCHAN = header["NCHAN"]

    # get the sample length (ie time per sample)
    TSAMP = header["TSAMP"]

    # get the WCS header for coordinate transformations
    wcsheader = wcs.WCS(header)

    # get the time series and image plane dimensions of the cube
    Nsamples = int(cube.shape[0])
    nx = int(cube.shape[1])
    ny = int(cube.shape[2])
    print("cube shape : "+str(cube.shape))

    # set the block length for sub sampling the time series data
    block_len = 256 // 4

    # replace with noise and high S/N pixel(s)
    mock = False
    mock = True

    if mock:
        #print("Loading vela")
        vela = np.load("/home/cflynn/Dropbox/nayab/vela.npy")
        #print("loaded")
        vela = np.mean(vela, axis=0)[0:Nsamples]
        #plt.plot(vela)
        #plt.show()
        #sys.exit()
    
    if mock:
        print("Replacing with mock data")
        print("Gaussian noise")
        cube = np.random.normal(0.0,5.0,(Nsamples,nx,ny))

        #print("Inserting two events, SNR = 100 per sample, two samples wide")

        #cube[Nsamples//2,0,0] += 500.0
        #cube[Nsamples//2+1,0,0] += 500.0
        
        #cube[Nsamples//2+50,nx//2,ny//2] += 500.0
        #cube[Nsamples//2+51,nx//2,ny//2] += 500.0
        #cube[Nsamples//2+52,nx//2,ny//2] += 500.0
        #cube[Nsamples//2+53,nx//2,ny//2] += 500.0

        vela = np.squeeze(vela)
        vela -= np.median(vela)
        
        for i in range(len(vela)):
            cube[i,nx//2,ny//2] += vela[i] / 2.0

        ts = cube[:,nx//2,ny//2]
        plt.figure(figsize=(14,6))
        plt.plot(ts)
        plt.xlabel('time [samples]')
        for i in range(Nsamples//block_len + 2):
            plt.axvline(i*block_len,color='g')

        #sys.exit()
        
        
    # we'll always be dealing with square images so set npix
    npix = nx

    # set the number of DM trials to search
    Ndm = 1

    # set the number of boxcar trials to search
    Nbox = 8

    # set up header strings of final results
    cand_str_header = '#SNR\tlpix\tmpix\tboxc_w[ms]\ttime[s]\tdm\tiblk\trawsn'
    cand_str_wcs_header = cand_str_header + \
        "\ttot_samp\tobstime_sec\tmjd\t\tdm_pccm3\tra_deg\tdec_deg\n"

    # set up the candidates handler and output file
    outname = 'candidates.txt'
    ch = Cands_handler(outname=outname)

    verbose = False
    #verbose = True

    for jj in range(npix):
        for ii in range(npix):

            jj = ny//2
            ii = nx//2
            
            #if jj==128 and ii==128:
            #    verbose = True
            #else:
            #    verbose = False
        
            if verbose:
                print("==================================================================")
                print("Search pixel pair =", ii, jj)
                print(cand_str_wcs_header)

            # extract the time series data for this sky location
            time_series = cube[:,jj,ii]

            # measure noise in this time series for the S/N threshold
            noise = np.std(time_series)

            # do a 3-sigma clip on this to get remove bright events
            clip = False
            clip = True
            if clip:
                mask = time_series < 3.0*noise
                noise = np.std(time_series[mask])
        
            time_series /= noise
                
            # set up an array of the time corrseponding to each sample
            local_timeseries = np.zeros((Ndm,Nsamples))
            local_timeseries[0,:] += time_series

            # set this parameter for the event search agorithm
            keep_last_boxcar = True

            # set up a logger of the boxcar history
            boxcar_history = np.zeros((Ndm,Nbox))

            # set up a boxcar searcher instance
            searcher = Boxcar_and_threshold(nt=block_len,boxcar_history=boxcar_history,
                                            keep_last_boxcar=keep_last_boxcar)

            # set up weights for the accumulation of S/N over boxcar width
            dm_boxcar_norm_factors = np.ones((Ndm,Nbox)) / np.sqrt(np.arange(Nbox)+1.0)

            # compute the number of blocks to process
            Nblocks = Nsamples // block_len

            if verbose:
                print("Nblocks = ", Nblocks)
            
            # main loop over blocks in the time series
            for iblock in range(Nblocks):

                if verbose:
                    print()
                print("Searching iblock", iblock)

                # now search for candidate events in this time series
                # the 5 returned values in the cands array are:
                # cands[0] snr
                # cands[1] ibox
                # cands[2] idm
                # cands[3] it + iblock * nt
                # cands[4] ngroup

                
                #plt.plot(local_timeseries[0,iblock*block_len:(iblock+1)*block_len])
                #print(local_timeseries[0,iblock*block_len:(iblock+1)*block_len])
                #plt.show()
                #sys.exit()
                
                cands = searcher.boxcar_and_threshold(
                    dmt_out = local_timeseries[:,iblock*block_len:(iblock+1)*block_len],
                    threshold = threshold,
                    dm_boxcar_norm_factors = dm_boxcar_norm_factors,
                    iblock = iblock)


                #print(iblock, len(cands))
                #sys.exit()

                
                if (len(cands) > 0) and ((iblock >= 0 and iblock % 1 == 0) \
                                         or (iblock + 1 == Nblocks) \
                                         or (iblock + 2 == Nblocks)):

                    print()
                    print("=======================================================================================")
                    print("\tcandidates found in pixel ",ii,jj)
                    print("\tcandidates before clustering")
                    print("\tcandidate #,     snr,    ibox+1,    idm,     it+iblock*nt,    ngroup")
                    #print(cands)
                    for i in range(len(cands)):
                        cands[i][3] -= cands[i][1]//2
                        print('\t{0:6d}{1:12.2f}{2:6d}{3:12d}{4:12d}{5:12d}'.format(i,                     # candidate number
                                                                                    cands[i][0],               # SNR
                                                                                    int(cands[i][1]+1),        # ibox+1
                                                                                    int(cands[i][2]),          # idm
                                                                                    int(cands[i][3]),   # it+iblock*nt - tophat//w = total_sample
                                                                                    int(cands[i][4])))         # ngroup

                    print()
        
                    repr_cands = ch.cluster_cands(cands)

                    
                    final_cands = ch.add_physical_units_columns(cands=repr_cands,
                                                                fbottom=FCH1_HZ,
                                                                df = CH_BW_HZ,
                                                                nf = NCHAN,
                                                                tsamp=TSAMP,
                                                                mjd_start=MJD)

                    print()
                    print("\tcandidates after clustering")

                    print("\traw noise = ", noise)
                    
                    print("\tcandidate #,     snr,    ibox+1,    idm,     it+iblock*nt,    ngroup,   ncands_in_group,  BCW(ms), DM(pc/cc), time (s),  MJD")
                    for i in range(len(final_cands)):
                        print('\t{0:6d}{1:12.2f}{2:6d}{3:12d}{4:12d}{5:12d}{6:5d}\t{7:0.8f}\t{8:0.8f}\t{9:0.8f}\t{10:0.10f}'.format(i,
                                                                                                                                    final_cands[i][0],
                                                                                                                                    int(final_cands[i][1]+1),
                                                                                                                                    int(final_cands[i][2]),
                                                                                                                                    int(final_cands[i][3]),
                                                                                                                                    int(final_cands[i][4]),
                                                                                                                                    int(final_cands[i][5]),
                                                                                                                                    final_cands[i][6],
                                                                                                                                    final_cands[i][7],
                                                                                                                                    final_cands[i][8],
                                                                                                                                    final_cands[i][9]))
                    print()
                    print()
                    print(cand_str_wcs_header)
                
                    # info that we'll add to the "plan" for this pixel
                    # frequency info, time sampling, MJD, wcs stuff
                    fmin = FCH1_HZ
                    fmax = FCH1_HZ + NCHAN*CH_BW_HZ
                    time_per_sample = TSAMP*units.second
                    my_plan = plan(npix, Nsamples, time_per_sample, MJD, fmin, fmax, wcsheader)
                    #print(my_plan)
                
                    # keep track of the noise in the time series for later so that rawsn can be converted to true S/N
                    raw_noise_level = 1.0000000

                    if verbose:
                        print("raw_noise_level = ",raw_noise_level)
                
                    # vpix and upix are acurrently treated as in x- and y- on the sky
                    # need to test if this is true
                    vpix = ii
                    upix = jj

                    # reconstruct the spatial location value from the vpix and upix positions
                    npix_half = npix//2
                    location = ((npix_half+vpix)%npix)*npix + (npix_half+upix)%npix

                    # stuff that's in a candidate numpy array at this stage:
                    # SNR     boxcar  DM    total_sample   ngroup  ncluster        boxcar_ms       DM_pccc time_s  mjd_inf
                    # so create a data type so we can add "loc_2dfft", "iblock" and "rawsn"
                    final_cands_structured_array_dtype = np.dtype([('snr', 'f8'),
                                                                   ('boxcar', 'i4'),
                                                                   ('dm', 'f8'),
                                                                   ('total_sample', 'i4'),
                                                                   ('ngroup', 'i4'),
                                                                   ('ncluster', 'i4'),
                                                                   ('boxc_width', 'f8'),
                                                                   ('dm_pccc', 'f8'),
                                                                   ('time', 'f8'),
                                                                   ('mjd_inf', 'f8'),
                                                                   ('loc_2dfft', 'i4'),
                                                                   ('iblk', 'i4'),
                                                                   ('rawsn', 'f8')])

                    # append (one at a time) the location, block number and rawsn to the results array (final_cands)
                    final_cands_location = np.append(final_cands,location)
                    final_cands_location_iblk = np.append(final_cands_location,iblock)
                    final_cands_location_iblk_rawsn = np.append(final_cands_location_iblk,0.0)

                    # insert the data in the results numpy array into a structured numpy array
                    # this requires a reshape operation to create rows
                    # and then conversion of each row's data into a tuple
                    final_cands_location_iblk_rawsn = \
                        final_cands_location_iblk_rawsn.reshape(1,len(final_cands_location_iblk_rawsn))
                    final_cands_location_iblk_rawsn = [tuple(i) for i in final_cands_location_iblk_rawsn]

                    # insert the final candidate list into the structured array
                    final_cands_structured_array = np.array(final_cands_location_iblk_rawsn,
                                                            dtype=final_cands_structured_array_dtype)

                    print_candidates_with_wcs(final_cands_structured_array,
                                              iblock, my_plan, raw_noise_level)

                    print(final_cands_structured_array['total_sample'])
                    plt.axvline(final_cands_structured_array['total_sample'],color='r',alpha=0.5)
                    plt.text(final_cands_structured_array['total_sample'], 125.0,str(np.around(float(final_cands_structured_array['snr']),1)))
            plt.show()
            sys.exit()
                    
    return

#######################################################################################################
# check for a command line argument
nargs = len(sys.argv) - 1
if (nargs==2):
    filename = sys.argv[1]
    threshold_argv = float(sys.argv[2])
else:
    print("Needs a fits file and an SNR_threshold")
    print("e.g. ULP.py ULP*.fits 9")
    sys.exit()

search_cube(filename, threshold_argv)

        


