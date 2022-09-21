import ROOT
import numpy as np
import pylab
import sys
from operator import itemgetter
import os
import math
from scipy.optimize import curve_fit
from matplotlib.patches import Rectangle
import struct
from matplotlib import colors
import yaml
from natsort import natsorted, ns
import warnings
import time
import multiprocessing as mp
warnings.filterwarnings("ignore")


"""Script for floodmap Module TB
"""

def plot_parameters():
    pylab.rcParams[ 'lines.linewidth' ] = 2
    pylab.rcParams[ 'font.size' ] = 15
    pylab.rcParams[ 'axes.titlesize' ] = 19
    pylab.rcParams[ 'axes.labelsize' ] = 16
    pylab.rcParams[ 'ytick.major.pad' ] = 14
    pylab.rcParams[ 'xtick.major.pad' ] = 14
    pylab.rcParams[ 'legend.fontsize' ] = 16


def gaussian(x, a, x0, sigma):
 	return a * np.exp(-(x - x0)**2 / (2 * sigma**2))


def lorentzian(x, a, x0, gam):
    return a * gam**2 / ( gam**2 + ( x - x0 )**2)


def gaussians_fit(n, bins, cb = 8):
  bin_centres = ( bins[ :-1 ] + bins[ 1: ] ) / 2
  #pylab.show()

  maxValue = np.where( n == n.max() )[ 0 ][ 0 ]
  init = max(maxValue - cb, 0)
  end = min(maxValue + cb, len(n))
  bin_centres_cut = bin_centres[init : end]
  n_cut = n[init : end]
  mu = sum(bin_centres_cut * n_cut) / sum(n_cut)

  sigma = np.sqrt(sum(n_cut * (bin_centres_cut - mu)**2) / sum(n_cut))
  p1, pcov = curve_fit(gaussian, bin_centres_cut, n_cut, p0=[max(n), mu, sigma])
  perror = np.sqrt(np.diag(pcov))
  amplitud = p1[0]
  muopt = p1[1]
  sigmaopt = p1[2]

  for it in range(len(bins)):
      if bins[it+1] > muopt:
        break
  peak = it
  binSize = bins[2] - bins[1]
  binMin = max(0,peak - int( 10*sigmaopt/binSize ))
  binMax = min(peak + int( 10*sigmaopt/binSize ), len(bin_centres) - 1)
  x = bin_centres
  #x = np.linspace( 3, 8, 100 )
  y = gaussian( bin_centres, p1[0], p1[1], p1[2] )
  fwhm = p1[ 2 ] * math.sqrt( 8 * math.log( 2 ) )

  return x, y, amplitud, muopt, sigmaopt, perror, fwhm


def lorentizan_fit(n, bins, cb = 8):
  bin_centres = ( bins[ :-1 ] + bins[ 1: ] ) / 2
  #pylab.show()

  maxValue = np.where( n == n.max() )[ 0 ][ 0 ]
  init = max(maxValue - cb, 0)
  end = min(maxValue + cb, len(n))
  bin_centres_cut = bin_centres[init : end]
  n_cut = n[init : end]
  mu = sum(bin_centres_cut * n_cut) / sum(n_cut)

  sigma = np.sqrt(sum(n_cut * (bin_centres_cut - mu)**2) / sum(n_cut))
  p1, pcov = curve_fit(lorentzian, bin_centres_cut, n_cut, p0=[max(n), mu, sigma])
  perror = np.sqrt(np.diag(pcov))
  amplitud = p1[0]
  muopt = p1[1]
  sigmaopt = p1[2]

  #for it in range(len(bins)):
    #  if bins[it+1] > muopt:
    #    break
  #peak = it
  #binSize = bins[2] - bins[1]
  #binMin = max(0,peak - int( 10*sigmaopt/binSize ))
  #binMax = min(peak + int( 10*sigmaopt/binSize ), len(bin_centres) - 1)
  x = bin_centres
  #x = np.linspace( 3, 8, 100 )
  y = lorentzian( bin_centres, p1[0], p1[1], p1[2] )
  fwhm = p1[ 2 ] * 2

  return x, y, amplitud, muopt, sigmaopt, perror, fwhm


def parallel_read(file_to_read, energy_chid, mod_mapping):
    start_r = time.time()
    size_file = os.path.getsize(file_to_read)
    fd = open(file_to_read, "rb")
    data_all = fd.read(size_file)
    #data = fd.read(eventSize)
    dt = np.dtype('B, B, i8, f4, i, B, B, i8, f4, i')
    data_array = np.frombuffer(data_all, dtype=dt)
    print("Size data_array - {}".format(len(data_array)))

    #paralellization 
    cpu_count = mp.cpu_count() - 12 #num cores to use (leaving 12 free)
    file_size = len(data_array)
    chunk_size = file_size // cpu_count
    chunk_args = []
    for it in range(cpu_count):
        chunk_args.append((data_array[chunk_size*it:chunk_size*(it+1)], energy_chid, mod_mapping))
    del data_array
    #print(chunk_args)
    print(cpu_count, file_size, chunk_size)
    
    with mp.Pool(cpu_count) as p:
        # Run chunks in parallel
        chunk_results = p.starmap(process_chunk, chunk_args)
    
    data_dict = {}
    cnt_ev = 0
    for chunk in chunk_results:
        if cnt_ev != 0:
            copy_chunk = chunk.copy()
            for ev in copy_chunk.keys():
                chunk[ev + cnt_ev] = chunk.pop(ev)
        cnt_ev = max(chunk.keys()) + 1 #+1 avoid overlap between firs events of different chunks        
        data_dict.update(chunk)
    del chunk_results
    end_r = time.time()
    print("Time enlapsed reading: {} s".format(int(end_r - start_r)))
    return(data_dict)


    
def process_chunk(data_array, energy_chid, mod_mapping):

    cnt = 0
    eventSize = 36
    num_lines = 0
    first_line_event = True
    event = -1
    good_events = 0
    data_dict = {}
    energy_minimum = 0.2
    
    print("No te preocupes, saldrá bien :) :(o no)")
    for line in data_array:
        num_lines += 1
        line_split = line
        #print(line_split)
        if first_line_event:
            #if event > 200000:
            #    break
            #print("FIRST LINE")
            event += 1

            #if event == 13:
            #    for it in range(len(data_dict[12]["mod2"])):
            #        print(data_dict[12]["mod2"][it])
            #    exit(0)

            lines_event = 1
            num_ev_det1 = int(line_split[0])
            num_ev_det2 = int(line_split[5])
            total_lines = num_ev_det1 + num_ev_det2 - 1
            first_line_event = False
            ch1_list = []
            ch2_list = []
            mod1_list = []
            mod2_list = []
            mod1_energy = 0
            mod2_energy = 0
            mod1_numEn = 0
            mod2_numEn = 0
            mod1_totalCh = 0
            mod2_totalCh = 0
            data_dict_event_1 = {}
            data_dict_event_2 = {}
        ch1_id = int(line_split[4])
        ch2_id = int(line_split[-1])
        ch1_tstp = float(line_split[2])
        ch2_tstp = float(line_split[7])
        ch1_energy = float(line_split[3])
        ch2_energy = float(line_split[8])
        ch1_mm = mod_mapping[ch1_id]
        ch2_mm = mod_mapping[ch2_id]

        if ch1_mm not in data_dict_event_1.keys():
            data_dict_event_1[ch1_mm] = [0,0,0]    #energy of the mm, number of energy channels, total number of channels activated
        if ch2_mm not in data_dict_event_2.keys():
            data_dict_event_2[ch2_mm] = [0,0,0]    #energy of the mm, number of energy channels, total number of channels activated
        if ch1_id not in ch1_list:
            data_dict_event_1[ch1_mm].append([ch1_tstp, ch1_energy, ch1_id])

            if ch1_id in energy_chid:
                data_dict_event_1[ch1_mm][0] += ch1_energy
                data_dict_event_1[ch1_mm][1] += 1
                mod1_numEn += 1
            data_dict_event_1[ch1_mm][2] += 1
            mod1_totalCh += 1
            ch1_list.append(ch1_id)
        else:
            pass
        if ch2_id not in ch2_list:
            data_dict_event_2[ch2_mm].append([ch2_tstp, ch2_energy, ch2_id])

            if ch2_id in energy_chid:
                data_dict_event_2[ch2_mm][0] += ch2_energy
                data_dict_event_2[ch2_mm][1] += 1
                mod2_numEn += 1
            data_dict_event_2[ch2_mm][2] += 1
            mod2_totalCh += 1
            ch2_list.append(ch2_id)
        else:
            pass
        if lines_event == total_lines:
            good_event_mod1 = False
            good_event_mod2 = False
            #print("hola")
            #print(data_dict_event_1)
            #print(data_dict_event_2)
            good_data_1 = []
            good_data_2 = []

            for mm in data_dict_event_1.keys():
                #print("mm {}".format(mm))
                mod1_numEn = data_dict_event_1[mm][1]
                mod1_totalCh = data_dict_event_1[mm][2]
                #print(data_dict_event)
                if mod1_numEn > 5 and mod1_totalCh > mod1_numEn:
                    mod1_energy = data_dict_event_1[mm][0]
                    mod1_list = data_dict_event_1[mm][3:]
                    good_data_1.append([mod1_list, mod1_energy, mm])
                    #print(mod2_list)
                    good_event_mod1 = True

            for mm in data_dict_event_2.keys():
                #print("mm {}".format(mm))
                mod2_numEn = data_dict_event_2[mm][1]
                mod2_totalCh = data_dict_event_2[mm][2]
                #print(data_dict_event)
                if mod2_numEn > 4 and mod2_totalCh > mod2_numEn:
                    mod2_energy = data_dict_event_2[mm][0]
                    mod2_list = data_dict_event_2[mm][3:]
                    good_data_2.append([mod2_list, mod2_energy, mm])
                    #print(mod2_list)
                    good_event_mod2 = True

            if good_event_mod1 and good_event_mod2:
                #print(good_data_1)
                if event not in data_dict.keys():
                    data_dict[event] = {"mod1": [], "mod2": []}
                for mm in good_data_1:
                    data_dict[event]["mod1"].append([mm[0] + [mm[1]] + [mm[2]]])
                for mm in good_data_2:
                    data_dict[event]["mod2"].append([mm[0] + [mm[1]] + [mm[2]]])
                good_events += 1

            #print("EVENT {}".format(event))
            #print(data_dict[event])
            #print("-----------")
            first_line_event = True

        else:
            lines_event += 1

    #data_dict format: data_dict[event][module][list_of_minimodules][channel_list + total_energy + minimodule_id]

    print("Number of lines in the file: {}".format(num_lines))
    print("Number of events in file: {}".format(event))
    print("Number of GOOD events in file: {}".format(good_events))

    
    
    return(data_dict)

def sum_spectrum_cut(data_dict, centroid_mapping, file_name):
    mM_dict = {"mod1": {}, "mod2": {}}
    for ev in data_dict.keys():
        for det in data_dict[ev].keys():
            for mM in data_dict[ev][det]:
                mM_num = mM[0][-1]
                mM_energy = mM[0][-2]
                mM_ch_list = mM[0][:-2]
                x, y = centroid_calculation(mM_ch_list, centroid_mapping)

                if mM_num not in mM_dict[det].keys():
                    mM_dict[det][mM_num] = {"energy": [], "x": [], "y": []}
                mM_dict[det][mM_num]["energy"].append(mM_energy)
                mM_dict[det][mM_num]["x"].append(x)
                mM_dict[det][mM_num]["y"].append(y)

    mM_enCut = {"mod1": {}, "mod2": {}}
    for det in mM_dict.keys():
        fig1 = pylab.figure(2, (15,15))
        cnt_mM = 1
        x_total_module = []
        y_total_module = []
        for mM in sorted(mM_dict[det].keys()):
            if mM not in mM_enCut[det].keys():
                mM_enCut[det][mM] = []
            pylab.subplot(4,4,mM)
            n, bins, patches = pylab.hist(mM_dict[det][mM]["energy"], bins = 200, range = [0, 300], histtype='step', fill=False, label = "Det: {}\nmM: {}".format(det,mM))
            x_gauss, y_gauss, amplitud, muopt, sigmaopt, perror, fwhm = gaussians_fit( n, bins, cb = 6 )
            pylab.text( min( x_gauss ) + 0.015 * ( max( x_gauss ) - min( x_gauss ) ), amplitud*0.6, "Centroid: {} au\nER: {}%".format(round(muopt, 2), round(fwhm/muopt*100,2)))
            pylab.plot( x_gauss , y_gauss)
            low_energy = muopt - 2 * sigmaopt
            up_energy = muopt + 2 * sigmaopt
            mM_enCut[det][mM] = {"lower": low_energy, "upper": up_energy}
            pylab.axvspan( low_energy, up_energy, facecolor = '#00FF00', alpha = 0.3 )
            pylab.xlabel("Energy (au)")
            pylab.legend(loc = 0)
            cnt_mM += 1
            filt_peak = [i for i in range(len(mM_dict[det][mM]["energy"])) if mM_dict[det][mM]["energy"][i] > low_energy and mM_dict[det][mM]["energy"][i] < up_energy]
            x_filt = [mM_dict[det][mM]["x"][i] for i in filt_peak]
            y_filt = [mM_dict[det][mM]["y"][i] for i in filt_peak]
            x_total_module.extend(x_filt)
            y_total_module.extend(y_filt)
            #x_total_module.extend(mM_dict[det][mM]["x"])
            #y_total_module.extend(mM_dict[det][mM]["y"])
        pylab.savefig(file_name.replace(".ldat","_EnergyModule" + str(det) + ".png"))
        fig1 = pylab.figure(100, (15,15))
        pylab.hist2d(x_total_module, y_total_module, bins = 500, range = [[0, 104], [0, 104]], cmap="viridis")
        #pylab.hist2d(x_total_module, y_total_module, bins = 500, range = [[0, 104], [0, 104]], cmap="jet", , cmax = 100, norm = colors.LogNorm()) #cmap="viridis" cmap="jet", norm = colors.LogNorm()
        pylab.xlabel('X position (pixelated) [mm]')
        pylab.ylabel('Y position (monolithic) [mm]')
        pylab.colorbar()
        pylab.savefig(file_name.replace(".ldat","_FloodModule" + str(det) + ".png"))
        pylab.close(2)
        pylab.close(100)
        #pylab.show()
    return mM_enCut


def CTR_slab(data_dict, mM_enCut, ALLSM_time_ch, file_name):
    range_low = -3000
    range_top = 3000
    CTR_per_slab = {}
    for ev in data_dict.keys():
        ev_time_dict = {}
        for det in data_dict[ev].keys():
            for mM in data_dict[ev][det]:
                mM_num = mM[0][-1]
                mM_energy = mM[0][-2]
                if mM_energy > mM_enCut[det][mM_num]["lower"] and mM_energy < mM_enCut[det][mM_num]["upper"]:
                    mM_ch_list = mM[0][:-2]
                    time_ch = [(ch[0], ch[2]) for ch in mM_ch_list if ch[2] in ALLSM_time_ch]
                    if det not in ev_time_dict.keys():
                        ev_time_dict[det] = time_ch
                    else:
                        ev_time_dict[det].extend(time_ch)
        if len(ev_time_dict.keys()) == 2:
            time1 = ev_time_dict["mod1"][0][1]
            time2 = ev_time_dict["mod2"][0][1]
            delay = ev_time_dict["mod1"][0][0] - ev_time_dict["mod2"][0][0]
            mix_channel = str(time1) + "-" + str(time2)
            if mix_channel not in CTR_per_slab.keys():
                CTR_per_slab[mix_channel] = []
            CTR_per_slab[mix_channel].append(delay)
    cnt = 1
    CTR_skew_corr = []
    counts_good = 0
    counts_bad = 0
    for ch in CTR_per_slab.keys():
        #print("Number of coincidences: {} - Ch {}".format(len(CTR_per_slab[ch]), ch))
        #pylab.subplot(5,2,cnt)
        std_CTR = np.std(CTR_per_slab[ch])
        num_events = len(CTR_per_slab[ch])

        if std_CTR > 2000 or num_events < 20:
            #print("HERE: {} nE - {} std".format(num_events, std_CTR))
            #print(CTR_per_slab[ch])
            #fig1 = pylab.figure(30, (15,15))
            #n, bins, patches = pylab.hist(CTR_per_slab[ch], bins = 151, range = [range_low, range_top], histtype='step', fill=False, label = "Ch {}".format(ch))
            #pylab.show()
            continue

        fig1 = pylab.figure(20, (15,15))
        n, bins, patches = pylab.hist(CTR_per_slab[ch], bins = 151, range = [range_low, range_top], histtype='step', fill=False, label = "Ch {}".format(ch))

        try:
            x_gauss, y_gauss, amplitud, muopt, sigmaopt, perror, fwhm = gaussians_fit( n, bins, cb = 5 )
            pylab.plot( x_gauss , y_gauss)
            pylab.text( min( x_gauss ) + 0.015 * ( max( x_gauss ) - min( x_gauss ) ), amplitud*0.6, "CTR: {} ps\n{} cps\n{} mu\n{} std\n{} perr".format(int(fwhm),
            len(CTR_per_slab[ch]), muopt, std_CTR, perror))
            pylab.legend(loc = 0)
            pylab.xlim([range_low,range_top])
            CTR_skew_corr.extend(np.array(CTR_per_slab[ch]) - muopt)
            counts_good += num_events
            #pylab.show()
        except:
            cnt += 1
            #print("Error fitting timming")
            #print("Number of coincidences: {} - Ch {}".format(len(CTR_per_slab[ch]), ch))
            CTR_skew_corr.extend(np.array(CTR_per_slab[ch]) - 0) #np.mean(np.array(CTR_per_slab[ch]))
            #CTR_skew_corr.extend([1000]*len(CTR_per_slab[ch]))
            counts_bad += num_events
            continue
        cnt += 1
        #pylab.savefig(file_name.replace(".ldat","_" + ch + ".png"))
        #pylab.show()
        pylab.close(20)
    print("GOOD COUNTS: {}".format(counts_good))
    print("BAD COUNTS: {}".format(counts_bad))
    fig1 = pylab.figure(200, (15,15))
    n, bins, patches = pylab.hist(CTR_skew_corr, bins = 151, range = [range_low, range_top], histtype='step', fill=False, label = "Full CTR")
    x_gauss, y_gauss, amplitud, muopt, sigmaopt, perror, fwhm = gaussians_fit( n, bins, cb = 5 )
    pylab.plot( x_gauss , y_gauss)
    pylab.text( min( x_gauss ) + 0.015 * ( max( x_gauss ) - min( x_gauss ) ), amplitud*0.6, "CTR: {} ps\n{} cps".format(int(fwhm), round(len(CTR_skew_corr), 3)))
    pylab.legend(loc = 0)
    pylab.xlim([range_low,range_top])
    pylab.savefig(file_name.replace(".ldat","_FullCTR.png"))
    #pylab.show()



def centroid_calculation(event_trace, mapping):

    RTP_x = 1
    RTP_y = 2
    x_centroid = 0
    y_centroid = 0
    energy_x = 0
    energy_y = 0
    offset_x = 0.00001
    offset_y = 0.00001
    #print(event_trace)
    for ch_tr in event_trace:
        channel = ch_tr[2]
        energy_ch = ch_tr[1]
        en_t = mapping[channel][0]
        position = mapping[channel][1]
        #print(ch_tr)
        if en_t == 0:   #es señal de tiempo
            x_centroid += ((energy_ch+offset_x)**RTP_x)*position
            energy_x += (energy_ch+offset_x)**RTP_x
            #print("ES X (tiempo): {}-{}".format(energy_ch, position))
        elif en_t == 1: #es señal de energia
            y_centroid += ((energy_ch+offset_y)**RTP_y)*position
            energy_y += (energy_ch+offset_y)**RTP_y
            #print("ES Y (energia): {}-{}".format(energy_ch, position))
    x_centroid = x_centroid/energy_x
    y_centroid = y_centroid/energy_y
    #print("X - Y: {}-{}".format(x_centroid, y_centroid))
    #print("-------------------------")
    return x_centroid, y_centroid


def get_maps(mapping):
    SM1_time_ch = mapping["time_channels"]
    SM1_energy_ch = mapping["energy_channels"]
    FEM_num_ch = 256
    mM_SM = 16
    ALLSM_time_ch = []
    ALLSM_energy_ch = []
    mM_mapping = {}
    mM_num = 1
    slab_num = 1
    centroid_mapping = {}
    rc_num = 0
    mM_energyMapping = {1:1, 2:5, 3:9, 4:13, 5:2, 6:6, 7:10, 8:14, 9:3, 10:7, 11:11, 12:15, 13:4, 14:8, 15:12, 16:16}
    for sm in range(4):
        mM_num = 1
        for tch, ech in zip(SM1_time_ch, SM1_energy_ch):
            absolut_tch = tch + sm*FEM_num_ch
            absolut_ech = ech + sm*FEM_num_ch
            ALLSM_time_ch.append(absolut_tch)
            ALLSM_energy_ch.append(absolut_ech)
            mM_num_en = mM_energyMapping[mM_num]
            mM_mapping[absolut_tch] = mM_num
            mM_mapping[absolut_ech] = mM_num_en
            centroid_mapping[absolut_tch] = [0, round(1.6 + 3.2*(rc_num), 2)]
            centroid_mapping[absolut_ech] = [1, round(1.6 + 3.2*(31-rc_num), 2)]  #stablish 0 reference at the botom left of the floodmap
            rc_num += 1
            if slab_num%8 == 0:
                mM_num += 1
            if slab_num%32 == 0:
                rc_num = 0
            slab_num += 1
    #print(centroid_mapping[239])
    #print(mM_mapping)
    #print(ALLSM_time_ch)
    #print(ALLSM_energy_ch)
    #print(mM_mapping[229])
    #exit(0)
    return ALLSM_time_ch, ALLSM_energy_ch, mM_mapping, centroid_mapping


if __name__ == "__main__":
    print("-----------------------------------------------------------------")
    print("  Usage: python3 SuperModule_CTR.py [file]")
    print("-----------------------------------------------------------------")
    plot_parameters()
    start = time.time()
    try:
        fd = open("/home/tbpet/WorkingDir/macros/SuperModule/SM_mapping.yaml", 'r')
    except:
        print("No YAML file found in this directory...")
        exit(0)

    mapping = yaml.safe_load(fd)
    ALLSM_time_ch, ALLSM_energy_ch, mM_mapping, centroid_mapping = get_maps(mapping)
    list_datafiles = sys.argv[1:]
    list_datafiles = natsorted(list_datafiles, key=lambda y: y.lower())

    for data_file in list_datafiles:
        print("Reading file.... {}".format(data_file))
        data_dict = parallel_read(data_file, ALLSM_energy_ch, mM_mapping)
        mM_enCut = sum_spectrum_cut(data_dict, centroid_mapping, data_file)
        #CTR_slab(data_dict, mM_enCut, ALLSM_time_ch, data_file)
        #pylab.close()
        end = time.time()
        print("Total time enlapsed: {} s".format(int(end - start)))
