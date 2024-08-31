# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:13:24 2024

@author: w.shu
"""

import os
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def sulcus_name_whole(sulcus_name, atlas):
    """
    Returns the whole name of a ROI, given its abbreviated name. Useful if you 
    are not familiar with the abbreviations of the Brainvisa or the Desikan 
    nomenclature.

    Parameters
    ----------
    sulcus_name : STR
        The abbreviated name of the sulcus.
    atlas : STR
        The atlas for the cerebral parcellation. Possible options are 
        'brainvisa' or 'desikan'.

    Returns
    -------
    sulcus_name_whole : STR
        The whole scientific name of the sulcus.

    """
    assert atlas in ["brainvisa", "desikan"]
    
    if sulcus_name.endswith('left'):
        side  = 'left'
        truncated_sulcus_name = sulcus_name[:-5]
    else:
        side = 'right'
        truncated_sulcus_name = sulcus_name[:-6]
    
    brainvisa_dict = {'F.C.L.a.' : 'anterior lateral fissure',
                  'F.C.L.p.' : 'posterior lateral fissure',
                  'F.C.L.r.ant.' : 'anterior ramus of the lateral fissure', 
                  'F.C.L.r.asc.': 'ascending ramus of the lateral fissure',
                  'F.C.L.r.retroC.tr.' : 'retro-central transverse ramus of the lateral fissure',
                  'F.C.L.r.sc.post.' : 'posterior subcentral ramus of the lateral fissure',
                  'F.C.M.ant.' : 'calloso-marginal anterior fissure',
                  'F.C.M.post.' : 'calloso-marginal posterior fissure',
                  'F.Cal.ant.-Sc.Cal.' : 'calcarine fissure',
                  'F.Coll.' : 'collateral fissure',
                  'F.I.P.' : 'intraparietal sulcus',
                  'F.I.P.Po.C.inf.' : 'superior postcentral intraparietal superior sulcus',
                  'F.I.P.r.int.1' : 'primary intermediate ramus of the intraparietal sulcus', 
                  'F.I.P.r.int.2' : 'secondary intermediate ramus of the intraparietal sulcus', 
                  'F.P.O.' : 'parieto-occipital fissure',
                  'INSULA' : 'insula', 
                  'OCCIPITAL' : 'occipital lobe', 
                  'S.C.' : 'central sulcus', 
                  'S.C.sylvian.' : 'central sylvian sulcus', 
                  'S.Call.' : 'subcallosal sulcus', 
                  'S.Cu.' : 'cuneal sulcus',
                  'S.F.inf.' : 'inferior frontal sulcus', 
                  'S.F.inf.ant.' : 'anterior inferior frontal sulcus', 
                  'S.F.int.' : 'inferior frontal sulcus', 
                  'S.F.inter.' : 'intermediate frontal sulcus',
                  'S.F.marginal.' : 'marginal frontal sulcus', 
                  'S.F.median.' : 'median frontal sulcus', 
                  'S.F.orbitaire.' : 'orbital frontal sulcus',
                  'S.F.polaire.tr.' : 'polar frontal sulcus', 
                  'S.F.sup.' : 'superior frontal sulcus', 
                  'S.Li.ant.' : 'anterior intralingual sulcus', 
                  'S.Li.post.' : 'posterior intra-lingual sulcus',
                  'S.O.T.lat.ant.' : 'anterior occipito-temporal lateral sulcus', 
                  'S.O.T.lat.int.' : 'interior occipito-temporal lateral sulcus', 
                  'S.O.T.lat.med.' : 'median occipito-temporal lateral sulcus',
                  'S.O.T.lat.post.' : 'posterior occipito-temporal lateral sulcus', 
                  'S.O.p.' : 'occipito-polar sulcus', 
                  'S.Olf.' : 'olfactory sulcus', 
                  'S.Or.' : 'orbital sulcus', 
                  'S.Pa.int.' : 'internal parietal sulcus',
                  'S.Pa.sup.' : 'superior parietal sulcus', 
                  'S.Pa.t.' : 'transverse parietal sulcus', 
                  'S.Pe.C.inf.' : 'inferior precentral sulcus', 
                  'S.Pe.C.inter.' : 'intermediate precentral sulcus',
                  'S.Pe.C.marginal.' : 'marginal precentral sulcus', 
                  'S.Pe.C.median.' : 'median precentral sulcus', 
                  'S.Pe.C.sup.' : 'superior precentral sulcus', 
                  'S.Po.C.sup.' : 'superior postcentral sulcus',
                  'S.R.inf.' : 'inferior rostral sulcus', 
                  'S.Rh.' : 'rhinal sulcus', 
                  'S.T.i.ant.' : 'anterior inferior temporal sulcus', 
                  'S.T.i.post.' : 'posterior inferior temporal sulcus', 
                  'S.T.pol.' : 'polar temporal sulcus',
                  'S.T.s.' : 'superior temporal sulcus', 
                  'S.T.s.ter.asc.ant.' : 'anterior terminal ascending branch of the superior temporal sulcus', 
                  'S.T.s.ter.asc.post.' : 'posterior terminal ascending branch of the superior temporal sulcus',
                  'S.p.C.' : 'paracentral sulcus',
                  'S.s.P.' : 'subparietal sulcus'}
    if atlas == "brainvisa":
        whole_sulcus_name = side + ' ' + brainvisa_dict[truncated_sulcus_name]
    else: # the abbreviations in the Desikan parcellation are already explicit
        whole_sulcus_name = side + ' ' + truncated_sulcus_name[:-1]
    return whole_sulcus_name

def read_sulci(feature, atlas, dataset):
    """
    Creates a Dataframe filled with subjects' data, indexed by their ID (which 
    have been randomized in house). Columns of the csv must be, IN ORDER: 
        - Age at imaging (AGE)
        - Sex (SEX) 
    followed by all the ROIs (34/hemisphere for Desikan, 62/hemisphere for 
    BrainVISA).
    
    Warning: in the case of the BrainVISA atlas, which is more fine-grained 
    than the Desikan-Killiany atlas, some ROIs are discarded due to too many 
    missing values.
    
    Warning: all subjects of the toy example given in the github were 
    processed both for BrainVISA and for Desikan. The subjects' IDs remain stable.

    Parameters
    ----------
    feature : STR
        The neuroanatomical feature to extract. Possible options are 'opening'
        or 'thickness'.
    atlas : STR
        The atlas for the cerebral parcellation. Possible options are 
        'brainvisa' or 'desikan'.
    dataset : STR
        The dataset. Possible options are 'UKB' for UK Biobank, or 'Memento' 
        for Memento.

    Returns
    -------
    data : DATAFRAME 
        The DataFrame filled with subjects' data.
    """
    assert feature in ['opening', 'thickness']
    assert atlas in ['brainvisa', 'desikan']
    assert dataset in ['UKB', 'memento']
    if feature == 'opening' and atlas == 'desikan':
        raise AssertionError("Only the Desikan cortical thickness is available.")
    if dataset == 'memento' and (atlas, feature) != ('brainvisa', 'opening'):
        raise AssertionError("Only the Brainvisa sulcal opening is available on the Memento dataset.")
    
    data = pd.read_csv('data/{}_{}_{}.csv'.format(dataset, atlas, feature), 
                       sep=';', index_col=0)
    
    return data

def piecewise_linear(X, Y, feature):
    """
    Fits a continuous piecewise linear function on the (X, Y) data with the 
    Trust Region Reflective algorithm implemented in the scipy package. 
    Constraints on the parameters were defined, depending on the feature for 
    some of them: the tipping-point age is constrained to remain between 50 and
    75 years old to limit edge effects, and the slopes are to remain positive
    for the sulcal openings and negative for cortical thicknesses.
    
    Parameters
    ----------
    X : ARRAY
        The training input samples. Contains the age of the subjects.
    Y : ARRAY 
        The training output samples. Contains either the sulcal opening or the 
        cortical thickness of one sulcus.
    feature : STR 
        The considered feature. Possible options are 'opening' for sulcal 
        opening and 'thickness' for cortical thickness.
    
    Returns
    ----------
    a1 : FLOAT
        The slope of the first segment of the fitted piecewise linear model.
    b1 : FLOAT 
        The intercept of the first segment. 
    a2 : FLOAT 
        The slope of the second segment.
    b2 : FLOAT 
        The intercept of the second segment.
    error : FLOAT 
        The mean quadratic error between the fitted model and the training 
        output samples.
                    
    """
    assert feature in ['opening', 'thickness']
    
    def model(X, T, yT, a1, a2):
        b1, b2 = yT - a1*T, yT - a2*T
        return np.piecewise(X, [X<T], [lambda x : a1*x+b1, lambda x : a2*x+b2])
    
    if feature == 'opening':
        bounds_inf = [50, -np.inf, 0, 0]
        bounds_sup = [75, +np.inf, np.inf, np.inf]
    else:
        bounds_inf = [50, -np.inf, -np.inf, -np.inf]
        bounds_sup = [75, +np.inf, 0, 0]
        
    p, e = curve_fit(model, X, Y, bounds=(bounds_inf, bounds_sup))
    
    T, yT, a1, a2 = p
    b1, b2 = yT - a1*T, yT - a2*T
    Y_pred = np.array([a1*x+b1 if x<T else a2*x+b2 for x in X])
    error = mean_squared_error(Y, Y_pred)
    return a1, b1, a2, b2, error

def slope_change(a1, a2):
    """
    Computes the angle (in degrees) between two given slopes.
    
    Parameters
    ----------
    a1 : FLOAT 
        The first slope.
    a2 : FLOAT 
        The second slope.
        
    Returns
    ----------
    angle : FLOAT 
        The angle between the two slopes a1 and a2, in degrees.
    """
    angle = (np.arctan(a2)-np.arctan(a1)) / (np.pi/2) * 100
    return angle

def piecewise_linear_percentile(data, feature, sulci_names, percentile):
    """
    Fits a continuous two-segment piecewise linear model on a quantile line of 
    parcellated neuroanatomical features, with the age at imaging of the 
    subjects as the explanatory variable. To compute the mean squared error, 
    each quantile point was weighted with the number of subjects in the 
    matching age group. 
    The outputs include an array with all the parameters of the piecewise 
    linear models for each ROI, and a dictionary containing information on the
    bootstrap resample used for training.
    
    Parameters
    ----------
    data : DATAFRAME 
        The Dataframe filled with UK Biobank data, preferrably of only one 
        gender (M or F), as we model differently for men and women. Columns 
        must include "AGE" (age at imaging), then all the ROIs that are 
        present in 'sulci_names'. Should be a bootstrap resample of the 
        original UK Biobank data set.
    feature : STR 
        The considered neuroanatomical feature. Possible options are 'opening' 
        (for sulcal opening) and 'thickness' (for cortical thickness).
    sulci_names : LIST OF STR 
        The list of names of the ROIs on which to fit the piecewise linear 
        model.
    percentile : INT 
        The quantile line to consider. Must be between 0 and 100 exclusive.
    
    Returns
    ----------
    tab : DATAFRAME 
        The Dataframe filled with the parameters of the piecewise linear model 
        fitted on the data. Each line matches a ROI, and columns are 
        respectively: 
            - the slope of the first segment (a1)
            - the intercept of the first segment (b1)
            - the slope of the second segment (a2)
            - the intercept of the second segment (b2)
            - the age at breakpoint, or tipping-point age (T)
            - the angle between the two slopes (sc)
        Note that the last two values are in fact derived from the four 
        parameters of each model. The tipping-point age is theage at which the 
        two segments meet. The slope change is computed with the auxiliary 
        function slope_change.
    this_data_bootstrap : DICT 
        The dictionary filled with information on the data used for training. 
        For each ROI, called sulcus_name, this_data_bootstrap[sulcus_name] 
        yields a 2D array, of size (total_age_range, 2), where each line gives 
        an age and the value of the neuroanatomical feature at the wanted 
        percentile in this age group. The variable total_age_range is simply 
        the number of different ages available on the data.
            
            
    IMPORTANT: we consider that the parameter "percentile" as pertaining to 
    the sulcal opening. For example, if we target the 90th quantile line of the
    sulcal openings of the UK Biobank subjects, it is implied that we target 
    the at-risk subjects, those with larger sulcal openings. These same 
    subjects would have the thinnest cortex, which should match the 10th per-
    centile of the cortical thicknesses of the UK Biobank subjects. Thus, if 
    the feature considered is the cortical thickness, we invert the percentile.
    """
    assert 'AGE' in data.columns
    assert feature in ['thickness', 'opening']
    assert type(percentile) is int and 0 < percentile < 100
    
    tab = np.zeros((len(sulci_names), 6))
    this_data_bootstrap = dict()
    # inversion of the percentile if we consider the cortical thickness
    if feature == 'thickness':
        percentile = 100 - percentile
    for i, sulcus_name in enumerate(sulci_names):
        this_data_bootstrap_s = list()
        # during preprocessing, NaN values were replaced by 0
        mask_nan = data[sulcus_name] > 0.0
        X = data['AGE'][mask_nan].to_numpy()
        Y = data[sulcus_name][mask_nan].to_numpy()
        for x in set(X):
            value_for_this_age = np.percentile(Y[X==x], percentile)
            # weighting each age group by the number of the pertaining subjects
            # is equivalent to replacing their feature by the value at the 
            # percentile
            Y[X==x] = value_for_this_age
            this_data_bootstrap_s.append([x, value_for_this_age])
        this_data_bootstrap[sulcus_name] = np.array(this_data_bootstrap_s)
        a1, b1, a2, b2, _ = piecewise_linear(X, Y, feature)
        T = round((b2 - b1) / (a1 - a2), 2)
        sc = slope_change(a1, a2)
        tab[i] = a1, b1, a2, b2, T, sc
    
    # for storage reasons, we will have to convert the dictionary this_data_
    # bootstrap into a 3D array, so every 2D array that is 
    # this_data_bootstrap[sulcus_name] must be the same shape. To ensure this,
    # we add NaNs.
    total_age_range = max([this_data_bootstrap[sulcus_name].shape[0] for sulcus_name in sulci_names])
    for sulcus_name in sulci_names:
        nb_lines_to_add = total_age_range - this_data_bootstrap[sulcus_name].shape[0]
        if nb_lines_to_add > 0 :
            buffer = [[np.nan, np.nan]] * nb_lines_to_add
            this_data_bootstrap[sulcus_name] = np.append(this_data_bootstrap[sulcus_name], buffer, axis=0)
    return tab, this_data_bootstrap

def bootstrap(data, feature, sulci_names, foldername, frac_samples=0.8, n_bootstraps=10, percentile=90):
    """
    Runs a number of bootstrap resamples on which piecewise linear models are
    fitted for every ROI. The goal is to avoid overfitting and to compute 
    confidence intervals on each parameter. The resamples are created from the 
    original dataset (data). For each resample, we fit piecewise linear models 
    on the neuroanatomical features of every ROI. The resulting parameters are
    available in the output tabs, with the derived values, and info
    on every resample is available in data_bootstrap. During the process, the 
    data is saved under .txt and .npy format.
    
    Parameters
    ----------
    data : DATAFRAME 
        The Dataframe filled with UK Biobank data, preferrably of only one 
        gender (M or F), as we model differently for men and women. Columns 
        must include "AGE" (age at imaging), then all the ROIs that are 
        present in 'sulci_names'. 
    feature : STR 
        The considered neuroanatomical feature. Possible options are 'opening' 
        (for sulcal opening) and 'thickness' (for cortical thickness).
    sulci_names : LIST OF STR 
        The list of names of the ROIs on which to fit the piecewise linear model.
    foldername : STR 
        The path to which data is saved. Data includes, for each resample : 
            - the parameters of the piecewise linear modelling (in
            .txt format)
            - the 3D array (in .npy format) constituted of one 2D 
            array for each ROI (first dimension). Each 2D array is 
            of size (total_age_range, 2), where each line gives an 
            age and the value of the neuroanatomical feature at 
            the wanted percentile in this age group. The variable 
            total_age_range is simply the number of different ages 
            available on the data. 
    frac_samples : FLOAT 
        The fraction of subjects needed for the bootstrap, from which the size 
        of each resamples is computed. For each bootstrap sample, a sample with 
        replacement with this chosen size is drawn.
    n_bootstraps : INT 
        The number of bootstrap samples to draw.
    percentile : INT 
        The quantile line to consider. Must be between 0 and 100 exclusive.
    
    Returns
    ----------
    tabs : DICT 
        The dictionary filled with info on the parameters for each bootstrap 
        iteration. For example, tabs[i] gives the Dataframe tab filled with 
        the parameters of the piecewise linear model fitted on the data at the 
        i-th bootstrap iteration. See documentation of the function "piecewise_
        linear_percentile"", output "tab" for more info.
    data_bootstrap : DICT
        The dictionary filled with information on the data used for training 
        for each bootstrap iteration. For example, data_bootstrap[i] gives the 
        dictionary this_data_bootstrap which is described in detail in the 
        documentation of the function "piecewise_linear_percentile", among the 
        outputs.
    
    """
    assert 'AGE' in data.columns
    assert feature in ['thickness', 'opening']
    assert type(percentile) is int and 0 < percentile < 100
    assert 0 < frac_samples <= 1 
    assert type(n_bootstraps) is int and n_bootstraps > 0
    
    n_samples = int(frac_samples*data.shape[0])
    assert n_samples != 0
    tabs, data_bootstrap = dict(), dict()
    i = 0
    while i < n_bootstraps:
        sample = resample(data, n_samples=n_samples)
        while True:
            try:
                tab, data_bootstrap_i = piecewise_linear_percentile(sample, feature, sulci_names, percentile)
                break
            except RuntimeError: 
                # if the optimization algo fails, take a new sample
                sample = resample(data, n_samples=n_samples)
        if not np.isnan(tab).any():
            tabs[i] = tab
            data_bootstrap[i] = data_bootstrap_i
            i += 1
            np.savetxt(foldername + '/{}.txt'.format(i), 
                       tab, delimiter=',')
            np.save(foldername + '/{}.npy'.format(i), 
                    np.array([data_bootstrap_i[s] for s in sulci_names]))             
    return tabs, data_bootstrap

def display_linear_bootstrap(data, tabs, data_bootstrap, foldername, feature, 
                             atlas, percentile, gender=None, display=False):
    """
    Saves and optionally displays the plots with:
        - Data points of the specified quantile on the UK Biobank, including 
        the quantile variability observed during bootstrapping;
        - The plot of the piecewise linear model, with the parameters 
        computed during bootstrapping.


    Parameters
    ----------
    data : DATAFRAME
        The Dataframe filled with UK Biobank data, preferrably of only one 
        gender (M or F), as we model differently for men and women. Columns 
        must include "AGE" (age at imaging), then all the ROIs that are 
        present in 'sulci_names'. Should be a bootstrap resample of the 
        original UK Biobank data set.
    tabs : DICT 
        The dictionary filled with info on the parameters for each bootstrap 
        iteration. For example, tabs[i] gives the Dataframe tab filled with 
        the parameters of the piecewise linear model fitted on the data at the 
        i-th bootstrap iteration. See documentation of the function "piecewise_
        linear_percentile"", output "tab" for more info.
    data_bootstrap : DICT
        The dictionary filled with information on the data used for training 
        for each bootstrap iteration. For example, data_bootstrap[i] gives the 
        dictionary this_data_bootstrap which is described in detail in the 
        documentation of the function "piecewise_linear_percentile", among the 
        outputs.
    foldername : STR
        The path where to save the plots.
    feature : STR 
        The considered neuroanatomical feature. Possible options are 'opening' 
        (for sulcal opening) and 'thickness' (for cortical thickness).
    atlas : STR
        The atlas for the cerebral parcellation. Possible options are 
        'brainvisa' or 'desikan'.
    percentile : INT 
        The quantile line to consider. Must be between 0 and 100 exclusive.
    gender : STR, optional
        If the input "data" only contains one gender, this parameter can be 
        set to either "F" or "M" to add to the title of the plots. The default 
        is None.
    display : BOOL, optional
        Whether to display the plots while saving. The default is False.

    """
    assert 'AGE' in data.columns
    assert feature in ['thickness', 'opening']
    assert type(percentile) is int and 0 < percentile < 100
    assert atlas in ['brainvisa', 'desikan']
    assert type(display) is bool

    # Create adequate title ending for each plot
    title_ending = feature
    title_ending = "{} {}".format(atlas, title_ending)
    if gender is not None:
        assert gender in ['F', 'M']
        title_ending += " {}".format(gender)
        
    # Inversion of the percentile if we consider the cortical thickness
    if feature == 'thickness':
        percentile = 100 - percentile
    
    # Initialization
    sulci_names = np.array(data.columns[1:])
    n_bootstraps = len(data_bootstrap)
    tab = np.median(np.array([tabs[i] for i in range(n_bootstraps)]), axis=0)
    horizontal_line_width = 0.25
    color_segment = '#2187bb'
    color_data = '#f44336'
    if not display:
        plt.ioff()
    plt.rcParams.update({'font.size': 12})
    try:
        tab_ = np.array(tab[tab.columns[:-1]])
    except AttributeError:
        tab_ = tab
    
    # Plotting
    for i, sulcus_name in enumerate(sulci_names):
        title = "{} \n({})".format(sulcus_name_whole(sulcus_name, atlas), title_ending)
        a1, b1, a2, b2, T, _ = tab_[i]
        mask_nan = data[sulcus_name] > 0.0 
        X = data['AGE'][mask_nan].to_numpy()
        Y = data[sulcus_name][mask_nan].to_numpy()    
        T = (b2 - b1) / (a1 - a2)
        X_model = np.linspace(min(X), max(X), 100)
        Y_model = np.array([a1*x+b1 if x<T else a2*x+b2 for x in X_model])
        fig, ax1 = plt.subplots()
        
        for x in set(X):
            xs_ys_bootstrap = [data_bootstrap[b][sulcus_name][data_bootstrap[b][sulcus_name][:, 0]==x] for b in range(n_bootstraps)]
            ys_bootstrap = [el[0, 1] for el in xs_ys_bootstrap if el.size>0]
            med = np.percentile(ys_bootstrap, 50)
            top = np.percentile(ys_bootstrap, 95)
            bottom = np.percentile(ys_bootstrap, 5)
            left = x - horizontal_line_width / 2
            right = x + horizontal_line_width / 2
            plt.plot([x, x], [top, bottom], color=color_segment)
            plt.plot([left, right], [top, top], color=color_segment)
            plt.plot([left, right], [bottom, bottom], color=color_segment)
            plt.plot(x, med, '+', color=color_segment)
            plt.plot(x, np.percentile(Y[X==x], percentile), 'o', color=color_data)
        plt.plot(X_model, Y_model, c='black', linewidth=3)
        plt.xlabel("Age at imaging (years)")
        if feature == 'opening':
            plt.ylabel("Sulcal opening (mm)")
        else:
            plt.ylabel('Cortical thickness (mm)')
        # Adding a histogram of the subjects' age distribution in overlay
        ax2 = ax1.twinx()
        ax2.hist(X, np.arange(min(X)-0.5, max(X)+0.5), density=True, label='number of subjects', alpha=0.5)
        plt.plot(np.nan, np.nan, 'o', color=color_data, label='{}th percentile'.format(percentile))
        ax2.set_ylabel('Population density')
        plt.title(title)
        plt.legend()        
        plt.savefig(foldername + '/Figure{}.png'.format(i+1))
        plt.tight_layout()
        if not display:
            plt.close()
    if not display:
        plt.ion()
    return None

def extract_models_from_csv(sulci_names, n_bootstraps, foldername):
    """
    Retrieves both dictionaries with info on each bootstrap iteration from the
    files saved during the process.
    
    Parameters
    ----------
    sulci_names : LIST OF STR 
        The list of names of the ROIs on which the piecewise linear models were 
        fitted.
    n_bootstraps : INT 
        The number of bootstraps that were drawn.
    foldername : STR 
        The path where the .txt and the .npy files are saved, under the 
        format "iteration number".txt and "iteration number".npy.
    
    Returns
    ----------
    tabs : DICT 
        The dictionary filled with info on the parameters for each bootstrap 
        iteration. For example, tabs[i] gives the Dataframe tab filled with the 
        parameters of the piecewise linear model fitted on the data at the i-th 
        bootstrap iteration. See documentation of the function "piecewise_
        linear_percentile"", output "tab" for more info.
    data_bootstrap : DICT 
        The dictionary filled with information on the data used for training 
        for each bootstrap iteration. For example, data_bootstrap[i] gives the 
        dictionary this_data_bootstrap which is described in detail in the 
        documentation of the function "piecewise_linear_percentile"", among 
        the outputs.
    """
    assert type(n_bootstraps) is int and n_bootstraps > 0 
    
    tabs = dict()
    data_bootstrap = dict()
    for i in range(n_bootstraps):
        tabs[i] = np.loadtxt(foldername + '/{}.txt'.format(i+1), delimiter=',')
        d = np.load(foldername + '/{}.npy'.format(i+1))
        data_bootstrap[i] = {sulcus_name: d[s] for s, sulcus_name in enumerate(sulci_names)}
    return tabs, data_bootstrap

def save_coeff(col_names, sulci_names_, coeff, filename):
    """
    Saves the model parameters of each sulcus in a csv, in a format accepted by 
    the Morphologist visualization software. The header (first line) is: "sulci" 
    followed by the names of the model parameters (e.g. 'a1', 'b1', etc). The 
    other lines will be comprised of the name of the sulcus followed by the 
    values of the model parameters, in the order listed by the header.
    
    Parameters
    ----------
    col_names : LIST OF STR 
        The names of the model parameters.
    sulci_names_ : LIST OF STR 
        The names of the sulci.
    coeff : ARRAY 
        The array containing the values of the model parameters, of size (number 
        of sulci, number of model parameters). In other words, each line of 
        coeff contains the model parameters of one sulcus.
    filename : STR 
        The path where to save the resulting csv. Please note that it must 
        include the name of the file you want to save, and it must end with 'csv'.
    """
    sulci_names_ = sulci_names_.reshape(-1, 1)
    if len(coeff.shape) == 1:
        coeff = coeff.reshape(-1, 1)
    txt = np.concatenate((sulci_names_, coeff), axis=1)
    header = ['sulci'] + col_names
    header = ' '.join(header)
    np.savetxt(filename, txt, fmt=['%s']+['%.10f' for _ in range(len(col_names))], 
               header=header, comments='')  
    return None

def process(output_dir, feature, dataset, atlas, n_bootstraps, percentile, gender=None):
    """
    Allows to run the entire process on a given dataset, on one percentile or 
    a list of percentiles. It entails, for each percentile: 
        - the bootstrap resampling
        - the fitting of the piecewise linear models 
        - the adequate plots.
    

    Parameters
    ----------
    output_dir : STR
        The path of the folder to which to save everything.
    feature : STR
        The neuroanatomical feature to extract. Possible options are 'opening'
        or 'thickness'.
    dataset : STR
        The dataset. Possible options are 'UKB' for UK Biobank, or 'Memento' 
        for Memento.
    atlas : STR
        The atlas for the cerebral parcellation. Possible options are 
        'brainvisa' or 'desikan'.
    n_bootstraps : INT 
        The number of bootstrap samples to draw.
    percentile : INT or LIST OF INT
        The quantile line(s) to consider. Must be between 0 and 100 exclusive.
    gender : STR, optional
        If the input "data" only contains one gender, this parameter can be 
        set to either "F" or "M" to add to the title of the plots. The default 
        is None.

    """
    assert feature in ['thickness', 'opening']
    assert dataset in ['UKB', 'memento']
    assert atlas in ['brainvisa', 'desikan']
    assert type(n_bootstraps) is int and n_bootstraps > 0
        
    if type(percentile) is int:
        percentile = [percentile]
    for i in range(len(percentile)):
        assert type(percentile[i]) is int and 0 < percentile[i] < 100
    
    if gender is not None:
        assert gender in ['F', 'M']
        
    
    # Create folder. Example for output_dir="root", dataset="UKB", gender="F", 
    # feature="opening", atlas="brainvisa":
    # root/MODELLING/dataset (F)/opening (brainvisa)
    output_dir0 = output_dir + "/MODELLING"
    os.makedirs(output_dir0, exist_ok=True)
    output_dir0 += "/{} ".format(dataset)
    if gender is not None:
        output_dir0 += "({})".format(gender)
    os.makedirs(output_dir0, exist_ok=True)
    output_dir0 += "/{} ({})".format(feature, atlas)
    os.makedirs(output_dir0, exist_ok=True)

    data = read_sulci(feature, atlas, dataset)
    if gender is not None:
        if gender == "F":
            index_gender = 0
        else:
            index_gender = 1
        data = data[data.SEX==index_gender]
    data.pop('SEX')
    sulci_names = np.array(data.drop(columns='AGE').columns)
    data.sort_values('AGE', inplace=True)
    
    # Run the bootstrap
    for p in percentile:
        foldername = output_dir0 + "/centile{}".format(p)
        assert not os.path.exists(foldername), "The folder {} already exists. Please delete it or change its name to avoid accidental overwriting.".format(foldername)
        os.makedirs(foldername)
        # Training and saving the trained data
        tabs, data_bootstrap = bootstrap(data, feature, sulci_names, foldername, 0.8, n_bootstraps, p)
        # Retrieve the trained data
        tabs, data_bootstrap = extract_models_from_csv(sulci_names, n_bootstraps, foldername)
        # Save the plots
        plt.ioff()
        display_linear_bootstrap(data, tabs, data_bootstrap, foldername, feature, atlas, p, gender, False)
        plt.ion()
        # Compute the final values of the parameters
        tab_final = np.median(np.array([tabs[i] for i in range(n_bootstraps)]), axis=0)
        # Computing the standard deviation of the tipping-point age during bootstrapping
        std_boot = np.array([np.std([tabs[j][s][4] for j in range(n_bootstraps)]) for s in range(len(sulci_names))]).reshape(-1, 1)
        tab_final = np.concatenate((tab_final, std_boot), axis=1)
        # Computing the tipping-point age 
        tab_final[:, 4] = (tab_final[:, 3] - tab_final[:, 1]) / (tab_final[:, 0] - tab_final[:, 2])
        # Computing the slope change
        tab_final[:, 5] = slope_change(tab_final[:, 0], tab_final[:, 2])
        
        # Save the final values of the parameters in an adequate format
        save_coeff(['a1', 'b1', 'a2', 'b2', 'T', 'slope_change', 'std'], sulci_names, tab_final, foldername + '/tab_centile{}.csv'.format(p))
        
    return None



if __name__ == "__main__":
    output_dir = "."
    # Example 1 : training on the 90th percentile of the Brainvisa sulcal opening, on UKB women
    process(output_dir, "opening", "UKB", "brainvisa", 2, 5, "F")
    
    # Example 2 : training on the 80th and the 45th percentiles of the Desikan thickness, on UKB men
    process(output_dir, "thickness", "UKB", "desikan", 2, [20, 55], "M")
    
